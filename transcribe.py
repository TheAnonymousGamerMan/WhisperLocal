"""
Local speech-to-text using OpenAI's pretrained Whisper model -- runs
entirely on your own machine (GPU if you have one), no API calls, no
per-minute cost. This is meant as a point of comparison against the
from-scratch VoiceAI project: Whisper was pretrained on 680,000+ hours of
audio, which is why it generalizes to novel/casual speech in a way a
model trained from scratch on a couple hours of your own voice can't yet.

Run:
    python transcribe.py                        # persistent session, keypress start/stop
    python transcribe.py --once                  # record + transcribe once, then exit
    python transcribe.py --seconds 6              # fixed 6s recordings instead of keypress
    python transcribe.py --file clip.wav           # transcribe one file, then exit
    python transcribe.py --output note.txt          # always overwrite this file with the latest result
    python transcribe.py --model medium              # use a bigger/more accurate model (see sizes below)
    python transcribe.py --list-devices               # show every mic sounddevice can see
    python transcribe.py --input-device 3               # force mic index 3 instead of the system default

Model sizes (speed vs. accuracy trade-off -- all run fine on an RTX 3080):
    tiny    fastest,  least accurate  (~1GB VRAM)
    base    fast,     decent          (~1GB VRAM)   <- default
    small   good balance              (~2GB VRAM)
    medium  quite accurate            (~5GB VRAM)
    large-v3 most accurate, slowest   (~10GB VRAM)

Every transcription is also appended, timestamped, to transcripts.txt in
this folder, same as VoiceAI does.
"""
import argparse
from datetime import datetime
from pathlib import Path

import torch
import whisper

from keypress import read_key
from record import record_from_mic, record_until_keypress, load_audio_file, list_input_devices, parse_device_arg

HERE = Path(__file__).resolve().parent
TRANSCRIPT_LOG = HERE / "transcripts.txt"


def log_and_save(text, source, output_path):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    TRANSCRIPT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(TRANSCRIPT_LOG, "a", encoding="utf-8") as log_file:
        log_file.write(f"[{timestamp}] ({source}) {text}\n")

    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="base",
                         help="Whisper model size: tiny/base/small/medium/large-v3 (default: base).")
    parser.add_argument("--language", type=str, default="en",
                         help="Force this language instead of auto-detecting (default: en). Pass '' to auto-detect.")
    parser.add_argument("--file", type=str, default=None,
                         help="Transcribe this one audio file, then exit.")
    parser.add_argument("--seconds", type=float, default=None,
                         help="Record fixed-length clips instead of press-a-key start/stop.")
    parser.add_argument("--once", action="store_true",
                         help="Record/transcribe a single time then exit, instead of looping.")
    parser.add_argument("--output", "-o", type=str, default=None,
                         help="Always overwrite this .txt file with the latest transcription.")
    parser.add_argument("--input-device", type=str, default=None,
                         help="Mic to record from: an index or a name/substring from --list-devices.")
    parser.add_argument("--list-devices", action="store_true",
                         help="Print every audio device sounddevice can see, then exit.")
    args = parser.parse_args()

    if args.list_devices:
        list_input_devices()
        return

    input_device = parse_device_arg(args.input_device)
    language = args.language if args.language else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Loading Whisper model '{args.model}' on {device}...")
    model = whisper.load_model(args.model, device=device)
    print("Model loaded.\n")

    fp16 = device == "cuda"

    def transcribe_waveform(waveform):
        result = model.transcribe(waveform, language=language, fp16=fp16)
        return result["text"].strip()

    # --file: one-shot, no live loop.
    if args.file:
        waveform, _ = load_audio_file(args.file)
        text = transcribe_waveform(waveform)
        print(f"Transcription: {text}")
        log_and_save(text, source=args.file, output_path=args.output)
        return

    # Live mic: loads the model once, then keeps transcribing until you quit.
    while True:
        if args.seconds is not None:
            key = read_key(f"Press any key to record {args.seconds:.1f}s (or 'q' to quit): ")
            if key.lower() == "q":
                break
            waveform = record_from_mic(duration_sec=args.seconds, device=input_device)
            source = f"microphone ({args.seconds:.1f}s)"
        else:
            key = read_key("Press any key to start recording (or 'q' to quit): ")
            if key.lower() == "q":
                break
            waveform = record_until_keypress(stop_prompt="Recording... press any key to stop.", device=input_device)
            source = "microphone (keypress)"

        if waveform.size == 0:
            print("No audio captured.\n")
            continue

        text = transcribe_waveform(waveform)
        print(f"Transcription: {text}\n")
        log_and_save(text, source=source, output_path=args.output)

        if args.once:
            break

    print("Session ended.")


if __name__ == "__main__":
    main()
