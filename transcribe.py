"""
Local speech-to-text using OpenAI's pretrained Whisper model -- runs
entirely on your own machine (GPU if you have one), no API calls, no
per-minute cost. This is meant as a point of comparison against the
from-scratch VoiceAI project: Whisper was pretrained on 680,000+ hours of
audio, which is why it generalizes to novel/casual speech in a way a
model trained from scratch on a couple hours of your own voice can't yet.

Two ways to run it:

1. Discrete recording (original mode) -- press a key, talk, press again:
    python transcribe.py                        # persistent session, keypress start/stop
    python transcribe.py --once                  # record + transcribe once, then exit
    python transcribe.py --seconds 6              # fixed 6s recordings instead of keypress
    python transcribe.py --file clip.wav           # transcribe one file, then exit

2. Live mode -- continuous listening with automatic speech detection, no
   keypress per utterance, good for monitoring an ongoing conversation
   (e.g. a Discord call). By default captures your mic; add --loopback to
   also capture system audio (whatever's playing -- Discord's voices,
   etc, via Windows WASAPI loopback, no virtual cable needed):
    python transcribe.py --live                    # continuous, mic only
    python transcribe.py --live --loopback           # continuous, mic AND system audio (Discord + you)
    python transcribe.py --live --loopback --no-mic    # continuous, system audio only
    python transcribe.py --list-output-devices           # see output devices for --loopback-device
    python transcribe.py --live --loopback --loopback-device 5   # pin a specific output device

   If you route audio through a virtual mixer (Voicemeeter, VB-Cable),
   --loopback often can't tap it -- WASAPI loopback frequently doesn't
   work on virtual playback devices even though the mixer itself shows
   audio arriving fine. Use --discord-device instead, pointed at the
   mixer's corresponding recording device -- free Voicemeeter calls it
   "Voicemeeter Output"; Banana/Potato instead expose "Voicemeeter Out
   B1"/"B2"/"B3" (whichever bus your Voicemeeter Input strip is routed
   to). That name can appear multiple times in --list-devices (once per
   Windows audio API), so pass the numeric index, not the name:
    python transcribe.py --list-devices                # find the right device and note its index
    python transcribe.py --live --discord-device 2       # mic AND that device, together (index example)

   In live mode, press any key (or Ctrl+C) to stop.

Other options:
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
import queue
import threading
from datetime import datetime
from pathlib import Path

import torch
import whisper

from keypress import read_key
from record import (
    record_from_mic, record_until_keypress, load_audio_file,
    list_input_devices, list_output_devices, parse_device_arg, live_vad_segments,
    test_loopback_capture,
)
import speaker_id

HERE = Path(__file__).resolve().parent
TRANSCRIPT_LOG = HERE / "transcripts.txt"


def log_and_save(text, source, output_path, speaker_label=None, speaker_sim=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    TRANSCRIPT_LOG.parent.mkdir(parents=True, exist_ok=True)
    tag = f" speaker={speaker_label} ({speaker_sim:.0%})" if speaker_label else ""
    with open(TRANSCRIPT_LOG, "a", encoding="utf-8") as log_file:
        log_file.write(f"[{timestamp}] ({source}){tag} {text}\n")

    if output_path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        prefix = f"[{speaker_label}] " if speaker_label else ""
        out_path.write_text(prefix + text, encoding="utf-8")


def _capture_worker(label, device, loopback, stop_event, result_queue, vad_kwargs):
    """Runs in a background thread: continuously listens on one source
    (mic or loopback) and pushes (label, waveform) onto result_queue for
    each detected speech segment. Errors (e.g. loopback unsupported) are
    reported and this source just stops, without killing the other one."""
    try:
        for waveform in live_vad_segments(device=device, loopback=loopback, stop_event=stop_event, **vad_kwargs):
            result_queue.put((label, waveform))
    except Exception as exc:
        print(f"\n[{label}] capture stopped due to an error: {exc}\n")


def _stop_listener(stop_event):
    """Background thread: blocks on a keypress, then signals stop_event
    so all capture threads and the main loop wind down."""
    read_key("Press any key to stop listening...\n")
    stop_event.set()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="base",
                         help="Whisper model size: tiny/base/small/medium/large-v3 (default: base).")
    parser.add_argument("--language", type=str, default="en",
                         help="Force this language instead of auto-detecting (default: en). Pass '' to auto-detect.")
    parser.add_argument("--file", type=str, default=None,
                         help="Transcribe this one audio file, then exit.")
    parser.add_argument("--seconds", type=float, default=None,
                         help="Record fixed-length clips instead of press-a-key start/stop (discrete mode only).")
    parser.add_argument("--once", action="store_true",
                         help="Record/transcribe a single time then exit, instead of looping (discrete mode only).")
    parser.add_argument("--output", "-o", type=str, default=None,
                         help="Always overwrite this .txt file with the latest transcription.")
    parser.add_argument("--input-device", type=str, default=None,
                         help="Mic to record from: an index or a name/substring from --list-devices.")
    parser.add_argument("--list-devices", action="store_true",
                         help="Print every audio (input) device sounddevice can see, then exit.")

    parser.add_argument("--live", action="store_true",
                         help="Continuous listening with automatic speech detection -- no keypress per "
                                "utterance. Press any key (or Ctrl+C) to stop.")
    parser.add_argument("--loopback", action="store_true",
                         help="In --live mode, also capture system audio (e.g. a Discord call) via WASAPI "
                                "loopback, alongside your mic (unless --no-mic).")
    parser.add_argument("--no-mic", action="store_true",
                         help="In --live mode, don't capture the mic -- only useful together with --loopback.")
    parser.add_argument("--loopback-device", type=str, default=None,
                         help="Output device to loop back: an index or name/substring from --list-output-devices. "
                                "Defaults to the system default output device.")
    parser.add_argument("--list-output-devices", action="store_true",
                         help="Print every output-capable device (for --loopback-device), then exit.")
    parser.add_argument("--loopback-test", type=float, default=None, metavar="SECONDS",
                         help="Debug --loopback: record SECONDS of raw loopback audio (no VAD, no Whisper), "
                                "report peak volume, then exit. Use this if --live --loopback isn't picking up "
                                "Discord -- it tells you whether audio is even arriving, separate from VAD tuning.")
    parser.add_argument("--discord-device", type=str, default=None,
                         help="Capture the 'discord' source as a normal input device (an index from "
                                "--list-devices, preferred since names can repeat across audio APIs) instead of "
                                "WASAPI loopback -- use this if you route audio through a virtual mixer like "
                                "Voicemeeter or VB-Cable (WASAPI loopback often doesn't work on those virtual "
                                "devices, but they provide a real recording device -- e.g. 'Voicemeeter Output', "
                                "or 'Voicemeeter Out B1' on Banana/Potato -- that carries the same audio). "
                                "Overrides --loopback.")
    parser.add_argument("--vad-threshold", type=float, default=0.02,
                         help="Live mode: RMS energy level that counts as 'someone's talking' (default: 0.02).")
    parser.add_argument("--vad-silence", type=float, default=0.8,
                         help="Live mode: seconds of quiet before a speech segment is considered finished (default: 0.8).")
    parser.add_argument("--min-speech", type=float, default=0.4,
                         help="Live mode: minimum seconds of actual speech to bother transcribing (default: 0.4) -- "
                                "filters out coughs/clicks.")
    args = parser.parse_args()

    if args.list_devices:
        list_input_devices()
        return
    if args.list_output_devices:
        list_output_devices()
        return

    loopback_device = parse_device_arg(args.loopback_device)

    if args.loopback_test is not None:
        test_loopback_capture(duration_sec=args.loopback_test, device=loopback_device,
                               save_path=HERE / "loopback_test.wav")
        return

    input_device = parse_device_arg(args.input_device)
    language = args.language if args.language else None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Loading Whisper model '{args.model}' on {device}...")
    model = whisper.load_model(args.model, device=device)
    print("Model loaded.\n")

    # A separate classifier per capture source ("mic" vs "discord"), built
    # from speakers scoped to that source (see speaker_id.SOURCES / the
    # --source flag on enroll.py) plus anyone scoped to "both". This means
    # e.g. someone who's only ever enrolled for Discord is never compared
    # against (or mistaken for) someone who's only ever on the mic, and
    # it's less computation per segment than one combined classifier once
    # more people are enrolled. "both" is the unfiltered fallback used for
    # --file, where there's no capture source to scope by.
    speaker_profiles = speaker_id.load_profiles()
    classifiers = {
        "both": speaker_id.train_classifier(speaker_profiles),
        "mic": speaker_id.train_classifier(speaker_id.filter_profiles_by_source(speaker_profiles, "mic")),
        "discord": speaker_id.train_classifier(speaker_id.filter_profiles_by_source(speaker_profiles, "discord")),
    }
    if speaker_profiles:
        for kind in ("mic", "discord"):
            scoped = speaker_id.filter_profiles_by_source(speaker_profiles, kind)
            if classifiers[kind]:
                print(f"Speaker ID ({kind}): on ({', '.join(classifiers[kind]['speakers'])})")
            elif scoped:
                print(f"Speaker ID ({kind}): only '{', '.join(scoped.keys())}' enrolled for this source -- "
                      "need at least 2 to tell apart. Run enroll.py to add another.")
            else:
                print(f"Speaker ID ({kind}): nobody enrolled for this source yet.")
        print()
    else:
        print("Speaker ID: no speakers enrolled yet -- run enroll.py to add some. Transcribing without speaker tags for now.\n")

    def identify_speaker(waveform, kind="both"):
        classifier = classifiers.get(kind)
        if not classifier:
            return None, None
        speaker_label, confidence, _ = speaker_id.identify(waveform, 16000, classifier=classifier)
        return speaker_label, confidence

    fp16 = device == "cuda"

    def transcribe_waveform(waveform):
        result = model.transcribe(waveform, language=language, fp16=fp16)
        return result["text"].strip()

    def report(text, source, speaker_label, speaker_sim):
        if speaker_label:
            print(f"[{source}] Transcription: [{speaker_label}] {text}  ({speaker_sim:.0%} confident)\n")
        else:
            print(f"[{source}] Transcription: {text}\n")
        log_and_save(text, source=source, output_path=args.output,
                     speaker_label=speaker_label, speaker_sim=speaker_sim)

    # --file: one-shot, no live loop.
    if args.file:
        waveform, _ = load_audio_file(args.file)
        text = transcribe_waveform(waveform)
        speaker_label, speaker_sim = identify_speaker(waveform)
        report(text, args.file, speaker_label, speaker_sim)
        return

    if args.live:
        discord_device = parse_device_arg(args.discord_device)
        has_discord_source = args.loopback or discord_device is not None
        if args.no_mic and not has_discord_source:
            print("--no-mic with no --loopback/--discord-device means nothing would be captured. "
                  "Add one of those, or drop --no-mic.")
            return

        vad_kwargs = dict(
            energy_threshold=args.vad_threshold,
            silence_duration=args.vad_silence,
            min_speech_duration=args.min_speech,
        )

        stop_event = threading.Event()
        result_queue = queue.Queue()
        threads = []

        if not args.no_mic:
            t = threading.Thread(
                target=_capture_worker,
                args=("mic", input_device, False, stop_event, result_queue, vad_kwargs),
                daemon=True,
            )
            threads.append(t)
        if discord_device is not None:
            # --discord-device: a normal input device (e.g. Voicemeeter Output),
            # captured exactly like the mic -- no WASAPI loopback involved.
            t = threading.Thread(
                target=_capture_worker,
                args=("discord", discord_device, False, stop_event, result_queue, vad_kwargs),
                daemon=True,
            )
            threads.append(t)
        elif args.loopback:
            t = threading.Thread(
                target=_capture_worker,
                args=("discord", loopback_device, True, stop_event, result_queue, vad_kwargs),
                daemon=True,
            )
            threads.append(t)

        for t in threads:
            t.start()

        stop_thread = threading.Thread(target=_stop_listener, args=(stop_event,), daemon=True)
        stop_thread.start()

        print("Live session started. Listening...\n")
        try:
            while not stop_event.is_set() or not result_queue.empty():
                try:
                    source, waveform = result_queue.get(timeout=0.3)
                except queue.Empty:
                    continue
                if waveform.size == 0:
                    continue
                text = transcribe_waveform(waveform)
                if not text:
                    continue  # VAD caught something but Whisper heard nothing worth transcribing
                speaker_label, speaker_sim = identify_speaker(waveform, kind=source)
                report(text, source, speaker_label, speaker_sim)
        except KeyboardInterrupt:
            stop_event.set()

        for t in threads:
            t.join(timeout=2.0)
        print("Session ended.")
        return

    # Discrete mode: loads the model once, then keeps transcribing until you quit.
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
        speaker_label, speaker_sim = identify_speaker(waveform, kind="mic")
        report(text, source, speaker_label, speaker_sim)

        if args.once:
            break

    print("Session ended.")


if __name__ == "__main__":
    main()
