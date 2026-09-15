# WhisperLocal

Local speech-to-text using OpenAI's pretrained **Whisper** model, running
entirely on your own machine -- no API calls, no per-minute cost, nothing
sent anywhere. This is a companion to `VoiceAI` (the from-scratch model)
so you can compare: Whisper was pretrained on 680,000+ hours of audio,
which is why it should handle casual, off-script speech (like "I like
playing video games") far better than a model trained on a couple hours
of your own voice from scratch.

The open-source Whisper model itself is free (MIT licensed) -- you only
pay if you use OpenAI's *hosted* Whisper API instead of running it
yourself, which is what this project avoids entirely.

## Setup

```
cd WhisperLocal
pip install -r requirements.txt
```

That installs `openai-whisper`, which needs `torch` (you already have
this from VoiceAI) and downloads model weights the first time you run
each size. No `ffmpeg` install needed -- this project captures/loads
audio itself and hands Whisper a ready-made array, sidestepping Whisper's
usual ffmpeg dependency.

The first time you use a given `--model` size, it'll download the weights
(cached afterward) -- expect a short pause and some disk usage the first
run only.

## Transcribe

```
python transcribe.py
```

Same UX as VoiceAI's `transcribe.py`: loads the model once, then press
any key to start recording, press any key again to stop, prints the
transcription, and loops -- press `q` at the start prompt to quit.

Other options:
```
python transcribe.py --once                  # record/transcribe a single time then exit
python transcribe.py --seconds 6              # fixed 6s recordings instead of keypress start/stop
python transcribe.py --file path/to/clip.wav   # transcribe one existing file, then exit
python transcribe.py --output note.txt          # always overwrite this file with the latest result
python transcribe.py --model medium              # bigger/more accurate model (see sizes below)
python transcribe.py --list-devices               # show every mic sounddevice can see
python transcribe.py --input-device 3               # force mic index 3 instead of the system default
python transcribe.py --language ""                    # auto-detect language instead of forcing English
```

Every transcription is appended, timestamped, to `transcripts.txt` in
this folder.

## Model sizes

All of these will run fine on your RTX 3080 (10GB VRAM):

| Model      | Speed        | Accuracy         | Rough VRAM |
|------------|--------------|------------------|------------|
| `tiny`     | fastest      | least accurate   | ~1GB       |
| `base`     | fast         | decent (default) | ~1GB       |
| `small`    | good balance | good             | ~2GB       |
| `medium`   | slower       | quite accurate   | ~5GB       |
| `large-v3` | slowest      | most accurate    | ~10GB      |

Start with the default (`base`) to confirm everything works, then try
`--model small` or `--model medium` -- for a live-mic assistant, `small`
or `medium` are usually the sweet spot between speed and accuracy.

## Notes

- Same mic caveat as VoiceAI: this records from whatever Windows calls
  the default input device. If that's a virtual/routed device
  (VoiceMeeter, OBS, a Discord virtual cable) rather than your physical
  mic, check `--list-devices` and pin the right one with `--input-device`
  if needed.
- No speaker identification here -- that was a VoiceAI-specific feature
  you built by hand. Whisper doesn't do that out of the box (that's a
  separate "diarization" problem).
- `--language en` is the default so it skips language auto-detection and
  transcribes a little faster; pass `--language ""` if you want to speak
  other languages.
