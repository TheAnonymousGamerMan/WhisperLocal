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

## Live mode (Discord calls, or any continuous conversation)

The default mode above is discrete: press a key, talk, press again. For
something like a Discord call, that's awkward -- you want it just
listening continuously and cutting up the audio into segments on its
own. `--live` does that: it listens non-stop, automatically detects when
someone starts/stops talking (voice-activity detection, energy-based),
and transcribes each chunk of speech as soon as it ends, with no keys to
press mid-conversation.

```
python transcribe.py --live --loopback
```

This is the command for "transcribe a Discord call": `--loopback`
captures your speakers/headphones output (i.e. what Discord is playing
-- the other person) via Windows' WASAPI loopback, no virtual audio
cable software needed. This part uses the `soundcard` package rather
than `sounddevice` -- PortAudio (what `sounddevice` wraps) doesn't
actually expose WASAPI loopback, so if you set this project up before
this feature was added, run `pip install -r requirements.txt` again to
pick up `soundcard`. **Your own mic is captured at the same time by
default** -- `--live` alone always includes the mic unless you pass
`--no-mic`, so `--live --loopback` gets both sides of the call
simultaneously, each on its own background thread, tagged in the output
so you can tell them apart:

```
[mic] Transcription: [zach] yeah that sounds good to me  (91% confident)
[discord] Transcription: [elliott] okay let's do that then  (88% confident)
```

**If you route your audio through a virtual mixer** (Voicemeeter,
VB-Cable, or similar) rather than playing straight out of a normal
Windows output device, use `--discord-device` instead of `--loopback`.
WASAPI loopback frequently just doesn't work on virtual playback
devices -- it silently captures nothing even though Voicemeeter's own
meters show audio arriving fine -- `--loopback-test` (below) confirms
this if you hit it. Virtual mixers get around this by also exposing a
separate, real *recording* device that carries the exact same audio,
meant for exactly this purpose. Point `--discord-device` at it and it's
captured the same reliable way the mic is, no loopback involved at all:

```
python transcribe.py --list-devices
python transcribe.py --live --discord-device 2       # index from --list-devices, mic AND that device together
```

What to look for in `--list-devices` depends on your Voicemeeter
edition:
- Basic (free) Voicemeeter: a device literally named "Voicemeeter
  Output".
- Voicemeeter Banana/Potato: no plain "Voicemeeter Output" -- instead
  look for "Voicemeeter Out B1" / "B2" / "B3". Which one carries your
  audio depends on which B-bus button is lit on the "Voicemeeter Input"
  strip inside the Voicemeeter app itself (B1 is the common default) --
  try B1 first, and B2/B3 if that's silent.
- The same name can appear several times in `--list-devices` (once per
  Windows audio API -- MME, DirectSound, WASAPI, WDM-KS), which makes it
  ambiguous to pass by name. **Use the numeric index shown on the left**
  instead of typing the name.

`--discord-device` takes an index or a name substring, same as
`--input-device`, and overrides `--loopback` if both are given.

Other live-mode flags:
```
python transcribe.py --live                          # mic only, no Discord/system audio
python transcribe.py --live --loopback --no-mic       # Discord/system audio only, skip your mic
python transcribe.py --list-output-devices             # list output devices, to pick one for --loopback-device
python transcribe.py --live --loopback --loopback-device 5   # capture a specific output device instead of the system default
python transcribe.py --live --discord-device 2 --no-mic          # virtual-mixer audio only (by index), skip mic
python transcribe.py --live --vad-threshold 0.03        # require louder audio to count as "speech" (less sensitive)
python transcribe.py --live --vad-silence 1.2             # wait longer before cutting a segment on silence
python transcribe.py --live --min-speech 0.3                # allow shorter utterances through (default 0.4s)
python transcribe.py --loopback-test 5                          # debug --loopback: record 5s raw, report peak volume, exit
```

Press any key (or Ctrl+C) to stop a live session; it finishes
transcribing whatever's still queued up before exiting.

Notes specific to live mode:
- `--loopback` defaults to your system's default *output* device (i.e.
  whatever your speakers/headphones normally play). If Discord is
  routed somewhere else (a specific headset, a virtual cable), run
  `--list-output-devices` and pass the right index with
  `--loopback-device`. Note `--list-output-devices` indices come from
  `soundcard` and `--list-devices` indices come from `sounddevice` --
  they're separate numbering, don't mix them up.
- If it's cutting off the ends of words or splitting one sentence into
  two, raise `--vad-silence` (waits longer before deciding someone
  stopped talking). If it's picking up background noise as speech,
  raise `--vad-threshold`.
- Speaker ID still runs per-segment same as discrete mode, so as long as
  you and whoever you're talking to are both enrolled (see below), each
  transcribed line gets tagged with who said it, on top of which source
  (`mic` vs `discord`) it came from.
- If `--live --loopback` isn't picking up Discord at all, run
  `python transcribe.py --loopback-test 5` -- it records 5 seconds of
  raw loopback audio (no VAD, no Whisper) and reports the peak volume,
  plus saves `loopback_test.wav` so you can listen back. A near-zero
  peak has two common causes: Discord (or Windows) has that app's
  output routed to a *different* device than the one `--loopback` is
  listening to (check Windows Settings > System > Sound > Volume mixer
  and Discord's own Settings > Voice & Video > Output Device, then
  match `--loopback-device` to it), or the device is a virtual mixer
  (Voicemeeter, VB-Cable) that WASAPI loopback can't tap at all -- use
  `--discord-device` instead, as above. A healthy peak but still no
  live transcriptions points at `--vad-threshold` being too high
  instead.

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

## Speaker identification

Separate from Whisper entirely, and built from scratch (no cloud API, no
pretrained speaker model) -- see `speaker_id.py` for the approach. It
needs far less data than transcription because it's a much simpler
problem: instead of learning an open-ended mapping from sound to
arbitrary text, it just has to tell a small, fixed set of enrolled voices
apart using traits (pitch, timbre/spectral shape) that stay fairly
constant no matter what someone says.

Two stages: each clip gets turned into a compact acoustic fingerprint
(mel-band statistics + pitch statistics, computed by hand same as
VoiceAI's features), then a small softmax regression classifier --
`logits = X @ W + b`, trained from scratch via gradient descent (manual
forward/backward pass, no autodiff framework) -- learns to tell the
enrolled fingerprints apart. That's real matrix-multiplication-based
learning, same spirit as VoiceAI, just for a much simpler problem. It
retrains from `data/speaker_profiles.json` in well under a second every
time `transcribe.py` starts, so there's no separate "train" step to
remember and nothing to go stale -- adding a new speaker is still just
enrollment.

Enroll each person two ways -- live mic, or from existing audio files:

```
python enroll.py --speaker zach
python enroll.py --speaker alex
python enroll.py --list                    # see who's enrolled and how many clips each has
python enroll.py --speaker zach --clips 8    # add more clips to an existing speaker later
python enroll.py --speaker zach --file some_folder/               # enroll every .wav directly in that folder
python enroll.py --speaker zach --file clip1.wav clip2.wav        # or specific files
python enroll.py --speaker zach --file "old_clips/*.wav"          # or a glob pattern (quote it)
```

Live mic mode shows you a sentence from `sentences.txt` to read for each
clip -- the actual content doesn't matter for speaker ID (only how you
sound does), it's just there so you have something natural to read
instead of freestyling. `--file` mode skips recording entirely and just
processes whatever you point it at -- a folder (every `.wav` directly
inside it), specific files, or a glob pattern, any mix of those at once.
Handy if you already have recordings of someone, including reusing
VoiceAI's own training clips, e.g.
`python enroll.py --speaker zach --file ../VoiceAI/data/clips`.

5-10 clips (5-10 seconds of actual speech each) per person is a
reasonable starting point. Each speaker works through `data/sentences.txt`
independently, remembers where they left off (`data/sentence_progress.json`),
and wraps back to the start once they've gone through it all -- add your
own lines to that file anytime, or press 's' during enrollment to skip a
sentence. Enrollment fingerprints are stored in `data/speaker_profiles.json`;
recordings loaded via `--file` are expected under `data/clips/<name>/` if
you want to keep per-person source audio organized, though any path works.

Once at least 2 speakers are enrolled (the classifier needs at least 2 to
have anything to tell apart), `transcribe.py` automatically tags every
result, e.g. `[zach] I like playing video games (94% confident)` -- no
flags needed, it trains itself at startup from whatever's in
`data/speaker_profiles.json`.

### Scoping speakers to mic vs. Discord (`--source`)

In `--live` mode, someone who's only ever on your mic (you) and someone
who's only ever on Discord (everyone else on the call) should never be
compared against each other -- it's wasted computation, and a small
chance of a wrong tag. `--source` scopes a speaker to one side so each
gets its own, smaller classifier under the hood:

```
python enroll.py --speaker zach --source mic --clips 8              # you: mic-only
python enroll.py --speaker quinn --file data/clips/quinn --source discord   # Discord-only
python enroll.py --speaker elliott --source discord --clips 0                # rescope an existing speaker, no new clips
```

`--source` is `mic`, `discord`, or `both` (the default -- unscoped,
compared against everyone). It only takes effect the first time a
speaker is enrolled; to change an already-enrolled speaker's scope later
without adding clips, pass `--source` together with `--clips 0` and no
`--file`. `--list` shows each speaker's current scope:

```
python enroll.py --list
  zach: 108 clip(s), source=mic
  elliott: 113 clip(s), source=discord
```

`transcribe.py --live` then prints separate Speaker ID status lines for
`mic` and `discord`, and tags each transcribed segment using only the
classifier for the source it actually came from.

Expect this to work best when everyone enrolls with the same mic/room
setup they'll actually use it in -- background noise and mic
characteristics are themselves part of what gets picked up as a "voice
fingerprint," so a big mismatch between enrollment and real use can hurt
accuracy. With 2-3 people and a consistent setup, a handful of clips each
should give clearly separated matches; if two people's scores come back
close together, enroll a few more clips for each.

## Notes

- Same mic caveat as VoiceAI: this records from whatever Windows calls
  the default input device. If that's a virtual/routed device
  (VoiceMeeter, OBS, a Discord virtual cable) rather than your physical
  mic, check `--list-devices` and pin the right one with `--input-device`
  if needed.
- Whisper itself has no speaker identification built in (that's a
  separate "diarization" problem) -- the tagging you see comes entirely
  from this project's own `speaker_id.py`, not from Whisper.
- `--language en` is the default so it skips language auto-detection and
  transcribes a little faster; pass `--language ""` if you want to speak
  other languages.
