"""
Enroll a speaker for speaker identification (see speaker_id.py for how the
matching works). Two ways to add clips:

1. Live mic, with a sentence from sentences.txt shown for each clip (what
   you actually say doesn't matter for speaker ID, only how you sound --
   this is just so you have something natural to read instead of
   freestyling). If sentences.txt is missing, it falls back to asking you
   to just talk about anything.
2. Existing audio via --file -- handy if you already have recordings of
   someone (a voice memo, an old clip, even VoiceAI's own training
   clips) and don't need to re-record them live. Point it at a folder
   and every .wav directly inside gets enrolled.

Run:
    python enroll.py --speaker zach                    # record clips tagged as "zach" (live mic)
    python enroll.py                                     # asks for a speaker name first
    python enroll.py --speaker zach --clips 8              # record 8 clips instead of the default 5
    python enroll.py --speaker zach --file some_folder/       # enroll every .wav in that folder
    python enroll.py --speaker zach --file clip1.wav clip2.wav   # or specific files
    python enroll.py --speaker zach --file "old_clips/*.wav"       # or a glob pattern (quote it so the
                                                                      shell doesn't expand it first)
    python enroll.py --list                                # show enrolled speakers and clip counts
    python enroll.py --list-devices                          # show every mic sounddevice can see
    python enroll.py --input-device 3                          # force mic index 3 instead of the system default
    python enroll.py --speaker quinn --file data/clips/quinn --source discord   # scope quinn to Discord-only
    python enroll.py --speaker zach --source mic --clips 0                       # (re)scope zach to mic-only, no new clips

--source (mic/discord/both, default: both) scopes a speaker to one
live-mode capture source, so e.g. someone who's only ever on Discord
never gets compared against someone who's only ever on the mic --
transcribe.py --live then uses a separate classifier per source (see
its --help). It only takes effect the first time a speaker is enrolled;
use --source with --clips 0 (and no --file) to change an existing
speaker's scope without adding clips.

You can run this multiple times for the same speaker to add more clips
later (it appends, doesn't overwrite) -- more clips generally means a more
reliable profile, but this needs far less data than transcription does:
5-10 short clips (5-10 seconds each) per person is a reasonable starting
point, from either source. Each speaker works through sentences.txt
independently (live-mic mode only) and your place is remembered across
runs, wrapping back to the start once you've gone through the whole list
(repeats are fine, more data just helps).
"""
import argparse
from pathlib import Path

from keypress import read_key
from record import record_until_keypress, load_audio_file, list_input_devices, parse_device_arg
import speaker_id
from sentence_bank import load_sentences, peek_sentence, advance


def _resolve_file_args(patterns):
    """Turns the values passed to --file into an actual list of .wav
    paths. Each value can be:
      - a folder -- every *.wav directly inside it is used
      - a glob pattern (e.g. "clips/*.wav") -- expanded here since
        PowerShell doesn't do that for you automatically
      - a literal file path
    """
    paths = []
    for pattern in patterns:
        p = Path(pattern)
        if p.is_dir():
            matches = sorted(p.glob("*.wav"))
            if not matches:
                print(f"No .wav files found in folder: {pattern}")
            paths.extend(matches)
        elif any(ch in pattern for ch in "*?["):
            base = p.parent if str(p.parent) != "." or "/" in pattern or "\\" in pattern else Path(".")
            matches = sorted(base.glob(p.name))
            if not matches:
                print(f"No files matched pattern: {pattern}")
            paths.extend(matches)
        else:
            if p.exists():
                paths.append(p)
            else:
                print(f"File/folder not found, skipping: {pattern}")
    return paths


def enroll_from_files(speaker, file_paths, source="both"):
    enrolled = 0
    for path in file_paths:
        try:
            waveform, _ = load_audio_file(path)
        except Exception as exc:
            print(f"Couldn't load {path}: {exc} -- skipping.")
            continue

        duration = len(waveform) / 16000
        if duration < 1.0:
            print(f"{path} is only {duration:.1f}s -- too short to be useful, skipping.")
            continue

        total_clips = speaker_id.enroll_clip(speaker, waveform, 16000, source=source)
        enrolled += 1
        print(f"Enrolled {path} ('{speaker}' now has {total_clips} enrollment clip(s) total).")

    print(f"\nEnrolled {enrolled}/{len(file_paths)} file(s) for '{speaker}'.")


def enroll_from_mic(speaker, num_clips, input_device, source="both"):
    sentences = load_sentences()

    print(f"\nEnrolling '{speaker}'.")
    if sentences:
        print(f"Reading from {len(sentences)} sentences -- just read what's shown, content doesn't matter for speaker ID.\n")
    else:
        print("No sentences.txt found -- just talk naturally, doesn't matter what about.\n")

    recorded = 0
    while recorded < num_clips:
        sentence, sent_idx, wrapped = (None, None, False)
        if sentences:
            sentence, sent_idx, wrapped = peek_sentence(speaker, sentences)
            if wrapped:
                print("(Back to the start of the list -- extra reps are still useful.)")
            print(f'\nRead this out loud:\n    "{sentence}"\n')

        key = read_key(
            f"[{recorded}/{num_clips}] Press any key to start recording "
            "('s' to skip, 'q' to stop early): "
        )
        if key.lower() == "q":
            break
        if key.lower() == "s" and sentences:
            advance(speaker, sent_idx)
            print("Skipped.\n")
            continue

        waveform = record_until_keypress(
            stop_prompt="Recording... press any key to stop.", device=input_device
        )
        if waveform.size == 0:
            print("No audio captured -- try again.\n")
            continue

        duration = len(waveform) / 16000
        if duration < 1.0:
            print(f"That was only {duration:.1f}s -- too short to be useful, try again with a bit more speech.\n")
            continue

        total_clips = speaker_id.enroll_clip(speaker, waveform, 16000, source=source)
        if sentences:
            advance(speaker, sent_idx)
        recorded += 1
        print(f"Saved clip {recorded}/{num_clips} ('{speaker}' now has {total_clips} enrollment clip(s) total).\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speaker", type=str, default=None,
                         help="Name/label for this speaker (any string). If omitted, you'll be asked.")
    parser.add_argument("--clips", type=int, default=5,
                         help="How many clips to record this run in live-mic mode (default: 5).")
    parser.add_argument("--file", type=str, nargs="+", default=None,
                         help="Enroll from existing audio instead of recording live. Accepts a folder "
                                "(every .wav directly inside it is used), file paths, and/or a glob "
                                "pattern like \"clips/*.wav\" (quote it) -- mix and match, any number of them.")
    parser.add_argument("--input-device", type=str, default=None,
                         help="Mic to record from: an index or a name/substring from --list-devices.")
    parser.add_argument("--list-devices", action="store_true",
                         help="Print every audio device sounddevice can see, then exit.")
    parser.add_argument("--list", action="store_true",
                         help="Print enrolled speakers, their clip counts, and their source scope, then exit.")
    parser.add_argument("--source", type=str, default="both", choices=list(speaker_id.SOURCES),
                         help="Scope this speaker to one --live capture source (default: both). Only takes "
                                "effect the first time a speaker is enrolled -- pass --source with --clips 0 "
                                "(and no --file) to (re)scope an existing speaker without adding clips.")
    args = parser.parse_args()

    if args.list_devices:
        list_input_devices()
        return

    if args.list:
        summary = speaker_id.profile_summary()
        if not summary:
            print("No speakers enrolled yet. Run: python enroll.py --speaker <name>")
        else:
            for speaker, (count, source) in summary.items():
                print(f"  {speaker}: {count} clip(s), source={source}")
        return

    speaker = args.speaker.strip() if args.speaker else input("Speaker name: ").strip()
    if not speaker:
        print("No speaker name given, exiting.")
        return

    existing = speaker_id.profile_summary()
    if speaker in existing and args.source != "both":
        speaker_id.set_speaker_source(speaker, args.source)
        print(f"'{speaker}' rescoped to source={args.source}.")

    if args.file:
        file_paths = _resolve_file_args(args.file)
        if not file_paths:
            print("No valid files found, exiting.")
            return
        enroll_from_files(speaker, file_paths, source=args.source)
    elif args.clips > 0:
        input_device = parse_device_arg(args.input_device)
        enroll_from_mic(speaker, args.clips, input_device, source=args.source)
    elif speaker not in existing:
        print(f"'{speaker}' isn't enrolled yet and --clips 0 was given with no --file, so there's nothing to do.")
        return

    total_clips, source = speaker_id.profile_summary().get(speaker, (0, args.source))
    print(f"\nDone. '{speaker}' has {total_clips} enrollment clip(s), source={source}.")
    print("Run transcribe.py -- it'll automatically tag speakers now.")


if __name__ == "__main__":
    main()
