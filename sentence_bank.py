"""
Loads reference sentences from sentences.txt and tracks, per speaker,
which one to read next during enrollment -- so enroll.py can show you a
sentence instead of asking you to freestyle. The content doesn't actually
matter for speaker ID (see speaker_id.py), this is purely so reading
clips feels natural and consistent instead of staring at a blank prompt.

Progress is persisted in sentence_progress.json so it picks up where you
left off across runs, and wraps back to the start once a speaker has gone
through the whole list (repeats are fine -- more clips just means a more
reliable profile).
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
SENTENCES_PATH = DATA_DIR / "sentences.txt"
PROGRESS_PATH = DATA_DIR / "sentence_progress.json"

_LINE_RE = re.compile(r"^\d+\.\s+(.*\S)\s*$")


def load_sentences():
    if not SENTENCES_PATH.exists():
        return []
    sentences = []
    with open(SENTENCES_PATH, "r", encoding="utf-8") as f:
        for line in f:
            match = _LINE_RE.match(line.strip())
            if match:
                sentences.append(match.group(1))
    return sentences


def _load_progress():
    if PROGRESS_PATH.exists():
        with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_progress(progress):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)


def peek_sentence(speaker, sentences):
    """Returns (sentence_text, index, wrapped) for the next sentence this
    speaker should read, WITHOUT marking it as used yet."""
    if not sentences:
        return None, None, False

    progress = _load_progress()
    idx = progress.get(speaker, 0)
    wrapped = idx >= len(sentences)
    if wrapped:
        idx = 0

    return sentences[idx], idx, wrapped


def advance(speaker, idx):
    progress = _load_progress()
    progress[speaker] = idx + 1
    _save_progress(progress)
