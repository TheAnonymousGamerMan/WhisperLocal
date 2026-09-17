"""
From-scratch speaker identification, decoupled from transcription entirely
-- it doesn't care what anyone said, only who was talking. That's why it
needs far less data than the transcription side: instead of learning an
open-ended mapping from sound to arbitrary text, it just has to tell a
small, fixed set of enrolled voices apart using traits (pitch range,
spectral "shape"/timbre) that stay fairly constant no matter what someone
says.

Two stages:
  1. Featurize each clip into a compact acoustic fingerprint -- per-mel-band
     mean/std (overall timbre/resonance) plus pitch (fundamental frequency)
     mean/std and voiced-frame ratio (pitch range and speech rhythm), all
     from scratch with numpy (log-mel via features.py, pitch via
     autocorrelation -- no external pitch-tracking library).
  2. Classify: a small softmax (multinomial logistic) regression --
     logits = X @ W + b -- trained from scratch via full-batch gradient
     descent (manual forward/backward, no autodiff framework), same
     matrix-multiplication spirit as VoiceAI's model. This replaces a
     plain nearest-centroid/cosine-similarity comparison with an actual
     learned decision boundary, and its softmax output doubles as a real
     (if not perfectly calibrated) confidence score.

Training is a handful of milliseconds at this data scale, so nothing is
cached to disk -- transcribe.py just retrains fresh from
data/speaker_profiles.json every time it starts, which means there's
nothing to keep in sync by hand. That also means adding a new speaker is
still just enrollment (plus this instant retrain) -- no manual retraining
step required.
"""
import json
from pathlib import Path

import numpy as np

from features import compute_log_mel, N_FFT, HOP_LENGTH

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
PROFILES_PATH = DATA_DIR / "speaker_profiles.json"

MIN_F0_HZ = 70.0
MAX_F0_HZ = 400.0


def _estimate_pitch_track(waveform, sample_rate, frame_length=N_FFT, hop_length=HOP_LENGTH,
                           min_f0=MIN_F0_HZ, max_f0=MAX_F0_HZ, voiced_threshold=0.3):
    """Simple autocorrelation-based F0 estimate per frame. Returns
    (f0_values_for_voiced_frames, voiced_ratio) -- unvoiced/silent frames
    (no clear periodicity) are excluded from the pitch stats, since their
    "pitch" is meaningless."""
    waveform = np.asarray(waveform, dtype=np.float32)
    if len(waveform) < frame_length:
        waveform = np.pad(waveform, (0, frame_length - len(waveform)))

    min_lag = int(sample_rate / max_f0)
    max_lag = int(sample_rate / min_f0)

    num_frames = 1 + (len(waveform) - frame_length) // hop_length
    f0s = []
    voiced_count = 0

    for i in range(num_frames):
        start = i * hop_length
        frame = waveform[start:start + frame_length]
        frame = frame - frame.mean()
        energy = np.sum(frame ** 2)
        if energy < 1e-6:
            continue  # near-silent frame, skip

        # autocorrelation via FFT (fast, still "just" correlation)
        n = len(frame)
        fft_frame = np.fft.rfft(frame, n=2 * n)
        acf = np.fft.irfft(fft_frame * np.conj(fft_frame))[:n]
        acf = acf / (acf[0] + 1e-9)

        if max_lag >= len(acf):
            continue
        search = acf[min_lag:max_lag]
        if search.size == 0:
            continue
        peak_idx = int(np.argmax(search))
        peak_val = search[peak_idx]

        if peak_val < voiced_threshold:
            continue  # not periodic enough to be confident voiced speech

        lag = min_lag + peak_idx
        f0 = sample_rate / lag
        f0s.append(f0)
        voiced_count += 1

    voiced_ratio = voiced_count / max(num_frames, 1)
    return np.array(f0s, dtype=np.float32), voiced_ratio


def extract_fingerprint(waveform, sample_rate):
    """Returns a 1-D numpy feature vector describing who's likely talking,
    independent of what they said."""
    log_mel = compute_log_mel(waveform, sample_rate, normalize=False)  # [T, n_mels], raw (un-normalized) scale
    mel_mean = log_mel.mean(axis=0)
    mel_std = log_mel.std(axis=0)

    f0s, voiced_ratio = _estimate_pitch_track(waveform, sample_rate)
    if f0s.size > 0:
        f0_mean = float(f0s.mean())
        f0_std = float(f0s.std())
    else:
        f0_mean = 0.0
        f0_std = 0.0

    fingerprint = np.concatenate([
        mel_mean, mel_std,
        np.array([f0_mean, f0_std, voiced_ratio], dtype=np.float32),
    ]).astype(np.float32)
    return fingerprint


SOURCES = ("mic", "discord", "both")


def load_profiles():
    if not PROFILES_PATH.exists():
        return {}
    with open(PROFILES_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {
        speaker: {
            "clips": [np.array(v, dtype=np.float32) for v in entry["clips"]],
            # "source" scopes a speaker to one live-mode capture source ("mic"
            # or "discord") so e.g. someone who only ever talks on Discord
            # never gets compared against (or confused with) someone who
            # only ever talks on the mic, and vice versa. Old profiles saved
            # before this existed default to "both" -- no behavior change
            # unless you opt in via `enroll.py --source`.
            "source": entry.get("source", "both"),
        }
        for speaker, entry in raw.items()
    }


def save_profiles(profiles):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw = {
        speaker: {
            "clips": [v.tolist() for v in entry["clips"]],
            "source": entry.get("source", "both"),
        }
        for speaker, entry in profiles.items()
    }
    with open(PROFILES_PATH, "w", encoding="utf-8") as f:
        json.dump(raw, f)


def enroll_clip(speaker, waveform, sample_rate, source=None):
    """Adds one more enrollment clip's fingerprint for `speaker` and
    persists it immediately (so a crashed/interrupted session doesn't
    lose earlier clips). `source` ("mic"/"discord"/"both") is only used
    the first time a speaker is enrolled -- it's ignored on later calls
    for a speaker that already exists, so re-running enrollment without
    --source doesn't silently reset it back to "both"."""
    fingerprint = extract_fingerprint(waveform, sample_rate)
    profiles = load_profiles()
    if speaker not in profiles:
        profiles[speaker] = {"clips": [], "source": source or "both"}
    profiles[speaker]["clips"].append(fingerprint)
    save_profiles(profiles)
    return len(profiles[speaker]["clips"])


def set_speaker_source(speaker, source):
    """Changes an already-enrolled speaker's source scope without
    touching their clips."""
    assert source in SOURCES, f"source must be one of {SOURCES}"
    profiles = load_profiles()
    if speaker not in profiles:
        raise KeyError(f"'{speaker}' isn't enrolled yet")
    profiles[speaker]["source"] = source
    save_profiles(profiles)


def filter_profiles_by_source(profiles, source):
    """Keeps only speakers scoped to `source` or to "both" -- this is
    what lets --live use a mic-only classifier for mic segments and a
    discord-only classifier for discord segments, so e.g. someone who
    only ever talks on Discord can never be mistaken for someone who
    only ever talks on the mic."""
    if source is None or source == "both":
        return profiles
    return {
        speaker: entry
        for speaker, entry in profiles.items()
        if entry.get("source", "both") in (source, "both")
    }


def profile_summary():
    profiles = load_profiles()
    return {speaker: (len(entry["clips"]), entry.get("source", "both")) for speaker, entry in profiles.items()}


# ---------------------------------------------------------------------
# Classifier: softmax regression trained from scratch via gradient
# descent. Pure matrix multiplication -- X @ W + b -- plus a manual
# backward pass, no autodiff framework.
# ---------------------------------------------------------------------

def _standardize_fit(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def _softmax(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def train_classifier(profiles=None, lr=0.5, iters=500, l2=1e-2, seed=0):
    """Trains a softmax regression classifier over all enrolled speakers'
    fingerprints. Needs at least 2 speakers with at least 1 clip each --
    returns None if that's not met (nothing to tell apart yet).

    Returns a dict {speakers, W, b, mean, std} or None.
    """
    if profiles is None:
        profiles = load_profiles()

    speakers = [s for s, entry in profiles.items() if entry["clips"]]
    if len(speakers) < 2:
        return None

    X_list, y_list = [], []
    for idx, speaker in enumerate(speakers):
        for fingerprint in profiles[speaker]["clips"]:
            X_list.append(fingerprint)
            y_list.append(idx)

    X = np.stack(X_list).astype(np.float64)
    y = np.array(y_list, dtype=np.int64)
    n, d = X.shape
    c = len(speakers)

    mean, std = _standardize_fit(X)
    Xn = (X - mean) / std

    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.01, size=(d, c))
    b = np.zeros(c)

    y_onehot = np.zeros((n, c))
    y_onehot[np.arange(n), y] = 1.0

    for _ in range(iters):
        logits = Xn @ W + b                       # forward: matrix multiply
        probs = _softmax(logits)
        grad_logits = (probs - y_onehot) / n       # softmax + cross-entropy gradient
        grad_W = Xn.T @ grad_logits + l2 * W        # backward: matrix multiply
        grad_b = grad_logits.sum(axis=0)
        W -= lr * grad_W
        b -= lr * grad_b

    return {"speakers": speakers, "W": W, "b": b, "mean": mean, "std": std}


def identify(waveform, sample_rate, classifier=None, profiles=None):
    """Returns (best_speaker_or_None, confidence, all_scores_dict).
    confidence is the classifier's softmax probability for the predicted
    speaker (0-1) -- a real, if not perfectly calibrated, probability,
    unlike a raw similarity score."""
    if classifier is None:
        classifier = train_classifier(profiles)
    if classifier is None:
        return None, 0.0, {}

    fingerprint = extract_fingerprint(waveform, sample_rate).astype(np.float64)
    x = (fingerprint - classifier["mean"]) / classifier["std"]
    logits = x @ classifier["W"] + classifier["b"]     # inference: matrix multiply
    probs = _softmax(logits.reshape(1, -1))[0]

    scores = {s: float(p) for s, p in zip(classifier["speakers"], probs)}
    best_idx = int(np.argmax(probs))
    best_speaker = classifier["speakers"][best_idx]
    return best_speaker, float(probs[best_idx]), scores
