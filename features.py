"""
From-scratch log-mel spectrogram extraction (manual STFT via framing +
Hamming window + rfft, triangular mel filterbank via matrix multiplication,
log). Same approach as VoiceAI's features.py, duplicated here so this
project stays self-contained. Used by speaker_id.py, not by Whisper itself
(Whisper does its own feature extraction internally).
"""
import numpy as np

N_FFT = 400
HOP_LENGTH = 160
N_MELS = 40
EPS = 1e-6


def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def build_mel_filterbank(sample_rate, n_fft=N_FFT, n_mels=N_MELS):
    n_freq_bins = n_fft // 2 + 1
    min_mel = hz_to_mel(0.0)
    max_mel = hz_to_mel(sample_rate / 2.0)
    mel_points = np.linspace(min_mel, max_mel, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bin_points = np.clip(bin_points, 0, n_freq_bins - 1)

    filterbank = np.zeros((n_mels, n_freq_bins), dtype=np.float32)
    for m in range(1, n_mels + 1):
        left, center, right = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        if center == left:
            center += 1
        if right == center:
            right += 1
        for k in range(left, center):
            filterbank[m - 1, k] = (k - left) / max(center - left, 1)
        for k in range(center, right):
            filterbank[m - 1, k] = (right - k) / max(right - center, 1)
    return filterbank


def frame_signal(waveform, n_fft=N_FFT, hop_length=HOP_LENGTH):
    if len(waveform) < n_fft:
        waveform = np.pad(waveform, (0, n_fft - len(waveform)))
    num_frames = 1 + (len(waveform) - n_fft) // hop_length
    frames = np.zeros((num_frames, n_fft), dtype=np.float32)
    for i in range(num_frames):
        start = i * hop_length
        frames[i] = waveform[start:start + n_fft]
    return frames


def compute_log_mel(waveform, sample_rate, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, normalize=True):
    waveform = np.asarray(waveform, dtype=np.float32)
    frames = frame_signal(waveform, n_fft, hop_length)

    window = np.hamming(n_fft).astype(np.float32)
    frames = frames * window

    spectrum = np.fft.rfft(frames, n=n_fft, axis=1)
    power = (np.abs(spectrum) ** 2) / n_fft

    filterbank = build_mel_filterbank(sample_rate, n_fft, n_mels)
    mel_energy = power @ filterbank.T  # [T, n_mels]
    log_mel = np.log(mel_energy + EPS)

    if normalize:
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + EPS)

    return log_mel.astype(np.float32)
