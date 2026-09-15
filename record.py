"""
Audio capture for local Whisper transcription: live mic (fixed-duration or
press-a-key start/stop) and existing audio files. Everything is converted
to mono float32 @ 16kHz numpy arrays, which is exactly what
whisper.transcribe() wants when you pass it an array directly (this lets
us skip Whisper's own ffmpeg-based loader entirely -- no ffmpeg install
needed for this project).

Live mic capture records at the input device's own native rate and
resamples down to 16000 Hz ourselves afterward, rather than asking
PortAudio to capture at 16000 Hz directly -- forcing real-time
sample-rate conversion during capture can subtly warp audio on some
Windows audio backends. This mirrors the fix used in the VoiceAI project.

Every capture function accepts an optional `device` (an index or a
name/substring, same as sounddevice.query_devices() rows) so you can pin a
specific input instead of trusting whatever Windows currently calls the
"default" device -- run --list-devices to see what's available.
"""
import numpy as np
import soundfile as sf

TARGET_SR = 16000


def list_input_devices():
    import sounddevice as sd

    print(sd.query_devices())
    try:
        default_in = sd.default.device[0]
        print(f"\nCurrent default input device index: {default_in}")
    except Exception:
        pass


def parse_device_arg(value):
    if value is None:
        return None
    value = value.strip()
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return value


def _print_capture_stats(waveform, sample_rate):
    if waveform.size == 0:
        print("Captured 0.00s -- nothing recorded.")
        return
    duration = len(waveform) / sample_rate
    peak = float(np.abs(waveform).max())
    warning = ""
    if peak < 0.01:
        warning = "  <-- near-silent! check your mic / default input device in Windows sound settings"
    elif peak > 0.98:
        warning = "  <-- clipping! you may be too close to the mic or input gain is too high"
    print(f"Captured {duration:.2f}s, peak volume {peak:.3f} (0=silence, 1=max){warning}")


def _resample_linear(waveform, orig_sr, target_sr):
    if orig_sr == target_sr:
        return waveform
    duration = len(waveform) / orig_sr
    n_target = int(round(duration * target_sr))
    orig_times = np.linspace(0.0, duration, num=len(waveform), endpoint=False)
    target_times = np.linspace(0.0, duration, num=n_target, endpoint=False)
    return np.interp(target_times, orig_times, waveform).astype(np.float32)


def _native_input_samplerate(fallback=TARGET_SR, device=None):
    import sounddevice as sd

    try:
        if device is not None:
            info = sd.query_devices(device)
        else:
            info = sd.query_devices(kind="input")
        sr = int(round(float(info["default_samplerate"])))
        name = info.get("name", "default input device")
        if sr <= 0:
            return fallback, name
        return sr, name
    except Exception as exc:
        print(f"(could not query native mic sample rate, falling back to {fallback}Hz: {exc})")
        return fallback, "unknown device"


def record_from_mic(duration_sec=3.0, sample_rate=TARGET_SR, device=None):
    import sounddevice as sd

    native_sr, device_name = _native_input_samplerate(fallback=sample_rate, device=device)
    print(f"Recording for {duration_sec:.1f}s (mic: {device_name} @ {native_sr}Hz native)...")
    audio = sd.rec(
        int(duration_sec * native_sr),
        samplerate=native_sr,
        channels=1,
        dtype="float32",
        device=device,
    )
    sd.wait()
    waveform = audio.reshape(-1)
    if native_sr != sample_rate:
        waveform = _resample_linear(waveform, native_sr, sample_rate)
        print(f"Resampled {native_sr}Hz -> {sample_rate}Hz.")
    print("Done.")
    _print_capture_stats(waveform, sample_rate)
    return waveform


def record_until_keypress(sample_rate=TARGET_SR, stop_prompt="Recording... press any key to stop.", device=None):
    import sounddevice as sd
    from keypress import read_key

    native_sr, device_name = _native_input_samplerate(fallback=sample_rate, device=device)
    frames = []

    def callback(indata, frame_count, time_info, status):
        frames.append(indata.copy())

    stream = sd.InputStream(
        samplerate=native_sr,
        channels=1,
        dtype="float32",
        callback=callback,
        device=device,
    )
    print(f"(mic: {device_name} @ {native_sr}Hz native)")
    with stream:
        read_key(stop_prompt)

    print("Done.")

    if not frames:
        waveform = np.zeros(0, dtype=np.float32)
    else:
        waveform = np.concatenate(frames, axis=0).reshape(-1)
        if native_sr != sample_rate:
            waveform = _resample_linear(waveform, native_sr, sample_rate)
            print(f"Resampled {native_sr}Hz -> {sample_rate}Hz.")

    _print_capture_stats(waveform, sample_rate)
    return waveform


def load_audio_file(path, target_sr=TARGET_SR):
    """Loads an audio file via soundfile (no ffmpeg needed for common
    formats like wav/flac/ogg), downmixes to mono, resamples to target_sr."""
    waveform, sr = sf.read(str(path), always_2d=False)
    waveform = np.asarray(waveform, dtype=np.float32)

    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)

    if sr != target_sr:
        waveform = _resample_linear(waveform, sr, target_sr)
        sr = target_sr

    return waveform, sr
