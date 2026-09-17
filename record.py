"""
Audio capture for local Whisper transcription: live mic or system-audio
loopback (fixed-duration, press-a-key, or continuous live streaming with
voice-activity detection), plus existing audio files. Everything is
converted to mono float32 @ 16kHz numpy arrays, which is exactly what
whisper.transcribe() wants when you pass it an array directly (this lets
us skip Whisper's own ffmpeg-based loader entirely -- no ffmpeg install
needed for this project).

Capture records at the device's own native rate and resamples down to
16000 Hz ourselves afterward, rather than asking PortAudio to capture at
16000 Hz directly -- forcing real-time sample-rate conversion during
capture can subtly warp audio on some Windows audio backends. This
mirrors the fix used in the VoiceAI project.

Loopback capture ("hear what's playing", e.g. a Discord call) uses
WASAPI's loopback mode on a normal *output* device (like your speakers or
headphones) -- Windows-only, no virtual cable software needed. This goes
through the `soundcard` package rather than `sounddevice`/PortAudio:
PortAudio's Python bindings (what `sounddevice` wraps) don't actually
expose WASAPI loopback despite some old docs/issues suggesting otherwise,
but `soundcard` supports it directly via
`get_microphone(..., include_loopback=True)`. Regular mic capture still
uses `sounddevice` as before. Run --list-devices to see input devices
(mics, via sounddevice) and --list-output-devices to see output devices
(via soundcard, usable with --loopback-device).
"""
import queue
import threading
from collections import deque

import numpy as np
import soundfile as sf

TARGET_SR = 16000
LOOPBACK_SR = 48000  # requested native rate for WASAPI loopback capture via `soundcard`


def list_input_devices():
    import sounddevice as sd

    print(sd.query_devices())
    try:
        default_in = sd.default.device[0]
        print(f"\nCurrent default input device index: {default_in}")
    except Exception:
        pass


def list_output_devices():
    """Lists output-capable devices (speakers/headphones/virtual cables)
    -- these are what --loopback-device expects, since loopback capture
    listens to what's being *played* on a device, not recorded from it.
    Uses `soundcard` (not `sounddevice`), since that's the library
    --loopback capture itself uses -- indices printed here are what
    --loopback-device expects, and won't necessarily match indices from
    --list-devices (which lists sounddevice's input devices instead)."""
    import soundcard as sc

    speakers = sc.all_speakers()
    try:
        default = sc.default_speaker()
    except Exception:
        default = None
    print("Output-capable devices (usable with --loopback-device):")
    for i, spk in enumerate(speakers):
        marker = "  <-- current default" if default is not None and spk.id == default.id else ""
        print(f"  [{i}] {spk.name}{marker}")


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


def _device_info(device, loopback):
    import sounddevice as sd

    if device is not None:
        return sd.query_devices(device)
    if loopback:
        out_idx = sd.default.device[1]
        return sd.query_devices(out_idx)
    return sd.query_devices(kind="input")


def _native_input_samplerate(fallback=TARGET_SR, device=None, loopback=False):
    try:
        info = _device_info(device, loopback)
        sr = int(round(float(info["default_samplerate"])))
        name = info.get("name", "default device")
        if sr <= 0:
            return fallback, name
        return sr, name
    except Exception as exc:
        print(f"(could not query native sample rate, falling back to {fallback}Hz: {exc})")
        return fallback, "unknown device"


def _open_stream(samplerate, channels, callback, device, loopback, blocksize=0):
    """Opens a plain sounddevice InputStream for mic capture. `blocksize`
    (frames) is 0 (let PortAudio choose) by default; live_vad_segments
    sets it explicitly so every callback block represents a known, fixed
    duration for the VAD timing logic.

    NOTE: this is mic-only. PortAudio's Python bindings don't actually
    support WASAPI loopback (there's no `loopback` option on
    WasapiSettings, despite what some old references suggest) -- loopback
    capture is handled separately via the `soundcard` package, see
    `_loopback_block_source` / `_resolve_loopback_speaker` below."""
    import sounddevice as sd

    if loopback:
        raise NotImplementedError(
            "Loopback capture isn't available through sounddevice/PortAudio. "
            "Use `live_vad_segments(loopback=True)` (i.e. `--live --loopback`), "
            "which captures loopback audio via the `soundcard` package instead."
        )

    return sd.InputStream(
        samplerate=samplerate,
        channels=channels,
        dtype="float32",
        callback=callback,
        device=device,
        blocksize=blocksize,
    )


def _resolve_loopback_speaker(device):
    """Resolves --loopback-device (an index from --list-output-devices, a
    name substring, or None for the system default) to a
    soundcard.Speaker. Raises a clear error if soundcard isn't installed
    or nothing matches."""
    try:
        import soundcard as sc
    except ImportError as exc:
        raise RuntimeError(
            "The `soundcard` package is required for --loopback capture. "
            "Run:\n    pip install soundcard\nthen try again."
        ) from exc

    speakers = sc.all_speakers()
    if not speakers:
        raise RuntimeError("No output devices found (soundcard.all_speakers() returned none).")
    if device is None:
        return sc.default_speaker()
    if isinstance(device, int):
        if device < 0 or device >= len(speakers):
            raise ValueError(
                f"No output device with index {device}. Run --list-output-devices to see valid indices."
            )
        return speakers[device]
    matches = [s for s in speakers if str(device).lower() in s.name.lower()]
    if not matches:
        raise ValueError(f"No output device matching '{device}'. Run --list-output-devices to see options.")
    return matches[0]


def _loopback_block_source(device, block_duration, stop_event):
    """Yields raw mono float32 blocks captured from a speaker's WASAPI
    loopback ("what's playing"), via the `soundcard` package. Unlike
    `sounddevice`/PortAudio, `soundcard` has genuine WASAPI loopback
    support on Windows (`get_microphone(..., include_loopback=True)`),
    which is why loopback capture uses a different library than mic
    capture does."""
    import soundcard as sc

    speaker = _resolve_loopback_speaker(device)
    mic = sc.get_microphone(id=speaker.id, include_loopback=True)
    block_size = max(1, int(round(LOOPBACK_SR * block_duration)))
    with mic.recorder(samplerate=LOOPBACK_SR) as recorder:
        while not stop_event.is_set():
            block = np.asarray(recorder.record(numframes=block_size), dtype=np.float32)
            if block.ndim > 1:
                block = block.mean(axis=1)
            else:
                block = block.reshape(-1)
            yield block


def record_from_mic(duration_sec=3.0, sample_rate=TARGET_SR, device=None, loopback=False):
    import sounddevice as sd

    native_sr, device_name = _native_input_samplerate(fallback=sample_rate, device=device, loopback=loopback)
    kind = "loopback" if loopback else "mic"
    print(f"Recording for {duration_sec:.1f}s ({kind}: {device_name} @ {native_sr}Hz native)...")

    frames = []
    done = threading.Event()

    def callback(indata, frame_count, time_info, status):
        frames.append(indata.copy())
        if sum(len(f) for f in frames) >= int(duration_sec * native_sr):
            done.set()

    channels = 2 if loopback else 1
    with _open_stream(native_sr, channels, callback, device, loopback):
        done.wait(timeout=duration_sec + 2.0)

    waveform = np.concatenate(frames, axis=0) if frames else np.zeros((0, channels), dtype=np.float32)
    if channels > 1:
        waveform = waveform.mean(axis=1)
    else:
        waveform = waveform.reshape(-1)

    if native_sr != sample_rate:
        waveform = _resample_linear(waveform, native_sr, sample_rate)
        print(f"Resampled {native_sr}Hz -> {sample_rate}Hz.")
    print("Done.")
    _print_capture_stats(waveform, sample_rate)
    return waveform


def record_until_keypress(sample_rate=TARGET_SR, stop_prompt="Recording... press any key to stop.",
                           device=None, loopback=False):
    from keypress import read_key

    native_sr, device_name = _native_input_samplerate(fallback=sample_rate, device=device, loopback=loopback)
    frames = []

    def callback(indata, frame_count, time_info, status):
        frames.append(indata.copy())

    channels = 2 if loopback else 1
    kind = "loopback" if loopback else "mic"
    print(f"({kind}: {device_name} @ {native_sr}Hz native)")
    with _open_stream(native_sr, channels, callback, device, loopback):
        read_key(stop_prompt)

    print("Done.")

    if not frames:
        waveform = np.zeros(0, dtype=np.float32)
    else:
        waveform = np.concatenate(frames, axis=0)
        if channels > 1:
            waveform = waveform.mean(axis=1)
        else:
            waveform = waveform.reshape(-1)
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


# ---------------------------------------------------------------------
# Continuous "live" capture with simple energy-based voice-activity
# detection (VAD) -- for a stream that never stops (a mic left open, or
# a Discord call via loopback), auto-detects when someone starts and
# stops talking instead of requiring a keypress per utterance.
# ---------------------------------------------------------------------

def vad_segments_from_blocks(block_iter, native_sr, target_sr=TARGET_SR, block_duration=0.05,
                              energy_threshold=0.01, silence_duration=0.8,
                              min_speech_duration=0.5, pre_roll=0.2, stop_event=None):
    """Pure segmentation logic, decoupled from sounddevice so it can be
    unit-tested with a synthetic block_iter. Consumes an iterable of raw
    mono float32 blocks (each `block_duration` seconds, at `native_sr`)
    and yields resampled (target_sr) mono float32 waveforms, one per
    detected speech segment.

    State machine: while quiet, keep a short rolling `pre_roll` buffer so
    the onset of speech isn't clipped. Once a block's RMS energy exceeds
    `energy_threshold`, start a segment (seeded with the pre-roll).
    Keep accumulating through brief dips, but once energy has stayed
    below the threshold for `silence_duration` seconds, end the segment.
    Segments shorter than `min_speech_duration` (false positives -- a
    cough, a click) are discarded rather than yielded.
    """
    pre_roll_blocks = deque(maxlen=max(1, int(round(pre_roll / block_duration))))
    segment = []
    silence_time = 0.0
    speaking = False
    active_blocks = 0  # blocks actually above threshold -- what min_speech_duration is judged against,
                        # NOT the padded segment length (pre-roll + hangover silence would otherwise let
                        # a brief cough/click sneak past the filter just from padding alone)

    for block in block_iter:
        if stop_event is not None and stop_event.is_set():
            return

        block = np.asarray(block, dtype=np.float32).reshape(-1)
        rms = float(np.sqrt(np.mean(block ** 2) + 1e-12))
        above = rms > energy_threshold

        if not speaking:
            pre_roll_blocks.append(block)
            if above:
                speaking = True
                segment = list(pre_roll_blocks)
                active_blocks = 1
                silence_time = 0.0
        else:
            segment.append(block)
            if above:
                active_blocks += 1
                silence_time = 0.0
            else:
                silence_time += block_duration
                if silence_time >= silence_duration:
                    waveform = np.concatenate(segment)
                    active_duration = active_blocks * block_duration
                    speaking = False
                    segment = []
                    active_blocks = 0
                    silence_time = 0.0
                    pre_roll_blocks.clear()
                    if active_duration >= min_speech_duration:
                        if native_sr != target_sr:
                            waveform = _resample_linear(waveform, native_sr, target_sr)
                        yield waveform

    # stream ended mid-utterance (e.g. stop requested) -- flush what we have
    if speaking and segment:
        waveform = np.concatenate(segment)
        active_duration = active_blocks * block_duration
        if active_duration >= min_speech_duration:
            if native_sr != target_sr:
                waveform = _resample_linear(waveform, native_sr, target_sr)
            yield waveform


def test_loopback_capture(duration_sec=5.0, device=None, save_path=None):
    """Records `duration_sec` seconds of raw WASAPI loopback audio,
    bypassing VAD and Whisper entirely, and reports peak/RMS -- for
    diagnosing "--loopback isn't picking up Discord" without guessing.
    A near-zero peak means no audio is reaching the captured device at
    all (almost always a device-routing issue: Discord/Windows sending
    audio to a different output device than the one being listened to --
    see Windows Settings > Sound > Volume mixer > App volume and device
    preferences, and Discord's own Voice & Video > Output Device). A
    healthy peak but no live transcriptions means it's a VAD sensitivity
    issue instead -- try a lower --vad-threshold."""
    import soundcard as sc

    speaker = _resolve_loopback_speaker(device)
    print(f"Recording {duration_sec:.1f}s of raw loopback audio from: {speaker.name}")
    print("Play/talk through it now (e.g. have someone talk in the Discord call) ...")

    mic = sc.get_microphone(id=speaker.id, include_loopback=True)
    numframes = max(1, int(round(duration_sec * LOOPBACK_SR)))
    data = np.asarray(mic.record(numframes=numframes, samplerate=LOOPBACK_SR), dtype=np.float32)
    mono = data.mean(axis=1) if data.ndim > 1 else data.reshape(-1)

    _print_capture_stats(mono, LOOPBACK_SR)
    peak = float(np.abs(mono).max()) if mono.size else 0.0
    if peak < 0.001:
        print(
            "\nPeak volume is essentially zero -- no audio is reaching this "
            "device at all. This is almost always because Discord (or "
            "Windows) is routing its audio to a DIFFERENT output device "
            "than the one being captured here, not a bug in this script. "
            "Check:\n"
            "  - Windows Settings > System > Sound > Volume mixer -- which "
            "device is Discord actually assigned to?\n"
            "  - Discord's own Settings > Voice & Video > Output Device\n"
            "  - Run --list-output-devices and try a different "
            "--loopback-device matching whichever device Discord is on\n"
            "  - Make sure system/app volume isn't muted or at 0"
        )
    else:
        print(
            "\nAudio is arriving fine. If --live --loopback still isn't "
            "producing transcriptions, try a lower --vad-threshold (the "
            "signal may just be quieter than the default threshold)."
        )

    if save_path:
        sf.write(str(save_path), mono, LOOPBACK_SR)
        print(f"Saved raw capture to {save_path} -- play it back to confirm what was actually captured.")

    return mono


def _live_block_source(device, loopback, native_sr, block_duration, stop_event):
    """Yields raw mono float32 blocks forever (until stop_event is set),
    for feeding into vad_segments_from_blocks. Every block represents
    exactly `block_duration` seconds -- the VAD timing logic assumes
    that. Loopback (`soundcard`) and mic (`sounddevice`) go through
    different libraries under the hood -- see `_loopback_block_source`
    and the module docstring for why."""
    if loopback:
        yield from _loopback_block_source(device, block_duration, stop_event)
        return

    block_size = max(1, int(round(native_sr * block_duration)))
    q = queue.Queue()

    def callback(indata, frame_count, time_info, status):
        q.put(indata.copy())

    stream = _open_stream(native_sr, 1, callback, device, False, blocksize=block_size)
    with stream:
        while not stop_event.is_set():
            try:
                block = q.get(timeout=0.2)
            except queue.Empty:
                continue
            yield block.reshape(-1)


def live_vad_segments(device=None, loopback=False, target_sr=TARGET_SR, block_duration=0.05,
                       energy_threshold=0.01, silence_duration=0.8, min_speech_duration=0.5,
                       pre_roll=0.2, stop_event=None):
    """Continuously listens (mic, or loopback if `loopback=True`) and
    yields one 16kHz mono float32 waveform per detected speech segment,
    for as long as `stop_event` (a threading.Event) is not set."""
    if stop_event is None:
        stop_event = threading.Event()

    if loopback:
        speaker = _resolve_loopback_speaker(device)
        native_sr, device_name = LOOPBACK_SR, speaker.name
    else:
        native_sr, device_name = _native_input_samplerate(fallback=target_sr, device=device, loopback=False)
    kind = "loopback (system audio)" if loopback else "mic"
    print(f"Listening continuously ({kind}: {device_name} @ {native_sr}Hz native). Press any key (or Ctrl+C) to stop.\n")

    blocks = _live_block_source(device, loopback, native_sr, block_duration, stop_event)
    yield from vad_segments_from_blocks(
        blocks, native_sr, target_sr=target_sr, block_duration=block_duration,
        energy_threshold=energy_threshold, silence_duration=silence_duration,
        min_speech_duration=min_speech_duration, pre_roll=pre_roll, stop_event=stop_event,
    )
