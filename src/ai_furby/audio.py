"""Microphone capture and speaker playback. Uses a simple loudness threshold to find where speech starts and ends."""

from __future__ import annotations

import logging
from collections import deque

import numpy as np
import sounddevice as sd

INPUT_SAMPLE_RATE = 16000  # what faster-whisper expects
BLOCK_SECONDS = 0.1
SILENCE_LEVEL = 1e-4  # samples quieter than this count as digital silence
NOISE_FLOOR = 0.002  # faint noise (about -54 dB) used in pauses instead of digital silence
MAX_PEAK = 0.8  # playback headroom so the device does not clip
NOISE_WINDOW_BLOCKS = 20  # about 2 s of quiet audio used to estimate room noise
NOISE_MULTIPLIER = 3.0  # speech must be this many times louder than the room noise
MIN_SPEECH_BLOCKS = 3  # 0.3 s of sustained loudness before recording starts


def find_input_device(name_part: str) -> int | None:
    return find_device(name_part, "input")


def find_device(name_part: str, kind: str) -> int | None:
    """Returns the index of an `input` or `output` device whose name contains `name_part`.

    Windows lists the same device under several host APIs, so WASAPI is preferred when it is available.
    """
    channel_key = "max_input_channels" if kind == "input" else "max_output_channels"
    matches = []
    for index, device in enumerate(sd.query_devices()):
        if device[channel_key] > 0 and name_part.lower() in device["name"].lower():
            host_api = sd.query_hostapis(device["hostapi"])["name"]
            matches.append((0 if "WASAPI" in host_api else 1, index))
    return min(matches)[1] if matches else None


def record_utterance(
    threshold: float = 0.01,
    silence_seconds: float = 1.0,
    max_seconds: float = 15.0,
    wait_seconds: float = 30.0,
    device: str | int | None = None,
) -> np.ndarray:
    """Waits for speech, records until a pause, and returns mono float32 audio at 16 kHz.

    Returns an empty array if no speech starts within `wait_seconds`.
    """
    if isinstance(device, str):
        device = find_input_device(device)
        if device is None:
            raise RuntimeError(f"No input device matches {device!r}; list devices with sounddevice.query_devices()")

    # WASAPI often refuses 16 kHz, so capture at the device's own rate and resample afterwards.
    device_rate = int(sd.query_devices(device, kind="input")["default_samplerate"])
    block = int(device_rate * BLOCK_SECONDS)
    silent_blocks_to_stop = int(silence_seconds / BLOCK_SECONDS)
    max_blocks = int(max_seconds / BLOCK_SECONDS)
    wait_blocks = int(wait_seconds / BLOCK_SECONDS)

    captured: list[np.ndarray] = []
    recent_levels: deque[float] = deque(maxlen=NOISE_WINDOW_BLOCKS)
    pending: list[np.ndarray] = []  # loud blocks that may be the start of speech, not yet confirmed
    speaking = False
    silent_run = 0

    with sd.InputStream(samplerate=device_rate, channels=1, dtype="float32", blocksize=block, device=device) as stream:
        for index in range(max_blocks + wait_blocks):
            data, _overflowed = stream.read(block)
            samples = data[:, 0].copy()
            level = float(np.sqrt(np.mean(np.square(samples))))
            if not speaking:
                recent_levels.append(level)
            # Mic gain changes with Windows settings, so judge speech against the current room noise, not a fixed number.
            loud = level > _adaptive_threshold(threshold, recent_levels)

            if not speaking:
                if index >= wait_blocks:
                    break
                if not loud:
                    pending.clear()
                    continue
                pending.append(samples)
                # Short clicks and bumps are loud for only a block or two; require a sustained burst before starting.
                if len(pending) >= MIN_SPEECH_BLOCKS:
                    speaking = True
                    captured.extend(pending)
                    pending.clear()
                continue

            captured.append(samples)
            silent_run = 0 if loud else silent_run + 1
            if silent_run >= silent_blocks_to_stop:
                break

    if not captured:
        return np.zeros(0, dtype=np.float32)
    return _resample(np.concatenate(captured), device_rate, INPUT_SAMPLE_RATE)


def _adaptive_threshold(minimum: float, recent_levels: deque[float]) -> float:
    """Speech must be well above the recent quiet level. `minimum` is the floor for very quiet rooms."""
    if len(recent_levels) < NOISE_WINDOW_BLOCKS // 2:
        return minimum
    noise = float(np.median(recent_levels))
    return max(minimum, noise * NOISE_MULTIPLIER)


def _resample(samples: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    if from_rate == to_rate:
        return samples
    target_length = int(len(samples) * to_rate / from_rate)
    old_positions = np.linspace(0.0, 1.0, num=len(samples))
    new_positions = np.linspace(0.0, 1.0, num=target_length)
    return np.interp(new_positions, old_positions, samples).astype(np.float32)


def play(samples: np.ndarray, sample_rate: int, device: str | None = None) -> None:
    """Plays int16 or float32 mono samples on the named output device (or the default) and waits until they finish."""
    output = find_device(device, "output") if device else None
    if device and output is None:
        raise RuntimeError(f"No output device matches {device!r}")
    if samples.dtype == np.int16:
        samples = samples.astype(np.float32) / 32768.0
    # Convert to the output device's own rate ourselves, so Windows doesn't resample in the middle of playback.
    device_rate = int(sd.query_devices(output, kind="output")["default_samplerate"])
    if device_rate != sample_rate:
        from math import gcd

        from scipy.signal import resample_poly

        divisor = gcd(device_rate, sample_rate)
        samples = resample_poly(samples, device_rate // divisor, sample_rate // divisor).astype(np.float32)
    # Resampling can overshoot full scale, and the device then clips the peaks. Keep headroom.
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak > MAX_PEAK:
        samples = samples * (MAX_PEAK / peak)
    # Bluetooth headsets can power down during digital silence, which clips the start of the next phrase.
    # Keep a faint noise floor in the pauses so the stream never goes fully quiet.
    rng = np.random.default_rng()
    quiet = np.abs(samples) < SILENCE_LEVEL
    samples = np.where(quiet, rng.uniform(-NOISE_FLOOR, NOISE_FLOOR, size=samples.shape).astype(np.float32), samples)
    try:
        _play_once(samples, device_rate, output)
    except sd.PortAudioError:
        # Bluetooth audio devices can appear or change while the program runs, and PortAudio keeps the old list.
        # Refresh the device list once and try again.
        log.warning("Playback failed; refreshing audio devices and retrying", exc_info=True)
        sd._terminate()
        sd._initialize()
        output = find_device(device, "output") if device else None
        _play_once(samples, device_rate, output)


def _play_once(samples: np.ndarray, rate: int, output: int | None) -> None:
    log.info("Playing on device %s at %d Hz", output if output is not None else "default", rate)
    sd.play(samples, samplerate=rate, device=output)
    sd.wait()
