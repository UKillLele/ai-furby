"""Microphone capture and speaker playback. Uses a simple loudness threshold to find where speech starts and ends."""

from __future__ import annotations

import numpy as np
import sounddevice as sd

INPUT_SAMPLE_RATE = 16000  # what faster-whisper expects
BLOCK_SECONDS = 0.1


def record_utterance(
    threshold: float = 0.02,
    silence_seconds: float = 1.0,
    max_seconds: float = 15.0,
    wait_seconds: float = 30.0,
    device: int | str | None = None,
) -> np.ndarray:
    """Waits for speech, records until a pause, and returns mono float32 audio.

    Returns an empty array if no speech starts within `wait_seconds`.
    """
    block = int(INPUT_SAMPLE_RATE * BLOCK_SECONDS)
    silent_blocks_to_stop = int(silence_seconds / BLOCK_SECONDS)
    max_blocks = int(max_seconds / BLOCK_SECONDS)
    wait_blocks = int(wait_seconds / BLOCK_SECONDS)

    captured: list[np.ndarray] = []
    speaking = False
    silent_run = 0

    with sd.InputStream(samplerate=INPUT_SAMPLE_RATE, channels=1, dtype="float32", blocksize=block, device=device) as stream:
        for index in range(max_blocks + wait_blocks):
            data, _overflowed = stream.read(block)
            samples = data[:, 0]
            loud = float(np.sqrt(np.mean(np.square(samples)))) > threshold

            if not speaking:
                if index >= wait_blocks:
                    break
                if loud:
                    speaking = True
                    captured.append(samples.copy())
                continue

            captured.append(samples.copy())
            silent_run = 0 if loud else silent_run + 1
            if silent_run >= silent_blocks_to_stop:
                break

    if not captured:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(captured)


def play(samples: np.ndarray, sample_rate: int) -> None:
    """Plays int16 or float32 mono samples and waits until they finish."""
    if samples.dtype == np.int16:
        samples = samples.astype(np.float32) / 32768.0
    sd.play(samples, samplerate=sample_rate)
    sd.wait()
