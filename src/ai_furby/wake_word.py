"""Always-on wake word listening with openWakeWord. Runs on small audio blocks, so it is cheap enough to leave on."""

from __future__ import annotations

import threading
import time
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from ai_furby.audio import capture_rate, find_input_device, is_alsa_device, open_microphone, remove_offset

MODEL_RATE = 16000  # openWakeWord expects 16 kHz audio
FRAME_SECONDS = 0.08  # openWakeWord scores audio in 80 ms frames (1280 samples at 16 kHz)
FRAME_SAMPLES = int(MODEL_RATE * FRAME_SECONDS)


def default_model_path() -> Path:
    """The trained "Hey Furby" model. Built with openWakeWord from data/wakeword/hey_furby_train.yml."""
    return Path(__file__).resolve().parents[2] / "models" / "hey_furby.onnx"


class WakeWordListener:
    def __init__(self, model_path: str | Path, threshold: float = 0.5, device: str | None = None) -> None:
        from openwakeword.model import Model

        self._model = Model(wakeword_models=[str(model_path)], inference_framework="onnx")
        self._threshold = threshold
        if not device:
            self._device = None
        elif is_alsa_device(device):
            self._device = device
        else:
            self._device = find_input_device(device)

    def wait_for_wake(self, timeout: float | None = None, stop: threading.Event | None = None) -> bool:
        """Blocks until the wake word is heard. Returns False if `timeout` passes or `stop` is set first."""
        device_rate = capture_rate(self._device)
        block = int(device_rate * FRAME_SECONDS)
        deadline = None if timeout is None else time.monotonic() + timeout
        self._model.reset()
        with open_microphone(self._device, device_rate, block) as stream:
            while (deadline is None or time.monotonic() < deadline) and not (stop and stop.is_set()):
                data, _overflowed = stream.read(block)
                frame = _to_model_rate(remove_offset(data[:, 0]), device_rate)
                scores = self._model.predict(frame)
                if any(score >= self._threshold for score in scores.values()):
                    return True
        return False


def _to_model_rate(samples: np.ndarray, rate: int) -> np.ndarray:
    if rate != MODEL_RATE:
        divisor = gcd(rate, MODEL_RATE)
        samples = resample_poly(samples, MODEL_RATE // divisor, rate // divisor)
    samples = samples[:FRAME_SAMPLES]
    return (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
