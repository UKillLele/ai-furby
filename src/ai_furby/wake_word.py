"""Always-on wake word listening with openWakeWord. Reads from the shared microphone feed, so it is cheap enough to leave on."""

from __future__ import annotations

import queue
import threading
import time
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

from ai_furby.audio import MicFeed, remove_offset

MODEL_RATE = 16000  # openWakeWord expects 16 kHz audio
FRAME_SECONDS = 0.08  # openWakeWord scores audio in 80 ms frames (1280 samples at 16 kHz)
FRAME_SAMPLES = int(MODEL_RATE * FRAME_SECONDS)


def default_model_path() -> Path:
    """The trained "Hey Furby" model. Built with openWakeWord from data/wakeword/hey_furby_train.yml."""
    return Path(__file__).resolve().parents[2] / "models" / "hey_furby.onnx"


class WakeWordListener:
    def __init__(self, model_path: str | Path, threshold: float = 0.5) -> None:
        from openwakeword.model import Model

        self._model = Model(wakeword_models=[str(model_path)], inference_framework="onnx")
        self._threshold = threshold

    def wait_for_wake(self, feed: MicFeed, timeout: float | None = None, stop: threading.Event | None = None) -> bool:
        """Blocks until the wake word is heard on `feed`. Returns False if `timeout` passes or `stop` is set first.

        The feed stays open afterwards, so speech right after the wake word still reaches the recorder.
        """
        frame_samples = int(feed.rate * FRAME_SECONDS)
        deadline = None if timeout is None else time.monotonic() + timeout
        self._model.reset()
        pending = np.zeros(0, dtype=np.float32)
        while (deadline is None or time.monotonic() < deadline) and not (stop and stop.is_set()):
            try:
                block = feed.read(timeout=0.5)
            except queue.Empty:
                continue
            pending = np.concatenate([pending, remove_offset(block)])
            while pending.size >= frame_samples:
                frame, pending = pending[:frame_samples], pending[frame_samples:]
                scores = self._model.predict(_to_model_rate(frame, feed.rate))
                if any(score >= self._threshold for score in scores.values()):
                    return True
        return False


def _to_model_rate(samples: np.ndarray, rate: int) -> np.ndarray:
    if rate != MODEL_RATE:
        divisor = gcd(rate, MODEL_RATE)
        samples = resample_poly(samples, MODEL_RATE // divisor, rate // divisor)
    samples = samples[:FRAME_SAMPLES]
    return (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
