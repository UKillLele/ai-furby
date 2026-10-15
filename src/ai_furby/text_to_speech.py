"""Local text-to-speech with Piper. Voice models are downloaded once into models/piper/."""

from __future__ import annotations

from typing import Any

import numpy as np

from ai_furby.config import Settings


class TextToSpeech:
    def __init__(self, settings: Settings, voice: Any | None = None) -> None:
        if voice is None:
            from piper import PiperVoice  # lazy: keeps imports light when TTS is not used

            voice = PiperVoice.load(settings.piper_model_path)
        self._voice = voice
        self.sample_rate: int = voice.config.sample_rate

    def synthesize(self, text: str) -> np.ndarray:
        """Returns mono int16 samples at `self.sample_rate`."""
        chunks = [np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16) for chunk in self._voice.synthesize(text)]
        if not chunks:
            return np.zeros(0, dtype=np.int16)
        return np.concatenate(chunks)
