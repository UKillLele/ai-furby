"""Speech-to-text with Voxtral, Mistral's hosted transcription model. Audio is sent as 16 kHz mono WAV."""

from __future__ import annotations

import io
import wave
from typing import Any

import numpy as np

from ai_furby.config import Settings
from ai_furby.mistral import API_BASE, auth_headers

SAMPLE_RATE = 16000  # what the recorder produces


def wav_bytes(audio: np.ndarray) -> bytes:
    """Mono float audio in [-1, 1] as a 16-bit WAV file in memory."""
    pcm = (np.clip(audio.astype(np.float32), -1.0, 1.0) * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()


class SpeechToText:
    """`client` can be injected so tests run without network access. It needs a `post` method like httpx."""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self._settings = settings
        self._client = client
        if client is None:
            auth_headers(settings)  # fail early with a clear message if the key is missing

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""
        client = self._client
        if client is None:
            import httpx  # lazy: keeps imports light when the network is not used

            client = httpx
        response = client.post(
            f"{API_BASE}/audio/transcriptions",
            headers=auth_headers(self._settings),
            data={"model": self._settings.transcribe_model},
            files={"file": ("speech.wav", wav_bytes(audio), "audio/wav")},
            timeout=60,
        )
        response.raise_for_status()
        return str(response.json().get("text", "")).strip()
