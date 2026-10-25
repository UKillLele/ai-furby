"""Text-to-speech with Voxtral's preset voice, shaped to sound like a Furby: pitched up and softened at the top."""

from __future__ import annotations

import base64
import io
import wave
from math import gcd
from typing import Any

import numpy as np

from ai_furby.config import Settings
from ai_furby.mistral import API_BASE, auth_headers

SAMPLE_RATE = 16000  # the shaping below was tuned at this rate
MAX_SPEECH_PEAK = 0.7  # fraction of full scale for the loudest sample
MAX_REFERENCE_SECONDS = 25  # the longest reference clip sent for cloning; 39 s was rejected


def read_wav(data: bytes) -> tuple[int, np.ndarray]:
    """Returns (sample rate, mono float samples in [-1, 1]) from 16-bit WAV bytes."""
    with wave.open(io.BytesIO(data), "rb") as wav:
        if wav.getsampwidth() != 2:
            raise ValueError("expected 16-bit WAV from the speech service")
        rate = wav.getframerate()
        channels = wav.getnchannels()
        samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return rate, samples


class TextToSpeech:
    """`client` can be injected so tests run without network access. It needs a `post` method like httpx."""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self._settings = settings
        self._client = client
        self.sample_rate = SAMPLE_RATE
        self._reference_b64 = _load_reference(settings.voice_reference_path)
        if client is None:
            auth_headers(settings)  # fail early with a clear message if the key is missing

    def synthesize(self, text: str) -> np.ndarray:
        """Returns mono int16 samples at `self.sample_rate`."""
        if not text.strip():
            return np.zeros(0, dtype=np.int16)
        client = self._client
        if client is None:
            import httpx  # lazy: keeps imports light when the network is not used

            client = httpx
        request = {"model": self._settings.speech_model, "input": text, "response_format": "wav"}
        if self._reference_b64:
            request["ref_audio"] = self._reference_b64  # cloned voice, sent with each request
        else:
            request["voice_id"] = self._settings.voice_id
        response = client.post(
            f"{API_BASE}/audio/speech",
            headers=auth_headers(self._settings),
            json=request,
            timeout=120,
        )
        response.raise_for_status()
        rate, speech = read_wav(base64.b64decode(response.json()["audio_data"]))
        if rate != SAMPLE_RATE:
            from scipy.signal import resample_poly

            divisor = gcd(rate, SAMPLE_RATE)
            speech = resample_poly(speech, SAMPLE_RATE // divisor, rate // divisor).astype(np.float32)
        speech = self._furby_voice(speech)
        # Keep headroom so no sample hits the int16 limit.
        peak = float(np.max(np.abs(speech))) if speech.size else 0.0
        if peak > 0:
            speech = speech * (self._settings.speech_peak / peak)
        return (np.clip(speech, -1.0, 1.0) * 32767).astype(np.int16)

    def _furby_voice(self, speech: np.ndarray) -> np.ndarray:
        if speech.size == 0:
            return speech
        if self._settings.voice_lowpass_hz < SAMPLE_RATE / 2:
            from scipy.signal import butter, sosfilt

            sos = butter(2, self._settings.voice_lowpass_hz, btype="low", fs=SAMPLE_RATE, output="sos")
            speech = sosfilt(sos, speech).astype(np.float32)
        return speech


def _load_reference(path: str) -> str | None:
    """Reads a reference clip for voice cloning, trims it to the length the service accepts, and encodes it."""
    if not path:
        return None
    with wave.open(path, "rb") as src:
        frames = src.readframes(int(MAX_REFERENCE_SECONDS * src.getframerate()))
        rate = src.getframerate()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as dst:
        dst.setnchannels(1)
        dst.setsampwidth(2)
        dst.setframerate(rate)
        dst.writeframes(frames)
    return base64.b64encode(buffer.getvalue()).decode("ascii")
