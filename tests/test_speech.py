import base64
import io
import wave

import numpy as np
import pytest

from ai_furby.config import Settings
from ai_furby.speech_to_text import SpeechToText, wav_bytes
from ai_furby.text_to_speech import MAX_SPEECH_PEAK, TextToSpeech

SETTINGS = Settings(
    furby_address=None,
    mistral_api_key="test-key",
    voice_reference_path="",
    voice_semitones=0.0,
    voice_lowpass_hz=8000.0,
)


class FakeTranscribe:
    def __init__(self, text: str) -> None:
        self._text = text
        self.received: list[bytes] = []

    def post(self, url, headers=None, data=None, files=None, **_kwargs):
        self.received.append(files["file"][1])
        return _Json({"text": self._text})


class _Json:
    def __init__(self, body: dict) -> None:
        self._body = body

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._body


def _wav_at(rate: int, samples: np.ndarray) -> bytes:
    pcm = (samples * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()


class FakeSpeech:
    def __init__(self, wav: bytes) -> None:
        self._wav = wav
        self.requests: list[dict] = []

    def post(self, url, headers=None, json=None, **_kwargs):
        self.requests.append(json)
        return _Json({"audio_data": base64.b64encode(self._wav).decode("ascii")})


def test_transcribe_sends_wav_and_returns_text():
    client = FakeTranscribe("  hey furby  ")
    stt = SpeechToText(SETTINGS, client=client)

    assert stt.transcribe(np.zeros(16000, dtype=np.float32)) == "hey furby"
    assert client.received[0][:4] == b"RIFF"


def test_transcribe_empty_audio_skips_the_request():
    client = FakeTranscribe("should not be used")
    stt = SpeechToText(SETTINGS, client=client)

    assert stt.transcribe(np.zeros(0, dtype=np.float32)) == ""
    assert client.received == []


def test_wav_bytes_round_trips_16_kHz_mono():
    audio = np.array([0.0, 0.5, -0.5], dtype=np.float32)
    with wave.open(io.BytesIO(wav_bytes(audio)), "rb") as wav:
        assert wav.getframerate() == 16000
        assert wav.getnchannels() == 1


def test_synthesize_returns_int16_at_16_khz_with_headroom():
    samples = np.array([0.1, 0.5, -1.0, 0.2], dtype=np.float32)
    client = FakeSpeech(_wav_at(16000, samples))
    tts = TextToSpeech(SETTINGS, client=client)

    out = tts.synthesize("hi")

    assert tts.sample_rate == 16000
    assert out.dtype == np.int16
    assert int(np.max(np.abs(out))) == int(MAX_SPEECH_PEAK * 32767)
    assert client.requests[0]["voice_id"] == SETTINGS.voice_id


def test_synthesize_resamples_other_rates():
    t = np.arange(24000) / 24000
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    tts = TextToSpeech(SETTINGS, client=FakeSpeech(_wav_at(24000, tone)))

    out = tts.synthesize("hi")

    assert abs(len(out) - 16000) <= 2  # one second at 16 kHz


def test_pitch_shift_keeps_length():
    t = np.arange(8000) / 16000
    tone = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    shifted = Settings(furby_address=None, mistral_api_key="k", voice_reference_path="",
                       voice_semitones=8.0, voice_lowpass_hz=8000.0)
    out = TextToSpeech(shifted, client=FakeSpeech(_wav_at(16000, tone))).synthesize("hi")

    assert len(out) == 8000


def test_empty_text_is_silence_without_a_request():
    client = FakeSpeech(_wav_at(16000, np.zeros(10, dtype=np.float32)))
    tts = TextToSpeech(SETTINGS, client=client)

    assert tts.synthesize("   ").size == 0
    assert client.requests == []


def test_missing_key_fails_early(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "")
    with pytest.raises(RuntimeError, match="MISTRAL_API_KEY is not set"):
        TextToSpeech(Settings(furby_address=None, mistral_api_key=""))


def test_cloned_voice_sends_reference_instead_of_voice_id(tmp_path):
    path = tmp_path / "reference.wav"
    path.write_bytes(_wav_at(16000, np.zeros(16000, dtype=np.float32)))
    client = FakeSpeech(_wav_at(16000, np.zeros(16000, dtype=np.float32)))
    settings = Settings(furby_address=None, mistral_api_key="k", voice_reference_path=str(path),
                        voice_semitones=0.0, voice_lowpass_hz=8000.0)

    TextToSpeech(settings, client=client).synthesize("hi")

    assert "ref_audio" in client.requests[0]
    assert "voice_id" not in client.requests[0]
