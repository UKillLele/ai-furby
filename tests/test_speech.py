from types import SimpleNamespace

import numpy as np

from ai_furby.config import Settings
from ai_furby.speech_to_text import SpeechToText
from ai_furby.text_to_speech import TextToSpeech


class FakeWhisper:
    def __init__(self, texts: list[str]) -> None:
        self._texts = texts
        self.received: list[np.ndarray] = []

    def transcribe(self, audio, **_kwargs):
        self.received.append(audio)
        return ([SimpleNamespace(text=t, no_speech_prob=0.1, avg_logprob=-0.2) for t in self._texts], None)


class FakeVoice:
    config = SimpleNamespace(sample_rate=22050)

    def synthesize(self, _text):
        yield SimpleNamespace(audio_int16_bytes=np.array([1, 2], dtype=np.int16).tobytes())
        yield SimpleNamespace(audio_int16_bytes=np.array([3], dtype=np.int16).tobytes())


def test_transcribe_joins_segments_and_passes_float32():
    whisper = FakeWhisper([" hello ", "there "])
    stt = SpeechToText(Settings(), model=whisper)

    assert stt.transcribe(np.zeros(16000, dtype=np.float64)) == "hello there"
    assert whisper.received[0].dtype == np.float32


def test_transcribe_empty_returns_empty_string():
    stt = SpeechToText(Settings(), model=FakeWhisper([]))
    assert stt.transcribe(np.zeros(10, dtype=np.float32)) == ""


def test_synthesize_concatenates_chunks_and_reports_sample_rate():
    tts = TextToSpeech(Settings(), voice=FakeVoice())

    assert tts.sample_rate == 22050
    assert tts.synthesize("hi").tolist() == [1, 2, 3]


def test_transcribe_drops_segments_whisper_is_unsure_about():
    class UnsureWhisper:
        def transcribe(self, _audio, **_kwargs):
            return (
                [
                    SimpleNamespace(text="Thank you for watching", no_speech_prob=0.9, avg_logprob=-0.3),
                    SimpleNamespace(text="can you hear me", no_speech_prob=0.05, avg_logprob=-0.2),
                    SimpleNamespace(text="maybe", no_speech_prob=0.1, avg_logprob=-2.5),
                ],
                None,
            )

    stt = SpeechToText(Settings(), model=UnsureWhisper())
    assert stt.transcribe(np.zeros(10, dtype=np.float32)) == "can you hear me"
