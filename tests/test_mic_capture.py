import numpy as np

from ai_furby.audio import ARECORD_RATE, _s32_to_float, capture_rate, is_alsa_device


def test_alsa_names_are_recognized():
    assert is_alsa_device("plughw:2,0")
    assert is_alsa_device("hw:1,0")
    assert not is_alsa_device("voicehat")
    assert not is_alsa_device(None)
    assert not is_alsa_device(3)


def test_alsa_device_uses_arecord_rate_without_querying_portaudio():
    assert capture_rate("plughw:2,0") == ARECORD_RATE


def test_s32_bytes_become_float_samples_in_range():
    raw = np.array([0, 2**30, -(2**31)], dtype="<i4").tobytes()
    samples = _s32_to_float(raw)
    assert samples.dtype == np.float32
    np.testing.assert_allclose(samples, [0.0, 0.5, -1.0])
