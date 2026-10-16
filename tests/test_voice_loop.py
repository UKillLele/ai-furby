from ai_furby.voice_loop import is_stop_phrase


def test_goodbye_with_punctuation_stops():
    assert is_stop_phrase("Goodbye, Furby.")
    assert is_stop_phrase("goodbye furby!")
    assert is_stop_phrase("  Bye,  Furby  ")


def test_ordinary_speech_does_not_stop():
    assert not is_stop_phrase("Hello, Furby.")
    assert not is_stop_phrase("Tell me a joke about goodbye.")
