from ai_furby.voice_loop import is_stop_phrase


def test_goodbye_with_punctuation_stops():
    assert is_stop_phrase("Goodbye, Furby.")
    assert is_stop_phrase("goodbye furby!")
    assert is_stop_phrase("  Bye,  Furby  ")


def test_ordinary_speech_does_not_stop():
    assert not is_stop_phrase("Hello, Furby.")
    assert not is_stop_phrase("Tell me a joke about goodbye.")


def test_misheard_name_still_stops():
    assert is_stop_phrase("Goodbye, Kirby.")
    assert is_stop_phrase("bye kirby")
    assert is_stop_phrase("good bye fury")


def test_bare_goodbye_does_not_stop():
    assert not is_stop_phrase("Goodbye.")
    assert not is_stop_phrase("Hello, Kirby.")
