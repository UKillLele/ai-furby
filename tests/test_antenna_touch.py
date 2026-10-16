from ai_furby.antenna_touch import is_antenna_movement


def test_antenna_pattern_is_recognized():
    assert is_antenna_movement("21000e0001000000")
    assert is_antenna_movement("21010e0001000000")
    assert is_antenna_movement("21020e0001000000")
    assert is_antenna_movement("21008e0001000000")


def test_pet_and_tickle_are_not_antenna_movement():
    assert not is_antenna_movement("2100040001000000")
    assert not is_antenna_movement("2100030001000000")


def test_idle_reading_is_not_antenna_movement():
    assert not is_antenna_movement("2100000001000000")
