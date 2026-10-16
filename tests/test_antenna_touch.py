from ai_furby.antenna_touch import is_antenna_movement


def test_antenna_movement_readings_are_recognized():
    assert is_antenna_movement("2100800001000000")  # byte 2 at 80
    assert is_antenna_movement("2102000001000000")  # byte 1 at 02, byte 2 at 00
    assert is_antenna_movement("2101000001000000")
    assert is_antenna_movement("21010e0001000000")


def test_pet_and_tickle_are_not_antenna_movement():
    assert not is_antenna_movement("2100040001000000")
    assert not is_antenna_movement("2100050001000000")
    assert not is_antenna_movement("2100030001000000")
    assert not is_antenna_movement("2100010001000000")


def test_idle_and_shake_readings_are_not_antenna_movement():
    assert not is_antenna_movement("2100000001000000")
    assert not is_antenna_movement("21000e0040000000")
