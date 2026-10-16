from ai_furby.antenna_touch import is_input


def test_antenna_pet_tickle_and_shake_all_count_as_input():
    assert is_input("2101000001000000")  # antenna
    assert is_input("2100040001000000")  # pet
    assert is_input("2100060001000000")  # tickle
    assert is_input("2100460001000000")  # shake


def test_at_rest_readings_do_not_count():
    assert not is_input("2100000001000000")
    assert not is_input("21000e0001000000")
