from ai_furby.mood_actions import ACTION_SECONDS, MOUTH_ACTIONS, pick_mouth_action


def test_every_mouth_action_has_a_measured_length():
    for action in MOUTH_ACTIONS:
        assert action in ACTION_SECONDS


def test_mouth_action_never_repeats_twice_in_a_row():
    picks = [pick_mouth_action() for _ in range(200)]
    assert all(a != b for a, b in zip(picks, picks[1:]))


def test_mouth_actions_come_from_the_approved_list():
    assert {pick_mouth_action() for _ in range(200)} <= set(MOUTH_ACTIONS)


def test_greetings_come_from_the_list():
    from ai_furby.mood_actions import GREETINGS, pick_greeting

    assert {pick_greeting() for _ in range(100)} <= set(GREETINGS)
