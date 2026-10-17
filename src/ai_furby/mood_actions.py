"""Built-in Furby actions that match the assistant's mood. IDs are (input, index, subindex, specific)
from PyFluff's docs/actionlist.md. Some actions include the Furby's own voice clips, so check them by ear."""

import random

MOOD_ACTIONS: dict[str, tuple[int, int, int, int]] = {
    "happy": (24, 4, 0, 1),  # laughing
    "excited": (9, 0, 0, 6),  # spinning noises
    "curious": (24, 1, 3, 0),  # flip upside down (movement, no voice when the Furby speaker is off)
    "sad": (10, 1, 1, 0),  # crying "yolo"; replaces crying shake, which closed the eyes
    "sleepy": (9, 2, 0, 1),  # shaking yawn
    "silly": (9, 3, 0, 4),  # shake noises
}

# Measured end-to-end length of each action in seconds (trigger to the Furby's SequenceEnded signal).
ACTION_SECONDS: dict[tuple[int, int, int, int], float] = {
    (26, 0, 0, 2): 4.00,  # mouth full mumbling (beak)
    (8, 1, 0, 7): 1.53,  # oo-nye talking (mouth)
    (24, 2, 1, 5): 4.49,  # silly singing (mouth)
    (9, 0, 0, 6): 4.33,  # spinning noises
    (9, 2, 0, 2): 2.27,  # crying shake
    (10, 1, 1, 0): 3.5,  # crying "yolo" (Bluetooth end signal: 3.06 to 4.30 s over 3 runs, mean 3.54 s)
    (24, 4, 0, 1): 4.27,  # laughing
    (24, 1, 3, 0): 4.23,  # flip upside down
    (9, 2, 0, 1): 3.58,  # shaking yawn
    (9, 3, 0, 4): 2.64,  # shake noises
    (15, 0, 1, 6): 4.92,  # smacking lips (mouth)
    (26, 0, 0, 3): 2.92,  # baby talk (mouth)
    (26, 0, 0, 4): 2.97,  # more baby talk (mouth)
    (8, 1, 0, 6): 3.22,  # talkin' bout (mouth)
    (30, 3, 0, 0): 12.19,  # put to sleep (speaking)
}

# Mouth actions for the rest of a reply, chosen by ear. Each one moves the mouth; they have a brief
# eyes-only pause at the end. Sing songing was rejected by ear, so it is left out. Gibberish is not rated yet.
MOUTH_ACTIONS: list[tuple[int, int, int, int]] = [
    (15, 0, 1, 6),  # smacking lips
    (26, 0, 0, 3),  # baby talk
    (26, 0, 0, 4),  # more baby talk
    (8, 1, 0, 6),  # talkin' bout
    (8, 1, 0, 7),  # oo-nye talking
    (26, 0, 0, 2),  # mouth full mumbling
    (24, 2, 1, 5),  # silly singing
    (30, 3, 0, 0),  # put to sleep (speaking; does not close the eyelids)
]

_last_mouth_action: tuple[int, int, int, int] | None = None


def pick_mouth_action() -> tuple[int, int, int, int]:
    """Picks a mouth action at random, never the same one twice in a row, so the expression doesn't go stale."""
    global _last_mouth_action
    choices = [action for action in MOUTH_ACTIONS if action != _last_mouth_action]
    _last_mouth_action = random.choice(choices)
    return _last_mouth_action



GREETINGS = (
    "What's up?",
    "What can I help you with?",
    "Hey, what's up?",
    "What do you need?",
    "I'm listening. What's up?",
)


def pick_greeting() -> str:
    """A short "what's up" line for when Furby is woken without a question."""
    return random.choice(GREETINGS)
