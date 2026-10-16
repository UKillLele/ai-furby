"""Built-in Furby actions that match the assistant's mood. IDs are (input, index, subindex, specific)
from PyFluff's docs/actionlist.md. Some actions include the Furby's own voice clips, so check them by ear."""

MOOD_ACTIONS: dict[str, tuple[int, int, int, int]] = {
    "happy": (24, 4, 0, 1),  # laughing
    "excited": (9, 0, 0, 6),  # spinning noises
    "curious": (24, 1, 3, 0),  # flip upside down (movement, no voice when the Furby speaker is off)
    "sad": (9, 2, 0, 2),  # crying shake
    "sleepy": (9, 2, 0, 1),  # shaking yawn
    "silly": (9, 3, 0, 4),  # shake noises
}

# Measured end-to-end length of each action in seconds (trigger to the Furby's SequenceEnded signal).
ACTION_SECONDS: dict[tuple[int, int, int, int], float] = {
    (26, 0, 0, 2): 4.00,  # mouth full mumbling (beak)
    (8, 1, 0, 7): 1.53,  # oo-nye talking (mouth)
    (9, 0, 0, 6): 4.33,  # spinning noises
    (9, 2, 0, 2): 2.27,  # crying shake
    (24, 4, 0, 1): 4.27,  # laughing
    (24, 1, 3, 0): 4.23,  # flip upside down
    (9, 2, 0, 1): 3.58,  # shaking yawn
    (9, 3, 0, 4): 2.64,  # shake noises
}

# Mouth movement for the rest of a reply, chosen by ear: short, with a brief eyes-only pause at the end.
# Its voice is silent while the Furby's speaker is off.
BEAK_ACTION = (8, 1, 0, 7)  # oo-nye talking
