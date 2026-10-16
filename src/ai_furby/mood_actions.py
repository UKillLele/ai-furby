"""Built-in Furby actions that match the assistant's mood. IDs are (input, index, subindex, specific)
from PyFluff's docs/actionlist.md. Some actions include the Furby's own voice clips, so check them by ear."""

MOOD_ACTIONS: dict[str, tuple[int, int, int, int]] = {
    "happy": (24, 4, 0, 1),  # laughing
    "excited": (9, 0, 0, 6),  # spinning noises
    "curious": (55, 2, 17, 0),  # ear point up
    "sad": (9, 2, 0, 2),  # crying shake
    "sleepy": (9, 2, 0, 1),  # shaking yawn
    "silly": (9, 3, 0, 4),  # shake noises
}
