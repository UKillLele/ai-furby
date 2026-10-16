"""Recognizes any input on the Furby from its raw sensor readings (hex, 8 bytes).

Any reading other than the at-rest baseline counts as input: antenna movement, pet, tickle, or shake.
The at-rest readings are the ones the Furby sends when nothing is touching it.
"""

from __future__ import annotations

AT_REST = {"2100000001000000", "21000e0001000000", "21000e0040000000"}


def is_input(raw_hex: str) -> bool:
    """True when a raw sensor reading is anything other than the at-rest baseline."""
    return raw_hex.lower() not in AT_REST
