"""Recognizes antenna movement in the Furby's raw sensor readings (hex, 8 bytes).

The second byte moves away from 00 when the antenna moves (captured values 01, 02, 10, 20, 80).
Byte 3 holds a value left by the last pet, tickle, or other touch, so it is not used to detect the antenna.
A separate pattern (byte 3 at 80 or 8e with byte 2 at 00) was also seen when the antenna was moved.
"""

from __future__ import annotations

ANTENNA_BYTE2_VALUES = (0x80, 0x8E)


def is_antenna_movement(raw_hex: str) -> bool:
    """True when a raw sensor reading looks like the antenna being moved."""
    data = bytes.fromhex(raw_hex)
    if len(data) < 3:
        return False
    if data[1] != 0:
        return True
    return data[2] in ANTENNA_BYTE2_VALUES
