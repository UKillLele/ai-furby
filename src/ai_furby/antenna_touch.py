"""Recognizes antenna movement in the Furby's raw sensor readings.

Captured readings (hex): antenna movement changes the second byte (00, 01, 02, 80) while the third stays 0e.
Pet and tickle change the third byte instead. Only the antenna pattern is used here.
"""

from __future__ import annotations

ANTENNA_THIRD_BYTE = 0x0E


ANTENNA_MOVED_THIRD_BYTE = 0x8E  # seen in some antenna captures with the second byte at 00


def is_antenna_movement(raw_hex: str) -> bool:
    """True when a raw sensor reading looks like the antenna being moved."""
    data = bytes.fromhex(raw_hex)
    if len(data) < 3:
        return False
    if data[2] == ANTENNA_MOVED_THIRD_BYTE:
        return True
    return data[1] != 0 and data[2] == ANTENNA_THIRD_BYTE
