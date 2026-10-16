"""Recognizes antenna movement in the Furby's raw sensor readings (hex, 8 bytes).

Captured antenna movement puts a nonzero value in byte 2 (80 or 8e), or a nonzero value in byte 1 with byte 2 at 00 or 0e.
Pet and tickle change byte 2 to other values (01 to 06), so they are not counted.
"""

from __future__ import annotations

ANTENNA_BYTE2_VALUES = (0x80, 0x8E)  # antenna movement, byte 1 at rest
ANTENNA_BYTE2_WITH_BYTE1 = (0x00, 0x0E)  # antenna movement, byte 1 nonzero


def is_antenna_movement(raw_hex: str) -> bool:
    """True when a raw sensor reading looks like the antenna being moved."""
    data = bytes.fromhex(raw_hex)
    if len(data) < 3:
        return False
    if data[2] in ANTENNA_BYTE2_VALUES:
        return True
    return data[1] != 0 and data[2] in ANTENNA_BYTE2_WITH_BYTE1
