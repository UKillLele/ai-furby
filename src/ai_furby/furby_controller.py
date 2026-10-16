"""Adapter around PyFluff. The rest of the assistant talks to this class, not to PyFluff directly."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

from ai_furby.config import Settings


log = logging.getLogger(__name__)

ACTION_REPEAT_SECONDS = 2.5  # roughly how long one built-in action takes; tune by watching the Furby


class FurbyController:
    """Thin async wrapper over PyFluff's FurbyConnect.

    `client_factory` exists so tests can inject a fake instead of a real BLE device.
    """

    def __init__(
        self,
        settings: Settings,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._settings = settings
        if client_factory is None:
            from pyfluff.furby import FurbyConnect  # imported lazily so tests run without Bluetooth

            client_factory = FurbyConnect
        self._client = client_factory()

    async def __aenter__(self) -> "FurbyController":
        await self.connect()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.disconnect()

    async def connect(self) -> None:
        await self._client.connect(
            address=self._settings.furby_address,
            timeout=self._settings.furby_connect_timeout,
        )

    async def disconnect(self) -> None:
        await self._client.disconnect()

    async def _write(self, operation: Callable[[], Awaitable[None]]) -> None:
        """Runs one command. If the Furby's Bluetooth link has dropped, reconnect from scratch and try once more."""
        try:
            await operation()
        except RuntimeError as error:
            if "Not connected" not in str(error):
                raise
            log.info("Furby link dropped; reconnecting")
            with contextlib.suppress(Exception):
                await self._client.disconnect()
            await self.connect()
            await operation()

    async def set_antenna_color(self, red: int, green: int, blue: int) -> None:
        _check_channel("red", red)
        _check_channel("green", green)
        _check_channel("blue", blue)
        await self._write(lambda: self._client.set_antenna_color(red, green, blue))

    async def set_backlight(self, enabled: bool) -> None:
        await self._client.set_lcd_backlight(enabled)

    async def trigger_action(self, input_id: int, index: int, subindex: int, specific: int) -> None:
        """Run a built-in Furby action. IDs come from PyFluff's docs/actionlist.md."""
        await self._write(lambda: self._client.trigger_action(input_id, index, subindex, specific))

    async def set_group_active(self, active: bool) -> None:
        """Turns the Furby's own behavior on or off (GeneralPlus command 0xe0, per PyFluff's protocol docs).

        Triggered actions only run while this is on, so they are played inside a short on-window.
        """
        command = bytes([0xE0, 0x00, 0x00, 0x00, 0x01 if active else 0x00])
        await self._write(lambda: self._client._write_gp(command))

    async def act_for(
        self,
        action: tuple[int, int, int, int],
        seconds: float,
        repeat_seconds: float | None = None,
    ) -> None:
        """Keeps the Furby moving for about `seconds`, repeating `action` back to back.

        The group is on only for this window, then turned off again. Each repeat waits for the action's
        measured length, so the movements run one after another without overlapping.
        """
        from ai_furby.mood_actions import ACTION_SECONDS

        step = repeat_seconds if repeat_seconds is not None else ACTION_SECONDS.get(action, ACTION_REPEAT_SECONDS)
        await self.set_group_active(True)
        try:
            remaining = seconds
            while remaining > 0:
                await self.trigger_action(*action)
                wait = min(step, remaining)
                await asyncio.sleep(wait)
                remaining -= wait
        finally:
            await self.set_group_active(False)

    def sensor_events(self) -> AsyncIterator[Any]:
        """Yield raw sensor readings (touch, shake, antenna joystick) as PyFluff produces them."""
        return self._client.sensor_stream()


def _check_channel(name: str, value: int) -> None:
    if not 0 <= value <= 255:
        raise ValueError(f"{name} must be between 0 and 255, got {value}")
