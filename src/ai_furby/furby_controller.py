"""Adapter around PyFluff. The rest of the assistant talks to this class, not to PyFluff directly."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any

from ai_furby.config import Settings


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

    async def set_antenna_color(self, red: int, green: int, blue: int) -> None:
        _check_channel("red", red)
        _check_channel("green", green)
        _check_channel("blue", blue)
        await self._client.set_antenna_color(red, green, blue)

    async def set_backlight(self, enabled: bool) -> None:
        await self._client.set_lcd_backlight(enabled)

    async def trigger_action(self, input_id: int, index: int, subindex: int, specific: int) -> None:
        """Run a built-in Furby action. IDs come from PyFluff's docs/actionlist.md."""
        await self._client.trigger_action(input_id, index, subindex, specific)

    def sensor_events(self) -> AsyncIterator[Any]:
        """Yield raw sensor readings (touch, shake, antenna joystick) as PyFluff produces them."""
        return self._client.sensor_stream()


def _check_channel(name: str, value: int) -> None:
    if not 0 <= value <= 255:
        raise ValueError(f"{name} must be between 0 and 255, got {value}")
