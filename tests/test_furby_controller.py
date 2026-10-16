import asyncio

import pytest

from ai_furby.config import Settings
from ai_furby.furby_controller import FurbyController


class FakeFurby:
    connected = True

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    async def connect(self, address=None, timeout=10.0, retries=3):
        self.calls.append(("connect", address, timeout))

    async def disconnect(self):
        self.calls.append(("disconnect",))

    async def set_antenna_color(self, red, green, blue):
        self.calls.append(("antenna", red, green, blue))

    async def set_lcd_backlight(self, enabled):
        self.calls.append(("backlight", enabled))

    async def trigger_action(self, input, index, subindex, specific):
        self.calls.append(("action", input, index, subindex, specific))


SETTINGS = Settings(furby_address="AA:BB:CC:DD:EE:FF", furby_connect_timeout=5.0, claude_model="claude-haiku-5-5")


def run(coro):
    return asyncio.run(coro)


def test_context_manager_connects_and_disconnects():
    fake = FakeFurby()

    async def scenario():
        async with FurbyController(SETTINGS, client_factory=lambda: fake):
            pass

    run(scenario())
    assert fake.calls == [("connect", "AA:BB:CC:DD:EE:FF", 5.0), ("disconnect",)]


def test_antenna_color_is_forwarded():
    fake = FakeFurby()
    controller = FurbyController(SETTINGS, client_factory=lambda: fake)
    run(controller.set_antenna_color(255, 0, 0))
    assert ("antenna", 255, 0, 0) in fake.calls


@pytest.mark.parametrize("bad", [-1, 256])
def test_antenna_color_rejects_out_of_range(bad):
    controller = FurbyController(SETTINGS, client_factory=FakeFurby)
    with pytest.raises(ValueError):
        run(controller.set_antenna_color(bad, 0, 0))


def test_trigger_action_passes_ids_through():
    fake = FakeFurby()
    controller = FurbyController(SETTINGS, client_factory=lambda: fake)
    run(controller.trigger_action(1, 2, 3, 4))
    assert ("action", 1, 2, 3, 4) in fake.calls


def test_act_for_repeats_action_inside_group_window():
    written: list[tuple] = []

    class Recorder(FakeFurby):
        async def _write_gp(self, data):
            written.append(("gp", data))

        async def trigger_action(self, input, index, subindex, specific):
            written.append(("action", input, index, subindex, specific))

    controller = FurbyController(SETTINGS, client_factory=Recorder)
    run(controller.act_for((9, 0, 0, 6), seconds=0.05, repeat_seconds=0.02))

    assert written[0] == ("gp", bytes.fromhex("e000000001"))
    assert written[-1] == ("gp", bytes.fromhex("e000000000"))
    assert sum(1 for item in written if item[0] == "action") >= 2


def test_reconnects_before_writing_when_link_dropped():
    events: list[str] = []

    class Dropping(FakeFurby):
        connected = False

        async def connect(self, address=None, timeout=10.0, retries=3):
            events.append("connect")
            self.connected = True

        async def trigger_action(self, input, index, subindex, specific):
            events.append("action")

    controller = FurbyController(SETTINGS, client_factory=Dropping)
    run(controller.trigger_action(9, 0, 0, 6))

    assert events == ["connect", "action"]
