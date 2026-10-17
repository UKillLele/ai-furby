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


def test_reconnects_and_retries_when_link_dropped():
    events: list[str] = []

    class Dropping(FakeFurby):
        def __init__(self) -> None:
            super().__init__()
            self.dropped_once = False

        async def connect(self, address=None, timeout=10.0, retries=3):
            events.append("connect")

        async def disconnect(self):
            events.append("disconnect")

        async def trigger_action(self, input, index, subindex, specific):
            if not self.dropped_once:
                self.dropped_once = True
                events.append("dropped")
                raise RuntimeError("Not connected to Furby")
            events.append("action")

    controller = FurbyController(SETTINGS, client_factory=Dropping)
    run(controller.trigger_action(9, 0, 0, 6))

    assert events == ["dropped", "disconnect", "connect", "action"]


def test_command_reconnects_with_a_single_quick_attempt():
    retries_seen: list[int] = []

    class Dropping(FakeFurby):
        def __init__(self) -> None:
            super().__init__()
            self.dropped_once = False

        async def connect(self, address=None, timeout=10.0, retries=3):
            retries_seen.append(retries)

        async def trigger_action(self, input, index, subindex, specific):
            if not self.dropped_once:
                self.dropped_once = True
                raise RuntimeError("Not connected to Furby")

    controller = FurbyController(SETTINGS, client_factory=Dropping)
    run(controller.trigger_action(9, 0, 0, 6))

    assert retries_seen == [1]


def test_reconnect_check_connects_when_disconnected_and_reports_once(capsys):
    from ai_furby.furby_worker import _reconnect_if_needed

    class Asleep(FakeFurby):
        connected = False

        async def connect(self, address=None, timeout=10.0, retries=3):
            raise RuntimeError("Device not found")

    class Back(FakeFurby):
        connected = False

        async def connect(self, address=None, timeout=10.0, retries=3):
            self.connected = True

    state = {"connected": True}
    controller = FurbyController(SETTINGS, client_factory=Asleep)
    run(_reconnect_if_needed(controller, state))
    run(_reconnect_if_needed(controller, state))
    assert "not reachable" in capsys.readouterr().out
    assert state["connected"] is False

    controller = FurbyController(SETTINGS, client_factory=Back)
    run(_reconnect_if_needed(controller, state))
    assert "reconnected" in capsys.readouterr().out
    assert state["connected"] is True


def test_reconnect_check_does_nothing_while_connected():
    from ai_furby.furby_worker import _reconnect_if_needed

    class Linked(FakeFurby):
        connected = True

        async def connect(self, address=None, timeout=10.0, retries=3):
            raise AssertionError("should not reconnect")

    controller = FurbyController(SETTINGS, client_factory=Linked)
    run(_reconnect_if_needed(controller, {"connected": True}))


def test_eyes_are_turned_back_on_after_reconnect():
    from ai_furby.furby_worker import _reconnect_if_needed

    backlight_calls: list[bool] = []

    class Back(FakeFurby):
        connected = False

        async def connect(self, address=None, timeout=10.0, retries=3):
            self.connected = True

        async def set_lcd_backlight(self, enabled):
            backlight_calls.append(enabled)

    controller = FurbyController(SETTINGS, client_factory=Back)
    run(_reconnect_if_needed(controller, {"connected": False, "eyes_on": True}))
    assert backlight_calls == [True]


def test_eyes_stay_off_after_reconnect_when_asleep():
    from ai_furby.furby_worker import _reconnect_if_needed

    backlight_calls: list[bool] = []

    class Back(FakeFurby):
        connected = False

        async def connect(self, address=None, timeout=10.0, retries=3):
            self.connected = True

        async def set_lcd_backlight(self, enabled):
            backlight_calls.append(enabled)

    controller = FurbyController(SETTINGS, client_factory=Back)
    run(_reconnect_if_needed(controller, {"connected": False, "eyes_on": False}))
    assert backlight_calls == []


def test_eyes_are_switched_on_before_every_action():
    order: list[str] = []

    class Recording(FakeFurby):
        async def set_lcd_backlight(self, enabled):
            order.append(f"eyes:{enabled}")

        async def trigger_action(self, input, index, subindex, specific):
            order.append("action")

    controller = FurbyController(SETTINGS, client_factory=Recording)
    run(controller.trigger_action(9, 0, 0, 6))
    run(controller.trigger_action(24, 4, 0, 1))

    assert order == ["eyes:True", "action", "eyes:True", "action"]
