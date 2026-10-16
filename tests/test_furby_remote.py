import asyncio
from queue import Queue

from ai_furby.config import Settings
from ai_furby.furby_remote import FurbyRemote

SETTINGS = Settings(furby_address=None)


def test_act_for_sends_group_on_repeats_then_group_off():
    commands: Queue = Queue()
    remote = FurbyRemote(SETTINGS, commands=commands)

    asyncio.run(remote.act_for((9, 0, 0, 6), seconds=0.05, repeat_seconds=0.02))

    sent = []
    while not commands.empty():
        sent.append(commands.get())
    assert sent[0] == ("group", True)
    assert sent[-1] == ("group", False)
    assert sum(1 for c in sent if c[0] == "action") >= 2


def test_antenna_color_is_forwarded():
    commands: Queue = Queue()
    remote = FurbyRemote(SETTINGS, commands=commands)

    asyncio.run(remote.set_antenna_color(255, 0, 0))

    assert commands.get() == ("antenna", 255, 0, 0)
