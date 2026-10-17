"""Child process that owns the Furby's Bluetooth link.

It runs in its own process and never imports the audio library. On Windows, audio loaded in the same process
breaks the Bluetooth connection, so the two must stay apart.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
from typing import Any

from ai_furby.config import Settings
from ai_furby.furby_controller import FurbyController

log = logging.getLogger(__name__)


def run_worker(commands: Any, settings: Settings, events: Any | None = None) -> None:
    """Process entry point. Reads commands from `commands` until it receives None.

    Raw sensor readings are sent back on `events` when given.
    """
    logging.basicConfig(level=logging.WARNING)
    # Ctrl+C reaches every process in the console. The parent shuts this process down cleanly with None instead.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    asyncio.run(_serve(commands, settings, events))


RECONNECT_SECONDS = 10.0  # how often to look for a Furby that is asleep or out of range


async def _reapply_eyes(furby: FurbyController, state: dict[str, bool]) -> None:
    """Keeps the eyes lit while the assistant is awake. The Furby can lose this setting on its own or after a reconnect."""
    if state.get("eyes_on"):
        with contextlib.suppress(Exception):
            await furby.set_backlight(True)


async def _reconnect_if_needed(furby: FurbyController, state: dict[str, bool]) -> None:
    """One check: if the Furby is not connected, try to connect. Prints once when it is lost and once when it returns."""
    if furby.connected:
        if not state["connected"]:
            print("Furby: reconnected.", flush=True)
            state["connected"] = True
            await _reapply_eyes(furby, state)
        return
    if state["connected"]:
        print("Furby: not reachable. It may be asleep; touch its antenna or power it on to reconnect.", flush=True)
        state["connected"] = False
    try:
        await furby.connect(retries=1)
    except Exception:
        log.debug("Furby still unreachable", exc_info=True)
        return
    print("Furby: reconnected.", flush=True)
    state["connected"] = True
    await _reapply_eyes(furby, state)


async def _keep_connected(furby: FurbyController, state: dict[str, bool]) -> None:
    while True:
        await _reconnect_if_needed(furby, state)
        if state["connected"]:
            await _reapply_eyes(furby, state)
        await asyncio.sleep(RECONNECT_SECONDS)


async def _serve(commands: Any, settings: Settings, events: Any | None = None) -> None:
    furby = FurbyController(settings)
    try:
        await furby.connect()
    except Exception:
        log.warning("Could not connect to the Furby yet; will keep trying", exc_info=True)
    state = {"connected": True, "eyes_on": False}  # eyes_on: what the assistant last asked for
    reconnect_task = asyncio.create_task(_keep_connected(furby, state))

    sensor_task = asyncio.create_task(_forward_sensors(furby, events)) if events is not None else None
    if events is not None:
        # Sleep mask on/off: the Furby sends these as GeneralPlus notifications (20:13 added, 20:14 removed).
        furby._client.add_gp_callback(lambda data: _forward_mask(data, events))

    while True:
        command = await asyncio.to_thread(commands.get)
        if command is None:
            break
        kind, *args = command
        try:
            if kind == "antenna":
                await furby.set_antenna_color(*args)
            elif kind == "group":
                await furby.set_group_active(*args)
            elif kind == "action":
                await furby.trigger_action(*args)
            elif kind == "backlight":
                state["eyes_on"] = bool(args[0])
                await furby.set_backlight(*args)
            else:
                log.warning("Unknown Furby command %r", kind)
        except Exception:  # one failed command should not stop the others
            log.warning("Furby command %r failed", kind, exc_info=True)

    # Leave the Furby in its quiet state rather than whatever state the last command set.
    with contextlib.suppress(Exception):
        await furby.set_group_active(False)
    reconnect_task.cancel()
    if sensor_task is not None:
        sensor_task.cancel()
    await asyncio.gather(furby.disconnect(), return_exceptions=True)


MASK_ON = bytes([0x20, 0x13])
MASK_OFF = bytes([0x20, 0x14])


def _forward_mask(data: bytes, events: Any) -> None:
    """Sends sleep mask events to the parent as "mask:on" / "mask:off"."""
    if data[:2] == MASK_ON:
        events.put("mask:on")
    elif data[:2] == MASK_OFF:
        events.put("mask:off")


async def _forward_sensors(furby: FurbyController, events: Any) -> None:
    """Sends each raw sensor reading to the parent process. Ends quietly if the sensor stream stops."""
    try:
        async for sample in furby._client.sensor_stream():
            events.put(sample.raw_data)
    except Exception:
        log.warning("Furby sensor stream stopped", exc_info=True)
