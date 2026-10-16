"""Child process that owns the Furby's Bluetooth link.

It runs in its own process and never imports the audio library. On Windows, audio loaded in the same process
breaks the Bluetooth connection, so the two must stay apart.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ai_furby.config import Settings
from ai_furby.furby_controller import FurbyController

log = logging.getLogger(__name__)


def run_worker(commands: Any, settings: Settings) -> None:
    """Process entry point. Reads commands from `commands` until it receives None."""
    logging.basicConfig(level=logging.WARNING)
    asyncio.run(_serve(commands, settings))


async def _serve(commands: Any, settings: Settings) -> None:
    furby = FurbyController(settings)
    try:
        await furby.connect()
    except Exception:
        log.warning("Could not connect to the Furby yet; will retry on the next command", exc_info=True)

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
            else:
                log.warning("Unknown Furby command %r", kind)
        except Exception:  # one failed command should not stop the others
            log.warning("Furby command %r failed", kind, exc_info=True)

    await asyncio.gather(furby.disconnect(), return_exceptions=True)
