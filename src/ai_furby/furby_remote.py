"""Main-process handle for the Furby. Sends commands to the Bluetooth child process."""

from __future__ import annotations

import asyncio
import multiprocessing
from typing import Any

from ai_furby.config import Settings
from ai_furby.furby_worker import run_worker
from ai_furby.mood_actions import ACTION_SECONDS

ACTION_REPEAT_SECONDS = 2.5  # used for actions whose length has not been measured


class FurbyRemote:
    def __init__(self, settings: Settings, commands: Any | None = None) -> None:
        self._process = None
        if commands is None:
            # "spawn" starts a clean interpreter, so the child does not inherit the audio library.
            context = multiprocessing.get_context("spawn")
            commands = context.Queue()
            self._process = context.Process(target=run_worker, args=(commands, settings), daemon=True)
        self._commands = commands

    async def __aenter__(self) -> "FurbyRemote":
        self.start()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        self.stop()

    def start(self) -> None:
        if self._process is not None:
            self._process.start()

    def stop(self) -> None:
        self._send("group", False)  # make sure the Furby is left quiet
        self._commands.put(None)
        if self._process is not None:
            self._process.join(timeout=10)
            if self._process.is_alive():
                self._process.terminate()

    def _send(self, *command: Any) -> None:
        self._commands.put(command)

    async def set_antenna_color(self, red: int, green: int, blue: int) -> None:
        self._send("antenna", red, green, blue)

    async def set_group_active(self, active: bool) -> None:
        self._send("group", active)

    async def trigger_action(self, input_id: int, index: int, subindex: int, specific: int) -> None:
        self._send("action", input_id, index, subindex, specific)

    async def act_for(
        self,
        action: tuple[int, int, int, int],
        seconds: float,
        repeat_seconds: float | None = None,
    ) -> None:
        """Keeps the Furby moving for about `seconds`, repeating `action` back to back, with the group on."""
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
