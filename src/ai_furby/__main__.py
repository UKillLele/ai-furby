"""Run the voice assistant: python -m ai_furby"""

import asyncio
import logging
import sys

from ai_furby.config import Settings


def _ignore_loop_shutdown_noise(unraisable) -> None:
    """Python's asyncio cleanup prints a harmless error when the program exits on Windows. Hide only that one."""
    if isinstance(unraisable.exc_value, AttributeError) and "'close'" in str(unraisable.exc_value):
        return
    sys.__unraisablehook__(unraisable)


def main() -> None:
    # Imported here, not at module level: the Bluetooth child process re-imports this module on Windows,
    # and it must not load the audio library.
    from ai_furby.voice_loop import run

    sys.unraisablehook = _ignore_loop_shutdown_noise
    logging.basicConfig(level=logging.WARNING)
    for name in ("ai_furby.audio", "ai_furby.voice_loop", "ai_furby.furby_worker"):
        logging.getLogger(name).setLevel(logging.INFO)  # which output device and mood each reply uses
    try:
        asyncio.run(run(Settings.from_env()))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
