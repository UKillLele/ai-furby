"""Run the voice assistant: python -m ai_furby"""

import asyncio
import logging

from ai_furby.config import Settings


def main() -> None:
    # Imported here, not at module level: the Bluetooth child process re-imports this module on Windows,
    # and it must not load the audio library.
    from ai_furby.voice_loop import run

    logging.basicConfig(level=logging.WARNING)
    try:
        asyncio.run(run(Settings.from_env()))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
