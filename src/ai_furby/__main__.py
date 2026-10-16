"""Run the voice assistant: python -m ai_furby"""

import asyncio
import logging

from ai_furby.config import Settings
from ai_furby.voice_loop import run


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    try:
        asyncio.run(run(Settings.from_env()))
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
