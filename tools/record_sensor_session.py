"""Record labeled Furby sensor readings so the touch detector can be tuned.

Usage: python tools/record_sensor_session.py LABEL [count] [seconds]
Each take records raw readings for `seconds` (default 6) after a short "GO" prompt, then saves them as JSON
under data/sensors/ with the label, so antenna, pet, tickle, and shake readings can be compared.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

from ai_furby.config import Settings
from ai_furby.furby_controller import FurbyController

OUT_DIR = Path("data/sensors")


async def take(furby, label: str, seconds: float) -> dict:
    rows: list[tuple[float, str]] = []
    start = time.monotonic()

    async def read():
        async for sample in furby._client.sensor_stream():
            rows.append((round(time.monotonic() - start, 3), sample.raw_data))

    task = asyncio.create_task(read())
    await asyncio.sleep(seconds)
    task.cancel()
    return {"label": label, "seconds": seconds, "readings": rows}


async def main() -> None:
    label = sys.argv[1]
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    async with FurbyController(Settings.from_env()) as furby:
        for n in range(count):
            print(f"Take {n + 1} of {count}: get ready to {label.upper()} ...", flush=True)
            await asyncio.sleep(2)
            print(f"   GO: {label.upper()} now ({seconds:.0f} s)", flush=True)
            take_result = await take(furby, label, seconds)
            path = OUT_DIR / f"{label}_{int(time.time())}_{n}.json"
            path.write_text(json.dumps(take_result, indent=1), encoding="utf-8")
            print(f"   saved {len(take_result['readings'])} readings", flush=True)
            await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
