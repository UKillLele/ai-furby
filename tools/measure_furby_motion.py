"""Measures how much the Furby's ears, eyes, beak, antenna, and body move during each known action.

Run: python tools/measure_furby_motion.py

The webcam is read live and analyzed frame by frame. Video is not saved; only the numbers are written to
data/furby_video/stats.json and stats.csv. Regions come from data/furby_video/rois.json.
"""

import asyncio
import csv
import json
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_furby.config import Settings  # noqa: E402
from ai_furby.furby_remote import FurbyRemote  # noqa: E402
from ai_furby.mood_actions import ACTION_SECONDS  # noqa: E402

OUT = ROOT / "data" / "furby_video"
ROIS: dict[str, list[int]] = json.loads((OUT / "rois.json").read_text())

CAMERA_INDEX = 0
WIDTH, HEIGHT = 1280, 720
CONNECT_SECONDS = 8.0  # time for the Bluetooth child process to connect before the first action
BASELINE_SECONDS = 1.5  # idle video just before each action, used as the "still" reference
TAIL_SECONDS = 1.0  # keep recording after the measured action length in case it runs long
REST_SECONDS = 2.0  # pause between actions so the previous motion dies down
MIN_MOTION = 0.5  # motion must be at least this much above the idle level to count

# Each action's name, from the comments in mood_actions.py. The sleep action is left out because it
# puts the Furby to sleep and would need a wake-up before the next test.
NAMES: dict[tuple[int, int, int, int], str] = {
    (26, 0, 0, 2): "mouth full mumbling",
    (8, 1, 0, 7): "oo-nye talking",
    (24, 2, 1, 5): "silly singing",
    (9, 0, 0, 6): "spinning noises",
    (9, 2, 0, 2): "crying shake",
    (24, 4, 0, 1): "laughing",
    (24, 1, 3, 0): "flip upside down",
    (9, 2, 0, 1): "shaking yawn",
    (9, 3, 0, 4): "shake noises",
    (15, 0, 1, 6): "smacking lips",
    (26, 0, 0, 3): "baby talk",
    (26, 0, 0, 4): "more baby talk",
    (8, 1, 0, 6): "talkin' bout",
}


class MotionMonitor:
    """Reads webcam frames on a background thread and keeps the frame-to-frame change inside each region."""

    def __init__(self) -> None:
        self._cam = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        self._cam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        self._cam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        self._lock = threading.Lock()
        self._samples: list[tuple[float, dict[str, float]]] = []
        self._previous: np.ndarray | None = None
        self._running = False
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        if not self._cam.isOpened():
            raise RuntimeError("The webcam did not open")
        self._running = True
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        self._thread.join(timeout=5)
        self._cam.release()

    def samples(self) -> list[tuple[float, dict[str, float]]]:
        with self._lock:
            return list(self._samples)

    def _loop(self) -> None:
        while self._running:
            ok, frame = self._cam.read()
            if not ok:
                continue
            now = time.monotonic()
            gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            if self._previous is not None:
                diff = cv2.absdiff(gray, self._previous)
                motion = {
                    name: float(diff[y0:y1, x0:x1].mean()) for name, (x0, y0, x1, y1) in ROIS.items()
                }
                with self._lock:
                    self._samples.append((now, motion))
            self._previous = gray


def analyze(samples: list[tuple[float, dict[str, float]]], start: float, end: float, fps: float) -> dict:
    """Per-region stats for one action: idle level, when motion starts, how long, and how strong."""
    idle = [m for t, m in samples if start - BASELINE_SECONDS <= t < start]
    during = [(t - start, m) for t, m in samples if start <= t < end]
    regions = {}
    for region in ROIS:
        idle_values = np.array([m[region] for m in idle]) if idle else np.array([0.0])
        idle_mean = float(idle_values.mean())
        threshold = idle_mean + max(3.0 * float(idle_values.std()), MIN_MOTION)
        values = np.array([m[region] for _, m in during]) if during else np.array([0.0])
        moving = values > threshold
        onset = next((round(t, 2) for t, m in during if m[region] > threshold), None)
        regions[region] = {
            "idle_mean": round(idle_mean, 2),
            "threshold": round(threshold, 2),
            "onset_s": onset,
            "moving_s": round(float(moving.sum()) / fps, 2),
            "peak": round(float(values.max()), 2),
            "mean": round(float(values.mean()), 2),
        }
    return regions


async def run_all(remote: FurbyRemote, monitor: MotionMonitor) -> list[dict]:
    results = []
    await asyncio.sleep(CONNECT_SECONDS)
    # The Bluetooth child exits when it cannot reach the Furby. Stop here rather than log actions that never ran.
    if remote._process is not None and not remote._process.is_alive():
        raise RuntimeError("The Furby did not connect; check that it is awake and nothing else holds Bluetooth")
    for action, measured in ACTION_SECONDS.items():
        if action not in NAMES:
            continue
        await asyncio.sleep(BASELINE_SECONDS)
        start = time.monotonic()
        await remote.set_group_active(True)
        await remote.trigger_action(*action)
        await asyncio.sleep(measured + TAIL_SECONDS)
        end = time.monotonic()
        await remote.set_group_active(False)
        await asyncio.sleep(REST_SECONDS)
        results.append({"name": NAMES[action], "action": list(action), "measured_seconds": measured,
                        "start": start, "end": end})
        print(f"done: {NAMES[action]}")
    return results


def main() -> None:
    monitor = MotionMonitor()
    monitor.start()
    time.sleep(2)  # let the camera settle before the first action
    remote = FurbyRemote(Settings.from_env())
    remote.start()
    try:
        runs = asyncio.run(run_all(remote, monitor))
    finally:
        remote.stop()
        samples = monitor.samples()
        monitor.stop()

    span = samples[-1][0] - samples[0][0] if len(samples) > 1 else 1.0
    fps = (len(samples) - 1) / span if span > 0 else 1.0
    stats = []
    for run in runs:
        regions = analyze(samples, run["start"], run["end"], fps)
        stats.append({k: v for k, v in run.items() if k not in ("start", "end")} | {"regions": regions})

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "stats.json").write_text(json.dumps({"fps": round(fps, 1), "actions": stats}, indent=2))
    with (OUT / "stats.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["action", "region", "idle_mean", "threshold", "onset_s", "moving_s", "peak", "mean"])
        for entry in stats:
            for region, values in entry["regions"].items():
                writer.writerow([entry["name"], region, values["idle_mean"], values["threshold"],
                                 values["onset_s"], values["moving_s"], values["peak"], values["mean"]])
    print(f"camera ran at about {fps:.1f} fps; wrote {OUT / 'stats.json'} and {OUT / 'stats.csv'}")


if __name__ == "__main__":
    main()
