"""Record short "Hey Furby" clips from the headset mic for training the wake word model.

Usage: python tools/record_wake_clips.py [count]
Each clip is 2 seconds, saved as 16 kHz mono WAV under data/wakeword/positive/.
"""

import sys
import time
import wave
from math import gcd
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

from ai_furby.audio import find_input_device

MIC = "Logitech G PRO X 2"
CLIP_SECONDS = 2.0
OUT_RATE = 16000
OUT_DIR = Path("data/wakeword/positive")


def record_clip(device: int) -> np.ndarray:
    rate = int(sd.query_devices(device, kind="input")["default_samplerate"])
    audio = sd.rec(int(rate * CLIP_SECONDS), samplerate=rate, channels=1, dtype="float32", device=device)
    sd.wait()
    x = audio[:, 0]
    divisor = gcd(rate, OUT_RATE)
    x = resample_poly(x, OUT_RATE // divisor, rate // divisor)
    return (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16)


def save(path: Path, samples: np.ndarray) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(OUT_RATE)
        wav.writeframes(samples.tobytes())


def main() -> None:
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = len(list(OUT_DIR.glob("*.wav")))
    device = find_input_device(MIC)
    for index in range(existing, existing + count):
        print(f"Clip {index + 1}: get ready...", flush=True)
        time.sleep(1.0)
        print("   SAY: Hey Furby", flush=True)
        samples = record_clip(device)
        save(OUT_DIR / f"hey_furby_{index:03d}.wav", samples)
        print(f"   saved (peak {np.max(np.abs(samples)) / 32767:.2f})", flush=True)
        time.sleep(0.8)
    print(f"Done. {len(list(OUT_DIR.glob('*.wav')))} clips in {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
