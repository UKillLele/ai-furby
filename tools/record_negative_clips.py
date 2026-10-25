"""Record negative speech for the wake word model: sound-alikes of "Hey Furby" and everyday phrases.

Usage: python tools/record_negative_clips.py
Each clip is 2 seconds, saved as 16 kHz mono WAV under data/wakeword/negative/. These are training negatives,
kept apart from data/wakeword/rejected/, which stays as the held-out test set.
"""

import time
from pathlib import Path

import numpy as np

from ai_furby.audio import find_input_device
from record_wake_clips import MIC, record_clip, save

OUT_DIR = Path("data/wakeword/negative")

PROMPTS = [
    "Hey Fury", "Hey Furry", "Hey Turby", "Hey Bury", "Hey Ferby", "Hey Fuzzy", "Hey Buddy", "Hey Bobby",
    "Hey Murphy", "Hey Flurby", "Furby", "Hey there", "Hey Jerry", "Hey Mary", "Hey Kirby",
    "Good morning", "What time is it", "Turn on the lights", "Can you help me", "Thanks a lot",
    "I like this song", "Let's go outside", "How was your day", "Where are my keys", "Be right back",
    "That sounds great", "Tell me a joke", "Okay, let's try again", "Put that over there", "Good night",
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    existing = len(list(OUT_DIR.glob("*.wav")))
    device = find_input_device(MIC)
    for index, phrase in enumerate(PROMPTS, start=existing):
        print(f"Clip {index + 1}: get ready...", flush=True)
        time.sleep(1.0)
        print(f"   SAY: {phrase}", flush=True)
        samples = record_clip(device)
        save(OUT_DIR / f"neg_{index:03d}.wav", samples)
        print(f"   saved (peak {np.max(np.abs(samples)) / 32767:.2f})", flush=True)
        time.sleep(0.8)
    print(f"Done. {len(list(OUT_DIR.glob('*.wav')))} clips in {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
