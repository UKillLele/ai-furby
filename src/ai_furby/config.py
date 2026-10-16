"""Settings read from environment variables. No secrets are stored in the repo."""

import os
from dataclasses import dataclass

DEFAULT_CLAUDE_MODEL = "claude-haiku-5-5"
DEFAULT_WHISPER_MODEL = "small"
DEFAULT_PIPER_MODEL = "models/piper/en_US-lessac-medium.onnx"
DEFAULT_MIC = "HD Pro Webcam C920"  # part of the device name; the webcam mic on this PC
DEFAULT_SPEAKER = "Speakers (Realtek"  # the PC speakers, so playback avoids the headset call mode


@dataclass(frozen=True)
class Settings:
    furby_address: str | None = None
    furby_connect_timeout: float = 10.0
    claude_model: str = DEFAULT_CLAUDE_MODEL
    whisper_model: str = DEFAULT_WHISPER_MODEL
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"
    piper_model_path: str = DEFAULT_PIPER_MODEL
    mic_device: str = DEFAULT_MIC
    speaker_device: str = DEFAULT_SPEAKER
    mic_threshold: float = 0.1  # speech loudness (RMS per 0.1 s); raise if noise triggers it, lower if you are not heard

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            furby_address=os.environ.get("FURBY_ADDRESS") or None,
            furby_connect_timeout=float(os.environ.get("FURBY_CONNECT_TIMEOUT", "10")),
            claude_model=os.environ.get("CLAUDE_MODEL") or DEFAULT_CLAUDE_MODEL,
            whisper_model=os.environ.get("WHISPER_MODEL") or DEFAULT_WHISPER_MODEL,
            whisper_device=os.environ.get("WHISPER_DEVICE") or "cuda",
            whisper_compute_type=os.environ.get("WHISPER_COMPUTE_TYPE") or "float16",
            piper_model_path=os.environ.get("PIPER_MODEL") or DEFAULT_PIPER_MODEL,
            mic_device=os.environ.get("MIC_DEVICE") or DEFAULT_MIC,
            speaker_device=os.environ.get("SPEAKER_DEVICE") or DEFAULT_SPEAKER,
            mic_threshold=float(os.environ.get("MIC_THRESHOLD", "0.1")),
        )
