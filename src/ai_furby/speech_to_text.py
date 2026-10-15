"""Local speech-to-text with faster-whisper. Runs on the GPU by default, CPU via WHISPER_DEVICE=cpu."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ai_furby.config import Settings


def _add_cuda_dll_dirs() -> None:
    """On Windows, the pip-installed NVIDIA runtime DLLs (cuBLAS, cuDNN) are not on the search path by default."""
    if sys.platform != "win32":
        return
    spec = importlib.util.find_spec("ctranslate2")  # locates this venv's site-packages without importing the library
    if spec is None or spec.origin is None:
        return
    site_packages = Path(spec.origin).resolve().parent.parent
    for bin_dir in (site_packages / "nvidia").glob("*/bin"):
        os.add_dll_directory(str(bin_dir))
        os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"  # CTranslate2 loads DLLs via PATH


class SpeechToText:
    def __init__(self, settings: Settings, model: Any | None = None) -> None:
        if model is None:
            _add_cuda_dll_dirs()
            from faster_whisper import WhisperModel  # lazy: model download happens on first use

            model = WhisperModel(
                settings.whisper_model,
                device=settings.whisper_device,
                compute_type=settings.whisper_compute_type,
            )
        self._model = model

    def transcribe(self, audio: np.ndarray) -> str:
        """`audio` is mono float32 at 16 kHz, values in [-1, 1]."""
        segments, _info = self._model.transcribe(
            audio.astype(np.float32),
            language="en",
            beam_size=1,
            vad_filter=True,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()
