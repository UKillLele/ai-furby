# AI Furby

A PC-hosted voice companion for a 2016 Furby Connect (LCD-eye model). The PC handles listening, thinking, and speaking. The Furby provides movement, expressions, lights, and touch input over Bluetooth LE via [PyFluff](https://github.com/martinwoodward/PyFluff).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Find your Furby's address with `python -m pyfluff.cli scan`, then set it:

```bash
set FURBY_ADDRESS=AA:BB:CC:DD:EE:FF
```

## Test

```bash
python -m pytest
```

## Layout

- `src/ai_furby/furby_controller.py`: adapter around PyFluff. The rest of the code uses this, not PyFluff directly.
- `src/ai_furby/config.py`: settings from environment variables.

Planned next: the assistant loop (mic, speech-to-text, LLM with persona and memory, TTS to speakers) and sensor events as input. API keys go in environment variables, never in the repo.

## Run the assistant

Needs an `ANTHROPIC_API_KEY` in your environment. The first run downloads the Whisper `small` model.

```bash
python -m ai_furby
```

Settings (all optional):

- `FURBY_ADDRESS`: Furby Bluetooth address. Leave unset to run audio-only.
- `CLAUDE_MODEL`: default `claude-haiku-5-5`.
- `WHISPER_MODEL`, `WHISPER_DEVICE`, `WHISPER_COMPUTE_TYPE`: default `small`, `cuda`, `float16`. For CPU, use `WHISPER_DEVICE=cpu` and `WHISPER_COMPUTE_TYPE=int8`.
- `PIPER_MODEL`: default `models/piper/en_US-lessac-medium.onnx`. Download voices with `python -m piper.download_voices --download-dir models/piper en_US-lessac-medium`.
