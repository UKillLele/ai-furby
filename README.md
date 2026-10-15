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
