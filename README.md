# AI Furby

A voice companion for a 2016 Furby Connect (LCD-eye model). The PC or Raspberry Pi listens for "Hey Furby," talks with Mistral's models, and moves the Furby's mouth, eyes, and antenna over Bluetooth LE through [PyFluff](https://github.com/martinwoodward/PyFluff).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Set these environment variables. Keep them out of the repository.

| Variable | Purpose |
|---|---|
| `MISTRAL_API_KEY` | Required. Mistral API key for speech-to-text, chat, and text-to-speech. |
| `FURBY_ADDRESS` | Your Furby's Bluetooth address. Leave unset to run audio-only. Find it with `python -m pyfluff.cli scan`. |
| `VOICE_ID` | Mistral voice ID for the Furby's voice. Defaults to the preset voice "James." |
| `WEB_TOKEN` | Enables the phone control page and sets its access token. |
| `WEB_PORT` | Port for the phone page. Defaults to 80. |

Optional settings: `CHAT_MODEL`, `TRANSCRIBE_MODEL`, `SPEECH_MODEL`, `MIC_DEVICE`, `SPEAKER_DEVICE`, `WAKE_WORD` (set to `0` to turn wake-word listening off), `VOICE_SEMITONES` and `VOICE_LOWPASS_HZ` (pitch and brightness shaping, both off by default for the custom voice).

## Wake word

The "Hey Furby" model is not in the repository. Train it with openWakeWord and place the result at `models/hey_furby.onnx`, with its `.data` file beside it. Without it, the assistant can't wake by voice.

## Run the assistant

```bash
python -m ai_furby
```

Say "Hey Furby" to wake it, and "Goodbye, Furby" to put it to sleep. The Furby sleeps after a quiet spell and wakes on the wake word or on two antenna touches.

When `WEB_TOKEN` is set, open `http://<address>/` on a phone on the same network to control the eyes, antenna, sleep, and moods. Every change needs the token.

## Test

```bash
python -m pytest
```

## Layout

- `src/ai_furby/voice_loop.py`: the main loop: listen, think, speak, move, sleep.
- `src/ai_furby/assistant.py`: the Furby persona and chat history, using Mistral chat.
- `src/ai_furby/speech_to_text.py` and `text_to_speech.py`: Voxtral transcription and the custom voice.
- `src/ai_furby/wake_word.py`: openWakeWord listener.
- `src/ai_furby/furby_controller.py`: adapter around PyFluff. The rest of the code uses this, not PyFluff directly.
- `src/ai_furby/furby_worker.py`: Bluetooth connection, background reconnect, and the eyes.
- `src/ai_furby/memory.py`: facts the Furby remembers, in a local SQLite file.
- `src/ai_furby/web_ui.py`: the phone control page.
- `tools/`: scripts for recording wake-word clips, testing movement with a webcam, and recording sensor data.

## Privacy

API keys, the Furby's address, and voice IDs come from environment variables. Recordings, the memory database, and models are kept out of git by `.gitignore`.
