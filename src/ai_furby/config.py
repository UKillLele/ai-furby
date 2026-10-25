"""Settings read from environment variables. No secrets are stored in the repo."""

import os
from dataclasses import dataclass

# Set FURBY_ADDRESS to your Furby's Bluetooth address. Without it, the assistant runs audio-only.
DEFAULT_CHAT_MODEL = "mistral-small-latest"
DEFAULT_TRANSCRIBE_MODEL = "voxtral-mini-transcribe-3"
DEFAULT_SPEECH_MODEL = "voxtral-mini-tts-3"
# Mistral's preset voice "James". Set VOICE_ID to use a custom voice from your Mistral account.
DEFAULT_VOICE_ID = "01a14675-dc30-778a-8d01-b5dead5a522a"
DEFAULT_VOICE_REFERENCE = ""  # empty: use the saved custom voice by ID instead of sending a clip each time
DEFAULT_VOICE_LOWPASS_HZ = 8000.0  # the clone already has the softened top end; 8000 means no filter
DEFAULT_SPEAKER = "Speakers (Realtek"  # PC speakers; empty string means the Windows default output
DEFAULT_MIC = "Logitech G PRO X 2"  # part of the device name; the headset mic on this PC


@dataclass(frozen=True)
class Settings:
    furby_address: str | None = None
    furby_connect_timeout: float = 10.0
    mistral_api_key: str = ""  # read from MISTRAL_API_KEY; never stored in the repo
    chat_model: str = DEFAULT_CHAT_MODEL
    transcribe_model: str = DEFAULT_TRANSCRIBE_MODEL
    speech_model: str = DEFAULT_SPEECH_MODEL
    voice_id: str = DEFAULT_VOICE_ID
    voice_reference_path: str = DEFAULT_VOICE_REFERENCE  # empty string uses the preset voice_id instead
    voice_lowpass_hz: float = DEFAULT_VOICE_LOWPASS_HZ
    mic_device: str = DEFAULT_MIC
    speaker_device: str = DEFAULT_SPEAKER
    wake_word_enabled: bool = True
    memory_path: str = "data/memory.db"
    wake_threshold: float = 0.5
    mic_threshold: float = 0.01  # floor for quiet rooms; speech also has to be well above the measured room noise
    web_port: int = 80  # phone control page; port 80 means no port number in the address
    web_token: str = ""  # required for the phone page; the page stays off without it

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            furby_address=os.environ.get("FURBY_ADDRESS") or None,
            furby_connect_timeout=float(os.environ.get("FURBY_CONNECT_TIMEOUT", "10")),
            mistral_api_key=os.environ.get("MISTRAL_API_KEY", ""),
            chat_model=os.environ.get("CHAT_MODEL") or DEFAULT_CHAT_MODEL,
            transcribe_model=os.environ.get("TRANSCRIBE_MODEL") or DEFAULT_TRANSCRIBE_MODEL,
            speech_model=os.environ.get("SPEECH_MODEL") or DEFAULT_SPEECH_MODEL,
            voice_id=os.environ.get("VOICE_ID") or DEFAULT_VOICE_ID,
            voice_reference_path=os.environ.get("VOICE_REFERENCE", DEFAULT_VOICE_REFERENCE),
            voice_lowpass_hz=float(os.environ.get("VOICE_LOWPASS_HZ", str(DEFAULT_VOICE_LOWPASS_HZ))),
            mic_device=os.environ.get("MIC_DEVICE") or DEFAULT_MIC,
            speaker_device=os.environ.get("SPEAKER_DEVICE") or DEFAULT_SPEAKER,
            wake_word_enabled=os.environ.get("WAKE_WORD", "1") != "0",
            memory_path=os.environ.get("MEMORY_PATH") or "data/memory.db",
            wake_threshold=float(os.environ.get("WAKE_THRESHOLD", "0.5")),
            mic_threshold=float(os.environ.get("MIC_THRESHOLD", "0.01")),
            web_port=int(os.environ.get("WEB_PORT") or 80),
            web_token=os.environ.get("WEB_TOKEN", ""),
        )
