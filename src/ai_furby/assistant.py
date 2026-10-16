"""Claude-backed conversation for the Furby. Keeps short-term history and a persona prompt."""

from __future__ import annotations

import re
from typing import Any

from ai_furby.config import Settings

PERSONA = (
    "You are Furby, a small, cheerful companion that lives on a desk. "
    "Reply in one to three short sentences, because your replies are spoken aloud. "
    "Never use markdown, lists, or emoji. Be warm, a little silly, and honest. "
    "After your reply, on its own final line, write MOOD: followed by one word for how you feel: "
    "happy, excited, curious, sad, sleepy, or silly."
)

MOODS = ("happy", "excited", "curious", "sad", "sleepy", "silly")

MAX_HISTORY_MESSAGES = 20  # keep the last N turns so replies stay fast and cheap

_MOOD_LINE = re.compile(r"\s*MOOD:\s*([A-Za-z]+)\s*$", re.IGNORECASE)


def split_mood(raw: str) -> tuple[str, str | None]:
    """Separates the spoken text from the trailing MOOD line. Unknown or missing moods return None."""
    match = _MOOD_LINE.search(raw)
    if not match:
        return raw.strip(), None
    mood = match.group(1).lower()
    return raw[: match.start()].strip(), (mood if mood in MOODS else None)


class Assistant:
    """Wraps the Anthropic client. `client` can be injected so tests run without network access."""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self._model = settings.claude_model
        if client is None:
            import os

            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set in this terminal. Open a new terminal after setting it, "
                    "or check that it is set as a user environment variable."
                )
            import anthropic  # imported lazily so tests run without the SDK configured

            client = anthropic.Anthropic()
        self._client = client
        self._history: list[dict[str, str]] = []
        self.last_mood: str | None = None

    def reply(self, user_text: str) -> str:
        self._history.append({"role": "user", "content": user_text})
        response = self._client.messages.create(
            model=self._model,
            max_tokens=300,
            system=PERSONA,
            messages=self._history[-MAX_HISTORY_MESSAGES:],
            output_config={"effort": "low"},
        )
        raw = "".join(block.text for block in response.content if block.type == "text").strip()
        text, self.last_mood = split_mood(raw)
        self._history.append({"role": "assistant", "content": raw})
        return text

    def reset(self) -> None:
        self._history.clear()
