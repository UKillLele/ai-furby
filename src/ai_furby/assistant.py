"""Claude-backed conversation for the Furby. Keeps short-term history and a persona prompt."""

from __future__ import annotations

import re
from typing import Any

from ai_furby.config import Settings

PERSONA = (
    "You are Furby, a small, cheerful companion that lives on a desk. "
    "Reply in one to three short sentences, because your replies are spoken aloud. "
    "Never use markdown, lists, or emoji. Be warm, a little silly, and honest. "
    "If the user tells you a lasting fact about themselves (their name, pets, favorites, people they care about), "
    "add a line FACT: followed by a short topic key, a vertical bar, and the fact, such as FACT: dog_name | Max. "
    "Only add one when the user said it directly, and never for things other people said, questions, or passing remarks. "
    "After your reply, on its own final line, write MOOD: followed by one word for how you feel: "
    "happy, excited, curious, sad, sleepy, or silly."
)

MOODS = ("happy", "excited", "curious", "sad", "sleepy", "silly")

MAX_HISTORY_MESSAGES = 20  # keep the last N turns so replies stay fast and cheap

_MOOD_LINE = re.compile(r"\s*MOOD:\s*([A-Za-z]+)\s*$", re.IGNORECASE)
_FACT_LINE = re.compile(r"\s*FACT:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def split_fact(raw: str) -> tuple[str, tuple[str, str] | None]:
    """Separates a FACT line from the reply. Returns (key, text) for the fact, or None."""
    match = _FACT_LINE.search(raw)
    if not match:
        return raw, None
    body = match.group(1)
    rest = raw[: match.start()] + raw[match.end():]
    if "|" in body:
        key, text = body.split("|", 1)
    else:
        key, text = body, body
    if not key.strip() or not text.strip():
        return rest, None
    return rest.strip(), (key.strip(), text.strip())


def split_mood(raw: str) -> tuple[str, str | None]:
    """Separates the spoken text from the trailing MOOD line. Unknown or missing moods return None."""
    match = _MOOD_LINE.search(raw)
    if not match:
        return raw.strip(), None
    mood = match.group(1).lower()
    return raw[: match.start()].strip(), (mood if mood in MOODS else None)


class Assistant:
    """Wraps the Anthropic client. `client` can be injected so tests run without network access."""

    def __init__(self, settings: Settings, client: Any | None = None, facts: Any | None = None) -> None:
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
        self._facts = facts  # a FactStore, or None when nothing is remembered
        self._history: list[dict[str, str]] = []
        self.last_mood: str | None = None
        self.last_fact: tuple[str, str] | None = None

    def reply(self, user_text: str) -> str:
        self._history.append({"role": "user", "content": user_text})
        response = self._client.messages.create(
            model=self._model,
            max_tokens=300,
            system=self._system_prompt(),
            messages=self._history[-MAX_HISTORY_MESSAGES:],
            output_config={"effort": "low"},
        )
        raw = "".join(block.text for block in response.content if block.type == "text").strip()
        raw, self.last_fact = split_fact(raw)
        text, self.last_mood = split_mood(raw)
        self._history.append({"role": "assistant", "content": raw})
        return text

    def _system_prompt(self) -> str:
        facts = self._facts.all() if self._facts is not None else []
        if not facts:
            return PERSONA
        listed = "\n".join(f"- {fact}" for fact in facts)
        return f"{PERSONA}\n\nThings the user asked you to remember:\n{listed}"

    def reset(self) -> None:
        self._history.clear()
