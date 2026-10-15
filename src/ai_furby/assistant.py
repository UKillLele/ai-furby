"""Claude-backed conversation for the Furby. Keeps short-term history and a persona prompt."""

from __future__ import annotations

from typing import Any

from ai_furby.config import Settings

PERSONA = (
    "You are Furby, a small, cheerful companion that lives on a desk. "
    "Reply in one to three short sentences, because your replies are spoken aloud. "
    "Never use markdown, lists, or emoji. Be warm, a little silly, and honest."
)

MAX_HISTORY_MESSAGES = 20  # keep the last N turns so replies stay fast and cheap


class Assistant:
    """Wraps the Anthropic client. `client` can be injected so tests run without network access."""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self._model = settings.claude_model
        if client is None:
            import anthropic  # imported lazily so tests run without the SDK configured

            client = anthropic.Anthropic()
        self._client = client
        self._history: list[dict[str, str]] = []

    def reply(self, user_text: str) -> str:
        self._history.append({"role": "user", "content": user_text})
        response = self._client.messages.create(
            model=self._model,
            max_tokens=300,
            system=PERSONA,
            messages=self._history[-MAX_HISTORY_MESSAGES:],
            output_config={"effort": "low"},
        )
        text = "".join(block.text for block in response.content if block.type == "text").strip()
        self._history.append({"role": "assistant", "content": text})
        return text

    def reset(self) -> None:
        self._history.clear()
