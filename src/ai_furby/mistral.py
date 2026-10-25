"""Shared settings for calls to Mistral's API: the address and the login header."""

from __future__ import annotations

from ai_furby.config import Settings

API_BASE = "https://api.mistral.ai/v1"


def auth_headers(settings: Settings) -> dict[str, str]:
    if not settings.mistral_api_key:
        raise RuntimeError(
            "MISTRAL_API_KEY is not set in this terminal. Open a new terminal after setting it, "
            "or check that it is set as a user environment variable."
        )
    return {"Authorization": f"Bearer {settings.mistral_api_key}"}
