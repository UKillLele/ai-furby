"""Settings read from environment variables. No secrets are stored in the repo."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    furby_address: str | None
    furby_connect_timeout: float

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            furby_address=os.environ.get("FURBY_ADDRESS") or None,
            furby_connect_timeout=float(os.environ.get("FURBY_CONNECT_TIMEOUT", "10")),
        )
