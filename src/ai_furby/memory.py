"""Small, durable memory of facts the user asks Furby to remember. Stored in a local SQLite file."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class FactStore:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS facts (id INTEGER PRIMARY KEY, text TEXT NOT NULL, learned_at TEXT NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    def add(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        learned_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._connect() as db:
            db.execute("INSERT INTO facts (text, learned_at) VALUES (?, ?)", (text, learned_at))

    def all(self) -> list[str]:
        with self._connect() as db:
            return [row[0] for row in db.execute("SELECT text FROM facts ORDER BY id")]
