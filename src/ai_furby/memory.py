"""Durable memory of facts about the user, stored in a local SQLite file.

Each fact has a key (its topic, such as "dog_name"). Saving the same key again replaces the old fact,
so a changed fact doesn't sit beside the outdated one.
"""

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
                "CREATE TABLE IF NOT EXISTS facts ("
                "id INTEGER PRIMARY KEY, key TEXT UNIQUE NOT NULL, text TEXT NOT NULL, learned_at TEXT NOT NULL)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path)

    def save(self, key: str, text: str) -> None:
        """Adds a fact, or replaces the fact stored under the same key."""
        key, text = key.strip().lower(), text.strip()
        if not key or not text:
            return
        learned_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._connect() as db:
            db.execute(
                "INSERT INTO facts (key, text, learned_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET text = excluded.text, learned_at = excluded.learned_at",
                (key, text, learned_at),
            )

    def all(self) -> list[str]:
        with self._connect() as db:
            return [row[0] for row in db.execute("SELECT text FROM facts ORDER BY id")]

    def forget_last(self) -> str | None:
        """Removes the most recently learned fact and returns its text, or None if there are none."""
        with self._connect() as db:
            row = db.execute("SELECT id, text FROM facts ORDER BY id DESC LIMIT 1").fetchone()
            if row is None:
                return None
            db.execute("DELETE FROM facts WHERE id = ?", (row[0],))
            return row[1]

    def forget_matching(self, words: str) -> int:
        """Removes every fact whose text contains `words` (case-insensitive). Returns how many were removed."""
        pattern = f"%{words.strip().lower()}%"
        with self._connect() as db:
            cursor = db.execute("DELETE FROM facts WHERE lower(text) LIKE ?", (pattern,))
            return cursor.rowcount
