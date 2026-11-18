from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class Cache:
    """Cache de respostas em SQLite. Chave = hash de (endpoint, modelo, mensagens, parâmetros)."""

    def __init__(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(p, check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.db.commit()

    def get(self, key: str) -> dict[str, Any] | None:
        row = self.db.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def set(self, key: str, value: dict[str, Any]) -> None:
        self.db.execute(
            "INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)",
            (key, json.dumps(value, ensure_ascii=False)),
        )
        self.db.commit()

    def close(self) -> None:
        self.db.close()
