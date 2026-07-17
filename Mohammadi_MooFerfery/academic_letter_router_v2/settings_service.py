from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from models import AppSettings


class SettingsStore:
    """Persist validated UI settings in SQLite.

    The profile key allows separate deployments or authenticated frontends to keep
    independent settings without changing the storage implementation.
    """

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout=10000")
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS app_settings (
                    profile TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def load(self, profile: str = "default") -> AppSettings:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM app_settings WHERE profile = ?", (profile,)
            ).fetchone()
        if row is None:
            return AppSettings()
        try:
            return AppSettings.model_validate_json(row[0])
        except Exception:
            return AppSettings()

    def save(self, settings: AppSettings, profile: str = "default") -> None:
        payload = json.dumps(settings.model_dump(), ensure_ascii=False, sort_keys=True)
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO app_settings(profile, payload, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(profile) DO UPDATE SET
                    payload = excluded.payload,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (profile, payload),
            )
