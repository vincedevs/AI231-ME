from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


class ReminderError(RuntimeError):
    """Expected reminder-storage failure with a safe user-facing message."""


@dataclass(frozen=True)
class Reminder:
    id: int
    task: str
    surface_text: str
    created_at_utc: str


def initialize_database(database_path: Path) -> int:
    """Create or validate Alfred's local database without removing existing data."""
    try:
        database_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(database_path, timeout=3.0) as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 3000")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS reminders (
                    id INTEGER PRIMARY KEY,
                    task TEXT NOT NULL,
                    surface_text TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    completed_at_utc TEXT
                )
                """
            )
            row = connection.execute(
                "SELECT value FROM schema_metadata WHERE key = 'schema_version'"
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO schema_metadata (key, value) VALUES ('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
                version = SCHEMA_VERSION
            else:
                try:
                    version = int(row[0])
                except (TypeError, ValueError) as error:
                    raise ReminderError(
                        "The reminder database has an invalid schema version."
                    ) from error
                if version != SCHEMA_VERSION:
                    raise ReminderError(
                        f"Unsupported reminder database schema version {version}; "
                        f"expected {SCHEMA_VERSION}."
                    )
    except sqlite3.Error as error:
        raise ReminderError(f"Unable to initialize the reminder database: {error}") from error
    return version


def reset_database(database_path: Path) -> int:
    """Initialize the schema and remove every reminder in one transaction."""
    version = initialize_database(database_path)
    try:
        with sqlite3.connect(database_path, timeout=3.0) as connection:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 3000")
            connection.execute("DELETE FROM reminders")
    except sqlite3.Error as error:
        raise ReminderError(f"Unable to reset the reminder database: {error}") from error
    return version


class ReminderStore:
    def __init__(self, database_path: Path, *, max_task_characters: int = 240) -> None:
        if max_task_characters <= 0:
            raise ValueError("max_task_characters must be positive")
        self.database_path = database_path
        self.max_task_characters = max_task_characters
        initialize_database(database_path)

    def create(self, value: Any) -> Reminder:
        if not isinstance(value, dict):
            raise ReminderError("I couldn't understand the reminder text.")
        surface = str(value.get("surface") or "").strip()
        task = str(value.get("text") or surface).strip()
        if not surface or not task:
            raise ReminderError("I couldn't understand the reminder text.")
        if len(surface) > self.max_task_characters or len(task) > self.max_task_characters:
            raise ReminderError("That reminder is too long to save.")
        created_at = datetime.now(UTC).isoformat()
        try:
            with sqlite3.connect(self.database_path, timeout=3.0) as connection:
                cursor = connection.execute(
                    """
                    INSERT INTO reminders (task, surface_text, created_at_utc)
                    VALUES (?, ?, ?)
                    """,
                    (task, surface, created_at),
                )
                reminder_id = int(cursor.lastrowid)
        except sqlite3.Error as error:
            raise ReminderError("I couldn't save that reminder.") from error
        return Reminder(reminder_id, task, surface, created_at)

    def list_active(self, limit: int | None = None) -> tuple[list[Reminder], int]:
        if limit is not None and limit <= 0:
            raise ValueError("Reminder list limit must be positive")
        try:
            with sqlite3.connect(self.database_path, timeout=3.0) as connection:
                query = """
                    SELECT id, task, surface_text, created_at_utc
                    FROM reminders
                    WHERE completed_at_utc IS NULL
                    ORDER BY id
                """
                parameters: tuple[int, ...] = ()
                if limit is not None:
                    query += " LIMIT ?"
                    parameters = (limit,)
                rows = connection.execute(query, parameters).fetchall()
                total = int(
                    connection.execute(
                        "SELECT COUNT(*) FROM reminders WHERE completed_at_utc IS NULL"
                    ).fetchone()[0]
                )
        except sqlite3.Error as error:
            raise ReminderError("I couldn't retrieve your reminders.") from error
        reminders = [Reminder(int(row[0]), str(row[1]), str(row[2]), str(row[3])) for row in rows]
        return reminders, total
