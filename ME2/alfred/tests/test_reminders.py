from __future__ import annotations

import sqlite3

import pytest

from alfred.reminders import ReminderError, ReminderStore, initialize_database, reset_database


def test_database_initialization_is_idempotent_and_preserves_rows(tmp_path) -> None:
    path = tmp_path / "nested" / "state.sqlite3"
    assert initialize_database(path) == 1
    store = ReminderStore(path)
    store.create({"surface": "Buy milk", "text": "buy milk"})
    assert initialize_database(path) == 1
    reminders, total = store.list_active(5)
    assert total == 1
    assert reminders[0].surface_text == "Buy milk"


def test_reminder_store_validates_text_and_applies_spoken_limit(tmp_path) -> None:
    store = ReminderStore(tmp_path / "state.sqlite3", max_task_characters=12)
    with pytest.raises(ReminderError, match="understand"):
        store.create({})
    with pytest.raises(ReminderError, match="too long"):
        store.create({"surface": "This is much too long"})
    store.create({"surface": "One"})
    store.create({"surface": "Two"})
    reminders, total = store.list_active(1)
    assert [item.surface_text for item in reminders] == ["One"]
    assert total == 2


def test_reminder_store_can_list_every_active_reminder(tmp_path) -> None:
    store = ReminderStore(tmp_path / "state.sqlite3")
    store.create({"surface": "One"})
    store.create({"surface": "Two"})
    reminders, total = store.list_active()
    assert [item.surface_text for item in reminders] == ["One", "Two"]
    assert total == 2


def test_database_reset_removes_reminders_but_preserves_schema(tmp_path) -> None:
    path = tmp_path / "state.sqlite3"
    store = ReminderStore(path)
    store.create({"surface": "Remove me"})
    assert reset_database(path) == 1
    reminders, total = store.list_active()
    assert reminders == []
    assert total == 0


def test_unknown_database_schema_is_rejected_without_deleting_data(tmp_path) -> None:
    path = tmp_path / "state.sqlite3"
    initialize_database(path)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE schema_metadata SET value = '99' WHERE key = 'schema_version'")
    with pytest.raises(ReminderError, match="schema version 99"):
        initialize_database(path)
