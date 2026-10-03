from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

from dataset_pipeline.common import slot_value
from dataset_pipeline.prepare import (
    adapt_previous_record,
    canonical_speaker,
    choose_validation_speakers,
    load_exclusions,
    load_supplement_overrides,
    materialize_audio,
    previous_candidates,
)
from dataset_pipeline.schema import INTENTS, SLOT_BY_INTENT, load_schema, validate_prepared_record

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def prepared_record(intent: str | None, *, speaker: str = "fixture:s1") -> dict:
    supported = intent is not None
    slots = {}
    if intent in SLOT_BY_INTENT:
        name = SLOT_BY_INTENT[intent]
        slots = {name: slot_value(name, "ten", "set it to ten")}
    return {
        "sample_id": f"sample-{speaker}-{intent}",
        "audio_path": "audio/x.wav",
        "source_dataset": "fixture",
        "source_id": "x",
        "speaker_id": speaker,
        "speaker_id_reliable": True,
        "recording_group_id": "recording-x",
        "utterance_group_id": "utterance-x",
        "transcript": "fixture command",
        "canonical_intent": intent,
        "slots": slots,
        "supported": supported,
        "is_synthetic": False,
        "is_supplemental": False,
        "mapping_rule": "fixture.v1",
    }


def test_frozen_schema() -> None:
    load_schema(PROJECT_ROOT / "configs/vcm_schema.json")
    assert len(INTENTS) == 19
    assert set(SLOT_BY_INTENT) == {
        "TIMER", "ALARM", "TEMPERATURE", "BRIGHTNESS", "COLOR", "CREATE_REMINDER"
    }


def test_missing_slot_is_valid_but_incorrect_slot_is_not() -> None:
    row = prepared_record("TIMER")
    row["slots"] = {}
    assert validate_prepared_record(row) == []
    row["slots"] = {"time": slot_value("time", "ten", row["transcript"])}
    assert validate_prepared_record(row)


def test_out_of_scope_contract() -> None:
    row = prepared_record(None)
    assert validate_prepared_record(row) == []
    row["canonical_intent"] = "LIGHT_ON"
    assert validate_prepared_record(row)


def test_stop_speakers_are_explicitly_unreliable() -> None:
    identity, reliable = canonical_speaker("STOP", "pseudo-12")
    assert identity == "stop:pseudo-12"
    assert not reliable


def test_audio_materialization_is_content_addressed_and_portable() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "source.wav"
        sf.write(source, np.zeros(1600, dtype=np.float32), 16000)
        first = materialize_audio(
            source, root / "prepared", {"sample_rate": 16000, "maximum_seconds": 12.0}
        )
        second = materialize_audio(
            source, root / "prepared", {"sample_rate": 16000, "maximum_seconds": 12.0}
        )
        assert first == second
        assert not Path(first["audio_path"]).is_absolute()
        assert (root / "prepared" / first["audio_path"]).is_file()


def test_previous_record_is_copied_as_training_only_supplement() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "old.wav"
        sf.write(source, np.zeros(1600, dtype=np.float32), 16000)
        old = prepared_record("WEATHER")
        old["audio_path"] = str(source)
        adapted = adapt_previous_record(
            old,
            root,
            root / "prepared",
            {"sample_rate": 16000, "maximum_seconds": 12.0},
        )
        assert adapted["split"] == "train"
        assert adapted["official_split"] is None
        assert adapted["is_supplemental"]
        assert not adapted["is_synthetic"]


def test_validation_search_is_speaker_disjoint_and_reproducible() -> None:
    rows = []
    for intent in ("WEATHER", "TIME"):
        for speaker in range(10):
            for repetition in range(5):
                row = prepared_record(intent, speaker=f"fixture:s{speaker}")
                row["sample_id"] += f"-{repetition}"
                rows.append(row)
    config = {
        "seed": 231,
        "validation": {
            "speaker_fraction": 0.2,
            "search_attempts": 100,
            "minimum_records_per_label": 5,
        },
    }
    selected_a, report_a = choose_validation_speakers(rows, config)
    selected_b, report_b = choose_validation_speakers(rows, config)
    assert selected_a == selected_b
    assert report_a == report_b
    assert all(report_a["validation_by_label"][intent] >= 5 for intent in ("WEATHER", "TIME"))


def test_dataset_configuration_is_pinned() -> None:
    config = json.loads((PROJECT_ROOT / "configs/dataset.json").read_text(encoding="utf-8"))
    assert len(config["source_revision"]) == 40
    assert len(config["source_shards_sha256"]) == 11
    assert config["expected_source_rows"]["holdout"] == 202
    assert config["expected_prepared_rows"]["holdout"] == 201
    assert not config["include_numerals"]
    assert config["validation"]["minimum_records_by_label"]["MESSAGE"] == 2
    assert config["validation"]["minimum_speakers_by_label"]["CREATE_REMINDER"] == 5
    assert config["validation"]["minimum_speakers_by_label"]["MESSAGE"] == 2
    assert (
        config["validation"]["minimum_recording_groups_by_label"]["CREATE_REMINDER"]
        == 10
    )
    assert config["validation"]["minimum_recording_groups_by_label"]["MESSAGE"] == 2


def test_dataset_exclusion_is_explicit_and_revision_pinned() -> None:
    config = json.loads((PROJECT_ROOT / "configs/dataset.json").read_text(encoding="utf-8"))
    exclusions = load_exclusions(PROJECT_ROOT / config["dataset_exclusions"])
    assert len(exclusions) == 1
    exclusion = next(iter(exclusions.values()))
    assert exclusion["source_revision"] == config["source_revision"]
    assert exclusion["duplicate_of"].startswith("holdout:")


def test_reviewed_supplement_overrides_are_explicit_and_span_aligned() -> None:
    path = PROJECT_ROOT / "configs/supplement_label_overrides.jsonl"
    overrides = load_supplement_overrides(path)
    assert len(overrides) == 36
    for row in overrides.values():
        assert row["canonical_intent"] == "CREATE_REMINDER"
        assert row["slots"]["task"].lower() in row["transcript"].lower()


def test_supplement_override_transforms_instead_of_duplicates() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        manifest = root / "previous" / "unsupported.jsonl"
        manifest.parent.mkdir(parents=True)
        row = prepared_record(None)
        row.update(
            {
                "sample_id": "old-reminder",
                "source_dataset": "slurp",
                "source_intent": "calendar_set",
                "source_id": "audio-1.flac",
                "utterance_group_id": "group-1",
                "recording_group_id": "recording-1",
                "speaker_id": "speaker-1",
                "transcript": "remind me to drink water",
            }
        )
        manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
        config = {
            "previous_dataset": {
                "root": str(root / "previous"),
                "supported_manifests": [],
                "unsupported_manifests": ["unsupported.jsonl"],
            }
        }
        override = {
            "source_dataset": "slurp",
            "source_intent": "calendar_set",
            "utterance_group_id": "group-1",
            "transcript": row["transcript"],
            "canonical_intent": "CREATE_REMINDER",
            "slots": {"task": "drink water"},
        }
        candidates, report = previous_candidates(
            root,
            config,
            {"slurp|calendar_set|group-1": override},
        )
        assert len(candidates) == 1
        assert candidates[0]["supported"]
        assert candidates[0]["canonical_intent"] == "CREATE_REMINDER"
        assert candidates[0]["slots"]["task"]["surface"] == "drink water"
        assert report["override_groups_matched"] == 1
