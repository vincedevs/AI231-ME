from __future__ import annotations

import hashlib
import io
import json
import math
import random
import shutil
import wave
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm

from .common import (
    normalize_text,
    sha256_file,
    slot_value,
    stable_digest,
    stable_id,
    write_json,
)
from .schema import INTENTS, SLOT_BY_INTENT, load_schema, validate_prepared_record

COMMAND_SPLITS = ("train", "test", "holdout")
MANIFEST_SPLITS = ("train", "validation", "test", "holdout")


def resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def canonical_source(value: str) -> str:
    key = "".join(character for character in value.lower() if character.isalnum())
    aliases = {
        "fluentspeechcommands": "fsc",
        "xelasettemperaturereal": "xela",
        "xelamultisensor": "xela",
        "commonvoiceen": "common_voice",
        "realvoice": "group_real",
        "groupsynthetic": "group_synthetic",
        "speechcommandsv2": "speech_commands",
        "timersandsuch": "timers_and_such",
    }
    return aliases.get(key, key or "unknown")


def canonical_speaker(source: str, speaker: str) -> tuple[str, bool]:
    source_key = canonical_source(source)
    raw = str(speaker or "").strip()
    if not raw:
        return "", False
    lowered = raw.lower()
    prefix = f"{source_key}:"
    identity = lowered if lowered.startswith(prefix) else f"{source_key}:{lowered}"
    # STOP has no true speaker identifiers. Its pseudo-identifiers must never
    # be used to claim speaker-disjoint validation.
    reliable = source_key != "stop"
    return identity, reliable


def waveform_to_pcm16(
    waveform: np.ndarray,
    source_rate: int,
    target_rate: int,
    maximum_seconds: float,
) -> bytes:
    values = np.asarray(waveform)
    if values.ndim == 2:
        values = values.mean(axis=1)
    values = values.astype(np.float32, copy=False)
    if int(source_rate) != target_rate:
        divisor = math.gcd(int(source_rate), target_rate)
        values = resample_poly(values, target_rate // divisor, int(source_rate) // divisor)
    values = values[: round(maximum_seconds * target_rate)]
    if not values.size:
        raise ValueError("Audio contains no samples")
    values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=-1.0)
    return np.rint(np.clip(values, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


def wav_bytes(pcm: bytes, sample_rate: int) -> bytes:
    destination = io.BytesIO()
    with wave.open(destination, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(pcm)
    return destination.getvalue()


def materialize_audio(
    source: bytes | Path,
    output_root: Path,
    audio_config: dict[str, Any],
) -> dict[str, Any]:
    if isinstance(source, bytes):
        waveform, source_rate = sf.read(io.BytesIO(source), dtype="float32", always_2d=True)
    else:
        waveform, source_rate = sf.read(source, dtype="float32", always_2d=True)
    sample_rate = int(audio_config["sample_rate"])
    pcm = waveform_to_pcm16(
        waveform,
        int(source_rate),
        sample_rate,
        float(audio_config["maximum_seconds"]),
    )
    pcm_sha256 = hashlib.sha256(pcm).hexdigest()
    encoded = wav_bytes(pcm, sample_rate)
    audio_sha256 = hashlib.sha256(encoded).hexdigest()
    relative = Path("audio") / pcm_sha256[:2] / f"{pcm_sha256}.wav"
    destination = output_root / relative
    if destination.is_file():
        if sha256_file(destination) != audio_sha256:
            raise ValueError(f"Existing materialized audio is inconsistent: {destination}")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".wav.tmp")
        temporary.write_bytes(encoded)
        temporary.replace(destination)
    return {
        # Paths are relative to the prepared dataset root so the entire
        # directory can be copied to the DGX or Raspberry Pi unchanged.
        "audio_path": str(relative),
        "audio_sha256": audio_sha256,
        "pcm_sha256": pcm_sha256,
        "sample_rate": sample_rate,
        "channels": 1,
        "duration_seconds": round((len(pcm) // 2) / sample_rate, 6),
    }


def verify_source_release(source_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    failures = []
    verified = {}
    for relative, expected in config["source_shards_sha256"].items():
        path = source_root / relative
        if not path.is_file():
            failures.append(f"missing {relative}")
            continue
        actual = sha256_file(path)
        verified[relative] = actual
        if actual != expected:
            failures.append(f"checksum mismatch {relative}: {actual}")
    if failures:
        raise ValueError("Source release verification failed: " + "; ".join(failures))
    cache_tree = source_root / ".cache" / "huggingface" / "trees" / (
        str(config["source_revision"]) + ".json"
    )
    if cache_tree.is_file():
        cached = json.loads(cache_tree.read_text(encoding="utf-8"))
        for relative, actual in verified.items():
            expected = cached.get("files", {}).get(relative, {}).get("lfs_sha256")
            if expected and expected != actual:
                raise ValueError(f"Cached Hugging Face revision disagrees for {relative}")
    return {"revision": config["source_revision"], "shards_sha256": verified}


def load_overrides(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path)
    lookup = {str(row["sample_key"]): row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError("Duplicate sample_key in label override file")
    for row in rows:
        if row["canonical_intent"] not in INTENTS:
            raise ValueError(f"Invalid override intent: {row['canonical_intent']}")
    return lookup


def load_exclusions(path: Path) -> dict[str, dict[str, Any]]:
    """Load explicit source-record exclusions made during dataset adjudication."""
    rows = read_jsonl(path)
    lookup = {str(row["sample_key"]): row for row in rows}
    if len(lookup) != len(rows):
        raise ValueError("Duplicate sample_key in dataset exclusion file")
    for row in rows:
        if not str(row.get("reason", "")).strip():
            raise ValueError(f"Dataset exclusion is missing its reason: {row}")
    return lookup


def supplement_override_key(row: dict[str, Any]) -> str:
    return "|".join(
        (
            canonical_source(str(row.get("source_dataset", ""))),
            str(row.get("source_intent", "")),
            str(row.get("utterance_group_id", "")),
        )
    )


def load_supplement_overrides(path: Path) -> dict[str, dict[str, Any]]:
    """Load reviewed relabeling decisions for previous-dataset utterance groups."""
    rows = read_jsonl(path)
    lookup: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = supplement_override_key(row)
        if not all(key.split("|")):
            raise ValueError(f"Incomplete supplement override provenance: {row}")
        if key in lookup:
            raise ValueError(f"Duplicate supplement override: {key}")
        intent = str(row.get("canonical_intent", ""))
        if intent not in INTENTS:
            raise ValueError(f"Invalid supplement override intent: {intent}")
        expected_slot = SLOT_BY_INTENT.get(intent)
        slots = dict(row.get("slots", {}))
        if set(slots) - ({expected_slot} if expected_slot else set()):
            raise ValueError(f"Invalid slots in supplement override {key}: {sorted(slots)}")
        if expected_slot and not str(slots.get(expected_slot, "")).strip():
            raise ValueError(f"Missing {expected_slot!r} in supplement override: {key}")
        lookup[key] = row
    return lookup


def record_slots(intent: str, surface: str, transcript: str) -> dict[str, Any]:
    slot_name = SLOT_BY_INTENT.get(intent)
    if slot_name is None or not str(surface or "").strip():
        return {}
    return {slot_name: slot_value(slot_name, str(surface), transcript)}


def source_group(source: str, speaker: str, transcript: str) -> str:
    # This conservative group joins close/far microphones and repeated takes
    # sharing a speaker and normalized prompt. It may group more than strictly
    # necessary, which is safer than allowing related audio across splits.
    return stable_digest(f"{canonical_source(source)}|{speaker}|{normalize_text(transcript)}")


def build_huggingface_records(
    source_root: Path,
    output_root: Path,
    config: dict[str, Any],
    overrides: dict[str, dict[str, Any]],
    exclusions: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_overrides: set[str] = set()
    seen_exclusions: set[str] = set()
    source_counts: Counter[str] = Counter()
    columns = [
        "audio",
        "file",
        "transcript",
        "command",
        "variation",
        "slot_value",
        "out_of_scope",
        "bucket",
        "speaker_id",
        "source",
        "is_synthetic",
        "accent_group",
        "duration_s",
        "note",
    ]
    for split in COMMAND_SPLITS:
        paths = sorted((source_root / "data").glob(f"{split}-*.parquet"))
        if not paths:
            raise FileNotFoundError(f"No Parquet shards found for {split}")
        expected = int(config["expected_source_rows"][split])
        progress = tqdm(total=expected, desc=f"Materialize {split}", unit="clip")
        for path in paths:
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=64, columns=columns):
                for row in batch.to_pylist():
                    source_counts[split] += 1
                    sample_key = f"{split}:{row['file']}"
                    exclusion = exclusions.get(sample_key)
                    if exclusion:
                        if exclusion.get("transcript") != row["transcript"]:
                            raise ValueError(f"Exclusion transcript mismatch: {sample_key}")
                        if sample_key in overrides:
                            raise ValueError(
                                f"Source record cannot be both relabeled and excluded: {sample_key}"
                            )
                        seen_exclusions.add(sample_key)
                        progress.update()
                        continue
                    override = overrides.get(sample_key)
                    if override:
                        if override.get("transcript") != row["transcript"]:
                            raise ValueError(f"Override transcript mismatch: {sample_key}")
                        seen_overrides.add(sample_key)
                    supported = not bool(row["out_of_scope"]) or override is not None
                    intent = str(override["canonical_intent"] if override else row["command"])
                    if not supported:
                        intent = None
                    elif intent not in INTENTS:
                        raise ValueError(f"Unexpected intent {intent!r} in {sample_key}")
                    transcript = str(row["transcript"] or "")
                    if override:
                        override_slots = dict(override.get("slots", {}))
                        slot_name = SLOT_BY_INTENT.get(str(intent))
                        surface = str(override_slots.get(slot_name, "")) if slot_name else ""
                    else:
                        surface = str(row["slot_value"] or "")
                    speaker, reliable = canonical_speaker(
                        str(row["source"]), str(row["speaker_id"] or "")
                    )
                    if not speaker:
                        speaker = f"unknown:{stable_digest(sample_key)[:20]}"
                    group = source_group(str(row["source"]), speaker, transcript)
                    audio = materialize_audio(row["audio"]["bytes"], output_root, config["audio"])
                    record = {
                        **audio,
                        "sample_id": stable_id("hf", sample_key),
                        "source_dataset": canonical_source(str(row["source"])),
                        "source_id": str(row["file"]),
                        "speaker_id": speaker,
                        "speaker_id_reliable": reliable,
                        "recording_group_id": group,
                        "utterance_group_id": group,
                        "transcript": transcript,
                        "canonical_intent": intent,
                        "slots": (
                            record_slots(str(intent), surface, transcript) if supported else {}
                        ),
                        "supported": supported,
                        "is_synthetic": bool(row["is_synthetic"]),
                        "is_supplemental": False,
                        "benchmark_origin": "huggingface",
                        "official_split": split,
                        "split": split,
                        "accent_group": str(row["accent_group"] or ""),
                        "variation": str(row["variation"] or ""),
                        "mapping_rule": "huggingface.option_b.adjudicated.v1",
                        "label_override": sample_key if override else None,
                        "ood_type": "dataset_out_of_scope" if not supported else "supported",
                        "category_partition": str(row["bucket"] or ""),
                    }
                    errors = validate_prepared_record(record)
                    if errors:
                        raise ValueError(f"Invalid prepared record {sample_key}: {errors}")
                    records.append(record)
                    progress.update()
        progress.close()
        if source_counts[split] != expected:
            raise ValueError(f"{split} has {source_counts[split]} rows; expected {expected}")
    missing_overrides = sorted(set(overrides) - seen_overrides)
    if missing_overrides:
        raise ValueError(
            f"Label overrides do not match the source release: {missing_overrides[:10]}"
        )
    missing_exclusions = sorted(set(exclusions) - seen_exclusions)
    if missing_exclusions:
        raise ValueError(
            f"Dataset exclusions do not match the source release: {missing_exclusions[:10]}"
        )
    return records, {
        "source_rows": dict(source_counts),
        "prepared_rows": dict(Counter(row["official_split"] for row in records)),
        "label_overrides_applied": len(seen_overrides),
        "exclusions_applied": len(seen_exclusions),
        "exclusions": [exclusions[key] for key in sorted(seen_exclusions)],
    }


def build_supplemental_synthetic_records(
    source_root: Path,
    output_root: Path,
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Materialize only the explicitly training-only supplemental TTS partition.

    The source shard also contains voice-disjoint test and holdout partitions.
    Those records are deliberately excluded: synthetic audio must not alter the
    frozen evaluation sets, and the supplemental shard is an ablation rather
    than part of the benchmark.
    """
    settings = config.get("supplemental_synthetic", {})
    if not settings.get("enabled", False):
        return [], {"enabled": False, "source_rows": 0, "selected": 0}

    relative = str(settings["shard"])
    path = source_root / relative
    if not path.is_file():
        raise FileNotFoundError(f"Supplemental synthetic shard is missing: {path}")
    expected_rows = int(settings["expected_rows"])
    expected_training_rows = int(settings["expected_training_rows"])
    selected_partition = str(settings.get("voice_split", "train"))
    columns = [
        "audio",
        "file",
        "transcript",
        "command",
        "variation",
        "slot_value",
        "out_of_scope",
        "bucket",
        "speaker_id",
        "source",
        "is_synthetic",
        "accent_group",
        "voice_split",
    ]
    records: list[dict[str, Any]] = []
    partitions: Counter[str] = Counter()
    progress = tqdm(total=expected_rows, desc="Materialize supplemental TTS", unit="clip")
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=64, columns=columns):
        for row in batch.to_pylist():
            partition = str(row["voice_split"] or "")
            partitions[partition] += 1
            progress.update()
            if partition != selected_partition:
                continue
            if bool(row["out_of_scope"]):
                raise ValueError("Supplemental synthetic training data contains OOS speech")
            if not bool(row["is_synthetic"]):
                raise ValueError("Supplemental synthetic shard contains non-synthetic speech")
            intent = str(row["command"])
            if intent not in INTENTS:
                raise ValueError(f"Unexpected supplemental synthetic intent: {intent!r}")
            transcript = str(row["transcript"] or "")
            sample_key = f"supplemental_synth:{row['file']}"
            speaker, _ = canonical_speaker(str(row["source"]), str(row["speaker_id"] or ""))
            if not speaker:
                speaker = f"synthetic:{stable_digest(sample_key)[:20]}"
            group = source_group(str(row["source"]), speaker, transcript)
            audio = materialize_audio(row["audio"]["bytes"], output_root, config["audio"])
            record = {
                **audio,
                "sample_id": stable_id("hf_supplemental_synth", sample_key),
                "source_dataset": canonical_source(str(row["source"])),
                "source_id": str(row["file"]),
                "speaker_id": speaker,
                "speaker_id_reliable": False,
                "recording_group_id": group,
                "utterance_group_id": group,
                "transcript": transcript,
                "canonical_intent": intent,
                "slots": record_slots(intent, str(row["slot_value"] or ""), transcript),
                "supported": True,
                "is_synthetic": True,
                "is_supplemental": True,
                "is_supplemental_synthetic": True,
                "benchmark_origin": "huggingface_supplemental_synth",
                "official_split": None,
                "split": "train",
                "accent_group": str(row["accent_group"] or ""),
                "variation": str(row["variation"] or ""),
                "mapping_rule": "huggingface.supplemental_synth.training_partition.v1",
                "label_override": None,
                "ood_type": "supported",
                "category_partition": str(row["bucket"] or ""),
            }
            errors = validate_prepared_record(record)
            if errors:
                raise ValueError(f"Invalid supplemental synthetic record {sample_key}: {errors}")
            records.append(record)
    progress.close()
    if sum(partitions.values()) != expected_rows:
        raise ValueError(
            f"Supplemental synthetic shard has {sum(partitions.values())} rows; "
            f"expected {expected_rows}"
        )
    if len(records) != expected_training_rows:
        raise ValueError(
            f"Supplemental synthetic {selected_partition!r} partition has {len(records)} "
            f"rows; expected {expected_training_rows}"
        )
    return records, {
        "enabled": True,
        "shard": relative,
        "source_rows": sum(partitions.values()),
        "partitions": dict(sorted(partitions.items())),
        "selected_partition": selected_partition,
        "selected": len(records),
    }


def build_synthetic_negative_records(
    output_root: Path,
    config: dict[str, Any],
    existing_pcm_hashes: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the pinned, training-only OOS negative ablation from Hugging Face.

    The records are marked supplemental and synthetic, so existing data
    conditions retain their membership. Only the explicit negative condition
    opts into them.
    """
    settings = config.get("synthetic_negatives", {})
    if not settings.get("enabled", False):
        return [], {"enabled": False, "source_rows": 0, "selected": 0}
    try:
        from datasets import Audio, load_dataset
    except ImportError as error:
        raise RuntimeError(
            "Synthetic-negative preparation requires the dataset extra. "
            "Run: uv sync --frozen --python 3.12 --extra dataset"
        ) from error

    repository = str(settings["repository"])
    revision = str(settings["revision"])
    config_name = str(settings["config"])
    split = str(settings.get("split", "train"))
    dataset = load_dataset(repository, name=config_name, split=split, revision=revision)
    if "audio" not in dataset.column_names:
        raise ValueError("Synthetic-negative Hugging Face split has no audio column")
    dataset = dataset.cast_column("audio", Audio(decode=False))
    expected_rows = int(settings["expected_rows"])
    if len(dataset) != expected_rows:
        raise ValueError(
            f"Synthetic-negative split has {len(dataset)} rows; expected {expected_rows}"
        )

    records: list[dict[str, Any]] = []
    seen_pcm_hashes = set(existing_pcm_hashes)
    ood_types: Counter[str] = Counter()
    skipped_duplicate_pcm = 0
    progress = tqdm(total=len(dataset), desc="Materialize synthetic negatives", unit="clip")
    for index, row in enumerate(dataset):
        audio_value = row["audio"]
        if not isinstance(audio_value, dict):
            raise TypeError(f"Synthetic-negative audio is not an Audio mapping at row {index}")
        audio_bytes = audio_value.get("bytes")
        audio_path = audio_value.get("path")
        if audio_bytes is not None:
            audio_source: bytes | Path = bytes(audio_bytes)
        elif audio_path:
            audio_source = Path(str(audio_path))
        else:
            raise ValueError(f"Synthetic-negative audio is unavailable at row {index}")
        source_id = str(row.get("file") or row.get("id") or f"{split}:{index:05d}")
        sample_key = f"synthetic_negatives:{source_id}"
        transcript = str(row.get("transcript") or "")
        ood_type = str(row.get("negative_type") or row.get("ood_type") or "synthetic_negative")
        audio = materialize_audio(audio_source, output_root, config["audio"])
        if audio["pcm_sha256"] in seen_pcm_hashes:
            # Exact PCM duplicates add no acoustic evidence and violate the
            # release-wide no-duplicate invariant. Keep the first source row
            # encountered and record this deterministic exclusion.
            skipped_duplicate_pcm += 1
            progress.update()
            continue
        seen_pcm_hashes.add(audio["pcm_sha256"])
        record = {
            **audio,
            "sample_id": stable_id("hf_synthetic_negative", sample_key),
            "source_dataset": "hf_synthetic_negatives",
            "source_id": source_id,
            "speaker_id": f"synthetic_negative:{stable_digest(sample_key)[:20]}",
            "speaker_id_reliable": False,
            "recording_group_id": stable_digest(sample_key),
            "utterance_group_id": stable_digest(sample_key),
            "transcript": transcript,
            "canonical_intent": None,
            "slots": {},
            "supported": False,
            "is_synthetic": True,
            "is_supplemental": True,
            "is_negative_ablation": True,
            "is_supplemental_synthetic": False,
            "benchmark_origin": "huggingface_synthetic_negatives",
            "official_split": None,
            "split": "train",
            "accent_group": "",
            "variation": "",
            "mapping_rule": "huggingface.synthetic_negatives.train_only.v1",
            "label_override": None,
            "ood_type": ood_type,
            "category_partition": str(row.get("bucket") or ""),
        }
        errors = validate_prepared_record(record)
        if errors:
            raise ValueError(f"Invalid synthetic-negative record {sample_key}: {errors}")
        records.append(record)
        ood_types[ood_type] += 1
        progress.update()
    progress.close()
    expected_training_rows = settings.get("expected_training_rows")
    if expected_training_rows is not None and len(records) != int(expected_training_rows):
        raise ValueError(
            "Synthetic-negative unique training rows are "
            f"{len(records)}; expected {expected_training_rows}"
        )
    return records, {
        "enabled": True,
        "repository": repository,
        "revision": revision,
        "config": config_name,
        "split": split,
        "source_rows": len(dataset),
        "selected": len(records),
        "skipped_duplicate_pcm": skipped_duplicate_pcm,
        "ood_types": dict(sorted(ood_types.items())),
        "policy": "training_only; synthetic_negatives/test is not materialized or calibrated",
    }


def previous_candidates(
    project_root: Path,
    config: dict[str, Any],
    supplement_overrides: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    previous = config["previous_dataset"]
    root = resolve(project_root, previous["root"])
    rows = []
    matched_override_groups: set[str] = set()
    overridden_records = 0
    for supported, names in (
        (True, previous["supported_manifests"]),
        (False, previous["unsupported_manifests"]),
    ):
        for name in names:
            for row in read_jsonl(root / name):
                if row.get("is_synthetic"):
                    continue
                source = str(row.get("source_dataset", ""))
                speaker, reliable = canonical_speaker(source, str(row.get("speaker_id", "")))
                candidate = dict(row)
                candidate["supported"] = supported
                candidate["canonical_intent"] = (
                    str(row["canonical_intent"]) if supported else None
                )
                candidate["speaker_id"] = speaker or f"unknown:{row['sample_id']}"
                candidate["speaker_id_reliable"] = reliable and bool(speaker)
                candidate["source_dataset"] = canonical_source(source)
                candidate["previous_root"] = root
                key = supplement_override_key(candidate)
                override = supplement_overrides.get(key)
                if override is not None:
                    if supported:
                        raise ValueError(f"Override targets an already-supported row: {key}")
                    if str(override["transcript"]) != str(candidate.get("transcript", "")):
                        raise ValueError(f"Supplement override transcript mismatch: {key}")
                    intent = str(override["canonical_intent"])
                    slot_name = SLOT_BY_INTENT.get(intent)
                    surface = str(override.get("slots", {}).get(slot_name, "")) if slot_name else ""
                    candidate["supported"] = True
                    candidate["canonical_intent"] = intent
                    candidate["slots"] = record_slots(
                        intent, surface, str(candidate.get("transcript", ""))
                    )
                    candidate["mapping_rule"] = "previous_dataset.reviewed_override.v1"
                    candidate["supplement_label_override"] = key
                    matched_override_groups.add(key)
                    overridden_records += 1
                rows.append(candidate)
    unique = {}
    for row in rows:
        unique.setdefault(str(row["sample_id"]), row)
    missing = sorted(set(supplement_overrides) - matched_override_groups)
    if missing:
        raise ValueError(f"Supplement overrides do not match previous data: {missing[:10]}")
    return list(unique.values()), {
        "override_groups_configured": len(supplement_overrides),
        "override_groups_matched": len(matched_override_groups),
        "override_records_matched": overridden_records,
    }


def interleave_groups(rows: list[dict[str, Any]], group_key, seed: int) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(group_key(row))].append(row)
    for key, values in groups.items():
        values.sort(key=lambda row: stable_digest(f"{seed}|{row['sample_id']}"))
    ordered = []
    while any(groups.values()):
        for key in sorted(groups, key=lambda item: stable_digest(f"{seed}|{item}")):
            if groups[key]:
                ordered.append(groups[key].pop(0))
    return ordered


def adapt_previous_record(
    row: dict[str, Any],
    project_root: Path,
    output_root: Path,
    audio_config: dict[str, Any],
) -> dict[str, Any]:
    source_audio = Path(str(row["audio_path"]))
    if not source_audio.is_absolute():
        source_audio = project_root / source_audio
    audio = materialize_audio(source_audio, output_root, audio_config)
    supported = bool(row["supported"])
    return {
        **audio,
        "sample_id": "supplement_" + stable_digest(str(row["sample_id"]))[:20],
        "source_dataset": str(row["source_dataset"]),
        "source_id": str(row.get("source_id", row["sample_id"])),
        "speaker_id": str(row["speaker_id"]),
        "speaker_id_reliable": bool(row["speaker_id_reliable"]),
        "recording_group_id": str(row.get("recording_group_id", row["sample_id"])),
        "utterance_group_id": str(row.get("utterance_group_id", row["sample_id"])),
        "transcript": str(row.get("transcript", "")),
        "canonical_intent": row["canonical_intent"],
        "slots": dict(row.get("slots", {})) if supported else {},
        "supported": supported,
        "is_synthetic": False,
        "is_supplemental": True,
        "benchmark_origin": "previous_dataset",
        "official_split": None,
        "split": "train",
        "accent_group": str(row.get("accent_group", "")),
        "variation": "",
        "mapping_rule": str(row.get("mapping_rule", "previous_dataset.reviewed.v1")),
        "label_override": row.get("supplement_label_override"),
        "ood_type": str(
            row.get("ood_type", "previous_out_of_scope" if not supported else "supported")
        ),
        "category_partition": str(row.get("category_partition", "previous_dataset")),
    }


def select_supplements(
    project_root: Path,
    output_root: Path,
    config: dict[str, Any],
    benchmark_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    settings = config["supplementation"]
    if not settings.get("enabled", True):
        return [], {"enabled": False, "selected": 0}
    evaluation = [row for row in benchmark_records if row["official_split"] in {"test", "holdout"}]
    evaluation_speakers = {row["speaker_id"] for row in evaluation if row["speaker_id_reliable"]}
    evaluation_pcm = {row["pcm_sha256"] for row in evaluation}
    used_pcm = {row["pcm_sha256"] for row in benchmark_records}
    used_source_ids = {
        (str(row["source_dataset"]), str(row["source_id"])) for row in benchmark_records
    }
    overrides_path = resolve(project_root, config["supplement_label_overrides"])
    supplement_overrides = load_supplement_overrides(overrides_path)
    previous, override_report = previous_candidates(
        project_root, config, supplement_overrides
    )
    reviewed_decisions: dict[str, dict[str, Any]] = {}

    def note_reviewed(row: dict[str, Any], decision: str) -> None:
        if not row.get("supplement_label_override"):
            return
        reviewed_decisions[str(row["sample_id"])] = {
            "sample_id": str(row["sample_id"]),
            "source_id": str(row["source_id"]),
            "speaker_id": str(row["speaker_id"]),
            "utterance_group_id": str(row["utterance_group_id"]),
            "transcript": str(row["transcript"]),
            "decision": decision,
        }

    for row in previous:
        if row.get("supplement_label_override"):
            decision = (
                "excluded_evaluation_speaker"
                if row["speaker_id_reliable"] and row["speaker_id"] in evaluation_speakers
                else "eligible"
            )
            note_reviewed(row, decision)
    candidates = [
        row
        for row in previous
        if not (row["speaker_id_reliable"] and row["speaker_id"] in evaluation_speakers)
    ]
    selected: list[dict[str, Any]] = []
    rejected = Counter()
    real_counts = Counter(
        row["canonical_intent"]
        for row in benchmark_records
        if row["official_split"] == "train" and row["supported"] and not row["is_synthetic"]
    )
    target = int(settings["target_real_records_per_intent"])
    for intent in INTENTS:
        pool = [row for row in candidates if row["supported"] and row["canonical_intent"] == intent]
        pool = interleave_groups(pool, lambda row: row["speaker_id"], int(config["seed"]))
        needed = max(0, target - real_counts[intent])
        for row in pool:
            if needed <= 0:
                break
            source_identity = (str(row["source_dataset"]), str(row["source_id"]))
            if source_identity in used_source_ids:
                rejected["duplicate_source_identity"] += 1
                note_reviewed(row, "rejected_duplicate_source_identity")
                continue
            try:
                adapted = adapt_previous_record(row, project_root, output_root, config["audio"])
            except (FileNotFoundError, RuntimeError, ValueError):
                rejected["unreadable_audio"] += 1
                note_reviewed(row, "rejected_unreadable_audio")
                continue
            if adapted["pcm_sha256"] in evaluation_pcm:
                rejected["evaluation_audio_match"] += 1
                note_reviewed(row, "rejected_evaluation_audio_match")
                continue
            if adapted["pcm_sha256"] in used_pcm:
                rejected["duplicate_audio"] += 1
                note_reviewed(row, "rejected_duplicate_audio")
                continue
            errors = validate_prepared_record(adapted)
            if errors:
                rejected["invalid_record"] += 1
                note_reviewed(row, "rejected_invalid_record")
                continue
            selected.append(adapted)
            used_pcm.add(adapted["pcm_sha256"])
            used_source_ids.add(source_identity)
            note_reviewed(row, "selected")
            needed -= 1

    unsupported_pool = interleave_groups(
        [row for row in candidates if not row["supported"]],
        lambda row: f"{row.get('category_partition', '')}|{row['speaker_id']}",
        int(config["seed"]),
    )
    unsupported_target = int(settings["target_unsupported_records"])
    unsupported_selected = 0
    for row in unsupported_pool:
        if unsupported_selected >= unsupported_target:
            break
        source_identity = (str(row["source_dataset"]), str(row["source_id"]))
        if source_identity in used_source_ids:
            rejected["duplicate_source_identity"] += 1
            continue
        try:
            adapted = adapt_previous_record(row, project_root, output_root, config["audio"])
        except (FileNotFoundError, RuntimeError, ValueError):
            rejected["unreadable_audio"] += 1
            continue
        if adapted["pcm_sha256"] in evaluation_pcm:
            rejected["evaluation_audio_match"] += 1
            continue
        if adapted["pcm_sha256"] in used_pcm:
            rejected["duplicate_audio"] += 1
            continue
        errors = validate_prepared_record(adapted)
        if errors:
            rejected["invalid_record"] += 1
            continue
        selected.append(adapted)
        used_pcm.add(adapted["pcm_sha256"])
        used_source_ids.add(source_identity)
        unsupported_selected += 1
    for decision in reviewed_decisions.values():
        if decision["decision"] == "eligible":
            decision["decision"] = "not_selected_target_met"
    return selected, {
        "enabled": True,
        "candidates": len(candidates),
        "selected": len(selected),
        "selected_supported_by_intent": dict(
            sorted(Counter(row["canonical_intent"] for row in selected if row["supported"]).items())
        ),
        "selected_unsupported": unsupported_selected,
        "rejected": dict(sorted(rejected.items())),
        "reviewed_label_overrides": {
            **override_report,
            "selected_records": sum(
                row["decision"] == "selected" for row in reviewed_decisions.values()
            ),
            "selected_groups": len(
                {
                    row["utterance_group_id"]
                    for row in reviewed_decisions.values()
                    if row["decision"] == "selected"
                }
            ),
            "decisions": [reviewed_decisions[key] for key in sorted(reviewed_decisions)],
        },
    }


def label_for_split(row: dict[str, Any]) -> str:
    return str(row["canonical_intent"] if row["supported"] else "UNSUPPORTED")


def choose_validation_speakers(
    development: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[set[str], dict[str, Any]]:
    settings = config["validation"]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    fixed_train = []
    for row in development:
        if row["is_synthetic"] or not row["speaker_id_reliable"]:
            fixed_train.append(row)
        else:
            groups[row["speaker_id"]].append(row)
    speakers = sorted(groups)
    if len(speakers) < 2:
        raise ValueError("At least two reliable development speakers are required")
    fraction = float(settings["speaker_fraction"])
    minimum = int(settings["minimum_records_per_label"])
    label_minimums = {
        str(label): int(value)
        for label, value in settings.get("minimum_records_by_label", {}).items()
    }
    speaker_minimums = {
        str(label): int(value)
        for label, value in settings.get("minimum_speakers_by_label", {}).items()
    }
    recording_group_minimums = {
        str(label): int(value)
        for label, value in settings.get("minimum_recording_groups_by_label", {}).items()
    }
    desired = max(1, min(len(speakers) - 1, round(len(speakers) * fraction)))
    totals = Counter(label_for_split(row) for rows in groups.values() for row in rows)
    speaker_label_counts: dict[str, Counter[str]] = {}
    speaker_label_groups: dict[str, dict[str, set[str]]] = {}
    total_speakers_by_label: Counter[str] = Counter()
    total_groups_by_label: dict[str, set[str]] = defaultdict(set)
    for speaker, rows in groups.items():
        counts = Counter(label_for_split(row) for row in rows)
        speaker_label_counts[speaker] = counts
        grouped: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            label = label_for_split(row)
            grouped[label].add(str(row["recording_group_id"]))
            total_groups_by_label[label].add(str(row["recording_group_id"]))
        speaker_label_groups[speaker] = grouped
        for label in counts:
            total_speakers_by_label[label] += 1

    best: tuple[
        float, set[str], Counter[str], Counter[str], dict[str, set[str]]
    ] | None = None
    for attempt in range(int(settings["search_attempts"])):
        shuffled = list(speakers)
        random.Random(int(config["seed"]) + attempt).shuffle(shuffled)
        chosen = set(shuffled[:desired])
        counts: Counter[str] = Counter()
        selected_speakers_by_label: Counter[str] = Counter()
        selected_groups_by_label: dict[str, set[str]] = defaultdict(set)
        for speaker in chosen:
            counts.update(speaker_label_counts[speaker])
            for label in speaker_label_counts[speaker]:
                selected_speakers_by_label[label] += 1
                selected_groups_by_label[label].update(speaker_label_groups[speaker][label])
        training_counts = totals - counts
        record_deficiency = sum(
            max(
                0,
                min(label_minimums.get(label, minimum), totals[label] - 1) - counts[label],
            )
            + max(
                0,
                min(label_minimums.get(label, minimum), totals[label] - 1)
                - training_counts[label],
            )
            for label in totals
        )
        speaker_deficiency = sum(
            max(
                0,
                min(required, total_speakers_by_label[label] - 1)
                - selected_speakers_by_label[label],
            )
            + max(
                0,
                min(required, total_speakers_by_label[label] - 1)
                - (total_speakers_by_label[label] - selected_speakers_by_label[label]),
            )
            for label, required in speaker_minimums.items()
        )
        recording_group_deficiency = sum(
            max(
                0,
                min(required, len(total_groups_by_label[label]) - 1)
                - len(selected_groups_by_label[label]),
            )
            + max(
                0,
                min(required, len(total_groups_by_label[label]) - 1)
                - (
                    len(total_groups_by_label[label])
                    - len(selected_groups_by_label[label])
                ),
            )
            for label, required in recording_group_minimums.items()
        )
        deficient = record_deficiency + speaker_deficiency + recording_group_deficiency
        distribution_error = sum(
            abs(counts[label] / max(totals[label], 1) - fraction) for label in totals
        )
        record_fraction = sum(counts.values()) / max(sum(totals.values()), 1)
        score = deficient * 1000.0 + distribution_error + abs(record_fraction - fraction)
        candidate = (
            score,
            chosen,
            counts,
            selected_speakers_by_label,
            selected_groups_by_label,
        )
        if best is None or candidate[0] < best[0]:
            best = candidate
    assert best is not None
    if best[0] >= 1000.0:
        raise ValueError(
            "Unable to create a speaker-disjoint validation split with the configured "
            "record, speaker, and recording-group minima"
        )
    return best[1], {
        "speaker_fraction": fraction,
        "minimum_records_per_label": minimum,
        "minimum_records_by_label": label_minimums,
        "minimum_speakers_by_label": speaker_minimums,
        "minimum_recording_groups_by_label": recording_group_minimums,
        "validation_speakers": len(best[1]),
        "eligible_speakers": len(speakers),
        "fixed_train_records": len(fixed_train),
        "validation_by_label": dict(sorted(best[2].items())),
        "validation_speakers_by_label": dict(sorted(best[3].items())),
        "validation_recording_groups_by_label": dict(
            sorted((label, len(values)) for label, values in best[4].items())
        ),
    }


def assign_splits(
    benchmark: list[dict[str, Any]], supplements: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    development = [row for row in benchmark if row["official_split"] == "train"] + supplements
    validation_speakers, report = choose_validation_speakers(development, config)
    assigned = []
    for original in benchmark + supplements:
        row = dict(original)
        if row["official_split"] in {"test", "holdout"}:
            row["split"] = row["official_split"]
        elif (
            not row["is_synthetic"]
            and row["speaker_id_reliable"]
            and row["speaker_id"] in validation_speakers
        ):
            row["split"] = "validation"
        else:
            row["split"] = "train"
        assigned.append(row)
    return assigned, report


def audit_assignments(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    by_split = {
        split: [row for row in records if row["split"] == split]
        for split in MANIFEST_SPLITS
    }
    expected = config["expected_prepared_rows"]
    if len(by_split["test"]) != int(expected["test"]):
        raise ValueError("Test membership changed")
    if len(by_split["holdout"]) != int(expected["holdout"]):
        raise ValueError("Holdout membership changed")
    if any(row["is_synthetic"] for row in by_split["validation"]):
        raise ValueError("Validation contains synthetic audio")
    reliable = {
        split: {row["speaker_id"] for row in rows if row["speaker_id_reliable"]}
        for split, rows in by_split.items()
    }
    overlaps = {}
    for left_index, left in enumerate(MANIFEST_SPLITS):
        for right in MANIFEST_SPLITS[left_index + 1 :]:
            overlap = sorted(reliable[left] & reliable[right])
            overlaps[f"{left}:{right}"] = overlap
            if overlap:
                raise ValueError(f"Speaker leakage between {left} and {right}: {overlap[:5]}")
    sample_ids = [row["sample_id"] for row in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Duplicate sample IDs in prepared dataset")
    pcm_hashes = [row["pcm_sha256"] for row in records]
    if len(pcm_hashes) != len(set(pcm_hashes)):
        raise ValueError("Duplicate decoded audio in prepared dataset")
    source_identities = [
        (str(row["source_dataset"]), str(row["source_id"])) for row in records
    ]
    if len(source_identities) != len(set(source_identities)):
        raise ValueError("Duplicate source identities in prepared dataset")
    recording_groups = {
        split: {
            (str(row["source_dataset"]), str(row["recording_group_id"]))
            for row in rows
        }
        for split, rows in by_split.items()
    }
    recording_group_overlaps = {}
    for left_index, left in enumerate(MANIFEST_SPLITS):
        for right in MANIFEST_SPLITS[left_index + 1 :]:
            overlap = recording_groups[left] & recording_groups[right]
            recording_group_overlaps[f"{left}:{right}"] = len(overlap)
            if overlap:
                raise ValueError(
                    f"Recording-group leakage between {left} and {right}: "
                    f"{sorted(overlap)[:5]}"
                )
    return {
        "splits": {
            split: {
                "records": len(rows),
                "supported": sum(row["supported"] for row in rows),
                "unsupported": sum(not row["supported"] for row in rows),
                "real": sum(not row["is_synthetic"] for row in rows),
                "synthetic": sum(row["is_synthetic"] for row in rows),
                "supplemental": sum(row["is_supplemental"] for row in rows),
                "supplemental_synthetic": sum(
                    bool(row.get("is_supplemental_synthetic", False)) for row in rows
                ),
                "slot_annotated": sum(bool(row["slots"]) for row in rows),
                "missing_slot_by_intent": dict(
                    sorted(
                        Counter(
                            row["canonical_intent"]
                            for row in rows
                            if row["supported"]
                            and row["canonical_intent"] in SLOT_BY_INTENT
                            and not row["slots"]
                        ).items()
                    )
                ),
                "speakers": len({row["speaker_id"] for row in rows}),
                "speakers_by_intent": dict(
                    sorted(
                        (
                            intent,
                            len(
                                {
                                    row["speaker_id"]
                                    for row in rows
                                    if row["supported"]
                                    and row["canonical_intent"] == intent
                                }
                            ),
                        )
                        for intent in INTENTS
                    )
                ),
                "recording_groups_by_intent": dict(
                    sorted(
                        (
                            intent,
                            len(
                                {
                                    row["recording_group_id"]
                                    for row in rows
                                    if row["supported"]
                                    and row["canonical_intent"] == intent
                                }
                            ),
                        )
                        for intent in INTENTS
                    )
                ),
                "by_intent": dict(
                    sorted(
                        Counter(
                            row["canonical_intent"] for row in rows if row["supported"]
                        ).items()
                    )
                ),
            }
            for split, rows in by_split.items()
        },
        "speaker_overlaps": overlaps,
        "recording_group_overlaps": recording_group_overlaps,
        "unique_audio": len(set(pcm_hashes)),
        "unique_source_identities": len(set(source_identities)),
        "records": len(records),
    }


def validate_prepared_dataset(output_root: Path) -> dict[str, Any]:
    release_path = output_root / "release.json"
    if not release_path.is_file():
        raise FileNotFoundError(f"Prepared release is missing: {release_path}")
    release = json.loads(release_path.read_text(encoding="utf-8"))
    release_artifacts = {
        "schema.json": "schema_sha256",
        "label_overrides.jsonl": "label_overrides_sha256",
        "dataset_exclusions.jsonl": "dataset_exclusions_sha256",
        "supplement_label_overrides.jsonl": "supplement_label_overrides_sha256",
        "preparation_config.json": "preparation_config_sha256",
    }
    for relative, release_key in release_artifacts.items():
        path = output_root / relative
        if not path.is_file() or sha256_file(path) != release[release_key]:
            raise ValueError(f"Prepared release artifact mismatch: {relative}")
    prepared_config = json.loads(
        (output_root / "preparation_config.json").read_text(encoding="utf-8")
    )
    if stable_digest(json.dumps(prepared_config, sort_keys=True)) != release["config_sha256"]:
        raise ValueError("Prepared configuration does not match release metadata")
    checked = 0
    for relative, expected in release["manifest_sha256"].items():
        path = output_root / relative
        if sha256_file(path) != expected:
            raise ValueError(f"Manifest checksum mismatch: {relative}")
        for row in read_jsonl(path):
            errors = validate_prepared_record(row)
            if errors:
                raise ValueError(f"Invalid record {row.get('sample_id')}: {errors}")
            audio = output_root / row["audio_path"]
            if not audio.is_file() or sha256_file(audio) != row["audio_sha256"]:
                raise ValueError(f"Audio checksum mismatch: {audio}")
            checked += 1
    return {"release": str(release_path), "records_checked": checked, "ok": True}


def prepare_dataset(
    project_root: Path,
    config: dict[str, Any],
    *,
    overwrite: bool,
    include_supplements: bool,
) -> dict[str, Any]:
    project_root = project_root.resolve()
    source_root = resolve(project_root, config["source_root"]).resolve()
    output_root = resolve(project_root, config["output_root"]).resolve()
    dataset_root = (project_root / "dataset").resolve()
    if output_root.parent != dataset_root:
        raise ValueError(f"Prepared output must be a direct child of {dataset_root}: {output_root}")
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} already exists; pass --overwrite to rebuild it")
        if output_root in {project_root, dataset_root, source_root}:
            raise ValueError(f"Refusing to remove unsafe output path: {output_root}")
        # Normalized files are content-addressed and checksum-verified by
        # materialize_audio. Preserve this cache across an intentional rebuild
        # so a late manifest/split failure does not require decoding every clip
        # again. Non-audio release artifacts are rebuilt from scratch.
        for child in output_root.iterdir():
            if child.name == "audio":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    output_root.mkdir(parents=True, exist_ok=True)
    source_release = verify_source_release(source_root, config)
    schema_path = resolve(project_root, config["schema"])
    load_schema(schema_path)
    overrides_path = resolve(project_root, config["label_overrides"])
    overrides = load_overrides(overrides_path)
    exclusions_path = resolve(project_root, config["dataset_exclusions"])
    exclusions = load_exclusions(exclusions_path)
    supplement_overrides_path = resolve(project_root, config["supplement_label_overrides"])
    load_supplement_overrides(supplement_overrides_path)
    benchmark, source_report = build_huggingface_records(
        source_root, output_root, config, overrides, exclusions
    )
    supplemental_synthetic, supplemental_synthetic_report = (
        build_supplemental_synthetic_records(source_root, output_root, config)
    )
    if include_supplements:
        supplements, supplement_report = select_supplements(
            project_root, output_root, config, benchmark
        )
    else:
        supplements, supplement_report = [], {"enabled": False, "selected": 0}
    synthetic_negatives, synthetic_negatives_report = build_synthetic_negative_records(
        output_root,
        config,
        {
            str(row["pcm_sha256"])
            for row in benchmark + supplements + supplemental_synthetic
        },
    )
    records, split_report = assign_splits(
        benchmark, supplements + supplemental_synthetic + synthetic_negatives, config
    )
    audit = audit_assignments(records, config)
    manifests = {}
    for split in MANIFEST_SPLITS:
        rows = sorted(
            (row for row in records if row["split"] == split),
            key=lambda row: str(row["sample_id"]),
        )
        relative = Path("manifests") / f"{split}.jsonl"
        write_jsonl(output_root / relative, rows)
        manifests[str(relative)] = sha256_file(output_root / relative)
    referenced_audio = {str(row["audio_path"]) for row in records}
    audio_root = output_root / "audio"
    if audio_root.is_dir():
        for path in audio_root.rglob("*.wav"):
            if str(path.relative_to(output_root)) not in referenced_audio:
                path.unlink()
        for directory in sorted(audio_root.rglob("*"), reverse=True):
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()
    shutil.copy2(schema_path, output_root / "schema.json")
    shutil.copy2(overrides_path, output_root / "label_overrides.jsonl")
    shutil.copy2(exclusions_path, output_root / "dataset_exclusions.jsonl")
    shutil.copy2(
        supplement_overrides_path,
        output_root / "supplement_label_overrides.jsonl",
    )
    write_json(output_root / "preparation_config.json", config)
    reports = output_root / "reports"
    write_json(reports / "summary.json", audit)
    write_json(reports / "source_release.json", source_release)
    write_json(reports / "source_conversion.json", source_report)
    write_json(reports / "supplement_selection.json", supplement_report)
    write_json(reports / "supplemental_synthetic.json", supplemental_synthetic_report)
    write_json(reports / "synthetic_negatives.json", synthetic_negatives_report)
    write_json(reports / "validation_split.json", split_report)
    write_json(
        reports / "label_changes.json",
        {"count": len(overrides), "overrides": list(overrides.values())},
    )
    release = {
        "dataset_name": config["dataset_name"],
        "dataset_version": config["dataset_version"],
        "seed": int(config["seed"]),
        "source_revision": config["source_revision"],
        "source_shards_sha256": source_release["shards_sha256"],
        "schema_sha256": sha256_file(output_root / "schema.json"),
        "label_overrides_sha256": sha256_file(overrides_path),
        "dataset_exclusions_sha256": sha256_file(exclusions_path),
        "supplement_label_overrides_sha256": sha256_file(supplement_overrides_path),
        "preparation_config_sha256": sha256_file(output_root / "preparation_config.json"),
        "config_sha256": stable_digest(json.dumps(config, sort_keys=True)),
        "manifest_sha256": manifests,
        "holdout_policy": "sealed_for_final_raspberry_pi_evaluation",
        "test_membership_frozen": True,
        "holdout_membership_frozen": True,
        "numerals_excluded": not bool(config.get("include_numerals", False)),
        "numerals_exclusion_reason": (
            "Numeral clips are auxiliary slot material and have no command-level intent label"
        ),
        "conditions": {
            "benchmark_real": "not synthetic and not supplemental",
            "benchmark_mixed": "not supplemental",
            "supplemented_real": "not synthetic",
            "supplemented_mixed": "all prepared records except supplemental synthetic",
            "supplemented_mixed_with_negatives": (
                "supplemented_mixed plus the pinned synthetic_negatives/train records"
            ),
            "supplemented_mixed_extended": "all prepared training records",
        },
        "summary": audit,
    }
    write_json(output_root / "release.json", release)
    return release
