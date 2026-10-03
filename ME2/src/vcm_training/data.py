from __future__ import annotations

import json
import math
import random
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.nn import functional as F
from torch.utils.data import Dataset, WeightedRandomSampler

from dataset_pipeline.common import normalize_text, stable_digest
from dataset_pipeline.schema import INTENTS, SLOT_BY_INTENT

from .audio import augment_waveform
from .models import SLOT_TO_INDEX

INTENT_TO_INDEX = {intent: index for index, intent in enumerate(INTENTS)}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def normalized_slot_text(record: dict[str, Any]) -> str:
    intent = record.get("canonical_intent")
    slot_name = SLOT_BY_INTENT.get(intent)
    if slot_name is None:
        return ""
    slot = record.get("slots", {}).get(slot_name, {})
    return normalize_text(str(slot.get("surface", "")))


def encode_slot(text: str) -> list[int]:
    return [SLOT_TO_INDEX[character] for character in text if character in SLOT_TO_INDEX]


def load_experiment_records(
    prepared_root: Path,
    split: str,
    data_condition: str,
    rejection_strategy: str,
) -> list[dict[str, Any]]:
    if split == "holdout":
        raise ValueError("Holdout is sealed and cannot be loaded by the training pipeline")
    available = {
        "benchmark_real",
        "benchmark_mixed",
        "supplemented_real",
        "supplemented_mixed",
        "supplemented_mixed_with_negatives",
        "supplemented_mixed_extended",
    }
    if data_condition not in available:
        raise ValueError(f"Unknown data condition {data_condition!r}; expected {sorted(available)}")
    records = read_jsonl(prepared_root / "manifests" / f"{split}.jsonl")
    if split == "train":
        records = [
            row
            for row in records
            if not (
                data_condition in {"benchmark_real", "benchmark_mixed"}
                and row["is_supplemental"]
            )
            and not (
                data_condition in {"benchmark_real", "supplemented_real"}
                and row["is_synthetic"]
            )
            and not (
                data_condition != "supplemented_mixed_with_negatives"
                and row.get("is_negative_ablation", False)
            )
            and not (
                data_condition != "supplemented_mixed_extended"
                and row.get("is_supplemental_synthetic", False)
            )
        ]
        if rejection_strategy == "confidence":
            records = [row for row in records if row["supported"]]
    elif split == "validation" and any(row.get("is_synthetic") for row in records):
        raise ValueError("Validation must remain real-only")
    identifiers = [str(row["sample_id"]) for row in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError(f"Duplicate sample IDs in prepared/{split}")
    return records


def resolve_audio_path(project_root: Path, record: dict[str, Any]) -> Path:
    path = Path(record["audio_path"])
    return path if path.is_absolute() else project_root / path


class CommandDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        project_root: Path,
        audio_config: dict[str, Any],
        augmentation_config: dict[str, Any] | None,
        seed: int,
    ) -> None:
        self.records = list(records)
        self.project_root = project_root
        self.sample_rate = int(audio_config["sample_rate"])
        self.maximum_samples = round(float(audio_config["maximum_seconds"]) * self.sample_rate)
        self.minimum_samples = int(audio_config["n_fft"])
        self.augmentation_config = augmentation_config
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _read_audio(self, path: Path) -> torch.Tensor:
        audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
        waveform = np.mean(audio, axis=1, dtype=np.float32)
        if int(source_rate) != self.sample_rate:
            divisor = math.gcd(int(source_rate), self.sample_rate)
            waveform = resample_poly(
                waveform,
                self.sample_rate // divisor,
                int(source_rate) // divisor,
            ).astype(np.float32)
        waveform = waveform[: self.maximum_samples]
        if not waveform.size:
            raise ValueError(f"Empty audio file: {path}")
        return torch.from_numpy(waveform.copy())

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        path = resolve_audio_path(self.project_root, record)
        waveform = self._read_audio(path)
        if self.augmentation_config is not None:
            item_seed = int(
                stable_digest(f"{self.seed}|{self.epoch}|{record['sample_id']}")[:16], 16
            )
            waveform = augment_waveform(
                waveform,
                self.sample_rate,
                self.augmentation_config,
                random.Random(item_seed),
            )
        supported = bool(record["supported"])
        intent_index = INTENT_TO_INDEX[str(record["canonical_intent"])] if supported else -1
        slot_text = normalized_slot_text(record) if supported else ""
        return {
            "waveform": waveform,
            "length": int(waveform.numel()),
            "intent_index": intent_index,
            "supported": supported,
            "slot_text": slot_text,
            "slot_tokens": encode_slot(slot_text),
            "sample_id": str(record["sample_id"]),
            "source_dataset": str(record.get("source_dataset", "")),
            "ood_type": str(record.get("ood_type", "supported")),
            "category_partition": str(record.get("category_partition", "supported")),
            "is_synthetic": bool(record.get("is_synthetic", False)),
            "is_supplemental": bool(record.get("is_supplemental", False)),
        }


def collate_commands(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    maximum = max(max(item["length"] for item in items), 400)
    waveforms = torch.stack(
        [F.pad(item["waveform"], (0, maximum - item["length"])) for item in items]
    )
    slot_sequences = [item["slot_tokens"] for item in items]
    flattened_slots = torch.tensor(
        [token for sequence in slot_sequences for token in sequence], dtype=torch.long
    )
    return {
        "waveform": waveforms,
        "lengths": torch.tensor([item["length"] for item in items], dtype=torch.long),
        "intent_targets": torch.tensor([item["intent_index"] for item in items], dtype=torch.long),
        "scope_targets": torch.tensor(
            [float(item["supported"]) for item in items], dtype=torch.float32
        ),
        "slot_targets": flattened_slots,
        "slot_target_lengths": torch.tensor(
            [len(sequence) for sequence in slot_sequences], dtype=torch.long
        ),
        "slot_target_sequences": slot_sequences,
        "slot_texts": [item["slot_text"] for item in items],
        "sample_ids": [item["sample_id"] for item in items],
        "source_datasets": [item["source_dataset"] for item in items],
        "ood_types": [item["ood_type"] for item in items],
        "category_partitions": [item["category_partition"] for item in items],
        "is_synthetic": [item["is_synthetic"] for item in items],
        "is_supplemental": [item["is_supplemental"] for item in items],
    }


def balanced_sampler(
    records: Sequence[dict[str, Any]],
    unsupported_fraction: float,
    seed: int,
) -> WeightedRandomSampler:
    intent_counts = Counter(str(row["canonical_intent"]) for row in records if row["supported"])
    unsupported_count = sum(not row["supported"] for row in records)
    has_unsupported = unsupported_count > 0
    supported_mass = 1.0 - unsupported_fraction if has_unsupported else 1.0
    weights = []
    for row in records:
        if row["supported"]:
            intent = str(row["canonical_intent"])
            weights.append(supported_mass / (len(intent_counts) * intent_counts[intent]))
        else:
            weights.append(unsupported_fraction / unsupported_count)
    generator = torch.Generator().manual_seed(int(seed))
    return WeightedRandomSampler(
        torch.tensor(weights, dtype=torch.double),
        num_samples=len(records),
        replacement=True,
        generator=generator,
    )


def dataset_summary(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "records": len(records),
        "supported": sum(bool(row["supported"]) for row in records),
        "unsupported": sum(not bool(row["supported"]) for row in records),
        "synthetic": sum(bool(row.get("is_synthetic")) for row in records),
        "supplemental": sum(bool(row.get("is_supplemental")) for row in records),
        "supplemental_synthetic": sum(
            bool(row.get("is_supplemental_synthetic")) for row in records
        ),
        "synthetic_negative_ablation": sum(
            bool(row.get("is_negative_ablation")) for row in records
        ),
        "slot_annotated": sum(bool(row.get("slots")) for row in records),
        "slot_annotated_by_intent": dict(
            sorted(
                Counter(
                    str(row["canonical_intent"])
                    for row in records
                    if row["supported"] and row.get("slots")
                ).items()
            )
        ),
        "by_intent": dict(
            sorted(
                Counter(str(row["canonical_intent"]) for row in records if row["supported"]).items()
            )
        ),
        "by_source": dict(
            sorted(Counter(str(row.get("source_dataset", "")) for row in records).items())
        ),
    }
