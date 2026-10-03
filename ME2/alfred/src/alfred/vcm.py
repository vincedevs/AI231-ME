from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .config import AlfredConfig
from .contracts import Decision, VcmResult

DEFAULT_SLOT_TOKENS = (
    "<blank>",
    *tuple(" abcdefghijklmnopqrstuvwxyz0123456789%'"),
)


def prepare_waveform(waveform: Any, minimum_samples: int, minimum_rms: float) -> Any | None:
    """Return finite clipped mono audio, or ``None`` for an invalid command.

    Silero VAD is Alfred's primary speech gate. This inexpensive second gate
    prevents digital silence, empty buffers, non-finite values, and extremely
    low-energy capture failures from reaching a model that was observed to be
    overconfident on pure silence at some durations.
    """
    import numpy as np

    audio = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if audio.size == 0 or not bool(np.isfinite(audio).all()):
        return None
    audio = audio.clip(-1, 1)
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    if rms < minimum_rms:
        return None
    if audio.size < minimum_samples:
        audio = np.pad(audio, (0, minimum_samples - audio.size))
    return audio


def greedy_ctc_decode(logits: Any, length: int, tokens: tuple[str, ...]) -> str:
    text, _ = greedy_ctc_decode_with_confidence(logits, length, tokens)
    return text


def greedy_ctc_decode_with_confidence(
    logits: Any, length: int, tokens: tuple[str, ...]
) -> tuple[str, float]:
    """Decode CTC and return a length-normalized emitted-token confidence.

    For each non-blank collapsed run, we retain the maximum posterior assigned
    to that emitted token. The geometric mean across emitted runs avoids the
    strong length bias of multiplying token probabilities. This score is useful
    as a configurable application gate, but is not claimed to be calibrated.
    """
    import numpy as np

    values = np.asarray(logits, dtype=np.float64)[: int(length)]
    if values.ndim != 2 or values.shape[0] == 0:
        return "", 0.0
    shifted = values - values.max(axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    indices = values.argmax(axis=-1).tolist()
    output: list[str] = []
    emitted_probabilities: list[float] = []
    previous = None
    for frame, index in enumerate(indices):
        if index != 0 and index != previous:
            output.append(tokens[int(index)])
            emitted_probabilities.append(float(probabilities[frame, int(index)]))
        elif index != 0 and index == previous:
            emitted_probabilities[-1] = max(
                emitted_probabilities[-1], float(probabilities[frame, int(index)])
            )
        previous = index
    text = "".join(output).strip()
    if not text or not emitted_probabilities:
        return text, 0.0
    confidence = float(np.exp(np.mean(np.log(np.clip(emitted_probabilities, 1e-12, 1.0)))))
    return text, confidence


def _softmax(values: Any) -> Any:
    import numpy as np

    array = np.asarray(values, dtype=np.float64)
    shifted = array - array.max()
    exponent = np.exp(shifted)
    return exponent / exponent.sum()


def interpret_intent_logits(
    logits: Any, temperature: float, rejection_strategy: str
) -> tuple[int, float, float | None]:
    """Return canonical index, calibrated confidence, and optional scope score."""
    import numpy as np

    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    if temperature <= 0:
        raise ValueError("VCM temperature must be positive")
    expected = 20 if rejection_strategy == "unknown_class" else 19
    if values.size != expected:
        raise ValueError(
            f"VCM strategy {rejection_strategy!r} requires {expected} intent logits; "
            f"received {values.size}"
        )
    canonical = _softmax(values[:19] / temperature)
    intent_index = int(canonical.argmax())
    confidence = float(canonical[intent_index])
    if rejection_strategy == "unknown_class":
        all_probabilities = _softmax(values / temperature)
        return intent_index, confidence, float(1.0 - all_probabilities[19])
    if rejection_strategy == "confidence":
        return intent_index, confidence, confidence
    if rejection_strategy == "binary_scope":
        return intent_index, confidence, None
    raise ValueError(f"Unsupported VCM rejection strategy: {rejection_strategy}")


class MockVcm:
    def __init__(self, result: VcmResult) -> None:
        self.result = result

    @classmethod
    def from_config(cls, config: AlfredConfig) -> MockVcm:
        values = config.document["vcm"]
        intent = values.get("mock_intent")
        slots = dict(values.get("mock_slots", {}))
        if intent is not None and intent not in config.intents:
            raise ValueError(f"Configured mock intent is not canonical: {intent}")
        return cls(
            VcmResult.from_mapping(
                {
                    "decision": values["mock_decision"],
                    "intent": intent,
                    "intent_confidence": values["mock_intent_confidence"],
                    "in_scope_score": values.get("mock_in_scope_score"),
                    "slots": slots,
                    "slot_confidences": {name: 1.0 for name in slots},
                }
            )
        )

    def predict(self, waveform: Any) -> VcmResult:
        del waveform
        return self.result


class OnnxVcm:
    """Decode the self-contained raw-waveform VCM into the application contract."""

    def __init__(
        self,
        model_path: Path,
        metadata_path: Path,
        *,
        expected_intents: tuple[str, ...],
        expected_slot_by_intent: dict[str, str],
        expected_schema_version: str,
        expected_sample_rate: int,
        minimum_waveform_rms: float,
        threads: int = 2,
    ) -> None:
        os.environ.setdefault("ORT_DISABLE_TELEMETRY", "1")
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise RuntimeError("onnxruntime is required for the real VCM") from error
        if not model_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError("The packaged VCM model and metadata are both required")
        if hasattr(ort, "disable_telemetry_events"):
            ort.disable_telemetry_events()
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if str(self.metadata.get("schema_version")) != expected_schema_version:
            raise ValueError("VCM metadata schema version does not match Alfred")
        if int(self.metadata.get("sample_rate", 0)) != expected_sample_rate:
            raise ValueError("VCM metadata sample rate does not match Alfred")
        self.intents = tuple(self.metadata["intent_labels"])
        if self.intents != expected_intents:
            raise ValueError("VCM intent labels must exactly match the canonical schema order")
        self.slot_by_intent = dict(self.metadata["slot_by_intent"])
        if self.slot_by_intent != expected_slot_by_intent:
            raise ValueError("VCM slot mapping does not match the canonical schema")
        self.slot_tokens = tuple(self.metadata.get("slot_tokens", DEFAULT_SLOT_TOKENS))
        if not self.slot_tokens or self.slot_tokens[0] != "<blank>":
            raise ValueError("VCM slot token index zero must be the CTC blank token")
        if len(set(self.slot_tokens)) != len(self.slot_tokens):
            raise ValueError("VCM slot tokens must be unique")
        self.temperature = float(self.metadata["temperature"])
        if self.temperature <= 0:
            raise ValueError("VCM temperature must be positive")
        self.scope_threshold = float(self.metadata["scope_threshold"])
        self.intent_threshold = float(self.metadata["minimum_intent_confidence"])
        if not 0 <= self.scope_threshold <= 1 or not 0 <= self.intent_threshold <= 1:
            raise ValueError("VCM confidence thresholds must be between 0 and 1")
        self.rejection_strategy = str(self.metadata["rejection_strategy"])
        if self.rejection_strategy not in {"confidence", "unknown_class", "binary_scope"}:
            raise ValueError(f"Unsupported VCM rejection strategy: {self.rejection_strategy}")
        self.minimum_samples = int(self.metadata.get("minimum_samples", 400))
        if self.minimum_samples <= 0:
            raise ValueError("VCM minimum_samples must be positive")
        self.minimum_waveform_rms = float(minimum_waveform_rms)
        if not 0 < self.minimum_waveform_rms <= 1:
            raise ValueError("VCM minimum waveform RMS must be greater than 0 and at most 1")
        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = int(threads)
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        inputs = {item.name for item in self.session.get_inputs()}
        outputs = {item.name for item in self.session.get_outputs()}
        if inputs != {"waveform", "lengths"}:
            raise ValueError(f"VCM has incompatible inputs: {sorted(inputs)}")
        required_outputs = {
            "intent_logits",
            "slot_logits",
            "scope_logits",
            "sequence_lengths",
        }
        if outputs != required_outputs:
            raise ValueError(f"VCM has incompatible outputs: {sorted(outputs)}")
        output_shapes = {item.name: item.shape for item in self.session.get_outputs()}
        expected_intent_outputs = 20 if self.rejection_strategy == "unknown_class" else 19
        intent_shape = output_shapes["intent_logits"]
        if len(intent_shape) != 2 or (
            isinstance(intent_shape[-1], int) and intent_shape[-1] != expected_intent_outputs
        ):
            raise ValueError(f"VCM has incompatible intent output shape: {intent_shape}")
        slot_shape = output_shapes["slot_logits"]
        if len(slot_shape) != 3 or (
            isinstance(slot_shape[-1], int) and slot_shape[-1] != len(self.slot_tokens)
        ):
            raise ValueError(f"VCM has incompatible slot output shape: {slot_shape}")

    def predict(self, waveform: Any) -> VcmResult:
        import numpy as np

        audio = prepare_waveform(
            waveform,
            minimum_samples=self.minimum_samples,
            minimum_rms=self.minimum_waveform_rms,
        )
        if audio is None:
            return VcmResult(
                decision=Decision.UNSUPPORTED,
                intent=None,
                intent_confidence=0.0,
                in_scope_score=0.0,
                slots={},
            )
        result = dict(
            zip(
                (item.name for item in self.session.get_outputs()),
                self.session.run(
                    None,
                    {
                        "waveform": audio[None, :],
                        "lengths": np.asarray([audio.size], dtype=np.int64),
                    },
                ),
                strict=True,
            )
        )
        intent_index, confidence, strategy_scope_score = interpret_intent_logits(
            result["intent_logits"][0], self.temperature, self.rejection_strategy
        )
        intent = self.intents[intent_index]

        if self.rejection_strategy == "binary_scope":
            scope_logit = float(result["scope_logits"][0].reshape(-1)[0])
            in_scope_score = float(1.0 / (1.0 + np.exp(-scope_logit)))
        else:
            if strategy_scope_score is None:
                raise RuntimeError("VCM rejection strategy did not produce a scope score")
            in_scope_score = strategy_scope_score

        if in_scope_score < self.scope_threshold:
            decision = Decision.UNSUPPORTED
        elif confidence < self.intent_threshold:
            decision = Decision.LOW_CONFIDENCE
        else:
            decision = Decision.EXECUTE

        slots: dict[str, Any] = {}
        slot_confidences: dict[str, float] = {}
        slot_name = self.slot_by_intent.get(intent)
        if slot_name is not None:
            slot_text, slot_confidence = greedy_ctc_decode_with_confidence(
                result["slot_logits"][0],
                int(result["sequence_lengths"][0]),
                self.slot_tokens,
            )
            if slot_text:
                slots[slot_name] = {"surface": slot_text}
                slot_confidences[slot_name] = slot_confidence
        return VcmResult(
            decision=decision,
            intent=intent,
            intent_confidence=confidence,
            in_scope_score=in_scope_score,
            slots=slots,
            slot_confidences=slot_confidences,
        )


def build_vcm(config: AlfredConfig) -> MockVcm | OnnxVcm:
    values = config.document["vcm"]
    if values["mode"] == "mock":
        return MockVcm.from_config(config)
    artifact = config.model_manifest.get("artifacts", {}).get("vcm")
    if not isinstance(artifact, dict):
        raise TypeError("Real VCM mode requires a packaged vcm artifact")
    model_path = config.local_path(str(artifact["path"]))
    metadata_path = config.local_path(str(artifact["metadata_path"]))
    return OnnxVcm(
        model_path,
        metadata_path,
        expected_intents=config.intents,
        expected_slot_by_intent=config.slot_by_intent,
        expected_schema_version=str(config.schema["schema_version"]),
        expected_sample_rate=int(config.document["audio"]["sample_rate"]),
        minimum_waveform_rms=float(values["minimum_waveform_rms"]),
        threads=int(values["onnx_threads"]),
    )
