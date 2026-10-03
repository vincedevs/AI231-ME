from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch

    from dataset_pipeline.schema import INTENTS, SLOT_BY_INTENT
    from vcm_training.experiment import initialize_study_identity
    from vcm_training.export import build_deployment_metadata, export_onnx
    from vcm_training.metrics import greedy_ctc_decode
    from vcm_training.models import ARCHITECTURES, SLOT_TO_INDEX, SLOT_TOKENS, build_model
    from vcm_training.training import compute_loss, repair_legacy_binary_scope_history


AUDIO_CONFIG = {
    "sample_rate": 16000,
    "maximum_seconds": 12.0,
    "n_fft": 400,
    "hop_length": 160,
    "n_mels": 40,
    "minimum_hertz": 20.0,
    "maximum_hertz": 7600.0,
}


@unittest.skipUnless(TORCH_AVAILABLE, "training dependencies are not installed")
class ModelContractTests(unittest.TestCase):
    def test_study_identity_rejects_changed_dataset_or_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "experiment"
            output.mkdir()
            release = root / "release.json"
            release.write_text('{"revision":"one"}\n', encoding="utf-8")
            identity = initialize_study_identity(output, {"seed": 231}, release)
            self.assertEqual(
                initialize_study_identity(output, {"seed": 231}, release), identity
            )
            with self.assertRaisesRegex(ValueError, "different dataset release"):
                initialize_study_identity(output, {"seed": 947}, release)

    def test_binary_scope_loss_handles_an_unsupported_only_batch(self) -> None:
        model = build_model("dscnn", "binary_scope", AUDIO_CONFIG)
        batch_size = 2
        batch = {
            "waveform": torch.randn(batch_size, 16000),
            "lengths": torch.full((batch_size,), 16000, dtype=torch.long),
            "intent_targets": torch.full((batch_size,), -1, dtype=torch.long),
            "scope_targets": torch.zeros(batch_size),
            "slot_target_sequences": [[], []],
        }
        loss, components = compute_loss(
            model,
            batch,
            "binary_scope",
            {
                "label_smoothing": 0.05,
                "slot_loss_weight": 0.5,
                "scope_loss_weight": 0.5,
            },
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(components["intent_loss"], 0.0)
        loss.backward()
        self.assertTrue(
            all(
                torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
                if parameter.grad is not None
            )
        )

    def test_legacy_history_repair_is_narrow_and_auditable(self) -> None:
        history = [
            {
                "epoch": 8,
                "training": {
                    "total_loss": float("nan"),
                    "intent_loss": float("nan"),
                    "slot_loss": 0.5,
                    "scope_loss": 0.25,
                },
                "validation": {"intent": {"macro": {"f1": 0.8}}},
            }
        ]
        repaired, repairs = repair_legacy_binary_scope_history(history, "binary_scope")
        self.assertIsNone(repaired[0]["training"]["total_loss"])
        self.assertIsNone(repaired[0]["training"]["intent_loss"])
        self.assertTrue(repairs)
        self.assertTrue(torch.isnan(torch.tensor(history[0]["training"]["total_loss"])))

        unexpected = [{"training": {"scope_loss": float("nan")}}]
        with self.assertRaises(FloatingPointError):
            repair_legacy_binary_scope_history(unexpected, "binary_scope")

    def test_every_architecture_uses_raw_waveform_contract(self) -> None:
        waveform = torch.randn(2, 16000)
        lengths = torch.tensor([16000, 12000])
        for architecture in ARCHITECTURES:
            with self.subTest(architecture=architecture):
                model = build_model(architecture, "binary_scope", AUDIO_CONFIG).eval()
                with torch.inference_mode():
                    intent, slot, scope, output_lengths = model(waveform, lengths)
                self.assertEqual(intent.shape, (2, 19))
                self.assertEqual(slot.shape[0], 2)
                self.assertEqual(slot.shape[2], 40)
                self.assertEqual(scope.shape, (2, 1))
                self.assertTrue(torch.all(output_lengths <= slot.shape[1]))

    def test_tiny_conformer_onnx_accepts_variable_audio_lengths(self) -> None:
        model = build_model("tiny_conformer", "binary_scope", AUDIO_CONFIG).eval()
        evaluation = {
            "onnx_opset": 17,
            "onnx_validation_seconds": [1.0, 2.0, 3.0],
            "onnx_absolute_tolerance": 0.0001,
            "onnx_relative_tolerance": 0.001,
            "latency_warmup_runs": 1,
            "latency_timed_runs": 1,
        }
        with tempfile.TemporaryDirectory() as temporary:
            report = export_onnx(
                model,
                Path(temporary) / "tiny_conformer.onnx",
                AUDIO_CONFIG,
                evaluation,
            )
        self.assertEqual(
            {row["audio_seconds"] for row in report["parity"]},
            {1.0, 2.0, 3.0},
        )
        self.assertTrue(all(row["close"] for row in report["parity"]))

    def test_unknown_class_is_internal_twentieth_output(self) -> None:
        model = build_model("dscnn", "unknown_class", AUDIO_CONFIG).eval()
        with torch.inference_mode():
            intent, _, _, _ = model(torch.randn(1, 16000), torch.tensor([16000]))
        self.assertEqual(intent.shape, (1, 20))

    def test_ctc_decoder_collapses_blanks_and_repetitions(self) -> None:
        tokens = [SLOT_TO_INDEX["a"], SLOT_TO_INDEX["a"], 0, SLOT_TO_INDEX["b"]]
        logits = torch.full((1, len(tokens), 40), -10.0)
        for step, token in enumerate(tokens):
            logits[0, step, token] = 10.0
        self.assertEqual(greedy_ctc_decode(logits, torch.tensor([4])), ["ab"])

    def test_deployment_metadata_preserves_the_frozen_contract(self) -> None:
        identity = {
            "architecture": "dscnn",
            "data_condition": "supplemented_real",
            "rejection_strategy": "binary_scope",
            "seed": 231,
        }
        metadata = build_deployment_metadata(
            audio_config=AUDIO_CONFIG,
            intent_labels=INTENTS,
            slot_by_intent=SLOT_BY_INTENT,
            slot_tokens=SLOT_TOKENS,
            rejection_strategy="binary_scope",
            temperature=1.2,
            scope_threshold=0.6,
            minimum_intent_confidence=0.7,
            experiment_identity=identity,
        )
        self.assertEqual(tuple(metadata["intent_labels"]), INTENTS)
        self.assertEqual(metadata["slot_by_intent"], SLOT_BY_INTENT)
        self.assertEqual(tuple(metadata["slot_tokens"]), SLOT_TOKENS)
        self.assertEqual(metadata["calibration_source"]["temperature"], "validation")


if __name__ == "__main__":
    unittest.main()
