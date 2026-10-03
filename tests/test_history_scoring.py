"""History workflow regression uses only synthetic text and injected backends."""

from contextlib import redirect_stdout
import copy
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from student_sim_cd import cache_rescore, history_interventions, history_scoring, inference, predict
from tests.test_history_interventions import row
from tests.test_inference import FakeBackend


class SyntheticTokenizer:
    eos_token_id = 256
    all_special_ids = [256]

    def encode(self, text, *, add_special_tokens=False, truncation=False):
        if add_special_tokens or truncation:
            raise AssertionError("Synthetic tokenizer received a changed canonical protocol")
        return list(text.encode("utf-8"))

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, truncation):
        if not tokenize or not add_generation_prompt or truncation:
            raise AssertionError("Synthetic prompt protocol changed")
        return list(messages[1]["content"].encode("utf-8"))


class SyntheticPoolBackend(FakeBackend):
    """Exactly 318 fixed candidates for 70 samples; no real inference occurs."""

    def generate(self, prompt, sample_id, attempt):
        index = int(sample_id[1:])
        number = attempt + 1 if index < 38 or attempt < 3 else 3
        text = f"print({number})\n"
        self.generations.append((sample_id, attempt))
        return {"attempt": attempt, "source": "greedy" if attempt == 0 else "sampled",
                "seed": attempt, "raw_text": text, "generated_token_ids": self.code_tokens(text),
                "eos_reached": True, "finish_reason": "eos"}


class ScoringOnlyBackend(FakeBackend):
    def generate(self, *args, **kwargs):
        raise AssertionError("No generation is permitted in an intervention")


def protocol_specification():
    return {
        "schema_version": history_scoring.PROTOCOL_VERSION,
        "new_generation": False, "student_execution": False,
        "seeds": list(history_scoring.SEEDS), "variants": list(history_interventions.VARIANTS),
        "expected_samples": 70, "expected_students": 17, "expected_candidates": 318,
        "fixed_method_parameters": copy.deepcopy(predict.METHOD_PARAMETERS),
        "matching_rule": history_interventions.MATCHING_RULE,
        "bootstrap": {"unit": "student", "paired": True, "repetitions": 2000, "seed": 20260929},
    }


class HistoryProtocolTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="history-protocol-test-")
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "protocol.json"

    def write(self, specification):
        self.path.write_bytes(history_scoring.json_bytes(specification))

    def test_fixed_protocol_is_accepted_and_exact_bytes_are_bound(self):
        specification = protocol_specification()
        self.write(specification)
        value, digest = history_scoring.protocol(self.path)
        self.assertEqual(value, specification)
        self.assertEqual(digest, inference.file_hash(self.path))

    def test_fixed_methods_seeds_cohort_pool_bootstrap_and_no_execution_cannot_change(self):
        mutations = [
            ("seeds", [20261002]), ("seeds", [20261004, 20261003, 20261002]),
            ("variants", ["reference_swapped"]), ("expected_samples", 69),
            ("expected_students", 16), ("expected_candidates", 317),
            ("new_generation", True), ("student_execution", True),
            ("matching_rule", "choose_using_labels"),
            ("bootstrap", {"unit": "submission", "paired": True, "repetitions": 2000, "seed": 20260929}),
        ]
        for key, value in mutations:
            with self.subTest(key=key, value=value):
                specification = protocol_specification()
                specification[key] = value
                self.write(specification)
                with self.assertRaises(ValueError):
                    history_scoring.protocol(self.path)
        for mutation in ("remove_method", "coefficient"):
            specification = protocol_specification()
            if mutation == "remove_method":
                specification["fixed_method_parameters"].pop("b")
            else:
                specification["fixed_method_parameters"]["b"]["alpha"] = 42
            self.write(specification)
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "Fixed method"):
                history_scoring.protocol(self.path)

    def test_import_does_not_load_model_dependencies(self):
        process = subprocess.run([sys.executable, "-B", "-c",
                                  "import sys; import student_sim_cd.history_scoring; "
                                  "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"],
                                 capture_output=True, text=True, timeout=10)
        self.assertEqual(process.returncode, 0, process.stderr)


class HistoryWorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="history-scoring-test-")
        self.addCleanup(temporary.cleanup)
        tokenizer_patch = patch.object(history_scoring.preflight, "load_local_tokenizer", return_value=SyntheticTokenizer())
        tokenizer_patch.start()
        self.addCleanup(tokenizer_patch.stop)
        self.root = Path(temporary.name).resolve()
        self.model = self.root / "synthetic-model"
        self.model.mkdir()
        (self.model / "config.json").write_text('{"max_position_embeddings":16384}\n', encoding="utf-8")
        (self.model / "tokenizer.json").write_text('{"synthetic":true}\n', encoding="utf-8")
        (self.model / "model.safetensors").write_bytes(b"synthetic-not-loadable-weights")
        self.config = inference.InferenceConfig(model_path=str(self.model), num_generations=4,
                                               max_context_tokens=16384, max_new_tokens=2048)
        self.rows = [row(f"q{i}", f"dev{i % 17}", "print(0)\n", day=20,
                         history_code=f"synthetic query history {i}\n") for i in range(70)]
        self.train = [row("original", "train-original", "original", day=4),
                      row("alternative-a", "train-a", "alternative-a", day=4),
                      row("alternative-b", "train-b", "alternative-b", day=4)]
        self.train_path = self.write_jsonl("training-inputs.jsonl", self.train)
        train_hash = inference.file_hash(self.train_path)
        original = self.train[0]
        self.reference_rows = [{
            "sample_id": query["sample_id"], "reference_kind": "matched",
            "reference_history": original["history"],
            "provenance": {"split": "train", "training_inputs_sha256": train_hash,
                "donor_sample_id": "original", "donor_student_id": "train-original",
                "donor_current_timestamp": original["current_timestamp"],
                "query_current_timestamp": query["current_timestamp"],
                "lab": query["lab"], "source_file": query["source_file"], "labels_used": False},
        } for query in self.rows]
        # Noncanonical whitespace is deliberate: untouched payloads must retain
        # their exact original bytes, not only equal decoded records.
        self.inputs_path = self.write_jsonl("inputs.jsonl", self.rows, spaced=True)
        self.refs_path = self.write_jsonl("references.jsonl", self.reference_rows, spaced=True)
        self.source = self.root / "original-source"
        original_manifest = inference.make_manifest(self.config, self.inputs_path, self.refs_path,
                                                    inference.model_fingerprint(self.model))
        source_run = self.source / "inference"
        source_hash = inference.prepare_output(source_run, original_manifest, False)
        with redirect_stdout(io.StringIO()):
            inference.run_inference(self.rows, inference.load_references(self.refs_path, {r["sample_id"] for r in self.rows}),
                                    self.config, source_run, source_hash, SyntheticPoolBackend())
        original_completed = {
            "schema_version": "student-sim-cd.experiment.v1", "status": "success", "exit_code": 0,
            "configuration": asdict(self.config),
            "implementation_sha256": {"inference.py": original_manifest["implementation_sha256"],
                                      "scoring.py": original_manifest["scoring_sha256"]},
            "source_sha256": {"inputs": original_manifest["inputs_sha256"], "references": original_manifest["references_sha256"]},
            "stages": [{"status": "success", "exit_code": 0, "name": name} for name in (
                "inference", "predict", "evaluate-base", "evaluate-history_d0", "evaluate-cd", "evaluate-b", "evaluate-copy")],
        }
        self.seal(self.source, original_completed)
        self.baseline_run = self.root / "completed-baseline"
        self.baseline_prepared = self.baseline_run / "prepared"
        with redirect_stdout(io.StringIO()):
            cache_rescore.prepare(source_run, self.inputs_path, self.refs_path, self.baseline_prepared)
            result = cache_rescore.score(self.baseline_prepared, replace(self.config, fence_policy="unwrap-single"),
                                         ScoringOnlyBackend())
        self.assertEqual((result["samples"], result["candidates"], result["scores"]), (70, 318, 318))
        self.release = "a" * 64
        complete = {"status": "success", "exit_code": 0, "source_root": "/synthetic/releases/" + self.release,
                    "stages": [{"status": "success", "exit_code": 0, "name": str(i)} for i in range(11)]}
        self.seal(self.baseline_run, complete)
        self.settings = protocol_specification()
        self.settings.update({"training_inputs_sha256": train_hash,
                              "baseline_release_sha256": self.release,
                              "baseline_sha256": {name: inference.file_hash(self.baseline_run / name)
                                  for name in ("manifest.json", "SUCCESS", "prepared/inputs.jsonl", "prepared/references.jsonl",
                                               "prepared/inference/manifest.json", "prepared/inference/candidates.jsonl",
                                               "prepared/inference/scores.jsonl")}})
        self.protocol_path = self.root / "protocol.json"
        self.protocol_path.write_bytes(history_scoring.json_bytes(self.settings))
        self.args = SimpleNamespace(protocol=self.protocol_path, baseline_run=self.baseline_run,
                                    train_inputs=self.train_path, model_path=self.model,
                                    prepared=self.root / "interventions", device=self.config.device)

    def seal(self, directory, manifest):
        (directory / "manifest.json").write_bytes(history_scoring.json_bytes(manifest))
        (directory / "SUCCESS").write_text(inference.file_hash(directory / "manifest.json") + "\n", encoding="ascii")

    def write_jsonl(self, name, rows, spaced=False):
        path = self.root / name
        if spaced:
            path.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=False) + "\n" for r in rows), encoding="utf-8")
        else:
            path.write_bytes(history_interventions.jsonl_bytes(rows))
        return path

    def prepare(self, tokenizer=None):
        with patch.object(history_scoring.preflight, "load_local_tokenizer", return_value=tokenizer or SyntheticTokenizer()):
            return history_scoring.prepare(self.args)

    def test_real_preparation_preserves_untouched_original_bytes_and_complete_coverage(self):
        with patch.object(inference, "HFBackend", side_effect=AssertionError("Preparation loaded model weights")):
            manifest = self.prepare()
        self.assertFalse(manifest["model_weights_loaded"])
        self.assertFalse(manifest["generation_performed"])
        self.assertFalse(manifest["labels_read"])
        self.assertTrue(manifest["formal_token_matching"])
        self.assertFalse((self.args.prepared / ".incomplete").exists())
        self.assertEqual((self.args.prepared / "SUCCESS").read_text(),
                         inference.file_hash(self.args.prepared / "manifest.json") + "\n")
        expected_files = {
            f"{variant}/{seed}/{field}.jsonl"
            for variant in history_interventions.VARIANTS for seed in history_scoring.SEEDS
            for field in ("inputs", "references", "assignments")
        } | {"prompt-audit.json", "donor-manifest.json", "history-token-lengths.json"}
        self.assertEqual(set(manifest["files_sha256"]), expected_files)
        for seed in history_scoring.SEEDS:
            real = self.args.prepared / "real_history_swapped" / str(seed)
            ref = self.args.prepared / "reference_swapped" / str(seed)
            self.assertEqual((real / "references.jsonl").read_bytes(), self.refs_path.read_bytes())
            self.assertEqual((ref / "inputs.jsonl").read_bytes(), self.inputs_path.read_bytes())
            self.assertNotEqual((real / "inputs.jsonl").read_bytes(), self.inputs_path.read_bytes())
            self.assertNotEqual((ref / "references.jsonl").read_bytes(), self.refs_path.read_bytes())
            for directory in (real, ref):
                self.assertEqual(len(inference.read_jsonl(directory / "assignments.jsonl")), 70)
        history_scoring.load_prepared(self.args)

    def test_previous_or_incomplete_preparation_is_preserved(self):
        self.args.prepared.mkdir()
        sentinel = self.args.prepared / ".incomplete"
        sentinel.write_bytes(b"preserve the old partial site")
        with self.assertRaisesRegex(ValueError, "new canonical directory"):
            self.prepare()
        self.assertEqual(sentinel.read_bytes(), b"preserve the old partial site")
        self.assertFalse((self.args.prepared / "manifest.json").exists())

    def test_canonical_candidate_token_change_fails_before_any_output_is_created(self):
        class ChangedTokenizer(SyntheticTokenizer):
            def encode(self, text, **kwargs):
                return super().encode(text, **kwargs) + [12] if text.startswith("print(") else super().encode(text, **kwargs)
        with self.assertRaisesRegex(ValueError, "Fixed candidate tokenization changed"):
            self.prepare(ChangedTokenizer())
        self.assertFalse(self.args.prepared.exists())

    def test_train_or_baseline_hash_tampering_is_rejected_without_loading_model(self):
        self.train_path.write_bytes(self.train_path.read_bytes() + b" ")
        with patch.object(inference, "HFBackend", side_effect=AssertionError("Model loaded")), self.assertRaisesRegex(ValueError, "training inputs SHA"):
            self.prepare()
        self.train_path.write_bytes(history_interventions.jsonl_bytes(self.train))
        baseline_scores = self.baseline_prepared / "inference/scores.jsonl"
        baseline_scores.write_bytes(baseline_scores.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "Completed baseline changed"):
            self.prepare()

    def test_complete_fresh_scoring_never_generates_and_retains_all_candidate_tokens(self):
        self.prepare()
        backend = ScoringOnlyBackend()
        with patch.object(inference, "HFBackend", side_effect=AssertionError("Synthetic test loaded a real model")), redirect_stdout(io.StringIO()):
            result = history_scoring.score(self.args, backend=backend)
        self.assertFalse(result["generation_performed"])
        self.assertFalse(result["model_loaded_once"])
        self.assertEqual(len(result["variants"]), 6)
        self.assertEqual(backend.generations, [])
        self.assertEqual(len(backend.scored), 6 * 318 * 4)
        baseline_candidates = inference.read_jsonl(self.baseline_prepared / "inference/candidates.jsonl")
        expected = {r["sample_id"]: r["candidates"] for r in baseline_candidates}
        for variant in history_interventions.VARIANTS:
            for seed in history_scoring.SEEDS:
                directory = self.args.prepared / variant / str(seed)
                candidates = inference.read_jsonl(directory / "inference/candidates.jsonl")
                self.assertEqual({r["sample_id"]: r["candidates"] for r in candidates}, expected)
                self.assertEqual(len(inference.read_jsonl(directory / "inference/scores.jsonl")), 318)
                manifest = predict._json((directory / "inference/manifest.json").read_bytes())
                self.assertEqual(manifest["cache_rescore"]["backend"], "injected_synthetic_backend")
                self.assertFalse(manifest["cache_rescore"]["old_scores_reused"])
                self.assertEqual(manifest["history_intervention"]["variant"], variant)
                self.assertEqual(manifest["history_intervention"]["permutation_seed"], seed)
                for method in history_scoring.METHODS:
                    self.assertEqual(len(inference.read_jsonl(directory / "predictions" / f"predictions.{method}.jsonl")), 70)

    def test_any_previous_score_directory_even_empty_stops_before_model_loading(self):
        self.prepare()
        previous = self.args.prepared / "reference_swapped" / str(history_scoring.SEEDS[-1]) / "inference"
        previous.mkdir()
        with patch.object(inference, "HFBackend", side_effect=AssertionError("Model loaded before fresh-output check")), self.assertRaisesRegex(ValueError, "Preserve prior/partial"):
            history_scoring.score(self.args)
        self.assertEqual(list(previous.iterdir()), [])
        first = self.args.prepared / "real_history_swapped" / str(history_scoring.SEEDS[0]) / "inference"
        self.assertFalse(first.exists())

    def test_changed_backend_completion_or_eos_tokens_never_get_score_success(self):
        self.prepare()
        backend = ScoringOnlyBackend()
        backend.code_tokens = lambda code: list(code.encode("utf-8")) + [999]
        with self.assertRaisesRegex(ValueError, "Canonical candidate/EOS tokens changed"):
            history_scoring.score(self.args, backend=backend)
        for variant in history_interventions.VARIANTS:
            for seed in history_scoring.SEEDS:
                directory = self.args.prepared / variant / str(seed)
                self.assertFalse((directory / "SCORE_SUCCESS").exists())
                self.assertFalse((directory / "inference/scores.jsonl").exists())

    def test_missing_prepared_payload_or_incomplete_marker_is_rejected(self):
        self.prepare()
        (self.args.prepared / ".incomplete").write_bytes(b"not finished")
        with self.assertRaisesRegex(ValueError, "incomplete preparation"):
            history_scoring.load_prepared(self.args)
        (self.args.prepared / ".incomplete").unlink()
        assignment = self.args.prepared / "reference_swapped" / str(history_scoring.SEEDS[-1]) / "assignments.jsonl"
        assignment.write_bytes(assignment.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "Prepared payload changed"):
            history_scoring.load_prepared(self.args)

    def test_rehashed_non_history_query_change_still_fails_lawful_reconstruction(self):
        self.prepare()
        name = f"real_history_swapped/{history_scoring.SEEDS[0]}/inputs.jsonl"
        payload = self.args.prepared / name
        rows = inference.read_jsonl(payload)
        rows[0]["current_code"] = "not the original frozen current code"
        payload.write_bytes(history_interventions.jsonl_bytes(rows))
        manifest = predict._json((self.args.prepared / "manifest.json").read_bytes())
        manifest["files_sha256"][name] = inference.file_hash(payload)
        self.seal(self.args.prepared, manifest)
        with self.assertRaisesRegex(ValueError, "lawful full-training donor"):
            history_scoring.load_prepared(self.args)

    def test_resealed_forged_token_lengths_and_coherent_assignments_are_rejected(self):
        self.prepare()
        lengths_path = self.args.prepared / "history-token-lengths.json"
        forged_lengths = {key: value + 1 for key, value in predict._json(lengths_path.read_bytes()).items()}
        manifest = predict._json((self.args.prepared / "manifest.json").read_bytes())
        forged = history_interventions.build_interventions(
            self.rows, self.reference_rows, self.train,
            training_inputs_bytes=self.train_path.read_bytes(),
            training_inputs_sha256=self.settings["training_inputs_sha256"],
            protocol_sha256=inference.file_hash(self.protocol_path),
            token_lengths=forged_lengths, tokenizer_fingerprint_sha256=manifest["tokenizer"]["sha256"],
        )
        lengths_path.write_bytes(history_scoring.json_bytes(forged_lengths))
        (self.args.prepared / "donor-manifest.json").write_bytes(history_scoring.json_bytes(forged["manifest"]))
        for variant in history_interventions.VARIANTS:
            for seed in history_scoring.SEEDS:
                bundle = forged["variants"][variant][str(seed)]
                directory = self.args.prepared / variant / str(seed)
                for field in ("inputs", "references", "assignments"):
                    content = bundle[field + "_bytes"]
                    if (variant, field) in {("real_history_swapped", "references"), ("reference_swapped", "inputs")}:
                        content = (self.baseline_prepared / (field + ".jsonl")).read_bytes()
                    (directory / (field + ".jsonl")).write_bytes(content)
        for name in manifest["files_sha256"]:
            manifest["files_sha256"][name] = inference.file_hash(self.args.prepared / name)
        self.seal(self.args.prepared, manifest)
        with self.assertRaisesRegex(ValueError, "token"):
            history_scoring.load_prepared(self.args)


if __name__ == "__main__":
    unittest.main()
