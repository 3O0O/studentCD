"""Prediction selection against synthetic inference caches; no model or labels."""

from contextlib import redirect_stdout
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from student_sim_cd import inference, predict, scoring


CODES = ("current\n", "history\n", "ordinary_cd\n", "interaction\n")
RAW = ((-1, -1, -1, -1), (-5, -5, -1, -11), (-4, -14, -1, -4), (-5, -13, -5, -1))


class SyntheticBackend:
    def prompt_tokens(self, messages):
        payload = json.loads(messages[1]["content"])
        condition = ("1" if payload["history"] else "0") + ("1" if payload["current_feedback"] else "0")
        return [inference.CONDITIONS.index(condition)]

    def code_tokens(self, code):
        return list(code.encode()) + [256]

    def generate(self, prompt, sample_id, attempt):
        return {"attempt": attempt, "source": "greedy" if attempt == 0 else "sampled",
                "seed": attempt, "raw_text": CODES[attempt + 1],
                "generated_token_ids": self.code_tokens(CODES[attempt + 1]),
                "eos_reached": True, "finish_reason": "eos"}

    def sequence_logp(self, prompt, completion):
        code = bytes(completion[:-1]).decode()
        return RAW[CODES.index(code)][prompt[0]]


class PredictionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="prediction-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.run_dir, self.output = self.root / "inference", self.root / "predictions"
        self.inputs = self.root / "inputs.test.jsonl"
        self.row = {
            "schema_version": "student-sim-cd.progfeed.v1", "sample_id": "sample-a",
            "student_id": "student-a", "lab": "lab1", "source_file": "answer.py",
            "current_timestamp": "2020-01-01T10:02:00", "problem_statement": "Synthetic task.",
            "current_code": CODES[0], "current_results": [],
            "history": [{"timestamp": "2020-01-01T10:00:00", "code": "older", "results": [], "feedback": []}],
            "feedback": [{"test_name": "synthetic", "function_name": "answer",
                          "assigned_type": "hint", "text": "Synthetic feedback."}],
        }
        self.rewrite(self.inputs, [self.row])
        config = inference.InferenceConfig(model_path="/unused/synthetic-model", num_generations=3,
                                           max_context_tokens=4096, max_new_tokens=64)
        manifest = inference.make_manifest(config, self.inputs, None, {"sha256": "0" * 64, "files": {}})
        self.run_hash = inference.prepare_output(self.run_dir, manifest, resume=False)
        with redirect_stdout(io.StringIO()):
            inference.run_inference([self.row], {}, config, self.run_dir, self.run_hash, SyntheticBackend())
        self.candidates, self.scores = self.run_dir / "candidates.jsonl", self.run_dir / "scores.jsonl"

    def rewrite(self, path, records, rehash=False):
        for row in records:
            if rehash:
                row["record_sha256"] = inference.object_hash({k: v for k, v in row.items() if k != "record_sha256"})
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")

    def select(self, include_copy=False):
        return predict.predict(self.run_dir, self.inputs, self.output, include_copy=include_copy)

    def test_actual_inference_schema_selects_four_different_fixed_weight_winners(self):
        # This unreferenced, invalid label file must never be opened.
        (self.root / "labels.test.jsonl").write_text("not JSON; SECRET NEXT SUBMISSION")
        report = self.select(include_copy=True)
        for method, expected_code in zip(("base", "history_d0", "cd", "b"), CODES):
            rows = inference.read_jsonl(self.output / ("predictions." + method + ".jsonl"))
            self.assertEqual(rows, [{"sample_id": "sample-a", "method": method, "predicted_code": expected_code}])
        self.assertEqual(inference.read_jsonl(self.output / "predictions.copy.jsonl")[0]["predicted_code"], CODES[0])
        self.assertEqual(report["method_parameters"], predict.METHOD_PARAMETERS)
        self.assertEqual(report["inference_run_sha256"], self.run_hash)
        self.assertEqual(report["source_sha256"]["inputs"], inference.file_hash(self.inputs))
        self.assertEqual(report["source_sha256"]["scores"], inference.file_hash(self.scores))
        for name, digest in report["output_sha256"].items():
            self.assertEqual(inference.file_hash(self.output / name), digest)
        self.assertFalse(report["labels_read"])
        self.assertFalse((self.output / ".incomplete").exists())

    def test_argmax_ties_use_candidate_id_not_cache_order(self):
        scores = inference.read_jsonl(self.scores)
        raw = scoring.LogProbs(-1, -1, -1, -1)
        for row in scores:
            row.update(asdict(raw), components=scoring.components(raw), scores_weight_1=scoring.method_scores(raw))
        self.rewrite(self.scores, list(reversed(scores)), rehash=True)
        candidates = inference.read_jsonl(self.candidates)
        candidates[0]["candidates"].reverse()
        self.rewrite(self.candidates, candidates, rehash=True)
        self.select()
        expected = min(CODES, key=lambda code: hashlib.sha256(code.encode()).hexdigest())
        for method in predict.METHOD_PARAMETERS:
            self.assertEqual(inference.read_jsonl(self.output / ("predictions." + method + ".jsonl"))[0]["predicted_code"], expected)

    def test_incomplete_score_pool_is_rejected_before_creating_output(self):
        self.rewrite(self.scores, inference.read_jsonl(self.scores)[:-1])
        with self.assertRaisesRegex(ValueError, "missing scores"):
            self.select()
        self.assertFalse(self.output.exists())

    def test_missing_sample_or_empty_candidate_pool_is_rejected(self):
        original = inference.read_jsonl(self.candidates)
        for rows in ([], [dict(original[0], candidates=[])]):
            self.rewrite(self.candidates, rows, rehash=True)
            with self.assertRaises(ValueError):
                self.select()
        self.assertFalse(self.output.exists())

    def test_duplicate_candidate_sample_and_score_ids_are_rejected(self):
        original_candidates, original_scores = inference.read_jsonl(self.candidates), inference.read_jsonl(self.scores)
        self.rewrite(self.candidates, original_candidates * 2)
        with self.assertRaisesRegex(ValueError, "duplicate candidate sample ID"):
            self.select()
        duplicate_pool = json.loads(json.dumps(original_candidates))
        duplicate_pool[0]["candidates"].append(duplicate_pool[0]["candidates"][0])
        self.rewrite(self.candidates, duplicate_pool, rehash=True)
        with self.assertRaisesRegex(ValueError, "duplicate candidate ID"):
            self.select()
        self.rewrite(self.candidates, original_candidates)
        self.rewrite(self.scores, original_scores + original_scores[:1])
        with self.assertRaisesRegex(ValueError, "duplicate score"):
            self.select()

    def test_code_content_and_record_hashes_are_independently_verified(self):
        rows = inference.read_jsonl(self.candidates)
        rows[0]["candidates"][0]["code"] = "edited cache text"
        self.rewrite(self.candidates, rows)
        with self.assertRaisesRegex(ValueError, "content hash mismatch"):
            self.select()
        self.rewrite(self.candidates, rows, rehash=True)
        with self.assertRaisesRegex(ValueError, "code content hash"):
            self.select()

    def test_manifest_content_and_input_file_hashes_are_verified(self):
        manifest_path = self.run_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["runtime_versions"]["synthetic-extra"] = "changed"
        manifest_path.write_text(inference.canonical_json(manifest))
        with self.assertRaisesRegex(ValueError, "run hash mismatch"):
            self.select()
        self.inputs.write_text(self.inputs.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "input file hash"):
            self.select()

    def test_nonfinite_raw_scores_and_derived_overflow_are_rejected(self):
        original = inference.read_jsonl(self.scores)
        for value in (float("nan"), float("inf"), float("-inf")):
            rows = json.loads(json.dumps(original))
            rows[0]["l11"] = value
            self.rewrite(self.scores, rows)
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.select()
        rows = json.loads(json.dumps(original))
        rows[0].update(l11=-1, l01=-1e308, l10=-1e308, l00=-1)
        self.rewrite(self.scores, rows, rehash=True)
        with self.assertRaisesRegex(ValueError, "finite"):
            self.select()

    def test_stored_diagnostic_scores_do_not_override_raw_scores(self):
        rows = inference.read_jsonl(self.scores)
        rows[0]["scores_weight_1"]["b"] = 123456
        self.rewrite(self.scores, rows, rehash=True)
        with self.assertRaisesRegex(ValueError, "stored method scores"):
            self.select()

    def test_unknown_scores_and_protocol_mismatches_are_rejected(self):
        rows = inference.read_jsonl(self.scores)
        for field, value in (("candidate_id", "0" * 64), ("token_count", 1234), ("eos_included", False)):
            changed = json.loads(json.dumps(rows))
            changed[0][field] = value
            self.rewrite(self.scores, changed, rehash=True)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.select()

    def test_copy_only_pool_is_reported_as_diagnostic(self):
        rows = inference.read_jsonl(self.candidates)
        rows[0]["candidates"] = rows[0]["candidates"][:1]
        identifier = rows[0]["candidates"][0]["candidate_id"]
        self.rewrite(self.candidates, rows, rehash=True)
        self.rewrite(self.scores, [row for row in inference.read_jsonl(self.scores) if row["candidate_id"] == identifier])
        report = self.select()
        self.assertEqual(report["copy_only_samples"], 1)
        self.assertEqual(report["reference_kinds"], {"empty_reference_diagnostic": 1})

    def test_no_overwrite_even_for_existing_empty_directory(self):
        self.output.mkdir()
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.select()
        self.assertEqual(list(self.output.iterdir()), [])

    def test_cli_outputs_evaluator_records_without_copy_unless_requested(self):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(predict.main(["--run-dir", str(self.run_dir), "--inputs", str(self.inputs),
                                           "--output", str(self.output)]), 0)
        self.assertFalse((self.output / "predictions.copy.jsonl").exists())
        self.assertEqual(len(inference.read_jsonl(self.output / "selections.jsonl")), 1)

    def test_jsonl_preserves_unicode_line_separators_inside_code(self):
        row = {"code": "# text\u0085comment\u2028preserved\u2029\n"}
        encoded = (inference.canonical_json(row) + "\n").encode()
        self.assertEqual(predict._records(encoded, "synthetic"), [row])

    def test_json_duplicate_fields_and_extra_label_fields_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate JSON field"):
            predict._records(b'{"sample_id":"a","sample_id":"b"}\n', "synthetic")
        rows = inference.read_jsonl(self.candidates)
        rows[0]["target_code"] = "not accepted, never used for selection"
        self.rewrite(self.candidates, rows, rehash=True)
        with self.assertRaisesRegex(ValueError, "record fields"):
            self.select()


if __name__ == "__main__":
    unittest.main()
