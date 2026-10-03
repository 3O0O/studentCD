"""Protocol tests use synthetic text and a fake backend; no GPU or downloads."""

from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from student_sim_cd.inference import (
    CONDITIONS, InferenceConfig, build_messages, canonical_json, ensure_context,
    generate_candidates, load_inputs, load_references, main, make_manifest,
    model_fingerprint, prepare_output, read_jsonl, run_inference, transform_code,
    validate_input,
)


def sample():
    return {
        "schema_version": "student-sim-cd.progfeed.v1",
        "sample_id": "sample-a", "student_id": "student-a", "lab": "lab1",
        "source_file": "answer.py", "current_timestamp": "2020-01-01T10:02:00",
        "problem_statement": "Print one integer.", "current_code": "print(0)\n",
        "history": [{"timestamp": "2020-01-01T10:00:00", "code": "print()\n",
                     "results": [], "feedback": []}],
        "feedback": [{"test_name": "test_print", "function_name": "answer",
                      "assigned_type": "hint", "text": "Check the required integer."}],
        "current_results": [{"test_name": "test_print", "function_name": "answer",
                             "status": "failed", "score": 0, "max_score": 1,
                             "testcase_mask": "0"}],
    }


class FakeBackend:
    """Code tokens are UTF-8 bytes + an explicit terminal marker."""

    def __init__(self):
        self.generations = []
        self.scored = []
        self.messages = []

    def prompt_tokens(self, messages):
        self.messages.append(messages)
        return list(messages[1]["content"].encode("utf-8"))

    def code_tokens(self, code):
        return list(code.encode("utf-8")) + [256]

    def generate(self, prompt, sample_id, attempt):
        self.generations.append((sample_id, attempt))
        text = "print(1)\n" if attempt < 2 else "print(2"
        complete = attempt < 2
        return {"attempt": attempt, "source": "greedy" if attempt == 0 else "sampled",
                "seed": attempt, "raw_text": text, "generated_token_ids": self.code_tokens(text),
                "eos_reached": complete, "finish_reason": "eos" if complete else "length_limit"}

    def sequence_logp(self, prompt, completion):
        self.scored.append((prompt, completion))
        return -float(len(completion)) - len(prompt) / 1000


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = InferenceConfig(model_path=str(self.root / "model"), num_generations=3,
                                      max_context_tokens=4096, max_new_tokens=100)

    def write_jsonl(self, name, rows):
        path = self.root / name
        path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")
        return path

    def test_real_input_requires_problem_and_rejects_labels(self):
        row = sample()
        validate_input(row)
        row["problem_statement"] = None
        with self.assertRaisesRegex(ValueError, "missing problem_statement"):
            validate_input(row)
        validate_input(row, require_problem=False)
        row["target_code"] = "NEVER SEEN"
        with self.assertRaisesRegex(ValueError, "target_code"):
            validate_input(row, require_problem=False)

    def test_nested_future_label_rejected(self):
        row = sample()
        row["history"][0]["target_results"] = []
        with self.assertRaisesRegex(ValueError, "target_results"):
            validate_input(row)
        row = sample()
        row["current_results"][0]["target_code"] = "secret"
        with self.assertRaisesRegex(ValueError, "target_code"):
            validate_input(row)

    def test_empty_submission_is_kept_as_eos_only_candidate(self):
        row = sample()
        row["current_code"] = ""
        validate_input(row)
        record = generate_candidates(row, {key: [1] for key in CONDITIONS}, FakeBackend(), self.config)
        candidate = record["candidates"][0]
        self.assertEqual(candidate["code"], "")
        self.assertEqual(candidate["completion_token_ids"], [256])
        self.assertEqual(candidate["token_count"], 1)

    def test_history_times_cannot_reach_current_or_future(self):
        row = sample()
        row["current_timestamp"] = "2020-01-01-10-02-00"
        row["history"][0]["timestamp"] = "2020-01-01-10-00-00"
        validate_input(row)
        for timestamp in ("2020-01-01-10-02-00", "2020-01-01-11-00-00"):
            row["history"][0]["timestamp"] = timestamp
            with self.assertRaisesRegex(ValueError, "precede current_timestamp"):
                validate_input(row)

    def test_duplicate_samples_and_nonfinite_json_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
            load_inputs(self.write_jsonl("duplicates.jsonl", [sample(), sample()]))
        path = self.root / "nan.jsonl"
        path.write_text('{"sample_id":"a","value":NaN}\n', encoding="utf-8")
        with self.assertRaises(ValueError):
            read_jsonl(path)

    def test_prompt_factorization_preserves_basic_results(self):
        row = sample()
        reference = [{"timestamp": "earlier", "code": "x = 4\n", "results": [], "feedback": []}]
        payloads = {condition: json.loads(build_messages(row, condition, reference)[1]["content"])
                    for condition in CONDITIONS}
        for condition, payload in payloads.items():
            self.assertEqual(payload["current_results"], row["current_results"])
            self.assertEqual(payload["current_code"], row["current_code"])
            self.assertEqual(payload["history"], row["history"] if condition[0] == "1" else reference)
            self.assertEqual(payload["current_feedback"], row["feedback"] if condition[1] == "1" else [])
            self.assertNotIn("student_id", payload)

    def test_reference_requires_complete_explicit_provenance(self):
        self.assertEqual(load_references(None, {"a"}), {})
        reference = {"sample_id": "sample-a", "reference_history": sample()["history"],
                     "reference_kind": "matched", "provenance": {"split": "train", "rule": "past-only"}}
        path = self.write_jsonl("references.jsonl", [reference])
        self.assertEqual(load_references(path, {"sample-a"})["sample-a"], reference)
        with self.assertRaisesRegex(ValueError, "missing"):
            load_references(path, {"sample-a", "sample-b"})
        reference["reference_history"] = []
        with self.assertRaisesRegex(ValueError, "nonempty history"):
            load_references(self.write_jsonl("empty.jsonl", [reference]), {"sample-a"})

    def test_code_transform_is_explicit_and_preserves_whitespace(self):
        raw = "```python\n  x = 1\n\n```\n"
        self.assertEqual(transform_code(raw, "preserve"), (raw, "none"))
        self.assertEqual(transform_code(raw, "unwrap-single"),
                         ("  x = 1\n\n", "unwrapped_single_outer_fence"))
        malformed = "Here is code:\n```python\nx = 1\n```\n"
        self.assertEqual(transform_code(malformed, "unwrap-single"), (malformed, "none"))
        code = "\n  x = 1  \n\n"
        self.assertEqual(transform_code(code, "unwrap-single"), (code, "none"))

    def test_context_overflow_is_an_error(self):
        ensure_context(4, 6, 10)
        with self.assertRaisesRegex(ValueError, "no truncation"):
            ensure_context(4, 7, 10)

    def test_generation_deduplicates_and_records_truncation(self):
        backend = FakeBackend()
        record = generate_candidates(sample(), {key: [1] for key in CONDITIONS}, backend, self.config)
        self.assertEqual(len(record["candidates"]), 2)
        self.assertEqual(record["candidates"][1]["sources"], ["greedy:0", "sampled:1"])
        self.assertIsNone(record["attempts"][2]["candidate_id"])
        self.assertIn("truncated", record["attempts"][2]["exclusion_reason"])
        for candidate in record["candidates"]:
            self.assertEqual(candidate["completion_token_ids"][-1], 256)
            self.assertEqual(candidate["token_count"], len(candidate["completion_token_ids"]))

    def test_resume_checks_input_model_and_config_content(self):
        inputs = self.write_jsonl("inputs.jsonl", [sample()])
        model = self.root / "model"
        model.mkdir()
        (model / "config.json").write_text('{}', encoding="utf-8")
        (model / "model.safetensors").write_bytes(b"synthetic-not-a-real-model")
        fingerprint = model_fingerprint(model)
        manifest = make_manifest(self.config, inputs, None, fingerprint)
        output = self.root / "output"
        run_hash = prepare_output(output, manifest, resume=False)
        self.assertEqual(prepare_output(output, manifest, resume=True), run_hash)
        with self.assertRaisesRegex(ValueError, "already has"):
            prepare_output(output, manifest, resume=False)
        for field in ("config_sha256", "inputs_sha256", "model"):
            changed = dict(manifest)
            changed[field] = "different"
            with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                prepare_output(output, changed, resume=True)
        (model / "model.safetensors").write_bytes(b"changed-weight-bytes")
        self.assertNotEqual(model_fingerprint(model)["sha256"], fingerprint["sha256"])

    def test_cached_run_does_not_regenerate_or_rescore_and_labels_stay_separate(self):
        self.write_jsonl("labels.test.jsonl", [{"target_code": "SECRET_REAL_NEXT_CODE"}])
        output = self.root / "output"
        output.mkdir()
        backend = FakeBackend()
        with redirect_stdout(io.StringIO()):
            result = run_inference([sample()], {}, self.config, output, "run-hash", backend)
        self.assertEqual(result, {"samples": 1, "candidates": 2, "scores": 2})
        self.assertEqual(len(backend.scored), 8)
        scores = read_jsonl(output / "scores.jsonl")
        self.assertEqual(scores[0]["reference_kind"], "empty_reference_diagnostic")
        self.assertTrue(all(name in scores[0] for name in ("l11", "l01", "l10", "l00")))
        self.assertNotIn("SECRET_REAL_NEXT_CODE", canonical_json(backend.messages))
        second = FakeBackend()
        with redirect_stdout(io.StringIO()):
            run_inference([sample()], {}, self.config, output, "run-hash", second)
        self.assertEqual(second.generations, [])
        self.assertEqual(second.scored, [])
        for _, completion in backend.scored:
            self.assertEqual(completion[-1], 256)

    def test_resume_rejects_tampered_or_wrong_run_cache(self):
        output = self.root / "output"
        output.mkdir()
        with redirect_stdout(io.StringIO()):
            run_inference([sample()], {}, self.config, output, "right", FakeBackend())
        with self.assertRaisesRegex(ValueError, "wrong run hash"):
            run_inference([sample()], {}, self.config, output, "wrong", FakeBackend())
        records = read_jsonl(output / "scores.jsonl")
        records[0]["l11"] = -999
        (output / "scores.jsonl").write_text("".join(canonical_json(row) + "\n" for row in records), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "record hash mismatch"):
            run_inference([sample()], {}, self.config, output, "right", FakeBackend())

    def test_cli_validation_is_not_a_model_result(self):
        row = sample()
        row["problem_statement"] = None
        path = self.write_jsonl("inputs.jsonl", [row])
        captured = io.StringIO()
        with redirect_stdout(captured):
            self.assertEqual(main(["--inputs", str(path), "--validate-only", "--allow-missing-problem"]), 0)
        result = json.loads(captured.getvalue())
        self.assertFalse(result["model_run"])
        self.assertEqual(result["missing_problems"], 1)

    def test_import_has_no_torch_or_transformers_dependency(self):
        result = subprocess.run([sys.executable, "-B", "-c",
                                 "import sys; import student_sim_cd.inference; "
                                 "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"],
                                check=False, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_invalid_generation_configuration(self):
        for changes in ({"temperature": float("nan")}, {"top_p": 2}, {"num_generations": 0},
                        {"max_new_tokens": 10000}):
            with self.assertRaises(ValueError):
                replace(self.config, **changes)


if __name__ == "__main__":
    unittest.main()
