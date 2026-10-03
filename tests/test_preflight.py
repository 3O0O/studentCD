"""Synthetic tokenizer tests; no Transformers install, GPU, labels, or weights."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from student_sim_cd.inference import CONDITIONS, canonical_json
from student_sim_cd.preflight import (
    PreflightConfig, load_local_tokenizer, main, measure, run_preflight,
    summarize, tokenizer_fingerprint,
)


def sample(sample_id="one", code="print(0)\n"):
    return {
        "schema_version": "student-sim-cd.progfeed.v1", "sample_id": sample_id,
        "student_id": "student-a", "lab": "lab03", "source_file": "answer.py",
        "current_timestamp": "2025-01-01-11-00-00", "current_code": code,
        "current_results": [], "problem_statement": "Print the requested integer.",
        "history": [{"timestamp": "2025-01-01-10-00-00", "code": "print()\n", "results": [], "feedback": []}],
        "feedback": [{"test_name": "f", "function_name": "f", "assigned_type": "nl", "text": "Check the number."}],
    }


class FakeTokenizer:
    eos_token_id = 256
    all_special_ids = [256, 257]
    chat_template = "synthetic-template-v1"

    def __init__(self):
        self.messages = []

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, truncation):
        assert tokenize and add_generation_prompt and truncation is False
        self.messages.append(messages)
        return list(canonical_json(messages).encode("utf-8")) + [257]

    def encode(self, text, *, add_special_tokens, truncation):
        assert add_special_tokens is False and truncation is False
        return [256] if text == "RESERVED" else list(text.encode("utf-8"))


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.model = self.root / "local-tokenizer"
        self.model.mkdir()
        (self.model / "tokenizer.json").write_text('{"synthetic":true}', encoding="utf-8")
        (self.model / "config.json").write_text('{"max_position_embeddings":32768}', encoding="utf-8")
        self.config = PreflightConfig(str(self.model))

    def records_file(self, name, rows):
        path = self.root / name
        path.write_text("".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8")
        return path

    def test_all_four_prompts_use_real_factorization_and_copy_includes_eos(self):
        row = sample()
        reference = {"sample_id": "one", "reference_history": [{"timestamp": "2025-01-01-09-00-00",
                     "code": "reference", "results": [], "feedback": []}],
                     "reference_kind": "matched", "provenance": {"split": "train"}}
        tokenizer = FakeTokenizer()
        records = measure([row], {"one": reference}, tokenizer)
        self.assertEqual(set(CONDITIONS), set(records[0]["prompt_tokens"]))
        self.assertEqual(len(row["current_code"].encode("utf-8")) + 1, records[0]["copy_code_plus_eos_tokens"])
        self.assertEqual(4, len(tokenizer.messages))
        for condition, messages in zip(CONDITIONS, tokenizer.messages):
            payload = json.loads(messages[1]["content"])
            self.assertEqual(payload["history"], row["history"] if condition[0] == "1" else reference["reference_history"])
            self.assertEqual(payload["current_feedback"], row["feedback"] if condition[1] == "1" else [])
            self.assertEqual(payload["current_code"], row["current_code"])

    def test_context_coverage_checks_copy_and_generation_without_combining_unrelated_maxima(self):
        records = []
        for sid, prompt, copy in (("long-prompt", 900, 10), ("long-code", 800, 900)):
            records.append({"sample_id": sid, "reference_kind": "matched", "max_prompt_tokens": prompt,
                            "prompt_tokens": {c: prompt for c in CONDITIONS},
                            "copy_code_plus_eos_tokens": copy, "copy_protocol_valid": True,
                            "prompt_plus_copy_tokens": {c: prompt + copy for c in CONDITIONS}})
        config = PreflightConfig(str(self.model), (1400, 1700), (100, 512))
        report = summarize(records, config, 1600)
        self.assertEqual(1700, report["minimum_context_for_all_copy_candidates"])
        self.assertEqual(1412, report["minimum_context_by_generation_budget"][1]["generation_only"])
        self.assertEqual(1700, report["minimum_context_by_generation_budget"][1]["generation_and_copy_candidate"])
        self.assertEqual(1, report["coverage"][0]["both_fit_all_branches"])
        self.assertEqual(["long-code"], report["coverage"][0]["overflow_sample_ids"])
        self.assertEqual(2, report["coverage"][-1]["both_fit_all_branches"])
        self.assertFalse(report["coverage"][-1]["within_declared_model_capacity"])

    def test_empty_code_and_reserved_tokens_are_reported_without_filtering(self):
        records = measure([sample("empty", ""), sample("reserved", "RESERVED")], {}, FakeTokenizer())
        self.assertEqual(2, len(records))
        self.assertEqual(1, records[0]["copy_code_plus_eos_tokens"])
        self.assertFalse(records[1]["copy_protocol_valid"])
        report = summarize(records, self.config, 32768)
        self.assertEqual(["reserved"], report["copy_protocol_invalid_sample_ids"])
        self.assertFalse(report["interpretation"]["samples_filtered"])

    def test_tokenizer_fingerprint_never_reads_weights_and_tracks_template(self):
        weight = self.model / "model.safetensors"
        weight.write_bytes(b"MUST_NOT_READ")
        before = tokenizer_fingerprint(self.model)
        self.assertNotIn(weight.name, before["files"])
        weight.write_bytes(b"DIFFERENT_WEIGHTS")
        self.assertEqual(before, tokenizer_fingerprint(self.model))
        (self.model / "chat_template.jinja").write_text("new template", encoding="utf-8")
        self.assertNotEqual(before["sha256"], tokenizer_fingerprint(self.model)["sha256"])

    def test_output_records_hashes_and_does_not_read_neighbor_label_file(self):
        inputs = self.records_file("inputs.jsonl", [sample()])
        label = self.root / "labels.test.jsonl"
        label.write_text("INVALID JSON; SECRET FUTURE", encoding="utf-8")
        output = self.root / "preflight"
        report = run_preflight(inputs, None, output, self.config, FakeTokenizer())
        self.assertEqual(1, report["samples"])
        self.assertEqual(12, len(report["coverage"]))
        self.assertEqual({"manifest.json", "report.json", "lengths.jsonl"}, {p.name for p in output.iterdir()})
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertIsNone(manifest["references_sha256"])
        self.assertTrue(manifest["loading"]["synthetic_tokenizer_injected"])
        self.assertIn("tokenizer.json", manifest["tokenizer_artifacts"]["files"])
        self.assertNotIn("SECRET FUTURE", "".join(p.read_text() for p in output.iterdir()))
        with self.assertRaisesRegex(ValueError, "only creates"):
            run_preflight(inputs, None, output, self.config, FakeTokenizer())

    def test_future_label_fields_and_incomplete_references_are_rejected(self):
        row = sample()
        row["target_code"] = "forbidden"
        with self.assertRaisesRegex(ValueError, "target_code"):
            run_preflight(self.records_file("bad.jsonl", [row]), None, self.root / "bad", self.config, FakeTokenizer())
        refs = self.records_file("empty-refs.jsonl", [])
        with self.assertRaisesRegex(ValueError, "missing"):
            run_preflight(self.records_file("good.jsonl", [sample()]), refs, self.root / "bad", self.config, FakeTokenizer())

    def test_local_loader_disables_network_and_remote_code(self):
        calls = []
        class AutoTokenizer:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                calls.append((args, kwargs))
                return FakeTokenizer()
        fake_module = types.SimpleNamespace(AutoTokenizer=AutoTokenizer)
        with patch.dict(sys.modules, {"transformers": fake_module}), patch.dict("os.environ", {}, clear=False):
            load_local_tokenizer(self.model)
            self.assertEqual({"local_files_only": True, "trust_remote_code": False}, calls[0][1])
        with self.assertRaisesRegex(ValueError, "local directory"):
            load_local_tokenizer(self.root / "not-a-hub-id")

    def test_cli_uses_tokenizer_only_and_reports_all_rows(self):
        inputs = self.records_file("inputs.jsonl", [sample(), sample("two", "")])
        with patch("student_sim_cd.preflight.load_local_tokenizer", return_value=FakeTokenizer()), redirect_stdout(io.StringIO()) as captured:
            status = main(["--inputs", str(inputs), "--model-path", str(self.model), "--output", str(self.root / "cli"),
                           "--context-limits", "8192", "16384", "--max-new-tokens", "512", "2048"])
        self.assertEqual(0, status)
        self.assertEqual(2, json.loads(captured.getvalue())["samples"])
        self.assertFalse(json.loads(captured.getvalue())["model_run"])

    def test_import_does_not_load_transformers_or_torch(self):
        result = subprocess.run([sys.executable, "-B", "-c",
                                 "import sys; import student_sim_cd.preflight; "
                                 "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"],
                                capture_output=True, text=True)
        self.assertEqual(0, result.returncode, result.stderr)

    def test_invalid_budget_and_missing_chat_template_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            PreflightConfig(str(self.model), (8192,), (0,))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            PreflightConfig(str(self.model), (8192, 8192), (100,))
        tokenizer = FakeTokenizer()
        tokenizer.chat_template = None
        with self.assertRaisesRegex(ValueError, "chat template"):
            measure([sample()], {}, tokenizer)


if __name__ == "__main__":
    unittest.main()
