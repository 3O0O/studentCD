"""Synthetic cache-rescore tests; no real model, GPU, downloads or code execution."""

from contextlib import redirect_stdout
from dataclasses import asdict, replace
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from student_sim_cd import cache_rescore, inference, predict
from tests.test_inference import FakeBackend, sample


class SourceBackend(FakeBackend):
    def __init__(self, raw=None, no_eos=()):
        super().__init__()
        self.raw = raw or ["```python\nprint(1)\n```\n", "print(1)\n",
                           "```py\nprint(0)\n```", "An explanation:\n```python\nprint(2)\n```\n"]
        self.no_eos = set(no_eos)

    def generate(self, prompt, sample_id, attempt):
        text, complete = self.raw[attempt], attempt not in self.no_eos
        return {"attempt": attempt, "source": "greedy" if attempt == 0 else "sampled",
                "seed": attempt, "raw_text": text,
                "generated_token_ids": self.code_tokens(text) if complete else self.code_tokens(text)[:-1],
                "eos_reached": complete, "finish_reason": "eos" if complete else "length_limit"}


class RescoreBackend(FakeBackend):
    def generate(self, *args, **kwargs):
        raise AssertionError("cache rescore attempted new generation")

    def sequence_logp(self, prompt, completion):
        return super().sequence_logp(prompt, completion) - 20.0


class CacheRescoreTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="cache-rescore-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.model = self.root / "synthetic-model"
        self.model.mkdir()
        (self.model / "config.json").write_text("{}\n", encoding="utf-8")
        (self.model / "model.safetensors").write_bytes(b"synthetic-not-loadable-model")
        self.config = inference.InferenceConfig(model_path=str(self.model), num_generations=4,
                                               max_context_tokens=4096, max_new_tokens=100)
        self.inputs = self.write_jsonl("inputs.jsonl", [sample()])
        self.references = self.write_jsonl("references.jsonl", [{
            "sample_id": "sample-a", "reference_history": sample()["history"],
            "reference_kind": "matched", "provenance": {"split": "train", "rule": "past-only"}}])
        self.source = self.root / "source-experiment"
        self.source_run = self.source / "inference"
        self.output = self.root / "new-rescore"
        self.build_source()

    def write_jsonl(self, name, records):
        path = self.root / name
        path.write_text("".join(inference.canonical_json(row) + "\n" for row in records), encoding="utf-8")
        return path

    def build_source(self, backend=None):
        if self.source_run.exists():
            for path in self.source_run.iterdir():
                path.unlink()
        manifest = inference.make_manifest(self.config, self.inputs, self.references,
                                           inference.model_fingerprint(self.model))
        run_hash = inference.prepare_output(self.source_run, manifest, False)
        refs = inference.load_references(self.references, {"sample-a"})
        with redirect_stdout(io.StringIO()):
            inference.run_inference([sample()], refs, self.config, self.source_run,
                                    run_hash, backend or SourceBackend())
        completed = {"schema_version": "student-sim-cd.experiment.v1", "status": "success", "exit_code": 0,
                     "configuration": asdict(self.config),
                     "implementation_sha256": {"inference.py": manifest["implementation_sha256"],
                                               "scoring.py": manifest["scoring_sha256"]},
                     "source_sha256": {"inputs": manifest["inputs_sha256"], "references": manifest["references_sha256"]},
                     "stages": [{"name": name, "status": "success", "exit_code": 0} for name in (
                         "inference", "predict", "evaluate-base", "evaluate-history_d0", "evaluate-cd", "evaluate-b", "evaluate-copy")]}
        (self.source / "manifest.json").write_text(inference.canonical_json(completed) + "\n", encoding="utf-8")
        (self.source / "SUCCESS").write_text(inference.file_hash(self.source / "manifest.json") + "\n", encoding="ascii")

    def prepare(self, **kwargs):
        return cache_rescore.prepare(self.source_run, self.inputs, self.references, self.output, **kwargs)

    def score(self, backend=None, **kwargs):
        with redirect_stdout(io.StringIO()):
            return cache_rescore.score(self.output, replace(self.config, fence_policy="unwrap-single"),
                                       backend or RescoreBackend(), **kwargs)

    def test_strict_whole_response_extraction_preserves_code_and_anomalies(self):
        for raw, expected in (("```python\n  x = 1  \n\n```\n", "  x = 1  \n\n"),
                              ("```py\r\nx = 1\r\n```\r\n", "x = 1\r\n")):
            self.assertEqual(cache_rescore.extract_code(raw)[0], expected)
        for raw in ("", "\n invalid python (  \n", "```\nx = 1\n```\n",
                    "```javascript\nx = 1\n```", " ```python\nx = 1\n```",
                    "```python\nx = 1\n```\nextra", "```python\nx=1\n```\n```py\nx=2\n```"):
            with self.subTest(raw=raw):
                self.assertEqual(cache_rescore.extract_code(raw)[:2], (raw, "none"))
                self.assertTrue(cache_rescore.extract_code(raw)[2])

    def test_preparation_preserves_raw_attempts_and_all_source_mappings(self):
        source_bytes = {path.name: path.read_bytes() for path in self.source_run.iterdir()}
        result = self.prepare()
        self.assertEqual((result["samples"], result["original_candidates"], result["candidates"], result["attempts"]),
                         (1, 5, 3, 4))
        record = inference.read_jsonl(self.output / "prepared.jsonl")[0]
        original = inference.read_jsonl(self.source_run / "candidates.jsonl")[0]
        self.assertEqual(record["attempts"], original["attempts"])
        self.assertEqual(len(record["mappings"]), 5)
        self.assertEqual(record["mappings"][1]["candidate_id"], record["mappings"][2]["candidate_id"])
        self.assertEqual(record["mappings"][0]["candidate_id"], record["mappings"][3]["candidate_id"])
        self.assertEqual(record["candidates"][0]["sources"], ["copy_current", "sampled:2"])
        self.assertEqual(source_bytes, {path.name: path.read_bytes() for path in self.source_run.iterdir()})
        self.assertFalse((self.output / "inference").exists())
        self.assertFalse(result["generation_performed"])

    def test_four_conditions_are_fresh_complete_and_predict_compatible_without_generation(self):
        self.prepare()
        backend = RescoreBackend()
        result = self.score(backend)
        self.assertEqual((result["samples"], result["candidates"], result["scores"]), (1, 3, 3))
        self.assertEqual(len(backend.scored), 12)
        self.assertEqual(backend.generations, [])
        self.assertTrue(all(completion[-1] == 256 for _, completion in backend.scored))
        new_scores = inference.read_jsonl(self.output / "inference/scores.jsonl")
        original_scores = {row["candidate_id"]: row for row in inference.read_jsonl(self.source_run / "scores.jsonl")}
        for row in new_scores:
            self.assertTrue(all(field in row for field in ("l11", "l01", "l10", "l00")))
            if row["candidate_id"] in original_scores:
                self.assertNotEqual(row["l11"], original_scores[row["candidate_id"]]["l11"])
        report = predict.predict(self.output / "inference", self.output / "inputs.jsonl", self.root / "predictions", True)
        self.assertEqual(report["samples"], 1)
        export = cache_rescore.export_greedy(self.output, self.root / "greedy.jsonl")
        self.assertEqual(export["samples"], 1)
        self.assertEqual(inference.read_jsonl(self.root / "greedy.jsonl"), [
            {"sample_id": "sample-a", "method": "greedy", "predicted_code": "print(1)\n"}])
        with self.assertRaisesRegex(ValueError, "already exists"):
            cache_rescore.export_greedy(self.output, self.root / "greedy.jsonl")

    def test_resume_reuses_only_verified_new_scores_and_rejects_fingerprint_changes(self):
        first = self.prepare()
        self.assertEqual(self.prepare(resume=True), first)
        self.score()
        backend = RescoreBackend()
        self.score(backend, resume=True)
        self.assertEqual(backend.scored, [])
        with self.assertRaisesRegex(ValueError, "already has"):
            self.prepare()
        with self.assertRaisesRegex(ValueError, "preserve"):
            cache_rescore.score(self.output, replace(self.config, fence_policy="unwrap-single", temperature=0.5),
                                RescoreBackend(), resume=True)
        (self.model / "model.safetensors").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "assets differ"):
            self.score(resume=True)

    def test_success_and_all_seven_stages_are_required_before_preparation(self):
        (self.source / "SUCCESS").write_text("0" * 64 + "\n", encoding="ascii")
        with self.assertRaisesRegex(ValueError, "SUCCESS"):
            self.prepare()
        self.assertFalse(self.output.exists())
        self.build_source()
        completed = predict._json((self.source / "manifest.json").read_bytes())
        completed["stages"][0]["status"] = "running"
        (self.source / "manifest.json").write_text(inference.canonical_json(completed), encoding="utf-8")
        (self.source / "SUCCESS").write_text(inference.file_hash(self.source / "manifest.json"), encoding="ascii")
        with self.assertRaisesRegex(ValueError, "seven"):
            self.prepare()

    def test_complete_original_scores_and_reference_hash_are_required(self):
        path = self.source_run / "scores.jsonl"
        rows = inference.read_jsonl(path)
        path.write_text("".join(inference.canonical_json(row) + "\n" for row in rows[:-1]), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing scores"):
            self.prepare()
        self.build_source()
        self.references.write_text(self.references.read_text() + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "reference bytes"):
            self.prepare()

    def test_source_attempt_relation_is_checked_beyond_the_existing_cache_validator(self):
        path = self.source_run / "candidates.jsonl"
        rows = inference.read_jsonl(path)
        rows[0]["attempts"][0]["raw_text"] = "unrelated code"
        rows[0].pop("record_sha256")
        rows[0]["record_sha256"] = inference.object_hash(rows[0])
        path.write_text(inference.canonical_json(rows[0]) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "raw attempt"):
            self.prepare()

    def test_empty_raw_code_is_kept_and_scored_as_exactly_one_eos(self):
        self.build_source(SourceBackend(raw=["", "", "", ""]))
        self.prepare()
        record = inference.read_jsonl(self.output / "prepared.jsonl")[0]
        self.assertIn("", [candidate["code"] for candidate in record["candidates"]])
        backend = RescoreBackend()
        self.score(backend)
        self.assertEqual(sum(completion == [256] for _, completion in backend.scored), 4)

    def test_truncated_greedy_is_preserved_but_never_silently_dropped_from_export(self):
        self.build_source(SourceBackend(no_eos=(0,)))
        self.prepare()
        self.score()
        with self.assertRaisesRegex(ValueError, "greedy attempt"):
            cache_rescore.export_greedy(self.output, self.root / "greedy.jsonl")
        self.assertFalse((self.root / "greedy.jsonl").exists())

    def test_reserved_token_and_nonfinite_score_fail_without_success(self):
        self.prepare()
        backend = RescoreBackend()
        backend.code_tokens = lambda code: (_ for _ in ()).throw(ValueError("reserved special token inside code"))
        with self.assertRaisesRegex(ValueError, "reserved special token"):
            self.score(backend)
        self.assertFalse((self.output / "inference/scores.jsonl").exists())
        self.assertFalse((self.output / "inference/SUCCESS").exists())
        nonfinite = RescoreBackend()
        nonfinite.sequence_logp = lambda prompt, completion: float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            self.score(nonfinite, resume=True)
        self.assertFalse((self.output / "inference/scores.jsonl").exists())

    def test_tampered_preparation_and_new_score_cache_are_rejected(self):
        self.prepare()
        path = self.output / "prepared.jsonl"
        rows = inference.read_jsonl(path)
        original = path.read_bytes()
        rows[0]["candidates"][0]["code"] = "tampered"
        path.write_text(inference.canonical_json(rows[0]) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "content hash"):
            self.score()
        path.write_bytes(original)
        self.score()
        path = self.output / "inference/scores.jsonl"
        scores = inference.read_jsonl(path)
        scores[0]["l11"] = -99999
        path.write_text("".join(inference.canonical_json(row) + "\n" for row in scores), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "record hash mismatch"):
            self.score(resume=True)

    def test_labels_are_not_read_and_import_requires_no_model_libraries(self):
        (self.root / "labels.jsonl").write_text("invalid future label bytes NEVER_READ", encoding="utf-8")
        self.prepare()
        backend = RescoreBackend()
        self.score(backend)
        self.assertNotIn("NEVER_READ", inference.canonical_json(backend.messages))
        result = subprocess.run([sys.executable, "-B", "-c",
                                 "import sys; import student_sim_cd.cache_rescore; "
                                 "assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_original_output_and_symlink_destinations_are_never_overwritten(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            cache_rescore.prepare(self.source_run, self.inputs, self.references, self.source / "new")
        link = self.root / "linked-source"
        link.symlink_to(self.source)
        with self.assertRaisesRegex(ValueError, "symlink"):
            cache_rescore.prepare(self.source_run, self.inputs, self.references, link / "new")


if __name__ == "__main__":
    unittest.main()
