"""Real standard-library replay of a synthetic complete 70/17 CPU pipeline."""

import ast
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
import importlib.util
import io
import json
from pathlib import Path
import shutil
import unittest

from student_sim_cd import candidate_support, candidate_support_analysis, inference, prompt_baselines
from tests import test_candidate_support as support_fixtures

Backend = support_fixtures.Backend
Tokenizer = support_fixtures.Tokenizer


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/verify_candidate_support_results.py"
specification = importlib.util.spec_from_file_location("independent_candidate_support_verifier", SCRIPT)
verifier = importlib.util.module_from_spec(specification)
specification.loader.exec_module(verifier)


class VerifyCandidateSupportResultsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = support_fixtures.CandidateSupportTests("test_prepare_checks_complete_cohort_without_model_or_labels")
        cls.fixture.setUp()
        cls.addClassCleanup(cls.fixture.doCleanups)
        cls.template = cls.fixture.root / "completed-support"
        cls.template.mkdir()
        cls.fixture.prepared = cls.template / "prepared"
        cls.labels = cls.fixture.root / "labels.dev.jsonl"
        cls.labels.write_bytes(b"".join(candidate_support.encoded({"sample_id": row["sample_id"],
            "target_code": "print(0)\n" if index % 7 == 0 else "print(1)\n"}) for index, row in enumerate(cls.fixture.rows)))
        protocol = json.loads((ROOT / "configs/candidate_support_v1.json").read_text())
        protocol.update(cls.fixture.specification, generation_config=asdict(cls.fixture.config),
                        labels_sha256=inference.file_hash(cls.labels))
        cls.fixture.specification = protocol
        with redirect_stdout(io.StringIO()):
            cls.fixture.prepare()
            candidate_support.run(cls.fixture.prepared, cls.fixture.config, backend=Backend(cls.fixture.config))
        references = inference.load_references(cls.fixture.prepared / "references.jsonl", {row["sample_id"] for row in cls.fixture.rows})
        main_manifest = json.loads((cls.fixture.prepared / "pools/balanced12/inference/manifest.json").read_text())
        protocol_sha = inference.file_hash(cls.fixture.prepared / "protocol.json")
        with redirect_stdout(io.StringIO()):
            strong = prompt_baselines.run(cls.fixture.rows, references, cls.fixture.config, Backend(cls.fixture.config),
                cls.template / "prompt-baselines", protocol_sha256=protocol_sha, model_identity=main_manifest["model"])
        (cls.template / "strong-prompt-audit.json").write_bytes(candidate_support.encoded(
            prompt_baselines.audit(cls.fixture.rows, references, Tokenizer(), cls.fixture.config)))
        external = []
        for method in candidate_support_analysis.EXTERNAL_METHODS:
            path = cls.template / "prompt-baselines" / (method + ".jsonl")
            external.append({"method": method, "predictions_path": str(path), "provenance": {
                "kind": "prompt_greedy" if method.endswith("_greedy") else "prompt_pool_reranking",
                "prompt_name": method.rsplit("_", 1)[0], "protocol_sha256": protocol_sha,
                "prediction_sha256": inference.file_hash(path), "backend": "injected_synthetic_backend",
                "generation_count_per_sample": 13, "random_count_per_sample": 12,
                "labels_used": False, "parameters_fitted": False, "student_execution": False,
                "generation_manifest_sha256": inference.file_hash(cls.template / "prompt-baselines/manifest.json"),
                "source_sha256": {"prompt_baselines": strong["implementation_sha256"],
                    "prepared_manifest": inference.file_hash(cls.fixture.prepared / "manifest.json")}}})
        (cls.template / "externals.json").write_bytes(candidate_support.encoded(external))
        with redirect_stdout(io.StringIO()):
            candidate_support_analysis.main(["--balanced-dir", str(cls.fixture.prepared / "pools/balanced12/inference"),
                "--control-dir", str(cls.fixture.prepared / "pools/11_only12/inference"),
                "--inputs", str(cls.fixture.prepared / "inputs.jsonl"), "--labels", str(cls.labels),
                "--externals-json", str(cls.template / "externals.json"), "--protocol", str(cls.fixture.prepared / "protocol.json"),
                "--output", str(cls.template / "analysis")])

    def setUp(self):
        self.output = self.fixture.root / ("verification-case-" + self._testMethodName)
        shutil.copytree(self.template, self.output)
        self.addCleanup(shutil.rmtree, self.output)

    def reseal_analysis(self):
        path = self.output / "analysis/analysis.json"
        (path.parent / "SUCCESS").write_text(inference.file_hash(path) + "\n")

    def edit_json(self, relative, operation):
        path = self.output / relative
        value = json.loads(path.read_text())
        operation(value)
        path.write_bytes(candidate_support.encoded(value))

    def edit_jsonl(self, relative, operation):
        path = self.output / relative
        rows = inference.read_jsonl(path)
        operation(rows)
        for row in rows:
            if "record_sha256" in row:
                row["record_sha256"] = inference.object_hash({name: value for name, value in row.items() if name != "record_sha256"})
        path.write_bytes(b"".join(candidate_support.encoded(row) for row in rows))

    def reseal_strong(self):
        directory = self.output / "prompt-baselines"
        manifest = json.loads((directory / "manifest.json").read_text())
        for name in manifest["files_sha256"]:
            manifest["files_sha256"][name] = inference.file_hash(directory / name)
        (directory / "manifest.json").write_bytes(candidate_support.encoded(manifest))
        (directory / "SUCCESS").write_text(inference.file_hash(directory / "manifest.json") + "\n")

    def test_complete_70_17_independent_replay_all_methods_and_statistics(self):
        report = verifier.verify(self.output, self.labels)
        self.assertEqual(report["status"], "verified")
        self.assertTrue(report["verified"])
        self.assertEqual((report["samples"], report["students"], report["samples_dropped"]), (70, 17, 0))
        self.assertEqual(report["generation_attempts"], 3080)
        self.assertEqual(report["raw_strong_new_attempts"], 1820)
        self.assertTrue(report["main_argmax_verified"])
        self.assertTrue(report["strong_own_prompt_argmax_verified"])
        self.assertTrue(report["student_means_bootstrap_subgroups_verified"])
        self.assertFalse(report["outer_completion_verified"])
        self.assertEqual(report["phase"], "analysis_child_before_outer_finalization")
        self.assertTrue(report["analysis_kind"].startswith("synthetic_"))
        self.assertFalse(report["model_scoring_performed_by_verifier"])
        self.assertEqual(len(report["strong_selections"]), 280)
        self.assertEqual(report["bootstrap_repetitions"], 2000)
        self.assertFalse(report["test_set_not_used"]["whole_pipeline_access_audited"])
        content = verifier.canonical(report)
        self.assertNotIn("print(0)\\n", content)
        self.assertNotIn('"raw_text"', content)
        self.assertNotIn('"generated_token_ids"', content)

    def test_standard_library_only_imports_and_strict_jsonl_delimiter(self):
        tree = ast.parse(SCRIPT.read_text())
        imports = {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        imports |= {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        self.assertFalse(imports & {"student_sim_cd", "torch", "transformers", "numpy", "subprocess", "socket"})
        self.assertEqual(verifier.records(b'{"sample_id":"x","value":"a\xe2\x80\xa8b"}\n', "unicode")[0]["value"], "a\u2028b")
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            verifier.parsed(b'{"x":1,"x":2}')
        with self.assertRaises(ValueError):
            verifier.parsed(b'{"x":NaN}')

    def test_wrong_main_raw_seed_refuses_even_after_record_rehash(self):
        self.edit_jsonl("prepared/pools/balanced12/inference/candidates.jsonl", lambda rows:
            rows[0]["attempts"][4]["raw_generation"].update(seed=7))
        with self.assertRaisesRegex(ValueError, "identity/seed"):
            verifier.verify(self.output, self.labels)

    def test_new_main_wrappers_must_match_durable_generation_source(self):
        reader = verifier.Reader(self.output)
        prepared = self.output / "prepared"
        inputs = {row["sample_id"]: row for row in inference.read_jsonl(prepared / "inputs.jsonl")}
        caches = {pool: {"records": {row["sample_id"]: row for row in inference.read_jsonl(
            prepared / "pools" / pool / "inference/candidates.jsonl")}} for pool in verifier.POOLS}
        self.assertEqual(verifier.generation_checkpoints(reader, prepared, inputs, asdict(self.fixture.config), 256, caches), 1260)
        caches["balanced12"]["records"]["q0"]["attempts"][4]["raw_generation"]["seed"] = 1
        with self.assertRaisesRegex(ValueError, "sealed durable draw"):
            verifier.generation_checkpoints(reader, prepared, inputs, asdict(self.fixture.config), 256, caches)

    def test_missing_durable_generation_pair_is_rejected(self):
        self.edit_jsonl("prepared/draw-checkpoints.jsonl", lambda rows: rows.pop())
        reader = verifier.Reader(self.output)
        prepared = self.output / "prepared"
        inputs = {row["sample_id"]: row for row in inference.read_jsonl(prepared / "inputs.jsonl")}
        with self.assertRaisesRegex(ValueError, "reservation/raw budget"):
            verifier.generation_checkpoints(reader, prepared, inputs, asdict(self.fixture.config), 256, {})

    def test_main_derived_score_and_copy_corruption_are_rejected(self):
        self.edit_jsonl("prepared/pools/balanced12/inference/scores.jsonl", lambda rows:
            rows[0]["scores_weight_1"].update(b=999))
        with self.assertRaisesRegex(ValueError, "fixed_methods"):
            verifier.verify(self.output, self.labels)

    def test_strong_prediction_cannot_change_under_completed_sha(self):
        self.edit_jsonl("prompt-baselines/student_revision_pool.jsonl", lambda rows:
            rows[0].update(predicted_code="independently wrong prediction\n"))
        self.reseal_strong()
        with self.assertRaisesRegex(ValueError, "strong actual prediction"):
            verifier.verify(self.output, self.labels)

    def test_strong_bad_attempt_seed_or_reservation_are_rejected(self):
        self.edit_jsonl("prompt-baselines/raw.jsonl", lambda rows: rows[0].update(seed=1))
        self.reseal_strong()
        with self.assertRaisesRegex(ValueError, "identity/seed"):
            verifier.verify(self.output, self.labels)

    def test_strong_reservation_prompt_mismatch_is_rejected(self):
        self.edit_jsonl("prompt-baselines/reservations.jsonl", lambda rows: rows[0].update(prompt_sha256="0" * 64))
        self.reseal_strong()
        with self.assertRaisesRegex(ValueError, "reservation/current prompt"):
            verifier.verify(self.output, self.labels)

    def test_incomplete_1820_attempt_cohort_cannot_pass(self):
        self.edit_jsonl("prompt-baselines/raw.jsonl", lambda rows: rows.pop())
        self.reseal_strong()
        with self.assertRaisesRegex(ValueError, "complete attempt"):
            verifier.verify(self.output, self.labels)

    def test_static_metric_rehash_and_bootstrap_interval_tampering_are_rejected(self):
        self.edit_json("analysis/analysis.json", lambda value:
            value["static_per_sample"]["balanced12"]["base"]["q0"]["metrics"].update(edit_location_f1=0.5))
        self.reseal_analysis()
        with self.assertRaisesRegex(ValueError, "independent difflib"):
            verifier.verify(self.output, self.labels)

    def test_primary_interval_tampering_is_rejected(self):
        self.edit_json("analysis/analysis.json", lambda value:
            value["primary"].update(paired_student_bootstrap_95_ci=[-0.5, 0.5]))
        self.reseal_analysis()
        with self.assertRaisesRegex(ValueError, "primary interaction"):
            verifier.verify(self.output, self.labels)

    def test_success_marker_missing_and_changed_labels_refuse(self):
        (self.output / "analysis/SUCCESS").write_text("0" * 64 + "\n")
        with self.assertRaisesRegex(ValueError, "Completion SHA"):
            verifier.verify(self.output, self.labels)

    def test_exclusive_cli_output_and_completed_outer_boundary(self):
        outer = {"status": "success", "exit_code": 0, "stages": [{"status": "success", "exit_code": 0}]}
        (self.output / "manifest.json").write_bytes(candidate_support.encoded(outer))
        (self.output / "SUCCESS").write_text(inference.file_hash(self.output / "manifest.json") + "\n")
        output = self.output / "verification.json"
        argv = ["--run-dir", str(self.output), "--labels", str(self.labels), "--output", str(output)]
        with redirect_stdout(io.StringIO()):
            self.assertEqual(verifier.main(argv), 0)
        value = json.loads(output.read_text())
        self.assertTrue(value["outer_completion_verified"])
        old = output.read_bytes()
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            verifier.main(argv)
        self.assertEqual(error.exception.code, 2)
        self.assertEqual(output.read_bytes(), old)


if __name__ == "__main__":
    unittest.main()
