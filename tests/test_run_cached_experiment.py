"""Runner orchestration tests with fake stages; no model or student execution."""

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("run_experiment", SCRIPTS / "run_experiment.py")
existing_runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(existing_runner)
with mock.patch.dict(sys.modules, {"run_experiment": existing_runner}):
    spec = importlib.util.spec_from_file_location("run_cached_experiment", SCRIPTS / "run_cached_experiment.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)


class CachedRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="cached-runner-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / "old/inference"
        self.source.mkdir(parents=True)
        (self.source / "manifest.json").write_text('{"config":{}}\n')
        (self.source.parent / "manifest.json").write_text("{}\n")
        for name in ("candidates", "scores"):
            (self.source / (name + ".jsonl")).write_text('{"synthetic":true}\n')
        self.paths = {}
        for name in ("inputs", "references", "labels"):
            path = self.root / (name + ".jsonl")
            path.write_text('{"synthetic":true}\n')
            self.paths[name] = path
        self.paths["model-path"] = self.root / "model"
        self.paths["model-path"].mkdir()
        self.protocol = json.loads((runner.DEFAULT_PROTOCOL).read_text())
        # Synthetic orchestration must remain testable after the real one-night
        # authorization expires; the shipped protocol retains its fixed date.
        self.protocol["night_resource_authorization"]["stop_by"] = (
            datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        self.protocol["inference"] = {}
        self.protocol["source_inference_manifest_sha256"] = runner.sha256(self.source / "manifest.json")
        self.protocol["source_experiment_manifest_sha256"] = runner.sha256(self.source.parent / "manifest.json")
        for name in ("candidates", "scores"):
            self.protocol["source_" + name + "_sha256"] = runner.sha256(self.source / (name + ".jsonl"))
        self.protocol["labels_sha256"] = runner.sha256(self.paths["labels"])
        self.paths["protocol"] = self.root / "protocol.json"
        self.paths["protocol"].write_text(json.dumps(self.protocol))
        self.output = self.root / "output"
        self.arguments = ["run", "--source-run", str(self.source), "--output", str(self.output),
                          "--device", "cuda:0", "--wall-seconds", "60"]
        for name, path in self.paths.items():
            self.arguments += ["--" + name, str(path)]

    def invoke(self, effect=None, arguments=None):
        if effect is None:
            def effect(command, **kwargs):
                kwargs["stdout"].write("Synthetic orchestration log; no real computation.\n")
                return subprocess.CompletedProcess(command, 0)
        with mock.patch.object(runner.subprocess, "run", side_effect=effect) as calls, \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            result = runner.main(arguments or self.arguments)
        return result, calls.call_args_list

    def test_labels_only_reach_evaluation_and_paired_comparison(self):
        result, calls = self.invoke()
        self.assertEqual(result, 0)
        self.assertEqual(len(calls), 11)
        for index, call in enumerate(calls):
            command = call.args[0]
            self.assertEqual(command[:3], [sys.executable, "-B", "-u"])
            if index < 4:
                self.assertNotIn("--labels", command)
                self.assertNotIn(str(self.paths["labels"]), command)
            else:
                self.assertIn("--labels", command)
            self.assertEqual(call.kwargs["cwd"], str(runner.SRC))
            self.assertTrue(0 < call.kwargs["timeout"] <= 60)
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                self.assertEqual(call.kwargs["env"][key], "4")
            self.assertEqual(call.kwargs["env"]["HF_HUB_OFFLINE"], "1")
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertFalse(manifest["generation_performed"])
        self.assertFalse(manifest["student_execution"])
        self.assertEqual(manifest["stages"][3]["name"], "greedy")
        self.assertEqual((self.output / "SUCCESS").read_text().strip(),
                         hashlib.sha256((self.output / "manifest.json").read_bytes()).hexdigest())

    def test_scoring_failure_preserves_logs_and_stops_downstream(self):
        index = []
        def effect(command, **kwargs):
            index.append(command)
            return subprocess.CompletedProcess(command, 19 if len(index) == 2 else 0)
        result, calls = self.invoke(effect)
        self.assertEqual(result, 19)
        self.assertEqual(len(calls), 2)
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["stages"][1]["exit_code"], 19)
        self.assertTrue(all(stage["status"] == "pending" for stage in manifest["stages"][2:]))
        self.assertFalse((self.output / "SUCCESS").exists())

    def test_timeout_fails_without_success_or_later_stages(self):
        result, calls = self.invoke(subprocess.TimeoutExpired("synthetic-score", 60))
        self.assertEqual(result, 1)
        self.assertEqual(len(calls), 1)
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "failed")
        self.assertFalse((self.output / "SUCCESS").exists())

    def test_elapsed_stages_share_one_deadline(self):
        with mock.patch.object(runner.time, "monotonic", side_effect=[0, 1, 1, 1, 40, 61]):
            result, calls = self.invoke()
        self.assertEqual(result, 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].kwargs["timeout"], 59)
        self.assertFalse((self.output / "SUCCESS").exists())

    def test_reservation_persistence_overhead_cannot_extend_the_granted_window(self):
        clock = {"monotonic": 100.0}
        original_acquire = runner.NightBudget.acquire

        def delayed_acquire(budget, requested):
            allowed = original_acquire(budget, requested)
            clock["monotonic"] += 70.0  # Reservation/output preparation consumes the grant.
            return allowed

        with mock.patch.object(runner.time, "monotonic", side_effect=lambda: clock["monotonic"]), \
             mock.patch.object(runner.NightBudget, "acquire", autospec=True, side_effect=delayed_acquire):
            result, calls = self.invoke()
        self.assertEqual(result, 1)
        self.assertEqual(calls, [])
        self.assertFalse((self.output / "SUCCESS").exists())
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(manifest["configuration"]["wall_seconds"], 60)
        ledger = json.loads(Path(manifest["configuration"]["budget_ledger"]).read_text())
        self.assertEqual(ledger["max_seconds"], 14400)
        self.assertLessEqual(ledger["entries"][0]["charged_seconds"], 60)

    def test_absolute_stop_by_is_rechecked_after_reservation_and_before_child_start(self):
        stop = runner.datetime.fromisoformat(self.protocol["night_resource_authorization"]["stop_by"]).timestamp()
        clock = {"wall": stop - 30}
        original_acquire = runner.NightBudget.acquire

        def expire_after_acquire(budget, requested):
            allowed = original_acquire(budget, requested)
            clock["wall"] = stop + 1  # Model a forward wall-clock adjustment or preparation delay.
            return allowed

        with mock.patch.object(runner.time, "time", side_effect=lambda: clock["wall"]), \
             mock.patch.object(runner.time, "monotonic", return_value=100), \
             mock.patch.object(runner.NightBudget, "acquire", autospec=True, side_effect=expire_after_acquire):
            result, calls = self.invoke()
        self.assertEqual(result, 1)
        self.assertEqual(calls, [])
        self.assertFalse((self.output / "SUCCESS").exists())

    def test_manifest_and_log_preparation_cannot_launch_a_child_after_deadline(self):
        # Lease start, loop check, elapsed start, then deadline reached while
        # writing stage metadata/opening its log. No subprocess may be started.
        with mock.patch.object(runner.time, "monotonic", side_effect=[0, 1, 1, 61]):
            result, calls = self.invoke()
        self.assertEqual(result, 1)
        self.assertEqual(calls, [])
        self.assertTrue((self.output / "logs/prepare.log").exists())
        self.assertFalse((self.output / "SUCCESS").exists())

    def test_invalid_budget_or_existing_output_never_starts_stages(self):
        for budget in ("0", "14401"):
            with mock.patch.object(runner.subprocess, "run") as call, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    runner.main(self.arguments + ["--wall-seconds", budget])
                call.assert_not_called()
            self.assertFalse(self.output.exists())
        self.output.mkdir()
        marker = self.output / "preserve"
        marker.write_text("existing result")
        with mock.patch.object(runner.subprocess, "run") as call, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                runner.main(self.arguments)
            call.assert_not_called()
        self.assertEqual(marker.read_text(), "existing result")

    def test_frozen_source_hash_mismatch_fails_before_output(self):
        (self.source.parent / "manifest.json").write_text('{"changed":true}\n')
        with mock.patch.object(runner.subprocess, "run") as call, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                runner.main(self.arguments)
            call.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_output_inside_old_experiment_is_rejected_before_any_write(self):
        nested = self.source.parent / "new-output"
        arguments = self.arguments + ["--output", str(nested)]
        with mock.patch.object(runner.subprocess, "run") as call, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                runner.main(arguments)
            call.assert_not_called()
        self.assertFalse(nested.exists())

    def test_expired_night_authorization_prevents_new_run(self):
        stop = runner.datetime.fromisoformat(self.protocol["night_resource_authorization"]["stop_by"])
        with mock.patch.object(runner.time, "time", return_value=stop.timestamp() + 1), \
             mock.patch.object(runner.subprocess, "run") as call, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                runner.main(self.arguments)
            call.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_prepare_and_score_parsers_reject_label_arguments(self):
        for command, fields in (("prepare", ("source-run", "inputs", "references", "output")),
                                ("score", ("source-run", "prepared", "model-path", "device"))):
            arguments = [command]
            for field in fields:
                arguments += ["--" + field, "synthetic"]
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                runner.main(arguments + ["--labels", "future-labels.jsonl"])


class CachedProvenanceTests(unittest.TestCase):
    """Real artifact checks with synthetic scores; no Torch import or GPU use."""

    def setUp(self):
        from student_sim_cd import cache_rescore, inference, predict
        from tests.test_cache_rescore import CacheRescoreTests
        self.cache, self.inference, self.predict = cache_rescore, inference, predict
        self.fixture = CacheRescoreTests("test_preparation_preserves_raw_attempts_and_all_source_mappings")
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.prepare()
        self.fixture.score()
        self.predictions = self.fixture.root / "predictions"
        predict.predict(self.fixture.output / "inference", self.fixture.inputs, self.predictions, True)
        cache_rescore.export_greedy(self.fixture.output, self.predictions / "predictions.greedy.jsonl")
        self.labels = self.fixture.root / "labels.jsonl"
        self.labels.write_text(inference.canonical_json({"sample_id": "sample-a", "target_code": "print(1)\n"}) + "\n")
        self.protocol = json.loads(runner.DEFAULT_PROTOCOL.read_text())
        original = predict._json((self.fixture.source_run / "manifest.json").read_bytes())
        self.protocol["inference"] = {key: original["config"][key] for key in self.protocol["inference"]}
        self.protocol["inference"]["fence_policy"] = "unwrap-single"
        self.protocol["expected_samples"] = self.protocol["expected_students"] = 1
        self.protocol["bootstrap"]["repetitions"] = 100
        self.protocol["source_inference_manifest_sha256"] = runner.sha256(self.fixture.source_run / "manifest.json")
        self.protocol["source_experiment_manifest_sha256"] = runner.sha256(self.fixture.source / "manifest.json")
        for name in ("candidates", "scores"):
            self.protocol["source_" + name + "_sha256"] = runner.sha256(self.fixture.source_run / (name + ".jsonl"))
        self.protocol["labels_sha256"] = runner.sha256(self.labels)
        self.protocol_path = self.fixture.root / "protocol.json"
        self.protocol_path.write_text(json.dumps(self.protocol))
        self.report_path = self.fixture.root / "paired-report.json"
        self.args = runner.argparse.Namespace(protocol=self.protocol_path, prepared=self.fixture.output,
            predictions=self.predictions, inputs=self.fixture.inputs, labels=self.labels, output=self.report_path)

    def edit_score_manifest(self, change):
        path = self.fixture.output / "inference/manifest.json"
        value = self.predict._json(path.read_bytes())
        change(value)
        path.write_text(self.inference.canonical_json(value) + "\n")

    def test_fixed_method_tie_and_eos_spec_cannot_be_silently_changed(self):
        import copy
        for field, replacement in (("fixed_method_parameters", {"b": {"alpha": 0, "beta": 2}}),
                                   ("selection", "choose first candidate"),
                                   ("sequence_protocol", "omit EOS and normalize length")):
            changed = copy.deepcopy(self.protocol)
            changed[field] = replacement
            self.protocol_path.write_text(json.dumps(changed))
            with self.subTest(field=field), self.assertRaises(ValueError):
                runner.read_protocol(self.protocol_path)

    def test_score_rejects_another_prepared_source_before_importing_torch(self):
        import copy
        loaded = self.cache._load_prepared(self.fixture.output)
        foreign = copy.deepcopy(loaded[0])
        foreign["source_sha256"]["candidates"] = "0" * 64
        args = runner.argparse.Namespace(protocol=self.protocol_path, prepared=self.fixture.output,
            source_run=self.fixture.source_run, model_path=self.fixture.model, device="cpu")
        fake_torch = mock.Mock()
        with mock.patch.object(self.cache, "_load_prepared", return_value=(foreign, *loaded[1:])), \
             mock.patch.object(self.cache, "score") as scoring_call, \
             mock.patch.dict(sys.modules, {"torch": fake_torch}), \
             self.assertRaisesRegex(ValueError, "Prepared source fingerprint"):
            runner.score(args)
        scoring_call.assert_not_called()
        fake_torch.set_num_threads.assert_not_called()

    def test_score_validates_real_prepared_chain_then_hands_only_fixed_config_to_backend(self):
        args = runner.argparse.Namespace(protocol=self.protocol_path, prepared=self.fixture.output,
            source_run=self.fixture.source_run, model_path=self.fixture.model, device="cpu")
        fake_torch = mock.Mock()
        with mock.patch.object(self.cache, "score", return_value={"synthetic": True}) as scoring_call, \
             mock.patch.dict(sys.modules, {"torch": fake_torch}):
            self.assertEqual(runner.score(args), {"synthetic": True})
        config = scoring_call.call_args.args[1]
        self.assertEqual(config.fence_policy, "unwrap-single")
        self.assertEqual(config.max_context_tokens, self.fixture.config.max_context_tokens)
        self.assertEqual(config.num_generations, 4)
        self.assertEqual(config.model_path, str(self.fixture.model))
        fake_torch.set_num_threads.assert_called_once_with(4)
        fake_torch.set_num_interop_threads.assert_called_once_with(4)

    def test_compare_synthetic_scores_are_explicit_and_greedy_origin_is_verified(self):
        result = runner.compare(self.args)
        self.assertFalse(result["model_scoring_performed"])
        self.assertEqual(result["scoring_backend"], "injected_synthetic_backend")
        self.assertTrue(result["analysis_kind"].startswith("synthetic_"))
        self.assertTrue(result["score_provenance"]["original_greedy_mapping_verified"])
        self.assertEqual(result["all"]["samples"], 1)
        self.assertEqual(result["candidate_diagnostics"]["candidates"], 3)
        self.assertTrue(any("not real model" in item for item in result["limitations"]))
        self.assertEqual(json.loads(self.report_path.read_text())["analysis_kind"], result["analysis_kind"])

    def test_compare_rejects_legacy_scores_without_new_cache_provenance(self):
        self.edit_score_manifest(lambda manifest: manifest.pop("cache_rescore"))
        with self.assertRaisesRegex(ValueError, "New score cache provenance"):
            runner.compare(self.args)
        self.assertFalse(self.report_path.exists())

    def test_compare_rejects_new_cache_linked_to_other_prepared_records(self):
        self.edit_score_manifest(lambda manifest: manifest["cache_rescore"].update(prepared_records_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "New score cache provenance"):
            runner.compare(self.args)
        self.assertFalse(self.report_path.exists())

    def test_compare_rejects_a_changed_original_even_with_recomputed_record_hash(self):
        path = self.fixture.source_run / "scores.jsonl"
        rows = self.predict._records(path.read_bytes(), "scores")
        rows[0]["l11"] -= 1
        rows[0]["record_sha256"] = self.inference.object_hash({key: value for key, value in rows[0].items()
                                                              if key != "record_sha256"})
        path.write_text("".join(self.inference.canonical_json(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "Source manifest differs"):
            runner.compare(self.args)
        self.assertFalse(self.report_path.exists())

    def test_compare_rejects_prediction_checksum_changes(self):
        path = self.predictions / "predictions.b.jsonl"
        path.write_text(path.read_text().replace("print(0)", "print(999)"))
        # Always change bytes, even if this synthetic selector chose print(1).
        with path.open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "Prediction output SHA256"):
            runner.compare(self.args)
        self.assertFalse(self.report_path.exists())

    def test_recomputed_prediction_manifest_cannot_hide_wrong_fixed_selection(self):
        path = self.predictions / "predictions.b.jsonl"
        path.write_text(self.inference.canonical_json({"sample_id": "sample-a", "method": "b",
                                                       "predicted_code": "unscored bogus text"}) + "\n")
        manifest_path = self.predictions / "manifest.json"
        manifest = self.predict._json(manifest_path.read_bytes())
        manifest["output_sha256"][path.name] = runner.sha256(path)
        manifest_path.write_text(self.inference.canonical_json(manifest) + "\n")
        with self.assertRaisesRegex(ValueError, "verified fixed selection"):
            runner.compare(self.args)
        self.assertFalse(self.report_path.exists())

    def test_greedy_must_equal_actual_attempt_zero_not_an_arbitrary_pool_prediction(self):
        path = self.predictions / "predictions.greedy.jsonl"
        path.write_text(self.inference.canonical_json({"sample_id": "sample-a", "method": "greedy",
                                                       "predicted_code": "wrong original greedy"}) + "\n")
        with self.assertRaisesRegex(ValueError, "actual greedy"):
            runner.compare(self.args)
        self.assertFalse(self.report_path.exists())

    def test_unknown_backend_and_uninstalled_hf_claim_cannot_be_real_evidence(self):
        original = (self.fixture.output / "inference/manifest.json").read_bytes()
        for backend in ("pretend_backend", "HFBackend"):
            (self.fixture.output / "inference/manifest.json").write_bytes(original)
            def change(manifest):
                manifest["cache_rescore"]["backend"] = backend
                if backend == "HFBackend":
                    manifest["runtime_versions"]["torch"] = "not-installed"
            self.edit_score_manifest(change)
            with self.subTest(backend=backend), self.assertRaises(ValueError):
                runner.compare(self.args)
        self.assertFalse(self.report_path.exists())

    def test_incomplete_prediction_export_is_never_accepted(self):
        (self.predictions / ".incomplete").write_text("preserve incomplete export")
        with self.assertRaisesRegex(ValueError, "Prediction export is incomplete"):
            runner.compare(self.args)
        self.assertFalse(self.report_path.exists())


if __name__ == "__main__":
    unittest.main()
