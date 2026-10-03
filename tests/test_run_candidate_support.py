"""Harmless real-child regressions for the separately approved generation suite."""

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


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_candidate_support.py"
spec = importlib.util.spec_from_file_location("run_candidate_support", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class CandidateSupportRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="candidate-suite-runner-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        outputs = self.root / "outputs"
        outputs.mkdir()
        baseline = outputs / "unchanged-baseline"
        baseline.mkdir()
        (baseline / "manifest.json").write_text('{"synthetic":"immutable baseline"}\n')
        model = self.root / "existing-model"
        model.mkdir()
        labels = self.root / "labels.jsonl"
        labels.write_text('{"synthetic":"labels only"}\n')
        protocol = self.root / "protocol.json"
        specification = json.loads(runner.DEFAULT_PROTOCOL.read_bytes())
        specification["labels_sha256"] = runner.sha256(labels)
        specification["baseline_sha256"]["manifest.json"] = runner.sha256(baseline / "manifest.json")
        protocol.write_text(json.dumps(specification))
        self.stop = datetime.now(timezone.utc) + timedelta(hours=2)
        self.args = runner.argparse.Namespace(baseline_run=str(baseline), labels=str(labels),
            model_path=str(model), protocol=str(protocol), output=str(outputs / "new-run"), device="cuda:0",
            stop_by=self.stop.isoformat(), max_seconds=60, wall_seconds=60)
        self.output = Path(self.args.output)
        self.ledger = outputs / runner.LEDGER_NAME
        patches = [mock.patch.object(runner, "SERVER_ROOT", self.root),
            mock.patch.object(runner, "server_identity", return_value={"release_sha256": "a" * 64,
                "host": "synthetic_control_plane", "cpu_affinity": [0, 1, 2, 3], "cpu_threads": 4,
                "pid": os.getpid(), "process_group": os.getpgrp(), "controlled_process_group": True}),
            mock.patch.dict(os.environ, {"STUDENT_SIM_CONTROLLED_PROCESS_GROUP": "1",
                "CUDA_VISIBLE_DEVICES": "GPU-d13915b1-0e5d-ed5f-d3e4-95eb30e91451",
                "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
                "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"})]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)

    def commands(self, first=None):
        child = [sys.executable, "-B", "-c", "print('synthetic control-plane stage',flush=True)"]
        return [("prepare", first or child), ("experiment", child), ("analyze", child)]

    def invoke(self, commands=None):
        with mock.patch.object(runner, "stage_commands", return_value=commands or self.commands()), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return runner.run(self.args)

    def manifest(self):
        return json.loads((self.output / "manifest.json").read_text())

    def test_real_children_success_match_marker_and_one_aggregate_lease(self):
        labels_before = Path(self.args.labels).read_bytes()
        baseline_before = (Path(self.args.baseline_run) / "manifest.json").read_bytes()
        self.assertEqual(self.invoke(), 0)
        manifest = self.manifest()
        self.assertEqual([stage["status"] for stage in manifest["stages"]], ["success"] * 3)
        self.assertEqual((self.output / "SUCCESS").read_text().strip(),
            hashlib.sha256((self.output / "manifest.json").read_bytes()).hexdigest())
        entries = json.loads(self.ledger.read_text())["entries"]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["status"], "finished")
        self.assertLessEqual(entries[0]["charged_seconds"], 60)
        self.assertEqual(manifest["max_new_generation_attempts"], 3080)
        self.assertFalse(manifest["configuration"]["automatic_resume"])
        self.assertFalse(manifest["student_execution"])
        self.assertEqual(Path(self.args.labels).read_bytes(), labels_before)
        self.assertEqual((Path(self.args.baseline_run) / "manifest.json").read_bytes(), baseline_before)

    def test_real_nonzero_child_stops_downstream_and_preserves_log(self):
        child = [sys.executable, "-B", "-c", "print('retained failure',flush=True);raise SystemExit(19)"]
        self.assertEqual(self.invoke(self.commands(child)), 19)
        manifest = self.manifest()
        self.assertEqual([stage["status"] for stage in manifest["stages"]], ["failed", "pending", "pending"])
        self.assertEqual(manifest["stages"][0]["exit_code"], 19)
        self.assertIn("retained failure", (self.output / "logs/prepare.log").read_text())
        self.assertFalse((self.output / "SUCCESS").exists())
        self.assertEqual(json.loads(self.ledger.read_text())["entries"][0]["status"], "finished")

    def test_real_timeout_kills_and_reaps_before_budget_finish(self):
        self.args.wall_seconds = 1
        child = [sys.executable, "-B", "-c", "import os,time;print(os.getpid(),flush=True);time.sleep(30)"]
        self.assertEqual(self.invoke(self.commands(child)), 1)
        pid = int((self.output / "logs/prepare.log").read_text())
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)
        self.assertFalse((self.output / "SUCCESS").exists())
        self.assertEqual([stage["status"] for stage in self.manifest()["stages"]], ["failed", "pending", "pending"])
        entry = json.loads(self.ledger.read_text())["entries"][0]
        self.assertEqual(entry["status"], "finished")
        self.assertGreaterEqual(entry["elapsed_seconds"], 0.9)

    def test_real_child_stays_in_reviewed_parent_process_group(self):
        child = [sys.executable, "-B", "-c", "import os;print(os.getpgrp(),flush=True)"]
        self.assertEqual(self.invoke(self.commands(child)), 0)
        self.assertEqual(int((self.output / "logs/prepare.log").read_text()), os.getpgrp())

    def test_exhausted_ledger_is_not_reset_and_no_child_or_output_created(self):
        budget = runner.NightBudget(self.ledger, stop_by=self.args.stop_by, max_seconds=60)
        self.assertEqual(budget.acquire(60), 60)
        budget._release()  # Interrupted runs retain their complete reservation.
        before = self.ledger.read_bytes()
        with mock.patch.object(runner.subprocess, "Popen", wraps=subprocess.Popen) as spawn:
            with self.assertRaisesRegex(ValueError, "budget exhausted"):
                self.invoke()
            spawn.assert_not_called()
        self.assertEqual(self.ledger.read_bytes(), before)
        self.assertFalse(self.output.exists())

    def test_active_aggregate_lease_refuses_second_owner(self):
        budget = runner.NightBudget(self.ledger, stop_by=self.args.stop_by, max_seconds=60)
        budget.acquire(60)
        self.addCleanup(budget._release)
        before = self.ledger.read_bytes()
        with mock.patch.object(runner.subprocess, "Popen", wraps=subprocess.Popen) as spawn:
            with self.assertRaisesRegex(ValueError, "active lease"):
                self.invoke()
            spawn.assert_not_called()
        self.assertEqual(self.ledger.read_bytes(), before)
        self.assertFalse(self.output.exists())

    def test_resource_arguments_are_explicit_aware_current_and_bounded(self):
        for field, value in (("stop_by", "2026-10-03T04:00:00"),
                ("stop_by", (datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat()),
                ("max_seconds", 43201), ("max_seconds", True), ("wall_seconds", 61)):
            with self.subTest(field=field, value=value):
                args = runner.argparse.Namespace(**vars(self.args))
                setattr(args, field, value)
                with self.assertRaises(ValueError):
                    runner.resource_arguments(args)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.parser().parse_args(["run", "--baseline-run", self.args.baseline_run,
                "--model-path", self.args.model_path, "--output", self.args.output,
                "--device", "cuda:0", "--labels", self.args.labels])

    def test_old_history_protocol_and_generation_budget_drift_are_rejected(self):
        data = json.loads(Path(self.args.protocol).read_text())
        for key, value in (("schema_version", "student-sim-cd.history-intervention.protocol.v1"),
                ("generation_performed", False), ("max_new_generation_attempts", 3081)):
            changed = dict(data, **{key: value})
            Path(self.args.protocol).write_text(json.dumps(changed))
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "Unsupported frozen"):
                runner.read_protocol(self.args.protocol)
        self.assertFalse(self.ledger.exists())

    def test_changed_labels_refuse_before_any_child_or_budget(self):
        Path(self.args.labels).write_text('{"synthetic":"different labels"}\n')
        with mock.patch.object(runner.subprocess, "Popen", wraps=subprocess.Popen) as spawn:
            with self.assertRaisesRegex(ValueError, "Labels differ"):
                self.invoke()
            spawn.assert_not_called()
        self.assertFalse(self.ledger.exists())
        self.assertFalse(self.output.exists())

    def test_cpu_environments_and_commands_exclude_labels_and_credentials(self):
        paths = {name: Path(getattr(self.args, name)) for name in ("baseline_run", "model_path", "protocol", "labels")}
        commands = runner.stage_commands(paths, self.output, self.args)
        with mock.patch.dict(os.environ, {"HOME": "/synthetic-home", "HF_TOKEN": "synthetic-not-a-credential",
                "HTTP_PROXY": "synthetic", "USE_TORCH": "0", "USE_TF": "0"}):
            for name, command in commands:
                self.assertEqual(command[:3], [sys.executable, "-B", "-u"])
                env = runner.stage_environment(name)
                for secret in ("HOME", "HF_TOKEN", "HTTP_PROXY"):
                    self.assertNotIn(secret, env)
                self.assertEqual(env["HF_HUB_OFFLINE"], "1")
                for variable in runner.THREAD_VARIABLES:
                    self.assertEqual(env[variable], "4")
                if name != "analyze":
                    self.assertNotIn("--labels", command)
                    self.assertNotIn(self.args.labels, command)
                if name != "experiment":
                    self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "")
                    self.assertEqual(env["USE_TORCH"], "0")
                    self.assertEqual(env["USE_TF"], "0")
                else:
                    self.assertEqual(env["CUDA_VISIBLE_DEVICES"], os.environ["CUDA_VISIBLE_DEVICES"])
                    self.assertNotIn("USE_TORCH", env)
                    self.assertNotIn("USE_TF", env)

    def test_production_resource_guard_requires_group_four_cores_and_one_uuid(self):
        env = {"STUDENT_SIM_CONTROLLED_PROCESS_GROUP": "1",
            "CUDA_VISIBLE_DEVICES": "GPU-d13915b1-0e5d-ed5f-d3e4-95eb30e91451",
            "HF_HUB_OFFLINE": "1", "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
            **{name: "4" for name in runner.THREAD_VARIABLES}}
        with mock.patch.dict(os.environ, env, clear=True), \
                mock.patch.object(runner.os, "sched_getaffinity", return_value={0, 1, 2, 3}, create=True):
            self.assertEqual(runner.controlled_resources("experiment")["cpu_affinity"], [0, 1, 2, 3])
            for name, value in (("STUDENT_SIM_CONTROLLED_PROCESS_GROUP", "0"),
                    ("CUDA_VISIBLE_DEVICES", "0"), ("CUDA_VISIBLE_DEVICES", env["CUDA_VISIBLE_DEVICES"] + ",1"),
                    ("OMP_NUM_THREADS", "5"), ("HF_HUB_OFFLINE", "0")):
                with self.subTest(name=name, value=value), mock.patch.dict(os.environ, {name: value}), self.assertRaises(ValueError):
                    runner.controlled_resources("experiment")
            with mock.patch.object(runner.os, "sched_getaffinity", return_value={0, 1, 2, 3, 4}), self.assertRaises(ValueError):
                runner.controlled_resources("experiment")
            with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "", "USE_TORCH": "0", "USE_TF": "0"}):
                self.assertEqual(runner.controlled_resources("prepare")["cuda_visible_devices"], "")
                self.assertEqual(runner.controlled_resources("analyze")["cpu_threads"], 4)

    def test_competing_directory_is_preserved_without_overwrite(self):
        original_mkdir = Path.mkdir
        preserved = b'{"synthetic":"different creator"}\n'
        def concurrent_creator(path, *args, **kwargs):
            if path == self.output:
                original_mkdir(path, *args, **kwargs)
                (path / "manifest.json").write_bytes(preserved)
                raise FileExistsError("Another creator won")
            return original_mkdir(path, *args, **kwargs)
        with mock.patch.object(Path, "mkdir", autospec=True, side_effect=concurrent_creator), \
                mock.patch.object(runner.subprocess, "Popen", wraps=subprocess.Popen) as spawn:
            self.assertEqual(self.invoke(), 1)
            spawn.assert_not_called()
        self.assertEqual((self.output / "manifest.json").read_bytes(), preserved)
        self.assertFalse((self.output / "SUCCESS").exists())

    def test_formal_prepare_uses_internal_cpu_tokenizer_not_synthetic_injection(self):
        from student_sim_cd import candidate_support, preflight, prompt_baselines
        self.output.mkdir()
        self.args.prepared = str(self.output / "prepared")
        rows, references, config = [object()] * 70, {}, object()
        audit = [{"synthetic_prompt": index} for index in range(140)]
        tokenizer = object()
        with mock.patch.object(runner, "load_config", return_value=config), \
                mock.patch.object(candidate_support, "prepare", return_value={"prepared": True}) as prepare, \
                mock.patch.object(candidate_support, "load_prepared", return_value={"rows": rows, "references": references}), \
                mock.patch.object(preflight, "load_local_tokenizer", return_value=tokenizer), \
                mock.patch.object(prompt_baselines, "audit", return_value=audit):
            result = runner.prepare(self.args)
        self.assertNotIn("tokenizer", prepare.call_args.kwargs)
        self.assertEqual(result["strong_prompts_audited"], 140)
        self.assertFalse(result["model_weights_loaded"])
        self.assertEqual(json.loads((self.output / "strong-prompt-audit.json").read_text()), audit)

    def experiment_mocks(self, *, mismatch=False):
        from student_sim_cd import candidate_support, inference, prompt_baselines
        self.output.mkdir()
        prepared = self.output / "prepared"
        prepared.mkdir()
        self.args.prepared = str(prepared)
        audit = [{"synthetic_prompt": index} for index in range(140)]
        (self.output / "strong-prompt-audit.json").write_text(json.dumps(audit))
        config = object()
        old = [None] * 8
        old[6] = {"model": {"sha256": "synthetic-fixed-model"}}
        bundle = {"rows": [object()] * 70, "references": {}, "manifest": {"protocol_sha256": "b" * 64}, "baseline": old}
        backend = mock.Mock(tokenizer=object())
        torch = mock.Mock()
        torch.get_num_threads.return_value = torch.get_num_interop_threads.return_value = 4
        patches = [mock.patch.object(runner, "load_config", return_value=config),
            mock.patch.object(runner, "idle_gpu", return_value={"synthetic_idle": True}),
            mock.patch.object(inference, "model_fingerprint", return_value=old[6]["model"]),
            mock.patch.object(inference, "HFBackend", return_value=backend),
            mock.patch.object(candidate_support, "load_prepared", return_value=bundle),
            mock.patch.object(candidate_support, "run", return_value={"backend": "HFBackend", "new_draws": 1260, "samples": 70, "students": 17}),
            mock.patch.object(prompt_baselines, "audit", return_value=audit[:-1] if mismatch else audit),
            mock.patch.object(prompt_baselines, "run", return_value={"backend": "HFBackend", "attempts": 1820, "samples": 70}),
            mock.patch.dict(sys.modules, {"torch": torch})]
        handles = [patch.start() for patch in patches]
        for patch in patches:
            self.addCleanup(patch.stop)
        return handles, backend, torch

    def test_one_shared_backend_and_thread_caps_for_all_new_generations(self):
        handles, backend, torch = self.experiment_mocks()
        result = runner.experiment(self.args)
        handles[3].assert_called_once()
        self.assertIs(handles[5].call_args.kwargs["backend"], backend)
        self.assertIs(handles[7].call_args.args[3], backend)
        self.assertFalse(handles[5].call_args.kwargs["resume"])
        self.assertFalse(handles[7].call_args.kwargs["resume"])
        torch.set_num_threads.assert_called_once_with(4)
        torch.set_num_interop_threads.assert_called_once_with(4)
        self.assertTrue(result["model_loaded_once"])
        self.assertFalse(result["labels_read"])

    def test_loaded_tokenizer_audit_mismatch_refuses_before_any_draw(self):
        handles, _, _ = self.experiment_mocks(mismatch=True)
        with self.assertRaisesRegex(ValueError, "audits differ"):
            runner.experiment(self.args)
        handles[5].assert_not_called()
        handles[7].assert_not_called()

    def prompt_artifacts(self):
        from student_sim_cd import prompt_baselines
        self.output.mkdir()
        prepared = self.output / "prepared"
        prepared.mkdir()
        self.args.prepared = str(prepared)
        (prepared / "manifest.json").write_text('{"synthetic":"prepared source"}\n')
        pool = prepared / "pools/balanced12/inference"
        pool.mkdir(parents=True)
        identity = {"model": {"synthetic": "fixed fingerprint"}, "runtime_versions": {"synthetic": "fixed runtime"}}
        (pool / "manifest.json").write_text(json.dumps(identity))
        directory = self.output / "prompt-baselines"
        directory.mkdir()
        names = {method + suffix + ".jsonl" for method in ("student_revision", "conservative_edit")
            for suffix in ("_greedy", "_pool")} | {"raw.jsonl", "scores.jsonl", "summary.json", "checkpoint-manifest.json", "reservations.jsonl"}
        for name in names:
            (directory / name).write_text('{"synthetic":"adapter routing only, not model result"}\n')
        manifest = {**identity, "status": "success", "samples": 70, "attempts": 1820,
            "attempts_per_prompt": 13, "random_attempts": 12, "labels_read": False,
            "student_execution": False, "protocol_sha256": runner.sha256(self.args.protocol),
            "config": runner.read_protocol(self.args.protocol)["generation_config"], "backend": "HFBackend",
            "implementation_sha256": runner.sha256(prompt_baselines.__file__),
            "files_sha256": {name: runner.sha256(directory / name) for name in names}}
        (directory / "manifest.json").write_text(json.dumps(manifest))
        (directory / "SUCCESS").write_text(runner.sha256(directory / "manifest.json") + "\n")
        return directory

    def test_analysis_adapter_passes_four_bound_private_predictions_and_cpu_labels(self):
        from student_sim_cd import candidate_support_analysis
        self.prompt_artifacts()
        def harmless_analyzer(argv):
            parsed = dict(zip(argv[::2], argv[1::2]))
            self.assertEqual(parsed["--labels"], self.args.labels)
            self.assertEqual(parsed["--inputs"], str(Path(self.args.prepared) / "inputs.jsonl"))
            self.assertEqual(parsed["--balanced-dir"], str(Path(self.args.prepared) / "pools/balanced12/inference"))
            self.assertEqual(parsed["--control-dir"], str(Path(self.args.prepared) / "pools/11_only12/inference"))
            external = json.loads(Path(parsed["--externals-json"]).read_text())
            self.assertEqual(len(external), 4)
            for item in external:
                self.assertEqual(item["provenance"]["prediction_sha256"], runner.sha256(item["predictions_path"]))
                self.assertFalse(item["provenance"]["labels_used"])
                self.assertFalse(item["provenance"]["parameters_fitted"])
                self.assertFalse(item["provenance"]["student_execution"])
                self.assertEqual(item["provenance"]["generation_count_per_sample"], 13)
            output = Path(parsed["--output"])
            output.mkdir()
            (output / "analysis.json").write_text('{"synthetic":"adapter only"}\n')
            (output / "SUCCESS").write_text(runner.sha256(output / "analysis.json") + "\n")
            return 0
        with mock.patch.object(candidate_support_analysis, "main", side_effect=harmless_analyzer) as analyze:
            def harmless_verifier(args):
                path = Path(args.output) / "independent-verification.json"
                path.write_text('{"status":"verified","synthetic":"adapter routing only"}\n')
                return path
            with mock.patch.object(runner, "independent_verification", side_effect=harmless_verifier) as verify:
                result = runner.analyze(self.args)
            verify.assert_called_once_with(self.args)
        analyze.assert_called_once()
        self.assertEqual(result["analysis"], str(self.output / "analysis/analysis.json"))
        self.assertTrue(result["labels_read"])
        self.assertFalse(result["student_execution"])

    def test_analysis_adapter_rejects_mutated_prediction_before_analyzer_or_specification(self):
        from student_sim_cd import candidate_support_analysis
        directory = self.prompt_artifacts()
        (directory / "student_revision_greedy.jsonl").write_text('{"synthetic":"changed"}\n')
        with mock.patch.object(candidate_support_analysis, "main") as analyze:
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                runner.analyze(self.args)
            analyze.assert_not_called()
        self.assertFalse((self.output / "externals.json").exists())
        self.assertFalse((self.output / "analysis").exists())

    def test_independent_verifier_is_in_process_and_output_never_overwritten(self):
        scripts = self.root / "scripts"
        scripts.mkdir()
        (scripts / "verify_candidate_support_results.py").write_text(
            "def verify(run_dir, labels_path):\n"
            "    return {'status':'verified','verified':True,'synthetic':'adapter only'}\n")
        self.output.mkdir()
        with mock.patch.object(runner, "ROOT", self.root), mock.patch.object(runner.subprocess, "Popen") as spawn:
            path = runner.independent_verification(self.args)
            spawn.assert_not_called()
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                runner.independent_verification(self.args)
            self.assertEqual(path.read_bytes(), before)
        self.assertTrue(json.loads(before)["verified"])

    def test_independent_verifier_failure_cannot_create_verified_artifact(self):
        scripts = self.root / "scripts"
        scripts.mkdir()
        (scripts / "verify_candidate_support_results.py").write_text(
            "def verify(run_dir, labels_path):\n"
            "    return {'status':'verified','verified':False}\n")
        self.output.mkdir()
        with mock.patch.object(runner, "ROOT", self.root), self.assertRaisesRegex(ValueError, "verification failed"):
            runner.independent_verification(self.args)
        self.assertFalse((self.output / "independent-verification.json").exists())


if __name__ == "__main__":
    unittest.main()
