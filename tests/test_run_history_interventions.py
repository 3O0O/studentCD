"""Real harmless subprocess and persistent-budget orchestration regressions."""

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
    spec = importlib.util.spec_from_file_location("run_history_interventions", SCRIPTS / "run_history_interventions.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)


class HistoryRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="history-runner-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        outputs = self.root / "outputs"
        outputs.mkdir()
        baseline = outputs / "immutable-baseline"
        baseline.mkdir()
        model = self.root / "existing-model"
        model.mkdir()
        train, labels = self.root / "train.jsonl", self.root / "labels.jsonl"
        train.write_text('{"synthetic":"train"}\n')
        labels.write_text('{"synthetic":"labels"}\n')
        protocol = json.loads(runner.DEFAULT_PROTOCOL.read_text())
        # Synthetic control-plane tests remain meaningful after today's real
        # authorization expires. The shipped protocol/date are never edited.
        stop_by = datetime.now(timezone.utc) + timedelta(hours=5)
        patched = mock.patch.object(runner, "STOP_BY", stop_by)
        patched.start()
        self.addCleanup(patched.stop)
        protocol["night_resource_authorization"]["stop_by"] = stop_by.isoformat()
        protocol["training_inputs_sha256"] = runner.sha256(train)
        protocol["labels_sha256"] = runner.sha256(labels)
        protocol_path = self.root / "frozen-protocol.json"
        protocol_path.write_text(json.dumps(protocol))
        self.args = runner.argparse.Namespace(baseline_run=str(baseline), train_inputs=str(train),
            labels=str(labels), model_path=str(model), protocol=str(protocol_path),
            output=str(outputs / "new-history-run"), device="cuda:0", wall_seconds=60)
        self.output = Path(self.args.output)
        self.ledger = outputs / runner.LEDGER_NAME
        environment = mock.patch.dict(os.environ, {
            "STUDENT_SIM_CONTROLLED_PROCESS_GROUP": "1",
            "CUDA_VISIBLE_DEVICES": "GPU-d13915b1-0e5d-ed5f-d3e4-95eb30e91451"})
        environment.start()
        self.addCleanup(environment.stop)
        affinity = mock.patch.object(runner.os, "sched_getaffinity", return_value={0, 1, 2, 3}, create=True)
        affinity.start()
        self.addCleanup(affinity.stop)

    def commands(self, *, first=None):
        simple = [sys.executable, "-B", "-c", "print('Harmless synthetic control-plane stage', flush=True)"]
        return [("prepare", first or simple), ("score", simple), ("analyze", simple)]

    def invoke(self, commands=None):
        with mock.patch.object(runner, "stage_commands", return_value=commands or self.commands()), \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            return runner.run(self.args)

    def manifest(self):
        return json.loads((self.output / "manifest.json").read_text())

    def test_real_subprocess_success_has_one_lease_and_matching_success_hash(self):
        source_bytes = Path(self.args.train_inputs).read_bytes()
        result = self.invoke()
        self.assertEqual(result, 0)
        manifest = self.manifest()
        self.assertEqual([stage["status"] for stage in manifest["stages"]], ["success"] * 3)
        self.assertEqual((self.output / "SUCCESS").read_text().strip(),
            hashlib.sha256((self.output / "manifest.json").read_bytes()).hexdigest())
        entries = json.loads(self.ledger.read_text())["entries"]
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["status"], "finished")
        self.assertLessEqual(entries[0]["charged_seconds"], 60)
        self.assertEqual(Path(self.args.train_inputs).read_bytes(), source_bytes)
        self.assertFalse(manifest["generation_performed"])
        self.assertFalse(manifest["training_text_exported"])

    def test_real_timeout_kills_and_reaps_child_before_budget_settlement(self):
        self.args.wall_seconds = 1
        child = [sys.executable, "-B", "-c", "import os,time; print(os.getpid(),flush=True); time.sleep(30)"]
        result = self.invoke(self.commands(first=child))
        self.assertEqual(result, 1)
        manifest = self.manifest()
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual([stage["status"] for stage in manifest["stages"]], ["failed", "pending", "pending"])
        self.assertFalse((self.output / "SUCCESS").exists())
        pid = int((self.output / "logs/prepare.log").read_text().strip())
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)
        entry = json.loads(self.ledger.read_text())["entries"][0]
        self.assertEqual(entry["status"], "finished")
        self.assertGreaterEqual(entry["elapsed_seconds"], 0.9)

    def test_real_nonzero_stage_preserves_log_and_stops_downstream(self):
        child = [sys.executable, "-B", "-c", "print('retained failure log',flush=True); raise SystemExit(19)"]
        self.assertEqual(self.invoke(self.commands(first=child)), 19)
        manifest = self.manifest()
        self.assertEqual(manifest["stages"][0]["exit_code"], 19)
        self.assertEqual([stage["status"] for stage in manifest["stages"]], ["failed", "pending", "pending"])
        self.assertIn("retained failure log", (self.output / "logs/prepare.log").read_text())
        self.assertFalse((self.output / "SUCCESS").exists())
        self.assertEqual(json.loads(self.ledger.read_text())["entries"][0]["status"], "finished")

    def test_exhausted_persistent_budget_refuses_before_subprocess_or_output(self):
        budget = runner.NightBudget(self.ledger, stop_by=runner.STOP_BY.isoformat(), max_seconds=14400)
        self.assertEqual(budget.acquire(14400), 14400)
        budget._release()  # An interrupted owner leaves its complete debit.
        before = self.ledger.read_bytes()
        with mock.patch.object(runner.subprocess, "Popen", wraps=subprocess.Popen) as spawn:
            with self.assertRaisesRegex(ValueError, "budget exhausted"):
                self.invoke()
            spawn.assert_not_called()
        self.assertEqual(self.ledger.read_bytes(), before)
        self.assertFalse(self.output.exists())

    def test_active_lease_refuses_second_runner_without_reset(self):
        budget = runner.NightBudget(self.ledger, stop_by=runner.STOP_BY.isoformat(), max_seconds=14400)
        budget.acquire(60)
        self.addCleanup(budget._release)
        before = self.ledger.read_bytes()
        with mock.patch.object(runner.subprocess, "Popen", wraps=subprocess.Popen) as spawn:
            with self.assertRaisesRegex(ValueError, "active lease"):
                self.invoke()
            spawn.assert_not_called()
        self.assertEqual(self.ledger.read_bytes(), before)
        self.assertFalse(self.output.exists())

    def test_child_keeps_parent_process_group(self):
        child = [sys.executable, "-B", "-c", "import os; print(os.getpgrp(),flush=True)"]
        self.assertEqual(self.invoke(self.commands(first=child)), 0)
        self.assertEqual(int((self.output / "logs/prepare.log").read_text()), os.getpgrp())

    def test_commands_and_environments_keep_labels_out_of_prepare_and_score(self):
        paths = {name: Path(getattr(self.args, name)) for name in
                 ("baseline_run", "train_inputs", "labels", "model_path", "protocol")}
        commands = runner.stage_commands(paths, self.output, self.args.device)
        for name, command in commands:
            self.assertEqual(command[:3], [sys.executable, "-B", "-u"])
            environment = runner.stage_environment(name)
            self.assertEqual(environment["HF_HUB_OFFLINE"], "1")
            for variable in runner.THREAD_VARIABLES:
                self.assertEqual(environment[variable], "4")
            if name != "analyze":
                self.assertNotIn("--labels", command)
                self.assertNotIn(self.args.labels, command)
            else:
                self.assertIn("--labels", command)
                self.assertIn("--analysis-output", command)
            if name == "score":
                self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], os.environ["CUDA_VISIBLE_DEVICES"])
                self.assertNotIn("USE_TORCH", environment)
                self.assertNotIn("USE_TF", environment)
            else:
                self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "")
                self.assertEqual(environment["USE_TORCH"], "0")
                self.assertEqual(environment["USE_TF"], "0")

    def test_no_reviewed_launcher_marker_refuses_before_budget(self):
        with mock.patch.dict(os.environ, {"STUDENT_SIM_CONTROLLED_PROCESS_GROUP": "0"}):
            with self.assertRaisesRegex(ValueError, "reviewed external GNU timeout"):
                self.invoke()
        self.assertFalse(self.ledger.exists())
        self.assertFalse(self.output.exists())

    def test_expired_fixed_authorization_refuses_before_budget(self):
        value = datetime.now(timezone.utc) - timedelta(seconds=1)
        data = json.loads(Path(self.args.protocol).read_text())
        data["night_resource_authorization"]["stop_by"] = value.isoformat()
        Path(self.args.protocol).write_text(json.dumps(data))
        with mock.patch.object(runner, "STOP_BY", value):
            with self.assertRaisesRegex(ValueError, "authorization has expired"):
                self.invoke()
        self.assertFalse(self.ledger.exists())
        self.assertFalse(self.output.exists())

    def test_protocol_cannot_enable_generation_or_change_fixed_deadline(self):
        data = json.loads(Path(self.args.protocol).read_text())
        data["new_generation"] = True
        Path(self.args.protocol).write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "Unsupported frozen"):
            runner.read_protocol(self.args.protocol)
        data["new_generation"] = False
        data["night_resource_authorization"]["stop_by"] = (runner.STOP_BY + timedelta(minutes=1)).isoformat()
        Path(self.args.protocol).write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "fixed 16:00"):
            runner.read_protocol(self.args.protocol)

    def test_competing_output_creator_is_preserved_without_manifest_overwrite(self):
        original_mkdir = Path.mkdir
        original = b'{"preserve":"another creator"}\n'

        def concurrent_creator(path, *args, **kwargs):
            if path == self.output:
                original_mkdir(path, *args, **kwargs)
                (path / "manifest.json").write_bytes(original)
                raise FileExistsError("Another creator won the directory")
            return original_mkdir(path, *args, **kwargs)

        with mock.patch.object(Path, "mkdir", autospec=True, side_effect=concurrent_creator), \
                mock.patch.object(runner.subprocess, "Popen", wraps=subprocess.Popen) as spawn:
            self.assertEqual(self.invoke(), 1)
            spawn.assert_not_called()
        self.assertEqual((self.output / "manifest.json").read_bytes(), original)
        self.assertFalse((self.output / "SUCCESS").exists())


if __name__ == "__main__":
    unittest.main()
