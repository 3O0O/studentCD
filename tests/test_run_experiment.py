"""Fake subprocess tests: no inference, GPU, network or student execution."""

from contextlib import redirect_stderr, redirect_stdout
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


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_experiment.py"
specification = importlib.util.spec_from_file_location("run_experiment", SCRIPT)
runner = importlib.util.module_from_spec(specification)
specification.loader.exec_module(runner)


class ExperimentRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="experiment-runner-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.inputs, self.references, self.labels = [self.root / name for name in ("inputs.jsonl", "references.jsonl", "labels.jsonl")]
        # Subprocesses are fake; no field from these files is interpreted.
        for path in (self.inputs, self.references, self.labels):
            path.write_text('{"synthetic":true}\n')
        self.model, self.output = self.root / "model", self.root / "output"
        self.model.mkdir()
        self.arguments = ["--inputs", str(self.inputs), "--references", str(self.references),
                          "--labels", str(self.labels), "--model-path", str(self.model),
                          "--output", str(self.output), "--device", "cuda:0", "--num-generations", "2",
                          "--max-new-tokens", "128", "--max-context-tokens", "4096", "--bootstrap", "20"]

    def invoke(self, subprocess_effect=None, arguments=None):
        if subprocess_effect is None:
            def subprocess_effect(command, **kwargs):
                kwargs["stdout"].write("Synthetic stage log.\n")
                return subprocess.CompletedProcess(command, 0)
        with mock.patch.object(runner.subprocess, "run", side_effect=subprocess_effect) as calls, \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            code = runner.main(self.arguments if arguments is None else arguments)
        return code, calls.call_args_list

    def manifest(self):
        return json.loads((self.output / "manifest.json").read_text())

    def test_success_chain_keeps_labels_only_in_evaluation_and_records_logs(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "3", "PYTHONPATH": "/old/server/src"}):
            code, calls = self.invoke()
            self.assertEqual(os.environ["PYTHONPATH"], "/old/server/src")
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 7)
        for index, call in enumerate(calls):
            command = call.args[0]
            self.assertEqual(command[:4], [sys.executable, "-B", "-u", "-m"])
            self.assertEqual(call.kwargs["cwd"], str(runner.SRC))
            self.assertEqual(call.kwargs["env"]["PYTHONPATH"], str(runner.SRC))
            self.assertEqual(call.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "3")
            self.assertEqual(call.kwargs["env"]["HF_HUB_OFFLINE"], "1")
            self.assertNotIn("timeout", call.kwargs)
            if index < 2:
                self.assertNotIn("--labels", command)
                self.assertNotIn(str(self.labels), command)
            else:
                self.assertEqual(command[4:6], ["student_sim_cd.evaluate", "evaluate"])
                self.assertEqual(command[command.index("--labels") + 1], str(self.labels))
        self.assertIn("--include-copy", calls[1].args[0])
        manifest = self.manifest()
        self.assertEqual(manifest["status"], "success")
        self.assertTrue(all(stage["status"] == "success" and stage["exit_code"] == 0 for stage in manifest["stages"]))
        for stage in manifest["stages"]:
            self.assertEqual((self.output / stage["log"]).read_text(), "Synthetic stage log.\n")
        self.assertEqual((self.output / "SUCCESS").read_text().strip(), hashlib.sha256((self.output / "manifest.json").read_bytes()).hexdigest())

    def test_inference_and_evaluation_failures_stop_later_stages(self):
        for failure_stage in (0, 2):
            self.output = self.root / ("failed-%d" % failure_stage)
            arguments = list(self.arguments)
            arguments[arguments.index("--output") + 1] = str(self.output)
            counter = []

            def fail(command, **kwargs):
                index = len(counter)
                counter.append(command)
                return subprocess.CompletedProcess(command, 23 if index == failure_stage else 0)

            code, calls = self.invoke(fail, arguments)
            self.assertEqual(code, 23)
            self.assertEqual(len(calls), failure_stage + 1)
            manifest = self.manifest()
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["stages"][failure_stage]["exit_code"], 23)
            self.assertTrue(all(stage["status"] == "pending" for stage in manifest["stages"][failure_stage + 1:]))
            self.assertFalse((self.output / "SUCCESS").exists())

    def test_subprocess_exception_is_recorded_without_success_marker(self):
        code, calls = self.invoke(OSError("synthetic process launch failure"))
        self.assertEqual(code, 1)
        self.assertEqual(len(calls), 1)
        manifest = self.manifest()
        self.assertEqual(manifest["status"], "failed")
        self.assertEqual(manifest["stages"][0]["status"], "failed")
        self.assertIsNone(manifest["stages"][0]["exit_code"])
        self.assertIn("process launch failure", manifest["error"])
        self.assertFalse((self.output / "SUCCESS").exists())

    def test_existing_output_is_preserved(self):
        self.output.mkdir()
        existing = self.output / "existing.txt"
        existing.write_text("preserve this")
        with mock.patch.object(runner.subprocess, "run") as call, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                runner.main(self.arguments)
            self.assertEqual(error.exception.code, 2)
            call.assert_not_called()
        self.assertEqual(existing.read_text(), "preserve this")
        self.assertEqual(list(self.output.iterdir()), [existing])

    def test_relative_paths_resolve_before_switching_child_working_directory(self):
        arguments = list(self.arguments)
        for name in ("inputs", "references", "labels", "model-path", "output"):
            index = arguments.index("--" + name) + 1
            arguments[index] = Path(arguments[index]).name
        original = Path.cwd()
        try:
            os.chdir(self.root)
            code, calls = self.invoke(arguments=arguments)
        finally:
            os.chdir(original)
        self.assertEqual(code, 0)
        command = calls[0].args[0]
        for name, path in (("inputs", self.inputs), ("references", self.references), ("model-path", self.model)):
            self.assertEqual(command[command.index("--" + name) + 1], str(path))
        self.assertEqual(command[command.index("--output") + 1], str(self.output / "inference"))
        self.assertEqual(calls[0].kwargs["cwd"], str(runner.SRC))

    def test_empty_reference_mode_requires_explicit_diagnostic_choice(self):
        arguments = list(self.arguments)
        index = arguments.index("--references")
        del arguments[index:index + 2]
        with mock.patch.object(runner.subprocess, "run") as call, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                runner.main(arguments)
            call.assert_not_called()
        code, calls = self.invoke(arguments=arguments + ["--empty-reference-diagnostic"])
        self.assertEqual(code, 0)
        self.assertNotIn("--references", calls[0].args[0])
        self.assertEqual(self.manifest()["reference_mode"], "empty_reference_diagnostic")

    def test_invalid_nonfinite_configuration_does_not_create_output(self):
        with redirect_stderr(io.StringIO()), mock.patch.object(runner.subprocess, "run") as call:
            with self.assertRaises(SystemExit):
                runner.main(self.arguments + ["--temperature", "nan"])
            call.assert_not_called()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()
