#!/usr/bin/env python3
"""Run one frozen inference -> prediction -> static evaluation experiment.

Use an externally approved device/CUDA_VISIBLE_DEVICES and an outer timeout or
tmux as needed. No GPU selection, downloads, installation, resume or code execution.
SUCCESS is written only after all seven subprocesses exit successfully.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


SRC = Path(__file__).resolve().parents[1] / "src"
METHODS = ("base", "history_d0", "cd", "b", "copy")
INFERENCE_OPTIONS = {
    "model_id": (str, "Qwen/Qwen2.5-Coder-7B-Instruct"), "num_generations": (int, 4),
    "max_new_tokens": (int, 1024), "max_context_tokens": (int, 8192),
    "temperature": (float, 0.8), "top_p": (float, 0.95), "seed": (int, 17),
    "dtype": (str, "bfloat16"), "fence_policy": (str, "preserve"),
}


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def save_manifest(path, manifest):
    encoded = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    temporary = path.with_suffix(".next.json")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
    os.replace(temporary, path)  # Only this run's own status manifest is updated.


def run(args):
    output = Path(args.output).resolve()
    if output.exists() or Path(args.output).is_symlink():
        raise ValueError("Output must be a new directory; no overwrite or resume")
    if (min(args.num_generations, args.max_new_tokens, args.bootstrap) < 1
            or args.max_new_tokens >= args.max_context_tokens or not math.isfinite(args.temperature)
            or args.temperature <= 0 or not math.isfinite(args.top_p) or not 0 < args.top_p <= 1):
        raise ValueError("Invalid frozen inference or bootstrap configuration")
    paths = {name: Path(getattr(args, name)).resolve(strict=True)
             for name in ("inputs", "labels", "model_path") + (("references",) if args.references else ())}
    if not paths["model_path"].is_dir() or any(not path.is_file() for name, path in paths.items() if name != "model_path"):
        raise ValueError("Inputs/references/labels must be files and model-path must be a directory")
    python = [sys.executable, "-B", "-u", "-m"]
    infer = python + ["student_sim_cd.inference", "--inputs", str(paths["inputs"]),
                      "--model-path", str(paths["model_path"]),
                      "--output", str(output / "inference"), "--device", args.device]
    if args.references:
        infer += ["--references", str(paths["references"])]
    for name in INFERENCE_OPTIONS:
        infer += ["--" + name.replace("_", "-"), str(getattr(args, name))]
    commands = [("inference", infer), ("predict", python + ["student_sim_cd.predict",
        "--run-dir", str(output / "inference"), "--inputs", str(paths["inputs"]),
        "--output", str(output / "predictions"), "--include-copy"])]
    for method in METHODS:
        commands.append(("evaluate-" + method, python + ["student_sim_cd.evaluate", "evaluate",
            "--inputs", str(paths["inputs"]), "--labels", str(paths["labels"]),
            "--predictions", str(output / "predictions" / ("predictions." + method + ".jsonl")),
            "--output", str(output / "evaluation" / method), "--bootstrap", str(args.bootstrap)]))
    environment = dict(os.environ, PYTHONPATH=str(SRC), PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    manifest = {
        "schema_version": "student-sim-cd.experiment.v1", "status": "running", "started_at": timestamp(),
        "python": sys.executable, "source_root": str(SRC.parent), "working_directory": str(SRC),
        "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES"),
        "reference_mode": "explicit_references" if args.references else "empty_reference_diagnostic",
        "configuration": {**{name: getattr(args, name) for name in INFERENCE_OPTIONS},
                          **{name: str(path) for name, path in paths.items()},
                          "device": args.device, "bootstrap": args.bootstrap},
        "source_sha256": {name: sha256(path) for name, path in paths.items() if name in ("inputs", "references")},
        "implementation_sha256": {path.name: sha256(path) for path in
            [Path(__file__), *[SRC / "student_sim_cd" / (name + ".py") for name in ("inference", "predict", "evaluate", "scoring")]]},
        "stages": [{"name": name, "command": command, "status": "pending", "exit_code": None,
                    "log": "logs/" + name + ".log"} for name, command in commands],
    }
    output.mkdir(parents=True)
    (output / "logs").mkdir()
    manifest_path, active, exit_code = output / "manifest.json", None, 0
    try:
        save_manifest(manifest_path, manifest)
        for active in manifest["stages"]:
            active.update(status="running", started_at=timestamp())
            save_manifest(manifest_path, manifest)
            print("Starting " + active["name"], flush=True)
            started = time.monotonic()
            with (output / active["log"]).open("x", encoding="utf-8") as log:
                result = subprocess.run(active["command"], cwd=str(SRC), env=environment,
                                        stdout=log, stderr=subprocess.STDOUT, check=False)
            active.update(exit_code=result.returncode, finished_at=timestamp(),
                          elapsed_seconds=round(time.monotonic() - started, 3),
                          status="success" if result.returncode == 0 else "failed")
            if result.returncode:
                exit_code = result.returncode if result.returncode > 0 else 128 - result.returncode
                raise RuntimeError(active["name"] + " failed; inspect " + active["log"])
        manifest.update(status="success", exit_code=0, finished_at=timestamp())
        save_manifest(manifest_path, manifest)
        with (output / "SUCCESS").open("x", encoding="utf-8") as stream:
            stream.write(sha256(manifest_path) + "\n")
    except (Exception, KeyboardInterrupt) as exc:
        exit_code = 130 if isinstance(exc, KeyboardInterrupt) else (exit_code or 1)
        if active is not None and active["status"] == "running":
            active.update(status="failed", finished_at=timestamp(), error=str(exc))
        manifest.update(status="failed", exit_code=exit_code, finished_at=timestamp(), error=str(exc))
        save_manifest(manifest_path, manifest)
        print(str(exc) or "Interrupted", file=sys.stderr, flush=True)
    return exit_code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "labels", "model-path", "output", "device"):
        parser.add_argument("--" + name, required=True)
    reference = parser.add_mutually_exclusive_group(required=True)
    reference.add_argument("--references")
    reference.add_argument("--empty-reference-diagnostic", action="store_true")
    for name, (kind, default) in INFERENCE_OPTIONS.items():
        choices = {"dtype": ("float32", "float16", "bfloat16"), "fence_policy": ("preserve", "unwrap-single")}.get(name)
        parser.add_argument("--" + name.replace("_", "-"), type=kind, default=default, choices=choices)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
