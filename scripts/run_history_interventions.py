#!/usr/bin/env python3
"""Run one bounded, past-only history intervention on an immutable cache.

The reviewed launcher must supply a fresh GNU timeout process group, a single
idle GPU, four CPU cores, and STUDENT_SIM_CONTROLLED_PROCESS_GROUP=1. Children
remain in that group: they never detach from the launcher's hard deadline.
Only the parent holds one shared budget lease. Prepare/score receive no labels;
training text stays on the server and is never printed by this orchestrator.
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from run_experiment import save_manifest, sha256, timestamp


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
from student_sim_cd.night_budget import NightBudget
from student_sim_cd.predict import _json, METHOD_PARAMETERS

DEFAULT_PROTOCOL = ROOT / "configs/history_interventions_v1.json"
STOP_BY = datetime.fromisoformat("2026-10-02T16:00:00+08:00")
MAX_SECONDS = 14400
LEDGER_NAME = "history-intervention-budget-20261002.json"
METHODS = ["base", "history_d0", "cd", "b", "copy"]
VARIANTS = ["real_history_swapped", "reference_swapped"]
SEEDS = [20261002, 20261003, 20261004]
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def read_protocol(path):
    protocol = _json(Path(path).read_bytes())
    expected = {
        "schema_version": "student-sim-cd.history-intervention.protocol.v1",
        "protocol": "history-interventions-v1", "new_generation": False,
        "student_execution": False, "expected_samples": 70,
        "expected_students": 17, "expected_candidates": 318,
        "methods": METHODS, "variants": VARIANTS, "seeds": SEEDS,
        "primary_metric": "equal-student macro edit_location_f1",
        "primary_comparison": "(B-base)_baseline - mean_seed(B-base)_real_history_swapped",
        "secondary_comparison": "(B-base)_baseline - mean_seed(B-base)_reference_swapped",
        "fixed_method_parameters": METHOD_PARAMETERS,
        "selection": "argmax; ties use lexicographically smallest candidate_id",
        "sequence_protocol": "canonical code tokens + exactly one tokenizer EOS; sum logp; no length normalization",
    }
    if not isinstance(protocol, dict) or any(
            protocol.get(key) != value or type(protocol.get(key)) is not type(value)
            for key, value in expected.items()):
        raise ValueError("Unsupported frozen history intervention protocol")
    authorization = protocol.get("night_resource_authorization")
    limits = {"hosts": ["4090-02"], "max_gpu_count": 1, "max_cpu_threads": 4,
              "max_total_gpu_wall_seconds": MAX_SECONDS,
              "requires_live_idle_and_lab_rule_check": True}
    if not isinstance(authorization, dict) or any(
            authorization.get(key) != value or type(authorization.get(key)) is not type(value)
            for key, value in limits.items()):
        raise ValueError("History intervention resource authorization differs")
    value = authorization.get("stop_by")
    if not isinstance(value, str):
        raise ValueError("A fixed timezone-aware history intervention deadline is required")
    stop_by = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stop_by.utcoffset() is None or stop_by != STOP_BY:
        raise ValueError("History intervention deadline must be the fixed 16:00 authorization")
    return protocol


def controlled_resources():
    if os.environ.get("STUDENT_SIM_CONTROLLED_PROCESS_GROUP") != "1":
        raise ValueError("Use the reviewed external GNU timeout process-group launcher")
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", gpu) is None:
        raise ValueError("Exactly one explicit GPU UUID must be supplied by the reviewed launcher")
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    if affinity is not None and len(affinity) != 4:
        raise ValueError("The reviewed launcher must restrict CPU affinity to four cores")
    return {"cpu_affinity": affinity, "process_group": os.getpgrp(),
            "pid": os.getpid(), "cuda_visible_devices": gpu,
            "cpu_threads": 4, "controlled_process_group": True,
            "hard_timeout": "external GNU timeout --signal=KILL; children do not detach"}


def canonical_existing(value, *, directory=False):
    path = Path(value).absolute()
    if path.is_symlink() or path.resolve(strict=True) != path:
        raise ValueError("Paths must be canonical and contain no symlinks")
    if not (path.is_dir() if directory else path.is_file()):
        raise ValueError("Expected an existing " + ("directory" if directory else "regular file"))
    return path


def stage_commands(paths, output, device):
    driver = [sys.executable, "-B", "-u", str(Path(__file__).resolve())]
    common = []
    for name in ("baseline_run", "train_inputs", "model_path", "protocol"):
        common.extend(["--" + name.replace("_", "-"), str(paths[name])])
    common.extend(["--output", str(output), "--prepared", str(output / "prepared"),
                   "--device", device])
    return [(name, driver + [name, *common] +
             (["--labels", str(paths["labels"]), "--analysis-output", str(output / "analysis.json")]
              if name == "analyze" else [])) for name in ("prepare", "score", "analyze")]


def stage_environment(name):
    environment = dict(os.environ, PYTHONPATH=str(SRC), PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
        HF_HUB_DISABLE_IMPLICIT_TOKEN="1", TOKENIZERS_PARALLELISM="false")
    environment.update({key: "4" for key in THREAD_VARIABLES})
    # Tokenizer imports occur in a separate CPU-only child, so USE_TORCH=0
    # cannot poison Transformers' backend availability in the scoring child.
    if name == "score":
        environment.pop("USE_TORCH", None)
        environment.pop("USE_TF", None)
    else:
        environment.update(USE_TORCH="0", USE_TF="0")
        environment["CUDA_VISIBLE_DEVICES"] = ""
    return environment


def stop_and_reap(process):
    if process.poll() is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    return process.wait()


def run(args):
    resources = controlled_resources()
    if args.device != "cuda:0":
        raise ValueError("The controlled run uses cuda:0 for its one explicit GPU UUID")
    if type(args.wall_seconds) is not int or not 1 <= args.wall_seconds <= MAX_SECONDS:
        raise ValueError("Aggregate wall time must be between 1 and 14400 seconds")
    paths = {name: canonical_existing(getattr(args, name), directory=name in {"baseline_run", "model_path"})
             for name in ("baseline_run", "train_inputs", "labels", "model_path", "protocol")}
    protocol = read_protocol(paths["protocol"])
    remaining_wall = min(args.wall_seconds, STOP_BY.timestamp() - time.time())
    if remaining_wall <= 0:
        raise ValueError("History intervention authorization has expired")
    output = Path(args.output).absolute()
    if (output.exists() or output.is_symlink() or output.resolve() != output
            or output.parent != paths["baseline_run"].parent):
        raise ValueError("Output must be a new canonical sibling of the immutable baseline run")
    commands = stage_commands(paths, output, args.device)
    source_hashes = {name: sha256(path) for name, path in paths.items() if path.is_file()}
    implementation_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in
        [Path(__file__), ROOT / "scripts/run_experiment.py", *sorted((SRC / "student_sim_cd").glob("*.py"))]}
    ledger = paths["baseline_run"].parent / LEDGER_NAME
    budget = NightBudget(ledger, stop_by=STOP_BY.isoformat(), max_seconds=MAX_SECONDS)
    lease_started = time.monotonic()
    allowed = budget.acquire(min(args.wall_seconds, STOP_BY.timestamp() - time.time()))
    deadline = lease_started + allowed
    manifest = {"schema_version": "student-sim-cd.history-intervention.experiment.v1",
        "status": "running", "started_at": timestamp(), "python": sys.executable,
        "source_root": str(ROOT), "protocol": protocol, "source_sha256": source_hashes,
        "implementation_sha256": implementation_hashes,
        "configuration": {**{key: str(path) for key, path in paths.items()},
            "output": str(output), "prepared": str(output / "prepared"), "device": args.device,
            "wall_seconds": allowed, "requested_wall_seconds": args.wall_seconds,
            "stop_by": STOP_BY.isoformat(), "budget_ledger": str(ledger)},
        "resources": resources, "generation_performed": False, "student_execution": False,
        "training_text_exported": False,
        "stages": [{"name": name, "command": command, "status": "pending", "exit_code": None,
                    "log": "logs/" + name + ".log"} for name, command in commands]}
    manifest_path, active, process, all_reaped, exit_code = output / "manifest.json", None, None, True, 0
    created_output = False

    def remaining_seconds():
        return min(deadline - time.monotonic(), STOP_BY.timestamp() - time.time())

    try:
        output.mkdir()
        created_output = True
        (output / "logs").mkdir()
        save_manifest(manifest_path, manifest)
        for active in manifest["stages"]:
            if remaining_seconds() <= 0:
                raise TimeoutError("Aggregate history intervention deadline reached")
            active.update(status="running", started_at=timestamp())
            save_manifest(manifest_path, manifest)
            print("Starting " + active["name"], flush=True)
            started = time.monotonic()
            with (output / active["log"]).open("xb") as log:
                remaining = remaining_seconds()
                if remaining <= 0:
                    raise TimeoutError("Deadline reached before history stage launch")
                process = subprocess.Popen(active["command"], cwd=str(SRC),
                    env=stage_environment(active["name"]), stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=False)
                all_reaped = False
                try:
                    returncode = process.wait(timeout=remaining)
                except BaseException:
                    stop_and_reap(process)
                    all_reaped = True
                    raise
                all_reaped = True
                process = None
            active.update(exit_code=returncode, finished_at=timestamp(),
                elapsed_seconds=round(time.monotonic() - started, 3),
                status="success" if returncode == 0 else "failed")
            save_manifest(manifest_path, manifest)
            if returncode:
                exit_code = returncode if returncode > 0 else 128 - returncode
                raise RuntimeError(active["name"] + " failed; inspect " + active["log"])
        if remaining_seconds() <= 0:
            raise TimeoutError("Deadline reached before completed-result verification")
        if any(sha256(paths[name]) != digest for name, digest in source_hashes.items()):
            raise ValueError("Frozen source/protocol bytes changed while running")
        manifest.update(status="success", exit_code=0, finished_at=timestamp())
        save_manifest(manifest_path, manifest)
        with (output / "SUCCESS").open("x", encoding="ascii") as stream:
            stream.write(sha256(manifest_path) + "\n")
    except (Exception, KeyboardInterrupt) as error:
        if process is not None and not all_reaped:
            stop_and_reap(process)
            all_reaped = True
        exit_code = 130 if isinstance(error, KeyboardInterrupt) else (exit_code or 1)
        if active is not None and active["status"] == "running":
            active.update(status="failed", finished_at=timestamp(), error=str(error))
        manifest.update(status="failed", exit_code=exit_code, finished_at=timestamp(),
                        error=type(error).__name__ + ": " + str(error))
        # A competing creator may have appeared after the initial absence
        # check. Failure to create our directory grants no right to its files.
        if created_output:
            save_manifest(manifest_path, manifest)
        print(str(error) or "Interrupted", file=sys.stderr, flush=True)
    finally:
        # If termination/reaping itself fails, retain the full reservation;
        # the external process-group timeout remains responsible for hard stop.
        if all_reaped:
            budget.finish()
    return exit_code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "prepare", "score", "analyze"):
        child = sub.add_parser(name)
        for field in ("baseline-run", "train-inputs", "model-path", "output", "device"):
            child.add_argument("--" + field, required=True)
        child.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
        if name in {"run", "analyze"}:
            child.add_argument("--labels", required=True)
        else:
            child.set_defaults(labels=None)
        if name != "run":
            child.add_argument("--prepared", required=True)
        if name == "analyze":
            child.add_argument("--analysis-output", required=True)
        else:
            child.set_defaults(analysis_output=None)
        if name == "run":
            child.add_argument("--wall-seconds", type=int, default=MAX_SECONDS)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            return run(args)
        # CPU children intentionally mask CUDA; the parent already validated
        # the one-GPU launcher. Never acquire a second budget lease here.
        if os.environ.get("STUDENT_SIM_CONTROLLED_PROCESS_GROUP") != "1":
            raise ValueError("Child stages require the reviewed controlled-process-group launcher")
        read_protocol(args.protocol)
        if STOP_BY.timestamp() <= time.time():
            raise ValueError("History intervention authorization has expired")
        from student_sim_cd import history_scoring
        result = getattr(history_scoring, args.command)(args)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
        return 0
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, "error: " + str(error) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
