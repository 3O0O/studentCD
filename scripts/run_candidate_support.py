"""Bounded fixed-cohort support/baseline suite; launch after explicit approval.

The reviewed external GNU timeout must own the whole process group. Child
stages never detach, and this entry never automatically resumes partial runs.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import pwd
import re
import socket
import stat
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
from student_sim_cd.night_budget import NightBudget

SERVER_ROOT = Path("/data/zzm110186486/projects/student-sim-cd")
DEFAULT_PROTOCOL = ROOT / "configs/candidate_support_v1.json"
MAX_SECONDS = 43200
LEDGER_NAME = "candidate-support-budget-20261002.json"
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON field")
        result[key] = value
    return result


def read_json(path):
    value = json.loads(Path(path).read_bytes(), object_pairs_hook=unique)
    json.dumps(value, allow_nan=False)
    return value


def canonical_existing(value, *, directory=False):
    path = Path(value).absolute()
    require(path.resolve(strict=True) == path and not path.is_symlink(), "Noncanonical or symlink input")
    info = path.lstat()
    require(info.st_uid == os.getuid() and (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)),
            "Input must be an owned canonical directory or regular file")
    return path


def deadline(value):
    require(isinstance(value, str), "An explicit timezone-aware stop-by is required")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.utcoffset() is not None and math.isfinite(parsed.timestamp()), "stop-by must have an explicit timezone")
    return parsed


def resource_arguments(args):
    require(type(args.max_seconds) is int and 1 <= args.max_seconds <= MAX_SECONDS,
            "Explicit aggregate max-seconds must be between 1 and 43200")
    require(type(args.wall_seconds) is int and 1 <= args.wall_seconds <= args.max_seconds,
            "Explicit wall-seconds must be positive and within aggregate max-seconds")
    stop = deadline(args.stop_by)
    require(stop.timestamp() > time.time(), "Explicit resource authorization has expired")
    return stop


def read_protocol(path):
    from student_sim_cd.prompt_baselines import PROMPTS
    value = read_json(path)
    expected = {"schema_version": "student-sim-cd.candidate-support.protocol.v1",
        "expected_samples": 70, "expected_students": 17, "seed": 20260929,
        "conditions": ["11", "01", "10", "00"], "balanced_random_per_condition": 3,
        "control_random_count": 12, "shared_greedy_condition": "11",
        "canonical_protocol": "cached-python-fence-v1", "student_execution": False,
        "generation_performed": True, "max_new_generation_attempts": 3080,
        "new_support_attempts": 1260, "new_prompt_baseline_attempts": 1820,
        "raw_output_policy": "preserved_before_cached_python_fence_v1"}
    require(isinstance(value, dict) and all(value.get(key) == expected_value
        and type(value.get(key)) is type(expected_value) for key, expected_value in expected.items()),
        "Unsupported frozen full-cohort candidate support protocol")
    require(value.get("test_set_used") is False, "Matched test is excluded from this development suite")
    require(isinstance(value.get("generation_config"), dict), "Frozen generation config is required")
    require(isinstance(value.get("prompt_baseline_texts"), dict), "Frozen baseline prompt texts are required")
    require(value["prompt_baseline_texts"] == PROMPTS, "Frozen baseline prompt texts differ from implementation")
    request = value.get("resource_request", {})
    require(isinstance(request, dict) and request.get("hosts") == ["4090-02"]
            and request.get("max_gpu_count") == 1 and request.get("max_cpu_threads") == 4
            and request.get("max_aggregate_wall_seconds") == MAX_SECONDS
            and request.get("requires_live_idle_and_lab_rule_check") is True,
            "Unsupported resource request; explicit reviewed launch approval is still required")
    return value


def server_identity(stage):
    """Validate fixed Linux account/root, existing interpreter and sealed release."""
    require(sys.platform == "linux" and socket.gethostname().split(".")[0] == "4090-02"
            and pwd.getpwuid(os.getuid()).pw_name == "zzm110186486", "Fixed 4090-02 Linux host/account required")
    canonical_existing(SERVER_ROOT, directory=True)
    require(Path(sys.prefix) == SERVER_ROOT / ".venv" and Path(sys.executable) == SERVER_ROOT / ".venv/bin/python",
            "Use the server's existing project interpreter")
    require(ROOT.parent == SERVER_ROOT / "releases" and re.fullmatch(r"[0-9a-f]{64}", ROOT.name),
            "Use a separately deployed content-hash release")
    canonical_existing(ROOT, directory=True)
    require(not (ROOT.lstat().st_mode & 0o222) and not (ROOT / ".incomplete").exists(), "Release must be sealed and complete")
    seal = canonical_existing(ROOT / ".deploy-manifest.json")
    require(sha256(seal) == ROOT.name, "Release manifest does not match its content-hash directory")
    expected_cwd = SERVER_ROOT if stage == "run" else SRC
    require(Path.cwd() == expected_cwd and Path.cwd().resolve() == expected_cwd, "Wrong canonical stage working directory")
    resources = controlled_resources(stage)
    return {**resources, "release_sha256": ROOT.name, "host": "4090-02", "user": "zzm110186486"}


def controlled_resources(stage):
    require(os.environ.get("STUDENT_SIM_CONTROLLED_PROCESS_GROUP") == "1", "Use the reviewed hard process-group launcher")
    affinity = sorted(os.sched_getaffinity(0))
    require(len(set(affinity)) == 4, "Exactly four CPU cores are required")
    for name in THREAD_VARIABLES:
        require(os.environ.get(name) == "4", "Four-thread environment setting differs")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_DISABLE_IMPLICIT_TOKEN"):
        require(os.environ.get(name) == "1", "Offline/no implicit authentication setting differs")
    require("HOME" not in os.environ, "Reviewed stage environment must not rewrite or inherit HOME")
    if stage in {"prepare", "analyze"}:
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == "" and os.environ.get("USE_TORCH") == "0"
                and os.environ.get("USE_TF") == "0", "CPU stage must mask model dependencies and CUDA")
    else:
        require(re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",
                             os.environ.get("CUDA_VISIBLE_DEVICES", "")), "Exactly one explicit GPU UUID is required")
    return {"cpu_affinity": affinity, "cpu_threads": 4, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pid": os.getpid(), "process_group": os.getpgrp(), "controlled_process_group": True,
        "hard_timeout": "external GNU timeout --signal=KILL; no detached children"}


def stage_environment(stage):
    # These values are set by the reviewed launcher; no credential/proxy/home
    # variable can enter a child through inherited shell or user configuration.
    allowed = {"PATH", "SHELL", "LANG", "LC_ALL", "TMPDIR", "XDG_CACHE_HOME", "HF_HOME", "HF_HUB_CACHE",
        "HF_TOKEN_PATH", "CUDA_VISIBLE_DEVICES", "CUDA_DEVICE_ORDER", "STUDENT_SIM_CONTROLLED_PROCESS_GROUP",
        "TOKENIZERS_PARALLELISM", "PYTHONDONTWRITEBYTECODE", "PYTHONNOUSERSITE", "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "HF_HUB_DISABLE_IMPLICIT_TOKEN",
        "WANDB_DISABLED", "WANDB_MODE", *THREAD_VARIABLES}
    env = {key: value for key, value in os.environ.items() if key in allowed}
    env.update(PYTHONPATH=str(SRC), PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
    for name in THREAD_VARIABLES:
        env[name] = "4"
    if stage in {"prepare", "analyze"}:
        env.update(CUDA_VISIBLE_DEVICES="", USE_TORCH="0", USE_TF="0")
    return env


def stage_commands(paths, output, args):
    common = ["--baseline-run", str(paths["baseline_run"]), "--model-path", str(paths["model_path"]),
        "--protocol", str(paths["protocol"]), "--output", str(output), "--prepared", str(output / "prepared"),
        "--device", args.device, "--stop-by", args.stop_by, "--max-seconds", str(args.max_seconds),
        "--wall-seconds", str(args.wall_seconds)]
    prefix = [sys.executable, "-B", "-u", str(Path(__file__).resolve())]
    return [(name, prefix + [name] + common + (["--labels", str(paths["labels"])] if name == "analyze" else []))
            for name in ("prepare", "experiment", "analyze")]


def save_manifest(path, value):
    encoded = (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    temporary = path.with_name(".manifest-" + str(os.getpid()) + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def stop_and_reap(process):
    if process.poll() is None:
        try:
            process.kill()
        except ProcessLookupError:
            pass
    return process.wait()


def run(args):
    stop = resource_arguments(args)
    resources = server_identity("run")
    require(args.device == "cuda:0", "The sole authorized GPU is visible as cuda:0")
    paths = {name: canonical_existing(getattr(args, name), directory=name in {"baseline_run", "model_path"})
             for name in ("baseline_run", "model_path", "protocol", "labels")}
    specification = read_protocol(paths["protocol"])
    require(paths["baseline_run"].parent == SERVER_ROOT / "outputs", "Baseline must be in the fixed server outputs directory")
    output = Path(args.output).absolute()
    require(output.parent == paths["baseline_run"].parent and output.resolve() == output
            and not output.exists() and not output.is_symlink(), "Use a new canonical result sibling; preserve prior runs")
    source_hashes = {name: sha256(path) for name, path in paths.items() if path.is_file()}
    source_hashes["baseline_manifest"] = sha256(paths["baseline_run"] / "manifest.json")
    require(source_hashes["labels"] == specification.get("labels_sha256"), "Labels differ from the frozen development source")
    require(source_hashes["baseline_manifest"] == specification.get("baseline_sha256", {}).get("manifest.json"),
            "Accepted baseline completion manifest changed")
    implementation = {str(path.relative_to(ROOT)): sha256(path) for path in
        [Path(__file__).resolve(), *sorted((SRC / "student_sim_cd").glob("*.py"))]}
    budget_path = paths["baseline_run"].parent / LEDGER_NAME
    budget = NightBudget(budget_path, stop_by=stop.isoformat(), max_seconds=args.max_seconds)
    lease_started = time.monotonic()
    allowed = budget.acquire(min(args.wall_seconds, stop.timestamp() - time.time()))
    end = lease_started + allowed
    manifest = {"schema_version": "student-sim-cd.candidate-support.experiment.v1", "status": "running",
        "started_at": stamp(), "python": sys.executable, "source_root": str(ROOT), "protocol": specification,
        "release_sha256": resources["release_sha256"], "source_sha256": source_hashes, "implementation_sha256": implementation,
        "configuration": {**{key: str(path) for key, path in paths.items()}, "output": str(output),
            "prepared": str(output / "prepared"), "device": args.device, "stop_by": stop.isoformat(),
            "max_seconds": args.max_seconds, "requested_wall_seconds": args.wall_seconds, "wall_seconds": allowed,
            "budget_ledger": str(budget_path), "automatic_resume": False}, "resources": resources,
        "generation_performed": True, "max_new_generation_attempts": 3080, "student_execution": False,
        "training_text_exported": False, "authorization_source": "explicit reviewed launcher parameters; approval retained in launch artifact",
        "stages": [{"name": name, "command": command, "status": "pending", "exit_code": None,
                    "log": "logs/" + name + ".log"} for name, command in stage_commands(paths, output, args)]}
    manifest_path, active, process, all_reaped, created, exit_code = output / "manifest.json", None, None, True, False, 0

    def remaining():
        return min(end - time.monotonic(), stop.timestamp() - time.time())

    try:
        output.mkdir()
        created = True
        (output / "logs").mkdir()
        save_manifest(manifest_path, manifest)
        for active in manifest["stages"]:
            require(remaining() > 0, "Aggregate deadline reached before stage")
            active.update(status="running", started_at=stamp())
            save_manifest(manifest_path, manifest)
            began = time.monotonic()
            with (output / active["log"]).open("xb") as log:
                left = remaining()
                if left <= 0:
                    raise TimeoutError("Aggregate deadline reached before child launch")
                process = subprocess.Popen(active["command"], cwd=str(SRC), env=stage_environment(active["name"]),
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=False)
                all_reaped = False
                try:
                    code = process.wait(timeout=left)
                except BaseException:
                    stop_and_reap(process)
                    all_reaped = True
                    raise
                all_reaped, process = True, None
            active.update(status="success" if code == 0 else "failed", exit_code=code,
                finished_at=stamp(), elapsed_seconds=round(time.monotonic() - began, 3))
            save_manifest(manifest_path, manifest)
            if code:
                exit_code = code if code > 0 else 128 - code
                raise RuntimeError(active["name"] + " failed; preserve its log and checkpoints")
        require(remaining() > 0, "Aggregate deadline reached before completion")
        require(all(sha256(paths[name]) == digest for name, digest in source_hashes.items() if name in paths),
                "Protocol or labels changed while running")
        require(sha256(paths["baseline_run"] / "manifest.json") == source_hashes["baseline_manifest"], "Baseline manifest changed")
        manifest.update(status="success", exit_code=0, finished_at=stamp())
        save_manifest(manifest_path, manifest)
        with (output / "SUCCESS").open("x", encoding="ascii") as stream:
            stream.write(sha256(manifest_path) + "\n")
    except (Exception, KeyboardInterrupt) as error:
        if process is not None and not all_reaped:
            stop_and_reap(process)
            all_reaped = True
        exit_code = 130 if isinstance(error, KeyboardInterrupt) else (exit_code or 1)
        if active is not None and active["status"] == "running":
            active.update(status="failed", finished_at=stamp(), error=type(error).__name__ + ": " + str(error))
        manifest.update(status="failed", exit_code=exit_code, finished_at=stamp(), error=type(error).__name__ + ": " + str(error))
        if created:
            save_manifest(manifest_path, manifest)
        print(str(error) or "Interrupted", file=sys.stderr, flush=True)
    finally:
        if all_reaped:
            budget.finish()
    return exit_code


def load_config(args):
    from student_sim_cd import inference
    specification = read_protocol(args.protocol)
    cached = read_json(Path(args.baseline_run) / "prepared/inference/manifest.json")["config"]
    require(specification["generation_config"] == cached, "Frozen config differs from accepted baseline")
    require(cached["fence_policy"] == "unwrap-single" and cached["device"] == args.device == "cuda:0"
            and cached["model_path"] == str(canonical_existing(args.model_path, directory=True)),
            "Only the existing fixed model/device/canonical extraction config is allowed")
    return inference.InferenceConfig(**cached)


def prepare(args):
    from student_sim_cd import candidate_support, preflight, prompt_baselines
    config = load_config(args)
    # An explicit tokenizer object is a synthetic-test injection in this
    # module. The formal path lets prepare load its own verified CPU tokenizer.
    result = candidate_support.prepare(args.baseline_run, args.prepared, protocol=Path(args.protocol), config=config)
    tokenizer = preflight.load_local_tokenizer(Path(args.model_path))
    bundle = candidate_support.load_prepared(args.prepared, config, tokenizer=tokenizer)
    prompts = prompt_baselines.audit(bundle["rows"], bundle["references"], tokenizer, config)
    require(len(prompts) == 140, "All seventy queries and both stronger prompts must pass CPU context audit")
    with (Path(args.output) / "strong-prompt-audit.json").open("x", encoding="utf-8") as stream:
        json.dump(prompts, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return {"support": result, "strong_prompt_audit_sha256": sha256(Path(args.output) / "strong-prompt-audit.json"),
            "strong_prompts_audited": len(prompts), "model_weights_loaded": False}


def idle_gpu():
    uuid = os.environ["CUDA_VISIBLE_DEVICES"]
    rows = list(csv.reader(io.StringIO(subprocess.check_output(["/usr/bin/nvidia-smi", "--id=" + uuid,
        "--query-gpu=uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True, timeout=15,
        stdin=subprocess.DEVNULL))))
    require(len(rows) == 1 and len(rows[0]) == 3 and rows[0][0].strip() == uuid, "Selected GPU is unavailable")
    memory, utilization = int(rows[0][1]), int(rows[0][2])
    apps = subprocess.check_output(["/usr/bin/nvidia-smi", "--id=" + uuid,
        "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True, timeout=15, stdin=subprocess.DEVNULL)
    require(0 <= memory <= 100 and utilization == 0 and not apps.strip(), "Selected GPU was occupied after preparation; stop without switching")
    return {"observed_at": stamp(), "uuid": uuid, "memory_used_mib": memory, "utilization_percent": utilization,
            "compute_apps": 0, "cpu_affinity": sorted(os.sched_getaffinity(0))}


def experiment(args):
    from student_sim_cd import candidate_support, inference, prompt_baselines
    config = load_config(args)
    bundle = candidate_support.load_prepared(args.prepared, config)
    require(inference.model_fingerprint(Path(args.model_path)) == bundle["baseline"][6]["model"], "Fixed model assets changed")
    observed = idle_gpu()
    with (Path(args.prepared) / "experiment-gpu-check.json").open("x", encoding="utf-8") as stream:
        json.dump(observed, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    require(torch.get_num_threads() == torch.get_num_interop_threads() == 4, "Torch thread caps differ")
    backend = inference.HFBackend(config)
    bundle = candidate_support.load_prepared(args.prepared, config, tokenizer=backend.tokenizer)
    prompts = prompt_baselines.audit(bundle["rows"], bundle["references"], backend.tokenizer, config)
    require(prompts == read_json(Path(args.output) / "strong-prompt-audit.json") and len(prompts) == 140,
            "CPU and loaded-tokenizer stronger prompt context audits differ; no draw is allowed")
    support = candidate_support.run(args.prepared, config, backend=backend, resume=False)
    baselines = prompt_baselines.run(bundle["rows"], bundle["references"], config, backend,
        Path(args.output) / "prompt-baselines", protocol_sha256=bundle["manifest"]["protocol_sha256"],
        model_identity=bundle["baseline"][6]["model"], resume=False)
    require(support.get("backend") == baselines.get("backend") == "HFBackend"
            and support.get("new_draws") == 1260 and support.get("samples") == baselines.get("samples") == 70
            and support.get("students") == 17 and baselines.get("attempts") == 1820,
            "The approved complete-cohort 3080-attempt generation suite is incomplete")
    return {"support": support, "prompt_baselines": baselines, "model_loaded_once": True, "labels_read": False}


def analyze(args):
    from student_sim_cd import candidate_support_analysis, prompt_baselines
    # Only the CPU analysis stage receives labels. Private prediction files
    # remain on the server; this specification contains paths and hashes.
    directory = Path(args.output) / "prompt-baselines"
    manifest_path = canonical_existing(directory / "manifest.json")
    manifest = read_json(manifest_path)
    protocol_hash = sha256(args.protocol)
    require((directory / "SUCCESS").read_text() == sha256(manifest_path) + "\n"
            and manifest.get("status") == "success" and manifest.get("samples") == 70
            and manifest.get("attempts") == 1820 and manifest.get("attempts_per_prompt") == 13
            and manifest.get("random_attempts") == 12 and manifest.get("labels_read") is False
            and manifest.get("student_execution") is False and manifest.get("protocol_sha256") == protocol_hash,
            "Stronger baseline completion/count/protocol markers differ")
    require(manifest.get("config") == read_protocol(args.protocol)["generation_config"], "Stronger baseline config changed")
    files = manifest.get("files_sha256")
    expected_files = {method + suffix + ".jsonl" for method in ("student_revision", "conservative_edit")
        for suffix in ("_greedy", "_pool")} | {"raw.jsonl", "scores.jsonl", "summary.json", "checkpoint-manifest.json", "reservations.jsonl"}
    require(isinstance(files, dict) and set(files) == expected_files, "Stronger baseline file inventory differs")
    require(all(sha256(canonical_existing(directory / name)) == digest for name, digest in files.items()),
            "Stronger baseline artifact changed after completion")
    require(manifest.get("backend") == "HFBackend", "Real stronger baseline backend provenance is required")
    support_manifest = read_json(Path(args.prepared) / "pools/balanced12/inference/manifest.json")
    require(manifest.get("model") == support_manifest.get("model")
            and manifest.get("runtime_versions") == support_manifest.get("runtime_versions")
            and manifest.get("implementation_sha256") == sha256(Path(prompt_baselines.__file__)),
            "Stronger baseline model/runtime/implementation provenance differs")
    external = []
    for method in sorted(name.removesuffix(".jsonl") for name in expected_files if name.endswith(("_greedy.jsonl", "_pool.jsonl"))):
        external.append({"method": method, "predictions_path": str(directory / (method + ".jsonl")),
            "provenance": {"kind": "prompt_greedy" if method.endswith("_greedy") else "prompt_pool_reranking",
                "prompt_name": method.rsplit("_", 1)[0], "protocol_sha256": protocol_hash,
                "prediction_sha256": files[method + ".jsonl"], "backend": manifest["backend"],
                "generation_count_per_sample": 13, "random_count_per_sample": 12,
                "labels_used": False, "parameters_fitted": False, "student_execution": False,
                "generation_manifest_sha256": sha256(manifest_path),
                "source_sha256": {"prompt_baselines": manifest["implementation_sha256"],
                                  "prepared_manifest": sha256(Path(args.prepared) / "manifest.json")}}})
    specification = Path(args.output) / "externals.json"
    with specification.open("x", encoding="utf-8") as stream:
        json.dump(external, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write("\n")
    output = Path(args.output) / "analysis"
    command = ["--balanced-dir", str(Path(args.prepared) / "pools/balanced12/inference"),
        "--control-dir", str(Path(args.prepared) / "pools/11_only12/inference"),
        "--inputs", str(Path(args.prepared) / "inputs.jsonl"), "--labels", str(args.labels),
        "--externals-json", str(specification), "--protocol", str(args.protocol), "--output", str(output)]
    require(candidate_support_analysis.main(command) == 0, "Static candidate support analysis failed")
    require((output / "SUCCESS").read_text() == sha256(output / "analysis.json") + "\n", "Analysis completion marker differs")
    verification = independent_verification(args)
    return {"analysis": str(output / "analysis.json"), "analysis_sha256": sha256(output / "analysis.json"),
            "independent_verification": str(verification), "independent_verification_sha256": sha256(verification),
            "student_execution": False, "labels_read": True}


def independent_verification(args):
    # Keep the independent stdlib implementation in this CPU child. A nested
    # subprocess could outlive the direct child at a partially spent budget's
    # inner deadline; no grandchildren are created here.
    path = canonical_existing(ROOT / "scripts/verify_candidate_support_results.py")
    name = "candidate_support_independent_result_verifier"
    specification = importlib.util.spec_from_file_location(name, path)
    require(specification is not None and specification.loader is not None, "Independent verifier is unavailable")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
        result = module.verify(Path(args.output), Path(args.labels))
    finally:
        sys.modules.pop(name, None)
    require(isinstance(result, dict) and result.get("status") == "verified" and result.get("verified") is True,
            "Independent result verification failed")
    output = Path(args.output) / "independent-verification.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return output


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    commands = value.add_subparsers(dest="command", required=True)
    for name in ("run", "prepare", "experiment", "analyze"):
        child = commands.add_parser(name)
        for field in ("baseline-run", "model-path", "output", "device"):
            child.add_argument("--" + field, required=True)
        child.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
        child.add_argument("--stop-by", required=True)
        child.add_argument("--max-seconds", required=True, type=int)
        child.add_argument("--wall-seconds", required=True, type=int)
        if name in {"run", "analyze"}:
            child.add_argument("--labels", required=True)
        else:
            child.set_defaults(labels=None)
        if name != "run":
            child.add_argument("--prepared", required=True)
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "run":
            return run(args)
        resource_arguments(args)
        server_identity(args.command)
        read_protocol(args.protocol)
        result = globals()[args.command](args)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
        return 0
    except (ValueError, OSError, KeyError) as error:
        print(type(error).__name__ + ": " + str(error), file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
