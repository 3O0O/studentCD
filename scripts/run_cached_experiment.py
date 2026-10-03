#!/usr/bin/env python3
"""Rescore an audited completed cache, then compare without executing code.

Each inference/selection subprocess receives no labels. Use only an externally
approved GPU and an existing model; this script never selects GPUs or downloads.
"""

import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_experiment import save_manifest, sha256, timestamp


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
from student_sim_cd.night_budget import NightBudget
METHODS = ("base", "history_d0", "cd", "b", "copy", "greedy")
DEFAULT_PROTOCOL = ROOT / "configs/cache_rescore_v1.json"


def read_protocol(path, source_run=None, *, preparation=None, prepared_dir=None):
    from student_sim_cd import cache_rescore, inference, predict
    from student_sim_cd.predict import _json
    protocol = _json(Path(path).read_bytes())
    if (protocol.get("schema_version") != "student-sim-cd.cached-experiment.protocol.v1"
            or protocol.get("protocol") != "cached-python-fence-v1"
            or protocol.get("new_generation") is not False
            or protocol.get("methods") != list(METHODS)):
        raise ValueError("Unsupported frozen cached-experiment protocol")
    for key in ("expected_samples", "expected_students"):
        if type(protocol.get(key)) is not int or protocol[key] < 1:
            raise ValueError("Invalid expected cohort size")
    settings = protocol.get("bootstrap", {})
    if (type(settings.get("repetitions")) is not int or settings["repetitions"] < 1
            or type(settings.get("seed")) is not int
            or settings.get("paired") is not True or settings.get("unit") != "student"):
        raise ValueError("A fixed paired student bootstrap is required")
    if protocol.get("fixed_method_parameters") != predict.METHOD_PARAMETERS:
        raise ValueError("Frozen method parameters differ from implemented fixed scoring")
    if protocol.get("selection") != "argmax; ties use lexicographically smallest candidate_id":
        raise ValueError("Frozen selection/tie protocol differs from implemented prediction")
    sequence = "canonical code tokens + exactly one tokenizer EOS; sum logp; no length normalization"
    if protocol.get("sequence_protocol") != sequence:
        raise ValueError("Frozen sequence/EOS protocol differs from implemented scoring")
    if preparation is not None and (source_run is None or prepared_dir is None):
        raise ValueError("Preparation provenance requires explicit source_run and prepared_dir")
    if source_run is not None:
        import hashlib
        source_run = Path(source_run).resolve(strict=True)
        files = {"manifest": source_run / "manifest.json",
                 "completion_manifest": source_run.parent / "manifest.json",
                 "candidates": source_run / "candidates.jsonl", "scores": source_run / "scores.jsonl"}
        keys = {"manifest": "source_inference_manifest_sha256",
                "completion_manifest": "source_experiment_manifest_sha256",
                "candidates": "source_candidates_sha256", "scores": "source_scores_sha256"}
        source_bytes = {name: file.read_bytes() for name, file in files.items()}
        for name, file in files.items():
            if hashlib.sha256(source_bytes[name]).hexdigest() != protocol.get(keys[name]):
                raise ValueError("Source manifest differs from frozen protocol: " + str(file))
        source_manifest = _json(source_bytes["manifest"])
        source_config = source_manifest["config"]
        for name, value in protocol["inference"].items():
            if name != "fence_policy" and source_config.get(name) != value:
                raise ValueError("Frozen source configuration mismatch: " + name)
        if preparation is not None:
            prepared_dir = Path(prepared_dir).resolve(strict=True)
            stored_hashes = preparation.get("source_sha256", {})
            for name, file in files.items():
                if (stored_hashes.get(name) != protocol[keys[name]]
                        or preparation.get("source_paths", {}).get(name) != str(file)):
                    raise ValueError("Prepared source fingerprint/path differs from frozen source: " + name)
            if (preparation.get("source_manifest") != source_manifest
                    or preparation.get("source_run_sha256") != inference.object_hash(source_manifest)
                    or preparation.get("source_completion_manifest") != _json(source_bytes["completion_manifest"])):
                raise ValueError("Prepared source manifests differ from the frozen completed experiment")
            source_bytes["success"] = (source_run.parent / "SUCCESS").read_bytes()
            if (stored_hashes.get("success") != hashlib.sha256(source_bytes["success"]).hexdigest()
                    or preparation.get("source_paths", {}).get("success") != str(source_run.parent / "SUCCESS")):
                raise ValueError("Prepared source SUCCESS fingerprint/path differs")
            cache_rescore._completion(source_bytes)
            for name, manifest_key in (("inputs", "inputs_sha256"), ("references", "references_sha256")):
                expected_hash = source_manifest.get(manifest_key)
                if stored_hashes.get(name) != expected_hash:
                    raise ValueError("Prepared " + name + " fingerprint differs from frozen inference")
                if name == "references" and expected_hash is None:
                    continue
                snapshot = prepared_dir / (name + ".jsonl")
                if sha256(snapshot) != expected_hash:
                    raise ValueError("Prepared " + name + " snapshot differs from frozen inference")
            source_bytes["inputs"] = (prepared_dir / "inputs.jsonl").read_bytes()
            _, _, old_pools, _, _, _ = predict._validated_run(source_bytes)
            original = {record["sample_id"]: record for record in predict._records(source_bytes["candidates"], "candidates")}
            rows = predict._records(source_bytes["inputs"], "inputs")
            expected_records = [cache_rescore._prepared_sample(row, original[row["sample_id"]],
                                                              old_pools[row["sample_id"]], source_config)
                                for row in rows]
            if inference.object_hash(expected_records) != preparation.get("records_payload_sha256"):
                raise ValueError("Prepared candidates/mappings do not reproduce the frozen original attempts")
            if (len(rows) != protocol["expected_samples"]
                    or len({row["student_id"] for row in rows}) != protocol["expected_students"]):
                raise ValueError("Prepared source does not retain the complete frozen cohort")
    return protocol


def prepare(args):
    from student_sim_cd.cache_rescore import prepare as prepare_cache
    from student_sim_cd.inference import load_inputs
    protocol = read_protocol(args.protocol, args.source_run)
    rows = load_inputs(Path(args.inputs))
    if (len(rows) != protocol["expected_samples"]
            or len({row["student_id"] for row in rows}) != protocol["expected_students"]):
        raise ValueError("Inputs do not contain the complete frozen cohort")
    return prepare_cache(args.source_run, args.inputs, args.references, args.output)


def score(args):
    from student_sim_cd.cache_rescore import _load_prepared, score as score_cache
    from student_sim_cd.inference import InferenceConfig
    from student_sim_cd.predict import _json
    preparation, _, _, _, _ = _load_prepared(Path(args.prepared))
    protocol = read_protocol(args.protocol, args.source_run,
                             preparation=preparation, prepared_dir=args.prepared)
    source_config = _json((Path(args.source_run) / "manifest.json").read_bytes())["config"]
    config = replace(InferenceConfig(**source_config), model_path=str(Path(args.model_path).resolve(strict=True)),
                     device=args.device, fence_policy=protocol["inference"]["fence_policy"])
    # Thread limits are explicit, and no model is imported on prepare/analysis paths.
    import torch
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
    return score_cache(args.prepared, config)


def greedy(args):
    from student_sim_cd.cache_rescore import export_greedy
    return export_greedy(args.prepared, args.output)


def compare(args):
    from student_sim_cd import cache_rescore, inference, predict, scoring
    from student_sim_cd.cache_rescore import _load_prepared
    from student_sim_cd.evaluate import pair_metrics
    from student_sim_cd.inference import read_jsonl
    from student_sim_cd.paired_analysis import analyze_predictions
    prepared_dir = Path(args.prepared).resolve(strict=True)
    preparation, preparation_hash, records, prepared_inputs, _ = _load_prepared(prepared_dir)
    source_run = Path(preparation["source_paths"]["manifest"]).parent
    protocol = read_protocol(args.protocol, source_run,
                             preparation=preparation, prepared_dir=prepared_dir)
    if sha256(Path(args.labels)) != protocol["labels_sha256"]:
        raise ValueError("Labels differ from the frozen evaluation file")
    score_dir, prediction_dir = prepared_dir / "inference", Path(args.predictions)
    sources = {"inputs": Path(args.inputs).read_bytes(),
               **{name: (score_dir / filename).read_bytes() for name, filename in
                  (("manifest", "manifest.json"), ("candidates", "candidates.jsonl"), ("scores", "scores.jsonl"))}}
    score_manifest = predict._json(sources["manifest"])
    evidence = score_manifest.get("cache_rescore", {})
    if (evidence.get("schema_version") != cache_rescore.SCHEMA_VERSION
            or evidence.get("extraction_protocol") != cache_rescore.EXTRACTION_PROTOCOL
            or evidence.get("generation_performed") is not False
            or evidence.get("old_scores_reused") is not False
            or evidence.get("prepared_run_sha256") != preparation_hash
            or evidence.get("prepared_records_sha256") != sha256(prepared_dir / cache_rescore.PREPARED_NAME)
            or evidence.get("source_run_sha256") != preparation["source_run_sha256"]
            or evidence.get("source_sha256") != preparation["source_sha256"]):
        raise ValueError("New score cache provenance does not match the frozen preparation/source")
    backend = evidence.get("backend")
    if backend not in ("HFBackend", "injected_synthetic_backend"):
        raise ValueError("Unknown scoring backend; cannot claim model-scoring evidence")
    if (evidence.get("implementation_sha256") != sha256(Path(cache_rescore.__file__))
            or score_manifest.get("implementation_sha256") != sha256(Path(inference.__file__))
            or score_manifest.get("scoring_sha256") != sha256(Path(scoring.__file__))):
        raise ValueError("Score implementation fingerprints differ from current reviewed code")
    original_manifest = preparation["source_manifest"]
    if score_manifest.get("model") != original_manifest["model"]:
        raise ValueError("New scoring model/tokenizer fingerprint differs from frozen source")
    current_config = score_manifest.get("config", {})
    if (current_config.get("fence_policy") != "unwrap-single"
            or any(current_config.get(name) != value for name, value in original_manifest["config"].items()
                   if name not in ("model_path", "device", "fence_policy"))
            or score_manifest.get("references_sha256") != original_manifest.get("references_sha256")
            or score_manifest.get("sequence_protocol") != protocol["sequence_protocol"]):
        raise ValueError("New scoring configuration/sequence/reference protocol differs")
    if backend == "HFBackend":
        versions = score_manifest.get("runtime_versions", {})
        if (versions != original_manifest.get("runtime_versions")
                or any(not isinstance(versions.get(name), str) or versions[name] == "not-installed"
                       for name in ("torch", "transformers", "tokenizers", "safetensors"))):
            raise ValueError("Real model-scoring runtime must match the installed frozen model environment")
    _, score_hash, pools, scores, copies, _ = predict._validated_run(sources)
    if score_manifest.get("inputs_sha256") != preparation["source_sha256"]["inputs"]:
        raise ValueError("Scored inputs differ from frozen preparation")
    if (prediction_dir / ".incomplete").exists() or (prediction_dir / ".incomplete").is_symlink():
        raise ValueError("Prediction export is incomplete; preserve it instead of accepting results")
    prediction_manifest = predict._json((prediction_dir / "manifest.json").read_bytes())
    if (prediction_manifest.get("schema_version") != predict.SCHEMA_VERSION
            or prediction_manifest.get("inference_manifest") != score_manifest
            or prediction_manifest.get("inference_run_sha256") != score_hash
            or prediction_manifest.get("method_parameters") != protocol["fixed_method_parameters"]
            or prediction_manifest.get("selection") != protocol["selection"]
            or prediction_manifest.get("common_candidate_pool") is not True
            or prediction_manifest.get("include_copy") is not True
            or prediction_manifest.get("labels_read") is not False
            or prediction_manifest.get("parameters_fitted") is not False
            or prediction_manifest.get("candidate_execution") is not False
            or prediction_manifest.get("samples") != protocol["expected_samples"]
            or prediction_manifest.get("candidates") != len(scores)):
        raise ValueError("Prediction manifest does not describe this complete new score cache")
    import hashlib
    source_hashes = {name: hashlib.sha256(content).hexdigest() for name, content in sources.items()}
    if prediction_manifest.get("source_sha256") != source_hashes:
        raise ValueError("Prediction source fingerprints differ from the new scored files")
    expected_outputs = {"predictions." + method + ".jsonl" for method in METHODS if method != "greedy"}
    expected_outputs.add("selections.jsonl")
    output_hashes = prediction_manifest.get("output_sha256", {})
    if set(output_hashes) != expected_outputs or any(sha256(prediction_dir / name) != checksum
                                                    for name, checksum in output_hashes.items()):
        raise ValueError("Prediction output SHA256 differs from completed prediction export")
    predictions = {method: predict._records((prediction_dir / ("predictions." + method + ".jsonl")).read_bytes(), method)
                   for method in METHODS}
    indexed = {}
    for method, rows in predictions.items():
        if (len(rows) != len(pools) or any(row.get("method") != method for row in rows)
                or {row.get("sample_id") for row in rows} != set(pools)):
            raise ValueError("Prediction method/cohort is incomplete or contains duplicates: " + method)
        indexed[method] = {row["sample_id"]: row for row in rows}
    for record in records:
        sample, pool = record["sample_id"], pools[record["sample_id"]]
        expected_pool = {candidate["candidate_id"]: candidate for candidate in record["candidates"]}
        if (set(pool) != set(expected_pool)
                or any(any(pool[key].get(field) != candidate[field] for field in ("code", "sources", "transform"))
                       for key, candidate in expected_pool.items())):
            raise ValueError("Scored candidate pool differs from prepared extraction/mappings")
        for method in METHODS:
            if method == "greedy":
                mappings = [mapping for mapping in record["mappings"] if mapping["attempt"] == 0]
                if (len(mappings) != 1 or mappings[0]["source"] != "greedy:0"
                        or not mappings[0]["eligible"] or mappings[0]["candidate_id"] not in pool):
                    raise ValueError("Actual original greedy has no eligible prepared/scored mapping")
                identifier = mappings[0]["candidate_id"]
            else:
                identifier = copies[sample] if method == "copy" else min(
                    pool, key=lambda candidate_id: (-scores[sample, candidate_id][method], candidate_id))
            if indexed[method][sample].get("predicted_code") != pool[identifier]["code"]:
                raise ValueError("Prediction differs from verified fixed selection or actual greedy: " + method)
    inputs, labels = read_jsonl(Path(args.inputs)), read_jsonl(Path(args.labels))
    result = analyze_predictions(
        inputs, labels, predictions,
        protocol=protocol["protocol"], expected_samples=protocol["expected_samples"],
        expected_students=protocol["expected_students"], comparisons=protocol["comparisons"],
        bootstrap=protocol["bootstrap"]["repetitions"], seed=protocol["bootstrap"]["seed"],
        method_kinds={"greedy": "original_greedy"})
    result["model_scoring_performed"] = backend == "HFBackend"
    result["scoring_backend"] = backend
    result["score_provenance"] = {"prepared_run_sha256": preparation_hash, "score_run_sha256": score_hash,
                                  "new_score_source_sha256": source_hashes,
                                  "prediction_manifest_sha256": sha256(prediction_dir / "manifest.json"),
                                  "original_greedy_mapping_verified": True}
    if backend == "injected_synthetic_backend":
        result["analysis_kind"] = "synthetic_" + result["analysis_kind"]
        result["limitations"].append("Injected synthetic backend: this report is offline protocol validation, not real model-scoring evidence.")
    if prepared_inputs != inputs:
        raise ValueError("Candidate diagnostics must use the same complete input file")
    by_sample = {row["sample_id"]: row for row in inputs}
    targets = {row["sample_id"]: row["target_code"] for row in labels}
    if {record["sample_id"] for record in records} != set(targets):
        raise ValueError("Candidate coverage differs from the complete evaluated cohort")
    per_sample = []
    for record in records:
        sample = record["sample_id"]
        values = [pair_metrics(by_sample[sample]["current_code"], item["code"], targets[sample])
                  for item in record["candidates"]]
        per_sample.append({"sample_id": sample, "student_id": by_sample[sample]["student_id"],
                           "candidates": len(values),
                           "observed_next_exact_in_pool": any(value["exact_next_code"] for value in values),
                           "oracle_static_edit_location_f1": max(value["edit_location_f1"] for value in values)})
    result["candidate_diagnostics"] = {
        "original_candidates": preparation["original_candidates"], "candidates": preparation["candidates"],
        "attempts": preparation["attempts"], "generation_performed": False,
        "extraction_reasons": dict(sorted(Counter(mapping["extraction_reason"] for record in records
                                             for mapping in record["mappings"]).items())),
        "copy_only_samples": sum(len(record["candidates"]) == 1 for record in records),
        "per_sample": per_sample,
        "labels_used_only_for_diagnostic": True,
        "limitation": "Observed-next text coverage and static oracle are evaluation-only; no correctness, execution or oracle selection."}
    result["protocol_sha256"] = sha256(Path(args.protocol))
    result["source_sha256"] = {"inputs": sha256(Path(args.inputs)), "labels": sha256(Path(args.labels)),
                               **{method: sha256(Path(args.predictions) / ("predictions." + method + ".jsonl"))
                                  for method in METHODS}}
    with Path(args.output).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return result


def run(args):
    output = Path(args.output).absolute()
    if output.exists() or output.is_symlink() or output.resolve() != output:
        raise ValueError("Output must be a new canonical directory; preserve existing results")
    if (type(args.wall_seconds) is not int or not 1 <= args.wall_seconds <= 14400):
        raise ValueError("Aggregate wall time must be between 1 and 14400 seconds")
    paths = {name: Path(getattr(args, name)).resolve(strict=True)
             for name in ("source_run", "inputs", "references", "labels", "model_path", "protocol")}
    protocol = read_protocol(paths["protocol"], paths["source_run"])
    original_experiment = paths["source_run"].parent
    if output == original_experiment or original_experiment in output.parents:
        raise ValueError("New output must be outside the original experiment; no writes to old results")
    if sha256(paths["labels"]) != protocol["labels_sha256"]:
        raise ValueError("Labels differ from the frozen evaluation file")
    stop_by = datetime.fromisoformat(protocol["night_resource_authorization"]["stop_by"])
    if stop_by.utcoffset() is None:
        raise ValueError("Night stop-by requires an explicit timezone")
    effective_wall = min(args.wall_seconds, stop_by.timestamp() - time.time())
    if effective_wall <= 0:
        raise ValueError("This night authorization has expired; do not start a new run")
    if not paths["source_run"].is_dir() or not paths["model_path"].is_dir():
        raise ValueError("Source inference run and existing model must be directories")
    if any(not paths[name].is_file() for name in ("inputs", "references", "labels", "protocol")):
        raise ValueError("Explicit input/reference/label/protocol files are required")
    driver = [sys.executable, "-B", "-u", str(Path(__file__).resolve())]
    module = [sys.executable, "-B", "-u", "-m"]
    common = ["--source-run", str(paths["source_run"]), "--protocol", str(paths["protocol"])]
    prepared = output / "prepared"
    commands = [
        ("prepare", driver + ["prepare", *common, "--inputs", str(paths["inputs"]),
                             "--references", str(paths["references"]), "--output", str(prepared)]),
        ("score", driver + ["score", *common, "--prepared", str(prepared),
                           "--model-path", str(paths["model_path"]), "--device", args.device]),
        ("predict", module + ["student_sim_cd.predict", "--run-dir", str(prepared / "inference"),
                             "--inputs", str(paths["inputs"]), "--output", str(output / "predictions"), "--include-copy"]),
        ("greedy", driver + ["greedy", "--prepared", str(prepared),
                             "--output", str(output / "predictions/predictions.greedy.jsonl")]),
    ]
    for method in METHODS:
        commands.append(("evaluate-" + method, module + ["student_sim_cd.evaluate", "evaluate",
            "--inputs", str(paths["inputs"]), "--labels", str(paths["labels"]),
            "--predictions", str(output / "predictions" / ("predictions." + method + ".jsonl")),
            "--output", str(output / "evaluation" / method),
            "--bootstrap", str(protocol["bootstrap"]["repetitions"])]))
    commands.append(("compare", driver + ["compare", "--inputs", str(paths["inputs"]),
        "--labels", str(paths["labels"]), "--predictions", str(output / "predictions"),
        "--protocol", str(paths["protocol"]), "--prepared", str(prepared),
        "--output", str(output / "paired-comparison.json")]))
    environment = dict(os.environ, PYTHONPATH=str(SRC), PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
        HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
        TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4")
    manifest = {"schema_version": "student-sim-cd.cached-experiment.v1", "status": "running",
        "started_at": timestamp(), "python": sys.executable, "source_root": str(ROOT),
        "configuration": {**{key: str(value) for key, value in paths.items()}, "device": args.device,
                          "wall_seconds": effective_wall, "requested_wall_seconds": args.wall_seconds,
                          "stop_by": stop_by.isoformat()},
        "source_sha256": {key: sha256(value) for key, value in paths.items() if value.is_file()},
        "implementation_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in
            [Path(__file__), ROOT / "scripts/run_experiment.py", *sorted((SRC / "student_sim_cd").glob("*.py"))]},
        "protocol": protocol, "cuda_visible_devices": environment.get("CUDA_VISIBLE_DEVICES"),
        "generation_performed": False, "student_execution": False,
        "stages": [{"name": name, "command": command, "status": "pending", "exit_code": None,
                    "log": "logs/" + name + ".log"} for name, command in commands]}
    budget_path = paths["source_run"].parents[1] / ("night-budget-" + stop_by.strftime("%Y%m%d") + ".json")
    budget = NightBudget(budget_path, stop_by=stop_by.isoformat(), max_seconds=14400)
    lease_started = time.monotonic()
    effective_wall = budget.acquire(effective_wall)
    manifest["configuration"].update(wall_seconds=effective_wall, budget_ledger=str(budget_path))
    try:
        output.mkdir(parents=True)
        (output / "logs").mkdir()
    except BaseException:
        budget.finish()
        raise
    manifest_path, active, exit_code = output / "manifest.json", None, 0
    # Reservation persistence and output preparation consume the same window;
    # adding the full grant to a later clock value would extend the deadline.
    deadline = lease_started + effective_wall

    def remaining_seconds():
        # Recheck wall time too, so a forward clock adjustment cannot permit a
        # child to begin after the absolute authorized stop-by timestamp.
        return min(deadline - time.monotonic(), stop_by.timestamp() - time.time())

    try:
        save_manifest(manifest_path, manifest)
        for active in manifest["stages"]:
            remaining = remaining_seconds()
            if remaining <= 0:
                raise TimeoutError("Aggregate experiment wall-time limit reached")
            active.update(status="running", started_at=timestamp())
            save_manifest(manifest_path, manifest)
            print("Starting " + active["name"], flush=True)
            started = time.monotonic()
            with (output / active["log"]).open("x", encoding="utf-8") as log:
                # Manifest fsync/log creation may consume the last available
                # seconds. Never launch using a timeout computed before them.
                remaining = remaining_seconds()
                if remaining <= 0:
                    raise TimeoutError("Experiment deadline reached before stage launch")
                result = subprocess.run(active["command"], cwd=str(SRC), env=environment,
                    stdout=log, stderr=subprocess.STDOUT, check=False, timeout=remaining)
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
    finally:
        # subprocess.run has reaped the active child before refunding unused time.
        # An abrupt kill cannot reach this settlement and retains the full debit.
        budget.finish()
    return exit_code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, fields in {
        "prepare": ("source-run", "inputs", "references", "output"),
        "score": ("source-run", "prepared", "model-path", "device"),
        "greedy": ("prepared", "output"),
        "compare": ("inputs", "labels", "predictions", "prepared", "output"),
        "run": ("source-run", "inputs", "references", "labels", "model-path", "output", "device"),
    }.items():
        child = sub.add_parser(name)
        for field in fields:
            child.add_argument("--" + field, required=True)
        if name != "greedy":
            child.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
        if name == "run":
            child.add_argument("--wall-seconds", type=int, default=14400)
    args = parser.parse_args(argv)
    try:
        result = globals()[args.command](args)
        if args.command == "run":
            return result
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
