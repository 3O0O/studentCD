"""Server-local history interventions; existing data stays on the server.

Preparation loads only an offline tokenizer. Scoring loads one existing model,
never reads labels and never generates. Analysis is a separate CPU stage.
"""

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import csv
import io
import re
import subprocess

from . import cache_rescore, history_interventions, inference, preflight, predict


SCHEMA_VERSION = "student-sim-cd.history-intervention-prepared.v1"
PROTOCOL_VERSION = "student-sim-cd.history-intervention.protocol.v1"
METHODS = ("base", "history_d0", "cd", "b", "copy")
SEEDS = (20261002, 20261003, 20261004)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(content):
    return hashlib.sha256(content).hexdigest()


def write(path, content):
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), "Preserve existing file: " + str(path))
    with path.open("xb") as stream:
        stream.write(content)


def json_bytes(value):
    return (inference.canonical_json(value) + "\n").encode("utf-8")


def protocol(path):
    raw = Path(path).read_bytes()
    value = predict._json(raw)
    require(value.get("schema_version") == PROTOCOL_VERSION, "Unsupported intervention protocol")
    require(value.get("new_generation") is False and value.get("student_execution") is False,
            "Generation and student execution are forbidden")
    require(value.get("seeds") == list(SEEDS) and value.get("variants") == list(history_interventions.VARIANTS),
            "Fixed two interventions and three assignment seeds required")
    require(value.get("expected_samples") == 70 and value.get("expected_students") == 17
            and value.get("expected_candidates") == 318, "Complete fixed cohort/candidate pool required")
    require(value.get("fixed_method_parameters") == predict.METHOD_PARAMETERS, "Fixed method parameters differ")
    require(value.get("matching_rule") == history_interventions.MATCHING_RULE, "Matching rule differs")
    require(value.get("bootstrap") == {"unit": "student", "paired": True, "repetitions": 2000, "seed": 20260929},
            "Fixed paired student bootstrap required")
    return value, sha(raw)


def baseline(root, settings):
    root = Path(root).resolve(strict=True)
    expected = settings["baseline_sha256"]
    require(set(expected) == {"manifest.json", "SUCCESS", "prepared/inputs.jsonl", "prepared/references.jsonl",
                             "prepared/inference/manifest.json", "prepared/inference/candidates.jsonl",
                             "prepared/inference/scores.jsonl"}, "Incomplete baseline hard anchors")
    for name, digest in expected.items():
        require(inference.file_hash(root / name) == digest, "Completed baseline changed: " + name)
    complete = predict._json((root / "manifest.json").read_bytes())
    require((root / "SUCCESS").read_bytes() == (expected["manifest.json"] + "\n").encode("ascii"),
            "Baseline SUCCESS does not bind its final manifest")
    require(complete.get("status") == "success" and complete.get("exit_code") == 0
            and len(complete.get("stages", [])) == 11
            and all(stage.get("status") == "success" and stage.get("exit_code") == 0
                    for stage in complete["stages"]), "Baseline final eleven stages are incomplete")
    require(complete.get("source_root", "").endswith("/releases/" + settings["baseline_release_sha256"]),
            "Baseline release identity differs")
    preparation, preparation_hash, records, rows, references = cache_rescore._load_prepared(root / "prepared")
    manifest, _, pools, scores, _, _ = predict._validated_run({
        "inputs": (root / "prepared/inputs.jsonl").read_bytes(),
        **{name: (root / ("prepared/inference/" + name + ".json" + ("l" if name != "manifest" else ""))).read_bytes()
           for name in ("manifest", "candidates", "scores")}})
    require(len(rows) == 70 and len({row["student_id"] for row in rows}) == 17 and len(scores) == 318,
            "Baseline cohort is incomplete")
    require(sum(len(record["candidates"]) for record in records) == 318, "Preparation pool incomplete")
    return root, preparation, preparation_hash, records, rows, references, manifest, pools


def _tokenizer_code(tokenizer, code):
    tokens = tokenizer.encode(code, add_special_tokens=False, truncation=False)
    require(isinstance(tokens, list) and all(type(token) is int for token in tokens), "Invalid code tokenization")
    require(type(tokenizer.eos_token_id) is int and not set(tokens) & set(tokenizer.all_special_ids),
            "Reserved special token inside candidate")
    return tokens + [tokenizer.eos_token_id]


def _prompt_audit(rows, refs, tokenizer, pools, config):
    result = []
    for row in rows:
        reference, _, _ = inference._reference(row, refs)
        tokens = {condition: tokenizer.apply_chat_template(inference.build_messages(row, condition, reference),
                  tokenize=True, add_generation_prompt=True, truncation=False) for condition in inference.CONDITIONS}
        require(all(isinstance(value, list) and value and all(type(item) is int for item in value)
                    for value in tokens.values()), "Invalid full prompt tokens")
        for value in tokens.values():
            inference.ensure_context(len(value), config.max_new_tokens, config.max_context_tokens)
        for candidate in pools[row["sample_id"]].values():
            completion = _tokenizer_code(tokenizer, candidate["code"])
            require(completion == candidate["completion_token_ids"], "Fixed candidate tokenization changed")
            for value in tokens.values():
                inference.ensure_context(len(value), len(completion), config.max_context_tokens)
        result.append({"sample_id": row["sample_id"],
                       "prompt_token_counts": {key: len(value) for key, value in tokens.items()},
                       "prompt_tokens_sha256": {key: inference.object_hash(value) for key, value in tokens.items()}})
    return result


def prepare(args):
    settings, protocol_hash = protocol(args.protocol)
    root, old_preparation, old_hash, records, rows, refs, old_manifest, pools = baseline(args.baseline_run, settings)
    train_bytes = Path(args.train_inputs).read_bytes()
    require(sha(train_bytes) == settings["training_inputs_sha256"], "Existing full training inputs SHA differs")
    train = predict._records(train_bytes, "training inputs")
    model_path = Path(args.model_path).resolve(strict=True)
    fingerprint = preflight.tokenizer_fingerprint(model_path)
    require(all(old_manifest["model"]["files"].get(name) == digest
                for name, digest in fingerprint["files"].items()), "Tokenizer differs from fixed baseline")
    tokenizer = preflight.load_local_tokenizer(model_path)
    config = inference.InferenceConfig(**old_manifest["config"])
    require(type(fingerprint["model_config_max_position_embeddings"]) is int
            and fingerprint["model_config_max_position_embeddings"] >= config.max_context_tokens,
            "Model context capacity cannot cover fixed protocol")
    histories = {inference.object_hash(row["history"]): row["history"] for row in rows + train if row["history"]}
    histories.update({inference.object_hash(row["reference_history"]): row["reference_history"] for row in refs.values()})
    lengths = {key: len(tokenizer.encode(inference.canonical_json(history), add_special_tokens=False, truncation=False))
               for key, history in histories.items()}
    built = history_interventions.build_interventions(rows, list(refs.values()), train,
        training_inputs_sha256=settings["training_inputs_sha256"], training_inputs_bytes=train_bytes,
        token_lengths=lengths, tokenizer_fingerprint_sha256=fingerprint["sha256"],
        protocol_sha256=protocol_hash, seeds=SEEDS)
    output = Path(args.prepared).absolute()
    require(not output.exists() and not output.is_symlink() and output.resolve() == output,
            "Preparation must use a new canonical directory")
    audit = {"baseline": _prompt_audit(rows, refs, tokenizer, pools, config)}
    serialized = {}
    for variant in history_interventions.VARIANTS:
        for seed in SEEDS:
            bundle = built["variants"][variant][str(seed)]
            key = variant + "/" + str(seed)
            # The untouched input/reference file retains its original bytes.
            if variant == "real_history_swapped":
                bundle["references_bytes"] = (root / "prepared/references.jsonl").read_bytes()
            else:
                bundle["inputs_bytes"] = (root / "prepared/inputs.jsonl").read_bytes()
            for field in ("inputs", "references", "assignments"):
                bundle[field + "_sha256"] = sha(bundle[field + "_bytes"])
            bundle["history_data_sha256"] = bundle["inputs_sha256" if variant == "real_history_swapped" else "references_sha256"]
            bundle["donor_assignment_sha256"] = bundle["assignments_sha256"]
            new_refs = {row["sample_id"]: row for row in bundle["references"]}
            audit[key] = _prompt_audit(bundle["inputs"], new_refs, tokenizer, pools, config)
            serialized[key] = bundle
    output.mkdir()
    write(output / ".incomplete", b"preserve until all preparation hashes are saved\n")
    files = {}
    for key, bundle in serialized.items():
        directory = output / key
        directory.mkdir(parents=True)
        for field in ("inputs", "references", "assignments"):
            name = key + "/" + field + ".jsonl"
            write(output / name, bundle[field + "_bytes"])
            files[name] = bundle[field + "_sha256"]
    write(output / "prompt-audit.json", json_bytes(audit))
    write(output / "donor-manifest.json", json_bytes(built["manifest"]))
    write(output / "history-token-lengths.json", json_bytes(lengths))
    for name in ("prompt-audit.json", "donor-manifest.json", "history-token-lengths.json"):
        files[name] = inference.file_hash(output / name)
    manifest = {"schema_version": SCHEMA_VERSION, "protocol_sha256": protocol_hash,
                "baseline_manifest_sha256": settings["baseline_sha256"]["manifest.json"],
                "training_inputs_sha256": sha(train_bytes), "tokenizer": fingerprint,
                "formal_token_matching": True, "files_sha256": files,
                "labels_read": False, "model_weights_loaded": False, "generation_performed": False,
                "student_execution": False, "data_residency": "training and donor history content stays on server"}
    write(output / "manifest.json", json_bytes(manifest))
    # This marker is ours in a fresh output and only removed after final sealing.
    (output / ".incomplete").unlink()
    write(output / "SUCCESS", (inference.file_hash(output / "manifest.json") + "\n").encode("ascii"))
    return manifest


def load_prepared(args):
    settings, protocol_hash = protocol(args.protocol)
    old = baseline(args.baseline_run, settings)
    directory = Path(args.prepared).resolve(strict=True)
    require(not (directory / ".incomplete").exists(), "Preserve incomplete preparation")
    raw = (directory / "manifest.json").read_bytes()
    manifest = predict._json(raw)
    require(manifest.get("schema_version") == SCHEMA_VERSION and manifest.get("formal_token_matching") is True
            and manifest.get("protocol_sha256") == protocol_hash
            and manifest.get("baseline_manifest_sha256") == settings["baseline_sha256"]["manifest.json"]
            and manifest.get("training_inputs_sha256") == settings["training_inputs_sha256"], "Preparation identity differs")
    require((directory / "SUCCESS").read_bytes() == (sha(raw) + "\n").encode("ascii"), "Preparation SUCCESS mismatch")
    expected = {variant + "/" + str(seed) + "/" + field + ".jsonl"
                for variant in history_interventions.VARIANTS for seed in SEEDS
                for field in ("inputs", "references", "assignments")} | {"prompt-audit.json", "donor-manifest.json", "history-token-lengths.json"}
    require(set(manifest["files_sha256"]) == expected, "Prepared file coverage incomplete")
    for name, digest in manifest["files_sha256"].items():
        require(inference.file_hash(directory / name) == digest, "Prepared payload changed: " + name)
    # Reconstruct every donor assignment from the original complete training
    # file before loading model weights. A rehashed partial or substituted
    # preparation cannot silently alter the single-factor intervention.
    train_bytes = Path(args.train_inputs).read_bytes()
    require(sha(train_bytes) == settings["training_inputs_sha256"], "Training source changed")
    lengths = predict._json((directory / "history-token-lengths.json").read_bytes())
    require(preflight.tokenizer_fingerprint(Path(args.model_path)) == manifest["tokenizer"], "Tokenizer changed")
    train = predict._records(train_bytes, "training inputs")
    tokenizer = preflight.load_local_tokenizer(Path(args.model_path))
    histories = {inference.object_hash(row["history"]): row["history"] for row in old[4] + train if row["history"]}
    histories.update({inference.object_hash(row["reference_history"]): row["reference_history"] for row in old[5].values()})
    actual_lengths = {key: len(tokenizer.encode(inference.canonical_json(history), add_special_tokens=False, truncation=False))
                      for key, history in histories.items()}
    require(lengths == actual_lengths, "Prepared token length table differs from the existing verified tokenizer")
    built = history_interventions.build_interventions(old[4], list(old[5].values()),
        train, training_inputs_bytes=train_bytes,
        training_inputs_sha256=settings["training_inputs_sha256"], token_lengths=lengths,
        tokenizer_fingerprint_sha256=manifest["tokenizer"]["sha256"], protocol_sha256=protocol_hash, seeds=SEEDS)
    require(built["manifest"] == predict._json((directory / "donor-manifest.json").read_bytes()),
            "Donor matching manifest does not reproduce the complete frozen assignment")
    for variant in history_interventions.VARIANTS:
        for seed in SEEDS:
            bundle = built["variants"][variant][str(seed)]
            for field in ("inputs", "references", "assignments"):
                expected_bytes = bundle[field + "_bytes"]
                if (variant, field) in {("real_history_swapped", "references"), ("reference_swapped", "inputs")}:
                    expected_bytes = (old[0] / ("prepared/" + field + ".jsonl")).read_bytes()
                require((directory / variant / str(seed) / (field + ".jsonl")).read_bytes() == expected_bytes,
                        "Prepared intervention does not reproduce its lawful full-training donor")
    return settings, protocol_hash, directory, manifest, old


def score(args, backend=None):
    settings, protocol_hash, prepared, preparation, old = load_prepared(args)
    root, old_preparation, old_hash, records, _, _, old_manifest, pools = old
    config = inference.InferenceConfig(**old_manifest["config"])
    require(str(Path(args.model_path).resolve(strict=True)) == config.model_path,
            "Use the exact existing baseline model path")
    require(args.device == config.device, "Fixed baseline device mapping required")
    require(inference.model_fingerprint(Path(config.model_path)) == old_manifest["model"], "Model assets changed")
    # No variant may have a previous scoring directory, even if empty.
    for variant in history_interventions.VARIANTS:
        for seed in SEEDS:
            output = prepared / variant / str(seed) / "inference"
            require(not output.exists() and not output.is_symlink(), "Preserve prior/partial score output")
    if backend is None:
        gpu_uuid = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        require(re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", gpu_uuid),
                "Exactly one reviewed physical GPU UUID required")
        require(len(os.sched_getaffinity(0)) <= 4 and all(os.environ.get(name) == "4"
                for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
                "Real scoring requires externally bounded four-core/four-thread execution")
        gpu_text = subprocess.check_output(["/usr/bin/nvidia-smi", "--query-gpu=uuid,memory.used,utilization.gpu",
                                           "--format=csv,noheader,nounits"], text=True, timeout=15)
        app_text = subprocess.check_output(["/usr/bin/nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                           "--format=csv,noheader,nounits"], text=True, timeout=15)
        devices = [list(map(str.strip, row)) for row in csv.reader(io.StringIO(gpu_text)) if row]
        selected = [row for row in devices if row[0] == gpu_uuid]
        require(len(selected) == 1 and len(selected[0]) == 3 and int(selected[0][1]) <= 100
                and int(selected[0][2]) == 0, "Reviewed GPU is now busy; no automatic alternative")
        apps = [list(map(str.strip, row)) for row in csv.reader(io.StringIO(app_text)) if row]
        require(all(len(row) == 2 and row[0] != gpu_uuid for row in apps), "Reviewed GPU has an existing application")
        require(all(Path("/proc") .joinpath(row[1]).stat().st_uid != os.getuid() for row in apps),
                "Account already has GPU applications; preserve them and do not exceed this phase's one-GPU limit")
        write(prepared / "score-gpu-check.json", json_bytes({"uuid": gpu_uuid, "memory_used_mib": int(selected[0][1]),
              "utilization_percent": int(selected[0][2]), "compute_apps": 0, "cpu_affinity": sorted(os.sched_getaffinity(0))}))
        import torch
        torch.set_num_threads(4)
        torch.set_num_interop_threads(4)
        require(torch.get_num_threads() == torch.get_num_interop_threads() == 4, "Torch thread limits differ")
    active = inference.HFBackend(config) if backend is None else backend
    results = {}
    for variant in history_interventions.VARIANTS:
        for seed in SEEDS:
            directory = prepared / variant / str(seed)
            rows = inference.load_inputs(directory / "inputs.jsonl")
            refs = inference.load_references(directory / "references.jsonl", {row["sample_id"] for row in rows})
            manifest = inference.make_manifest(config, directory / "inputs.jsonl", directory / "references.jsonl", old_manifest["model"])
            for name in ("config", "config_sha256", "runtime_versions", "implementation_sha256", "scoring_sha256", "model", "sequence_protocol"):
                require(manifest[name] == old_manifest[name], "Fixed baseline scoring changed: " + name)
            manifest["cache_rescore"] = dict(old_manifest["cache_rescore"],
                implementation_sha256=inference.file_hash(Path(__file__)),
                backend="HFBackend" if backend is None else "injected_synthetic_backend",
                generation_performed=False, old_scores_reused=False)
            manifest["history_intervention"] = {
                "variant": variant, "permutation_seed": seed, "protocol_sha256": protocol_hash,
                "donor_assignment_sha256": inference.file_hash(directory / "assignments.jsonl"),
                "baseline_manifest_sha256": settings["baseline_sha256"]["manifest.json"],
                "history_data_sha256": inference.file_hash(directory / ("inputs.jsonl" if variant == "real_history_swapped" else "references.jsonl"))}
            output = directory / "inference"
            run_hash = inference.prepare_output(output, manifest, False)
            candidates = []
            baseline_candidates = {row["sample_id"]: row for row in predict._records((root / "prepared/inference/candidates.jsonl").read_bytes(), "baseline candidates")}
            for row in rows:
                sample = row["sample_id"]
                items = []
                for candidate in baseline_candidates[sample]["candidates"]:
                    tokens = active.code_tokens(candidate["code"])
                    require(tokens == candidate["completion_token_ids"], "Canonical candidate/EOS tokens changed")
                    items.append(dict(candidate))
                _, kind, provenance = inference._reference(row, refs)
                record = {"sample_id": sample, "candidates": items, "attempts": baseline_candidates[sample]["attempts"],
                          "run_sha256": run_hash, "reference_kind": kind, "reference_provenance": provenance}
                record["record_sha256"] = inference.object_hash(record)
                candidates.append(record)
            write(output / "candidates.jsonl", b"".join(json_bytes(record) for record in candidates))
            require(not (output / "scores.jsonl").exists(), "Fresh scores only; no cache resume")
            result = inference.run_inference(rows, refs, config, output, run_hash, cache_rescore._NoGenerationBackend(active))
            require(result == {"samples": 70, "candidates": 318, "scores": 318}, "Complete four-condition scores required")
            predict.predict(output, directory / "inputs.jsonl", directory / "predictions", include_copy=True)
            write(directory / "SCORE_SUCCESS", json_bytes(result))
            results[variant + ":" + str(seed)] = result
    return {"variants": results, "generation_performed": False, "model_loaded_once": backend is None}


def analyze(args):
    from .intervention_analysis import InferenceArtifact, analyze_interventions
    settings, protocol_hash, prepared, preparation, old = load_prepared(args)
    root = old[0]

    def artifact(directory, original=False):
        inference_dir = directory / "inference"
        prediction_dir = root / "predictions" if original else directory / "predictions"
        return InferenceArtifact(inputs=(directory / "inputs.jsonl").read_bytes(),
            references=(directory / "references.jsonl").read_bytes(), manifest=(inference_dir / "manifest.json").read_bytes(),
            candidates=(inference_dir / "candidates.jsonl").read_bytes(), scores=(inference_dir / "scores.jsonl").read_bytes(),
            predictions={method: (prediction_dir / ("predictions." + method + ".jsonl")).read_bytes() for method in METHODS},
            assignments=None if original else (directory / "assignments.jsonl").read_bytes(),
            completion_manifest=(root / "manifest.json").read_bytes() if original else None,
            success=(root / "SUCCESS").read_bytes() if original else None)

    require(inference.file_hash(Path(args.labels)) == settings["labels_sha256"], "Fixed evaluation labels changed")
    result = analyze_interventions((root / "prepared/inputs.jsonl").read_bytes(), Path(args.labels).read_bytes(),
        artifact(root / "prepared", True),
        {variant: {str(seed): artifact(prepared / variant / str(seed)) for seed in SEEDS}
         for variant in history_interventions.VARIANTS}, protocol_sha256=protocol_hash,
        fixed_external_greedy=(root / "predictions/predictions.greedy.jsonl").read_bytes())
    result["donor_matching"] = predict._json((prepared / "donor-manifest.json").read_bytes())
    result["prepared_manifest_sha256"] = inference.file_hash(prepared / "manifest.json")
    result["baseline_scores"] = "reused_prior_completed_scores; not freshly recomputed in this phase"
    write(Path(args.analysis_output), json_bytes(result))
    return {"analysis_sha256": sha(json_bytes(result)), "samples": 70, "students": 17}
