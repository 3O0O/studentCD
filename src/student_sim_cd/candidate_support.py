"""Budget-matched candidate support, with immutable raw checkpoints.

Four conditions each supply three random draws; the control supplies twelve
11-condition random draws. Both pools share the original 11 greedy and copy.
Only text is handled. No labels, student execution, downloads or authentication
are accepted here. Resource authorization belongs to the controlled outer job.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
import hashlib
import math
import os
from pathlib import Path
import re
import stat
from time import monotonic

from . import cache_rescore, history_scoring, inference, preflight, predict
from .scoring import LogProbs, components, method_scores


SCHEMA_VERSION = "student-sim-cd.candidate-support.v1"
PROTOCOL_VERSION = "student-sim-cd.candidate-support.protocol.v1"
POOLS = ("balanced12", "11_only12")
RAW_FIELDS = {"attempt", "source", "seed", "raw_text", "generated_token_ids", "eos_reached", "finish_reason"}
WRAPPER_FIELDS = {"sample_id", "condition", "draw_id", "slot", "origin", "baseline_record_sha256",
                  "raw_generation", "prompt_tokens_sha256"}


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (inference.canonical_json(value) + "\n").encode("utf-8")


def canonical_path(path, *, exists=False):
    path = Path(path).absolute()
    require(path.resolve(strict=exists) == path and not path.is_symlink(), "Noncanonical/symlink path")
    return path


def read(path):
    path = canonical_path(path, exists=True)
    before = path.stat()
    require(stat.S_ISREG(before.st_mode) and before.st_uid == os.getuid(), "Require owned regular artifact")
    raw = path.read_bytes()
    after = path.stat()
    fields = ("st_dev", "st_ino", "st_mode", "st_uid", "st_size", "st_mtime_ns", "st_ctime_ns")
    require(all(getattr(before, key) == getattr(after, key) for key in fields), "Artifact changed while reading")
    return raw


def write(path, raw):
    path = canonical_path(path)
    require(not path.exists(), "Preserve existing artifact: " + path.name)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def exact(path, raw):
    if Path(path).exists():
        require(read(path) == raw, "Checkpoint differs; no overwrite: " + Path(path).name)
    else:
        write(path, raw)


@contextmanager
def lease(directory):
    path = canonical_path(directory, exists=True) / ".candidate-support.lock"
    require(not path.is_symlink(), "Symlink checkpoint lock")
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        require(stat.S_ISREG(os.fstat(descriptor).st_mode) and os.fstat(descriptor).st_uid == os.getuid(), "Unsafe checkpoint lock")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another candidate-support stage holds this checkpoint") from error
        yield
    finally:
        os.close(descriptor)


def protocol(value):
    raw = encoded(value) if isinstance(value, dict) else read(value)
    value = predict._json(raw)
    require(value.get("schema_version") == PROTOCOL_VERSION
            and value.get("expected_samples") == 70 and value.get("expected_students") == 17
            and value.get("seed") == 20260929 and value.get("conditions") == list(inference.CONDITIONS)
            and value.get("balanced_random_per_condition") == 3 and value.get("control_random_count") == 12
            and value.get("shared_greedy_condition") == "11"
            and value.get("canonical_protocol") == "cached-python-fence-v1"
            and value.get("student_execution") is False, "Frozen full-cohort support protocol differs")
    require(isinstance(value.get("baseline_release_sha256"), str) and len(value["baseline_release_sha256"]) == 64,
            "Baseline release hard anchor required")
    return value, raw, sha(raw)


def draw_seed(seed, sample_id, draw_id):
    """Exactly the original HFBackend per-attempt seed, including paired draws."""
    return int(inference.object_hash([seed, sample_id, draw_id])[:8], 16)


def draw_schedule(sample_ids):
    result = []
    for sample in sample_ids:
        result.extend((sample, "11", draw, draw) for draw in range(4, 13))
        for condition, start in (("01", 4), ("10", 7), ("00", 10)):
            result.extend((sample, condition, draw, slot) for slot, draw in enumerate(range(start, start + 3), 1))
    return result


def _settings(config, original):
    current = asdict(config)
    require(config.fence_policy == "unwrap-single"
            and all(current[key] == value for key, value in original.items()
                    if key not in {"model_path", "device", "fence_policy"}), "Original generation/scoring configuration changed")


def _prompt_tokens(rows, references, tokenizer, config):
    result, audits = {}, []
    require(type(tokenizer.eos_token_id) is int and tokenizer.eos_token_id in tokenizer.all_special_ids,
            "Single known tokenizer EOS required")
    for row in rows:
        history, kind, _ = inference._reference(row, references)
        require(kind == "matched", "No diagnostic reference fallback in the main cohort")
        prompts = {condition: tokenizer.apply_chat_template(inference.build_messages(row, condition, history),
            tokenize=True, add_generation_prompt=True, truncation=False) for condition in inference.CONDITIONS}
        for tokens in prompts.values():
            require(isinstance(tokens, list) and tokens and all(type(token) is int and token >= 0 for token in tokens),
                    "Full prompt must be a nonempty flat token sequence")
            inference.ensure_context(len(tokens), config.max_new_tokens, config.max_context_tokens)
        result[row["sample_id"]] = prompts
        audits.append({"sample_id": row["sample_id"], "prompt_token_counts": {key: len(tokens) for key, tokens in prompts.items()},
            "prompt_tokens_sha256": {key: inference.object_hash(tokens) for key, tokens in prompts.items()}})
    return result, audits


def _code_tokens(tokenizer, code):
    tokens = tokenizer.encode(code, add_special_tokens=False, truncation=False)
    require(isinstance(tokens, list) and all(type(token) is int and token >= 0 for token in tokens)
            and not set(tokens) & set(tokenizer.all_special_ids), "Reserved/invalid token in canonical candidate")
    return tokens + [tokenizer.eos_token_id]


def _raw(raw, sample, draw, config, eos_id, tokenizer=None, *, original=False):
    require(isinstance(raw, dict) and (set(raw) <= cache_rescore.ATTEMPT_FIELDS if original else set(raw) == RAW_FIELDS)
            and raw.get("attempt") == draw and type(raw.get("attempt")) is int
            and raw.get("source") == ("greedy" if draw == 0 else "sampled")
            and type(raw.get("seed")) is int and raw["seed"] == draw_seed(config.seed, sample, draw)
            and isinstance(raw.get("raw_text"), str) and type(raw.get("eos_reached")) is bool,
            "Raw draw does not match original seed/sampling identity")
    tokens = raw.get("generated_token_ids")
    require(isinstance(tokens, list) and tokens and all(type(token) is int and token >= 0 for token in tokens)
            and len(tokens) <= config.max_new_tokens
            and raw.get("finish_reason") == ("eos" if raw["eos_reached"] else "length_limit"), "Invalid raw token/finish budget")
    require((tokens[-1] == eos_id and eos_id not in tokens[:-1]) if raw["eos_reached"] else eos_id not in tokens,
            "Raw generation must have exactly one terminal EOS, or explicitly no EOS")
    require(raw["eos_reached"] or len(tokens) == config.max_new_tokens, "No-EOS draw must account for the entire length-limit budget")
    if tokenizer is not None:
        content = tokens[:-1] if raw["eos_reached"] else tokens
        require(tokenizer.decode(content, skip_special_tokens=False, clean_up_tokenization_spaces=False) == raw["raw_text"],
                "Raw generation text differs from its original tokens")


def prepare(baseline_run, output, *, protocol, config, tokenizer=None):
    """Validate frozen sources and every prompt before any model is loaded."""
    specification, raw_protocol, protocol_hash = globals()["protocol"](protocol)
    old = history_scoring.baseline(canonical_path(baseline_run, exists=True), specification)
    baseline_root, _, _, records, rows, references, old_manifest, pools = old
    _settings(config, old_manifest["config"])
    require("generation_config" not in specification or specification["generation_config"] == asdict(config),
            "Protocol generation configuration differs from actual execution")
    require("new_support_attempts" not in specification or specification["new_support_attempts"] == 1260,
            "Protocol new support budget must equal 1260")
    require("fixed_method_parameters" not in specification or specification["fixed_method_parameters"] == predict.METHOD_PARAMETERS,
            "Fixed method coefficients changed")
    require("methods" not in specification or specification["methods"] == ["base", "history_d0", "cd", "b", "copy"],
            "Fixed methods changed")
    output = canonical_path(output)
    require(not output.exists() and output != baseline_root and baseline_root not in output.parents,
            "Preparation must use a new directory outside the original result")
    artifacts = preflight.tokenizer_fingerprint(Path(config.model_path))
    require(all(old_manifest["model"]["files"].get(name) == digest for name, digest in artifacts["files"].items())
            and type(artifacts["model_config_max_position_embeddings"]) is int
            and artifacts["model_config_max_position_embeddings"] >= config.max_context_tokens, "Tokenizer/model context differs from baseline")
    injected = tokenizer is not None
    active = tokenizer if injected else preflight.load_local_tokenizer(Path(config.model_path))
    prompts, audits = _prompt_tokens(rows, references, active, config)
    baseline_rows = {row["sample_id"]: row for row in records}
    for row in rows:
        sample = row["sample_id"]
        record = baseline_rows[sample]
        require(len(record["attempts"]) == 4, "Original full 1-greedy/3-random budget missing")
        for draw, attempt in enumerate(record["attempts"]):
            _raw(attempt, sample, draw, config, active.eos_token_id, active, original=True)
            require(attempt["eos_reached"], "Frozen original greedy/random draw lacks EOS; cannot silently replace it")
            code = cache_rescore.extract_code(attempt["raw_text"])[0]
            require(sha(code.encode("utf-8")) in pools[sample], "Original extracted draw differs from baseline candidate")
        for candidate in pools[sample].values():
            tokens = _code_tokens(active, candidate["code"])
            require(tokens == candidate["completion_token_ids"], "Original canonical candidate/EOS tokenization changed")
            for prompt in prompts[sample].values():
                inference.ensure_context(len(prompt), len(tokens), config.max_context_tokens)
    require(len(draw_schedule([row["sample_id"] for row in rows])) == 1260, "Full new draw budget differs")
    inputs_raw, references_raw = read(baseline_root / "prepared/inputs.jsonl"), read(baseline_root / "prepared/references.jsonl")
    payloads = {"protocol.json": raw_protocol, "inputs.jsonl": inputs_raw, "references.jsonl": references_raw,
                "prompt-audit.json": encoded(audits)}
    manifest = {"schema_version": SCHEMA_VERSION, "stage": "prepared", "protocol_sha256": protocol_hash,
        "protocol": specification, "baseline_run": str(baseline_root), "baseline_manifest_sha256": specification["baseline_sha256"]["manifest.json"],
        "config": asdict(config), "tokenizer": artifacts, "eos_token_id": active.eos_token_id,
        "special_token_ids": sorted(active.all_special_ids), "prompt_audit_sha256": sha(payloads["prompt-audit.json"]),
        "files_sha256": {name: sha(raw) for name, raw in payloads.items()}, "samples": 70, "students": 17,
        "old_greedy": 70, "old_random": 210, "new_draws": 1260, "pool_random_count": 12,
        "generation_performed": False, "labels_read": False, "student_execution": False,
        "synthetic_tokenizer_injected": injected, "implementation_sha256": inference.file_hash(Path(__file__))}
    output.mkdir()
    for name, raw in payloads.items():
        write(output / name, raw)
    write(output / "manifest.json", encoded(manifest))
    write(output / "SUCCESS", (sha(encoded(manifest)) + "\n").encode("ascii"))
    return {"prepared_dir": str(output), "prepared_manifest_sha256": sha(encoded(manifest)), "samples": 70, "students": 17, "new_draws": 1260}


def _load(directory, config, tokenizer=None):
    directory = canonical_path(directory, exists=True)
    raw = read(directory / "manifest.json")
    manifest = predict._json(raw)
    require(manifest.get("schema_version") == SCHEMA_VERSION and manifest.get("stage") == "prepared"
            and read(directory / "SUCCESS") == (sha(raw) + "\n").encode("ascii"), "Preparation is incomplete")
    specification, raw_protocol, digest = protocol(directory / "protocol.json")
    require(digest == manifest["protocol_sha256"] and specification == manifest["protocol"]
            and manifest["implementation_sha256"] == inference.file_hash(Path(__file__))
            and manifest["config"] == asdict(config), "Prepared protocol/implementation/config changed")
    for name, digest in manifest["files_sha256"].items():
        require(name in {"protocol.json", "inputs.jsonl", "references.jsonl", "prompt-audit.json"}
                and sha(read(directory / name)) == digest, "Prepared artifact changed")
    require(set(manifest["files_sha256"]) == {"protocol.json", "inputs.jsonl", "references.jsonl", "prompt-audit.json"}, "Prepared inventory incomplete")
    old = history_scoring.baseline(Path(manifest["baseline_run"]), specification)
    require(read(directory / "inputs.jsonl") == read(old[0] / "prepared/inputs.jsonl")
            and read(directory / "references.jsonl") == read(old[0] / "prepared/references.jsonl"), "Prepared sources differ from original baseline")
    require(preflight.tokenizer_fingerprint(Path(config.model_path)) == manifest["tokenizer"], "Prepared tokenizer assets changed")
    if tokenizer is not None:
        prompts, audits = _prompt_tokens(old[4], old[5], tokenizer, config)
        require(encoded(audits) == read(directory / "prompt-audit.json"), "Full prompt tokenizer audit changed")
        require(tokenizer.eos_token_id == manifest["eos_token_id"] and sorted(tokenizer.all_special_ids) == manifest["special_token_ids"], "EOS/special tokens changed")
    else:
        prompts = None
    return directory, manifest, sha(raw), old, prompts


def load_prepared(prepared_dir, config, *, tokenizer=None):
    directory, manifest, digest, old, prompts = _load(prepared_dir, config, tokenizer)
    return {"prepared_dir": directory, "manifest": manifest, "manifest_sha256": digest,
            "baseline": old, "rows": old[4], "references": old[5], "prompt_tokens": prompts}


def _compatible(directory, config, old):
    model = inference.model_fingerprint(Path(config.model_path))
    require(model == old[6]["model"], "Fixed model weights/tokenizer changed")
    manifest = inference.make_manifest(config, directory / "inputs.jsonl", directory / "references.jsonl", model)
    for key in ("config", "config_sha256", "runtime_versions", "implementation_sha256", "scoring_sha256", "prompt_version", "sequence_protocol", "inputs_sha256", "references_sha256"):
        require(manifest[key] == old[6][key], "Old generation/scores are not exactly compatible: " + key)
    return manifest


def _event(path, value):
    path = canonical_path(path)
    if path.exists():
        read(path)
    inference.append_record(path, value)


def _events(directory, manifest_hash, schedule, config, eos_id):
    path = directory / "draw-checkpoints.jsonl"
    events = predict._records(read(path), "draw checkpoints") if path.exists() else []
    draws = {}
    require(len(events) % 2 == 0, "Interrupted reserved draw has no durable raw result; preserve it, no automatic regeneration")
    for position in range(0, len(events), 2):
        reservation, result = events[position:position + 2]
        require(position // 2 < len(schedule), "Generation checkpoint exceeds total 1260 attempts")
        sample, condition, draw, slot = schedule[position // 2]
        key = (sample, condition, draw)
        identity = {"sample_id": sample, "condition": condition, "draw_id": draw, "slot": slot,
                    "seed": draw_seed(config.seed, sample, draw), "prepared_manifest_sha256": manifest_hash}
        require(reservation == dict(identity, event="reserved") and result.get("identity") == identity
                and set(result) == {"event", "identity", "raw_generation", "elapsed_seconds", "record_sha256"}
                and result["event"] == "raw" and result["record_sha256"] == inference.object_hash({k: v for k, v in result.items() if k != "record_sha256"}),
                "Wrong/reordered/tampered draw checkpoint")
        require(type(result["elapsed_seconds"]) in (int, float) and math.isfinite(result["elapsed_seconds"]) and result["elapsed_seconds"] >= 0,
                "Invalid generation resource accounting")
        _raw(result["raw_generation"], sample, draw, config, eos_id)
        draws[key] = result["raw_generation"]
    return draws


def _wrapper(sample, condition, draw, slot, raw, origin, record_hash, prompt_hash):
    return {"sample_id": sample, "condition": condition, "draw_id": draw, "slot": slot,
            "origin": origin, "baseline_record_sha256": record_hash, "raw_generation": raw,
            "prompt_tokens_sha256": prompt_hash}


def _candidate_records(rows, references, records, draws, prompts, backend, config):
    old = {record["sample_id"]: record for record in records}
    result, mappings = {pool: [] for pool in POOLS}, []
    for row in rows:
        sample = row["sample_id"]
        wrappers = {}
        for draw, raw in enumerate(old[sample]["attempts"]):
            wrappers["11", draw] = _wrapper(sample, "11", draw, draw, raw, "baseline_cache",
                old[sample]["record_sha256"], inference.object_hash(prompts[sample]["11"]))
        for sid, condition, draw, slot in draw_schedule([sample]):
            wrappers[condition, draw] = _wrapper(sample, condition, draw, slot, draws[sid, condition, draw],
                "new_model_generation", None, inference.object_hash(prompts[sample][condition]))
        canonical_candidates, candidate_map = {}, {}
        for key, wrapper in wrappers.items():
            raw = wrapper["raw_generation"]
            code, transform, reason = cache_rescore.extract_code(raw["raw_text"])
            identifier, exclusion = None, None
            if raw["eos_reached"]:
                tokens = backend.code_tokens(code)
                require(isinstance(tokens, list) and tokens and all(type(token) is int and token >= 0 for token in tokens)
                        and tokens[-1] == backend.eos_id and backend.eos_id not in tokens[:-1], "Canonical code must have exactly one EOS")
                for prompt in prompts[sample].values():
                    inference.ensure_context(len(prompt), len(tokens), config.max_context_tokens)
                identifier = sha(code.encode("utf-8"))
                canonical_candidates.setdefault(identifier, {"candidate_id": identifier, "code": code,
                    "sources": [], "transform": transform, "completion_token_ids": tokens,
                    "token_count": len(tokens), "eos_included": True})
            else:
                exclusion = "no EOS; raw attempt retained, no automatic replacement"
            candidate_map[key] = identifier
            mappings.append({**wrapper, "candidate_id": identifier, "transform": transform,
                "extraction_reason": reason, "eligible": identifier is not None, "exclusion_reason": exclusion})
        copy_id = sha(row["current_code"].encode("utf-8"))
        copy_tokens = backend.code_tokens(row["current_code"])
        require(copy_tokens and copy_tokens[-1] == backend.eos_id and backend.eos_id not in copy_tokens[:-1], "Copy EOS protocol changed")
        for prompt in prompts[sample].values():
            inference.ensure_context(len(prompt), len(copy_tokens), config.max_context_tokens)
        copy_candidate = {"candidate_id": copy_id, "code": row["current_code"], "sources": ["copy_current"],
            "transform": "none", "completion_token_ids": copy_tokens, "token_count": len(copy_tokens), "eos_included": True}
        for pool in POOLS:
            keys = [("11", 0), *(("11", draw) for draw in range(1, 4))]
            keys += ([("11", draw) for draw in range(4, 13)] if pool == "11_only12"
                     else [(condition, draw) for _, condition, draw, _ in draw_schedule([sample]) if condition != "11"])
            chosen = {copy_id: dict(copy_candidate, sources=["copy_current"])}
            for condition, draw in keys:
                identifier = candidate_map[condition, draw]
                if identifier is not None:
                    if identifier not in chosen:
                        chosen[identifier] = dict(canonical_candidates[identifier], sources=[])
                    chosen[identifier]["sources"].append(("greedy" if draw == 0 else "sampled") + ":" + condition + ":" + str(draw))
            _, kind, provenance = inference._reference(row, references)
            result[pool].append({"sample_id": sample, "candidates": list(chosen.values()),
                "attempts": [wrappers[key] for key in keys], "reference_kind": kind, "reference_provenance": provenance})
    return result, mappings


def _generation_complete(directory, manifest_hash):
    raw = read(directory / "generation-manifest.json")
    value = predict._json(raw)
    require(value.get("schema_version") == SCHEMA_VERSION and value.get("stage") == "generated"
            and value.get("prepared_manifest_sha256") == manifest_hash
            and read(directory / "GENERATION_SUCCESS") == (sha(raw) + "\n").encode("ascii"), "Generation completion differs")
    for name, digest in value["files_sha256"].items():
        require(name in {"draw-checkpoints.jsonl", "source-mappings.jsonl", "pools/balanced12/candidates.jsonl", "pools/11_only12/candidates.jsonl"}
                and sha(read(directory / name)) == digest, "Generation final artifact changed")
    require(len(value["files_sha256"]) == 4 and value["new_draws"] == 1260, "Complete raw draw finalization required")
    return value


def _generation_cost(directory, draws, prompts):
    events = predict._records(read(directory / "draw-checkpoints.jsonl"), "generation timing")
    return {"generation_seconds": math.fsum(item["elapsed_seconds"] for item in events[1::2]),
        "new_generation_prompt_tokens": sum(len(prompts[sample][condition]) for sample, condition, _ in draws),
        "new_generated_tokens_including_eos_when_reached": sum(len(raw["generated_token_ids"]) for raw in draws.values()),
        "new_generation_attempts": len(draws), "generation_timing_scope": "durable raw draws; includes all completed attempts across explicit resumes"}


def _generate(directory, config, backend, resume):
    directory, preparation, prepared_hash, old, _ = _load(directory, config)
    if isinstance(backend, inference.HFBackend):
        _real_preconditions(preparation, old)
    _compatible(directory, config, old)
    require(type(backend.eos_id) is int and backend.eos_id == preparation["eos_token_id"], "Generation EOS changed")
    rows, references, records = old[4], old[5], old[3]
    prompts = {row["sample_id"]: {condition: backend.prompt_tokens(inference.build_messages(row, condition,
        inference._reference(row, references)[0])) for condition in inference.CONDITIONS} for row in rows}
    audits = [{"sample_id": row["sample_id"], "prompt_token_counts": {key: len(value) for key, value in prompts[row["sample_id"]].items()},
        "prompt_tokens_sha256": {key: inference.object_hash(value) for key, value in prompts[row["sample_id"]].items()}} for row in rows]
    require(encoded(audits) == read(directory / "prompt-audit.json"), "Full seventy-sample runtime preflight changed")
    if (directory / "GENERATION_SUCCESS").exists():
        require(resume, "Final generation exists; do not overwrite")
        return _generation_complete(directory, prepared_hash)
    path = directory / "draw-checkpoints.jsonl"
    require(resume or not path.exists(), "Checkpoint exists; explicit resume required")
    schedule = draw_schedule([row["sample_id"] for row in rows])
    draws = _events(directory, prepared_hash, schedule, config, backend.eos_id)
    for sample, condition, draw, slot in schedule[len(draws):]:
        identity = {"sample_id": sample, "condition": condition, "draw_id": draw, "slot": slot,
                    "seed": draw_seed(config.seed, sample, draw), "prepared_manifest_sha256": prepared_hash}
        _event(path, dict(identity, event="reserved"))
        started = monotonic()
        raw = backend.generate(prompts[sample][condition], sample, draw)
        # Preserve the actual raw generation durably before extraction/eligibility.
        event = {"event": "raw", "identity": identity, "raw_generation": raw, "elapsed_seconds": max(0, monotonic() - started)}
        event["record_sha256"] = inference.object_hash(event)
        _event(path, event)
        _raw(raw, sample, draw, config, backend.eos_id, getattr(backend, "tokenizer", None))
        draws[sample, condition, draw] = raw
    require(len(draws) == 1260, "Incomplete new generation budget")
    pools, mappings = _candidate_records(rows, references, records, draws, prompts, backend, config)
    artifacts = {"source-mappings.jsonl": b"".join(encoded(row) for row in mappings),
        **{"pools/" + pool + "/candidates.jsonl": b"".join(encoded(record) for record in pools[pool]) for pool in POOLS}}
    for name, raw in artifacts.items():
        exact(directory / name, raw)
    files = {name: sha(raw) for name, raw in artifacts.items()}
    files["draw-checkpoints.jsonl"] = sha(read(path))
    value = {"schema_version": SCHEMA_VERSION, "stage": "generated", "prepared_manifest_sha256": prepared_hash,
        "protocol_sha256": preparation["protocol_sha256"], "files_sha256": files,
        "new_draws": len(draws), "old_greedy": 70, "old_random": 210, "samples": 70, "students": 17,
        "candidate_counts": {pool: sum(len(row["candidates"]) for row in pools[pool]) for pool in POOLS},
        "cost_accounting": _generation_cost(directory, draws, prompts),
        "new_model_generation": True, "backend": "HFBackend" if isinstance(backend, inference.HFBackend) else "injected_synthetic_backend",
        "labels_read": False, "student_execution": False, "automatic_replacements": 0}
    exact(directory / "generation-manifest.json", encoded(value))
    write(directory / "GENERATION_SUCCESS", (sha(encoded(value)) + "\n").encode("ascii"))
    return value


def generate(prepared_dir, config, backend, *, resume=False):
    with lease(prepared_dir):
        return _generate(prepared_dir, config, backend, resume)


def _signed(record, run_hash):
    record = dict(record, run_sha256=run_hash)
    record["record_sha256"] = inference.object_hash({key: value for key, value in record.items() if key != "record_sha256"})
    return record


def _score(directory, config, backend, resume):
    started = monotonic()
    directory, preparation, prepared_hash, old, _ = _load(directory, config)
    if isinstance(backend, inference.HFBackend):
        _real_preconditions(preparation, old)
    generation = _generation_complete(directory, prepared_hash)
    final = directory / "SCORE_SUCCESS"
    require(resume or not final.exists(), "Final scoring exists; do not overwrite")
    manifest = _compatible(directory, config, old)
    require(backend.eos_id == preparation["eos_token_id"], "Scoring EOS changed")
    prompts = {row["sample_id"]: {condition: backend.prompt_tokens(inference.build_messages(row, condition,
        inference._reference(row, old[5])[0])) for condition in inference.CONDITIONS} for row in old[4]}
    audits = [{"sample_id": row["sample_id"], "prompt_token_counts": {key: len(value) for key, value in prompts[row["sample_id"]].items()},
        "prompt_tokens_sha256": {key: inference.object_hash(value) for key, value in prompts[row["sample_id"]].items()}} for row in old[4]]
    require(encoded(audits) == read(directory / "prompt-audit.json"), "Scoring prompt audit changed")
    schedule = draw_schedule([row["sample_id"] for row in old[4]])
    draws = _events(directory, prepared_hash, schedule, config, backend.eos_id)
    require(len(draws) == 1260, "Final generation has missing raw draws")
    generation_cost = _generation_cost(directory, draws, prompts)
    require(generation.get("cost_accounting") == generation_cost, "Generation cost does not reproduce raw checkpoints")
    for (sample, _, draw), raw in draws.items():
        _raw(raw, sample, draw, config, backend.eos_id, getattr(backend, "tokenizer", None))
    rebuilt, mappings = _candidate_records(old[4], old[5], old[3], draws, prompts, backend, config)
    require(read(directory / "source-mappings.jsonl") == b"".join(encoded(row) for row in mappings), "Source mapping does not reproduce original raw draws")
    for pool in POOLS:
        require(read(directory / ("pools/" + pool + "/candidates.jsonl")) == b"".join(encoded(row) for row in rebuilt[pool]),
                "Candidate support does not reproduce its fixed raw draw assignment")
    evidence = {"schema_version": SCHEMA_VERSION, "protocol_sha256": preparation["protocol_sha256"],
        "baseline_manifest_sha256": preparation["baseline_manifest_sha256"],
        "generation_manifest_sha256": sha(read(directory / "generation-manifest.json")),
        "pool_random_count": 12, "shared_greedy_condition": "11", "shared_copy": True,
        "baseline_scores_reused": True, "baseline_score_records": 318, "labels_read": False, "student_execution": False,
        "generation_performed": True, "scoring_performed": True, "backend": generation["backend"]}
    manifest["candidate_support"] = dict(evidence, pool="combined_unique")
    run_hash = inference.object_hash(manifest)
    combined = directory / "combined"
    require(resume or not combined.exists(), "Scoring checkpoint exists; explicit resume required")
    exact(combined / "manifest.json", encoded(manifest))
    payloads = {pool: predict._records(read(directory / ("pools/" + pool + "/candidates.jsonl")), "pool candidates") for pool in POOLS}
    merged = {}
    for pool in POOLS:
        require([row["sample_id"] for row in payloads[pool]] == [row["sample_id"] for row in old[4]], "Pool samples/order incomplete")
        for row in payloads[pool]:
            sample = row["sample_id"]
            if sample not in merged:
                merged[sample] = dict(row, candidates=[])
            existing = {item["candidate_id"] for item in merged[sample]["candidates"]}
            for candidate in row["candidates"]:
                if candidate["candidate_id"] not in existing:
                    merged[sample]["candidates"].append(candidate)
                    existing.add(candidate["candidate_id"])
    candidate_raw = b"".join(encoded(_signed(row, run_hash)) for row in merged.values())
    exact(combined / "candidates.jsonl", candidate_raw)
    baseline_scores = predict._records(read(old[0] / "prepared/inference/scores.jsonl"), "original scores")
    expected_old = {(row["sample_id"], item["candidate_id"]) for row in merged.values() for item in row["candidates"] if item["candidate_id"] in old[7][row["sample_id"]]}
    require(len(expected_old) == 318, "Every original candidate must remain in both-pool union")
    score_path = combined / "scores.jsonl"
    if not score_path.exists():
        write(score_path, b"".join(encoded(_signed(row, run_hash)) for row in baseline_scores))
    else:
        require(resume, "Scores checkpoint exists; explicit resume required")
        cached = predict._records(read(score_path), "cached scores")
        indexed = {(row["sample_id"], row["candidate_id"]): row for row in cached}
        for row in baseline_scores:
            require(indexed.get((row["sample_id"], row["candidate_id"])) == _signed(row, run_hash), "Reused baseline scores changed")
    result = inference.run_inference(old[4], old[5], config, combined, run_hash, cache_rescore._NoGenerationBackend(backend))
    sources = {"inputs": read(directory / "inputs.jsonl"), "manifest": read(combined / "manifest.json"),
        "candidates": read(combined / "candidates.jsonl"), "scores": read(score_path)}
    predict._validated_run(sources)
    score_rows = {(row["sample_id"], row["candidate_id"]): row for row in predict._records(sources["scores"], "combined scores")}
    newly_scored = [value for key, value in score_rows.items() if key not in expected_old]
    summaries = {}
    for pool in POOLS:
        target = directory / "pools" / pool / "inference"
        pool_manifest = dict(manifest, candidate_support=dict(evidence, pool=pool))
        pool_hash = inference.object_hash(pool_manifest)
        exact(target / "manifest.json", encoded(pool_manifest))
        candidates = b"".join(encoded(_signed(row, pool_hash)) for row in payloads[pool])
        scores = b"".join(encoded(_signed(score_rows[row["sample_id"], item["candidate_id"]], pool_hash))
            for row in payloads[pool] for item in row["candidates"])
        exact(target / "candidates.jsonl", candidates)
        exact(target / "scores.jsonl", scores)
        predict._validated_run({"inputs": sources["inputs"], "manifest": encoded(pool_manifest), "candidates": candidates, "scores": scores})
        summaries[pool] = {"samples": 70, "candidates": sum(len(row["candidates"]) for row in payloads[pool]),
            "manifest_sha256": sha(encoded(pool_manifest)), "inference_dir": str(target)}
    phase_seconds = max(0, monotonic() - started)
    if final.exists():
        phase_seconds = predict._json(read(final))["cost_accounting"]["scoring_phase_wall_seconds"]
        require(type(phase_seconds) in (int, float) and math.isfinite(phase_seconds) and phase_seconds >= 0,
                "Invalid completed scoring timing")
    cost = {**generation_cost, "scoring_phase_wall_seconds": phase_seconds,
        "scoring_timing_scope": "first successful scoring call, including validation; interrupted scoring wall time remains in outer job logs and budget",
        "new_unique_candidates_scored": len(newly_scored), "new_scoring_branch_count": 4 * len(newly_scored),
        "new_scoring_branch_prompt_tokens": sum(sum(row["prompt_token_counts"].values()) for row in newly_scored),
        "new_scoring_branch_completion_tokens": 4 * sum(row["token_count"] for row in newly_scored),
        "reused_baseline_scores": 318, "score_pool_policy": "one common four-branch pass per new UTF-8-unique candidate; old compatible scores reused"}
    report = {"schema_version": SCHEMA_VERSION, "samples": 70, "students": 17, "new_draws": 1260,
        "combined_candidates": result["candidates"], "combined_scores": result["scores"],
        "baseline_scores_reused": 318, "new_score_records": result["scores"] - 318,
        "pools": summaries, "cost_accounting": cost,
        "backend": generation["backend"], "labels_read": False, "student_execution": False}
    exact(final, encoded(report))
    return report


def score(prepared_dir, config, backend, *, resume=False):
    with lease(prepared_dir):
        return _score(prepared_dir, config, backend, resume)


def run(prepared_dir, config, *, backend=None, resume=False):
    """Load one existing HF model, generate and score; never run student code."""
    with lease(prepared_dir):
        _, preparation, _, old, _ = _load(prepared_dir, config)
        _compatible(Path(prepared_dir), config, old)
        if backend is None or isinstance(backend, inference.HFBackend):
            _real_preconditions(preparation, old)
        active = inference.HFBackend(config) if backend is None else backend
        _generate(prepared_dir, config, active, resume)
        return _score(prepared_dir, config, active, resume)


def _real_preconditions(preparation, old):
    require(preparation["synthetic_tokenizer_injected"] is False
            and old[6].get("cache_rescore", {}).get("backend") == "HFBackend", "Real execution cannot consume synthetic preparation")
    require(os.environ.get("STUDENT_SIM_CONTROLLED_PROCESS_GROUP") == "1"
            and re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
            and hasattr(os, "sched_getaffinity") and 1 <= len(os.sched_getaffinity(0)) <= 4
            and all(os.environ.get(name) == "4" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")),
            "Real model execution requires the reviewed bounded outer process group, one explicit GPU and four CPU cores/threads")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "run", "generate", "score"))
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--baseline-run", type=Path)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.stage == "prepare":
            require(args.protocol is not None and args.baseline_run is not None and not args.resume, "Preparation requires baseline/protocol and a new directory")
            settings, _, _ = protocol(args.protocol)
            old = history_scoring.baseline(args.baseline_run, settings)
            config = inference.InferenceConfig(**dict(old[6]["config"], model_path=str(args.model_path), device=args.device))
            result = prepare(args.baseline_run, args.prepared, protocol=args.protocol, config=config)
        else:
            require(args.protocol is None and args.baseline_run is None, "Later stages use only frozen preparation sources")
            preparation = predict._json(read(args.prepared / "manifest.json"))
            config = inference.InferenceConfig(**dict(preparation["config"], model_path=str(args.model_path), device=args.device))
            if args.stage == "run":
                result = run(args.prepared, config, resume=args.resume)
            else:
                # Model allocation also occurs under the checkpoint lock.
                with lease(args.prepared):
                    _, preparation, _, old, _ = _load(args.prepared, config)
                    _compatible(args.prepared, config, old)
                    _real_preconditions(preparation, old)
                    active = inference.HFBackend(config)
                    function = _generate if args.stage == "generate" else _score
                    result = function(args.prepared, config, active, args.resume)
        print(inference.canonical_json(result), flush=True)
        return 0
    except (ValueError, OSError, ImportError) as error:
        parser.exit(2, "candidate-support: %s\n" % error)


if __name__ == "__main__":
    raise SystemExit(main())
