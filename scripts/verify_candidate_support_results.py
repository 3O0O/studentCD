#!/usr/bin/env python3
"""Independently replay candidate-support selections and static statistics.

Only Python's standard library is imported. No project analysis/model/student
module, tokenizer, network, subprocess, or student execution is used. Private
text is read in place; the returned report contains numbers, IDs and hashes.
This runs inside the CPU analysis child before the outer runner writes SUCCESS.
"""

import argparse
from collections import defaultdict
import difflib
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import stat
import statistics


SCHEMA = "student-sim-cd.candidate-support-independent-verification.v1"
POOLS = ("balanced12", "11_only12")
METHODS = ("base", "history_d0", "cd", "b", "copy")
PROMPTS = ("student_revision", "conservative_edit")
EXTERNALS = tuple(name + suffix for name in PROMPTS for suffix in ("_greedy", "_pool"))
PARAMETERS = {"base": {"alpha": 0, "beta": 0}, "history_d0": {"alpha": 1, "beta": 0},
              "cd": {"alpha": 1, "beta": 1}, "b": {"alpha": 0, "beta": 1}}
SEQUENCE = "canonical code tokens + exactly one tokenizer EOS; sum logp; no length normalization"
SELECTION = "argmax; ties use lexicographically smallest candidate_id"
SEED = 20260929
KNOWN_BASELINE = "9bafd1d5b2dd6a8c3da22dce69c128848fcf9a56411bdb935a93d0ca0b2e9328"
BASELINE_FILES = {"manifest.json", "SUCCESS", "prepared/inputs.jsonl", "prepared/references.jsonl",
                  "prepared/inference/manifest.json", "prepared/inference/candidates.jsonl", "prepared/inference/scores.jsonl"}
CANDIDATE_FIELDS = {"candidate_id", "code", "sources", "transform", "completion_token_ids", "token_count", "eos_included"}
CANDIDATE_RECORD_FIELDS = {"sample_id", "candidates", "attempts", "reference_kind", "reference_provenance", "run_sha256", "record_sha256"}
SCORE_FIELDS = {"sample_id", "candidate_id", "l11", "l01", "l10", "l00", "token_count", "eos_included",
                "prompt_token_counts", "reference_kind", "components", "scores_weight_1", "run_sha256", "record_sha256"}
WRAPPER_FIELDS = {"sample_id", "condition", "draw_id", "slot", "origin", "baseline_record_sha256", "raw_generation", "prompt_tokens_sha256"}
FORBIDDEN = {"code", "current_code", "target_code", "predicted_code", "raw_text", "history", "reference_history",
             "raw_generation", "generated_token_ids", "completion_token_ids"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def object_sha(value):
    return sha(canonical(value).encode("utf-8"))


def digest(value, name):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), "Invalid SHA256: " + name)
    return value


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON field")
        result[key] = value
    return result


def parsed(raw):
    value = json.loads(raw, object_pairs_hook=unique)
    canonical(value)
    return value


def records(raw, name):
    lines = raw.decode("utf-8").split("\n")
    if lines[-1] == "":
        lines.pop()
    require(bool(lines) and all(line.strip() for line in lines), "Empty JSONL: " + name)
    values = [parsed(line) for line in lines]
    require(all(isinstance(value, dict) for value in values), "Non-object JSONL: " + name)
    return values


def index(values, keys, name):
    result = {}
    for value in values:
        key = tuple(value.get(name_) for name_ in keys)
        require(all(item is not None for item in key) and key not in result, "Missing/duplicate identity: " + name)
        result[key] = value
    return result


def equal(actual, expected, name):
    """Exact schema/strings/integers; only float roundoff permits 1e-12 absolute."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), "Dictionary fields differ: " + name)
        for key, value in expected.items():
            equal(actual[key], value, name + "." + str(key))
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), "List length/type differs: " + name)
        for number, (left, right) in enumerate(zip(actual, expected)):
            equal(left, right, name + "." + str(number))
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        require(type(actual) is type(expected) and actual == expected, "Value/type differs: " + name)
    elif isinstance(expected, (float, int)):
        require(type(actual) in (float, int) and math.isfinite(actual) and math.isfinite(expected)
                and (actual == expected if isinstance(expected, int) else abs(actual - expected) <= 1e-12),
                "Numeric value differs: " + name)
    else:
        raise ValueError("Unsupported comparison type: " + name)


class Reader:
    def __init__(self, root):
        self.root = self.path(root, directory=True)
        self.hashes, self.cache, self.total = {}, {}, 0

    @staticmethod
    def path(value, directory=False):
        path = Path(value).absolute()
        require(not path.is_symlink() and path.resolve(strict=True) == path, "Require canonical artifact path")
        info = path.stat()
        require(info.st_uid == os.getuid() and (stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)),
                "Require owned regular artifact/directory")
        return path

    def read(self, path):
        path = self.path(path)
        if path in self.cache:
            return self.cache[path]
        before = path.stat()
        require(before.st_size <= 256 * 1024 ** 2 and self.total + before.st_size <= 1024 ** 3, "Artifact read budget exceeded")
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as stream:
            raw = stream.read()
        after = path.stat()
        require(len(raw) == before.st_size and all(getattr(before, field) == getattr(after, field) for field in
                ("st_dev", "st_ino", "st_mode", "st_uid", "st_size", "st_mtime_ns", "st_ctime_ns")), "Artifact changed while reading")
        self.total += len(raw)
        self.cache[path] = raw
        key = str(path.relative_to(self.root)) if self.root in path.parents else "external:" + str(path)
        self.hashes[key] = sha(raw)
        return raw

    def json(self, path):
        return parsed(self.read(path))

    def complete(self, directory, manifest="manifest.json", marker="SUCCESS"):
        raw = self.read(directory / manifest)
        require(self.read(directory / marker) == (sha(raw) + "\n").encode("ascii"), "Completion SHA marker differs")
        require(not (directory / ".incomplete").exists(), "Preserve incomplete artifact directory")
        return parsed(raw)


def signed(row, run_hash, fields=None):
    require((fields is None or set(row) == fields) and row.get("run_sha256") == run_hash
            and row.get("record_sha256") == object_sha({key: value for key, value in row.items() if key != "record_sha256"}),
            "Record schema/run/content SHA differs")


def extract(raw):
    """Frozen cached-python-fence-v1, independently implemented."""
    require(isinstance(raw, str), "Raw generation text must be a string")
    if raw.startswith(("```python\n", "```py\n", "```python\r\n", "```py\r\n")):
        match = re.fullmatch(r"```(?:python|py)?\r?\n([\s\S]*?\r?\n)```(?:\r?\n)?", raw)
        if match and "```" not in match.group(1):
            return match.group(1), "unwrapped_single_outer_fence", "single_python_fence_extracted"
        return raw, "none", "python_fence_not_single_complete_response"
    return raw, "none", ("unsupported_or_unlabelled_fence_preserved" if raw.startswith("```") else
                         "non_whole_or_multiple_fence_preserved" if "```" in raw else "original_text_preserved")


def raw_attempt(raw, sid, draw, config, eos):
    tokens, complete = raw.get("generated_token_ids"), raw.get("eos_reached")
    require(type(raw.get("attempt")) is int and raw["attempt"] == draw
            and raw.get("source") == ("greedy" if draw == 0 else "sampled")
            and type(raw.get("seed")) is int and raw["seed"] == int(object_sha([config["seed"], sid, draw])[:8], 16)
            and isinstance(raw.get("raw_text"), str) and type(complete) is bool
            and isinstance(tokens, list) and tokens and all(type(item) is int and item >= 0 for item in tokens)
            and len(tokens) <= config["max_new_tokens"]
            and raw.get("finish_reason") == ("eos" if complete else "length_limit"), "Raw attempt identity/seed/token budget differs")
    require((tokens[-1] == eos and eos not in tokens[:-1]) if complete else
            (eos not in tokens and len(tokens) == config["max_new_tokens"]), "Raw attempt single-EOS/length-limit protocol differs")
    return extract(raw["raw_text"])[0] if complete else None


def derived(row):
    raw = [row[name] for name in ("l11", "l01", "l10", "l00")]
    require(all(type(value) in (int, float) and math.isfinite(value) and value <= 0 for value in raw), "Invalid four-condition log probability")
    l11, l01, l10, l00 = raw
    d0, d1 = l10 - l00, l11 - l01
    gamma = d1 - d0
    components = {"base": l11, "d0": d0, "d1": d1, "gamma": gamma}
    scores = {"base": l11 + 0.0 * d0 + 0.0 * gamma, "history_d0": l11 + 1.0 * d0 + 0.0 * gamma,
              "cd": l11 + 1.0 * d0 + 1.0 * gamma, "b": l11 + 0.0 * d0 + 1.0 * gamma,
              "joint": l11 + 0.0 * d0 + 1.0 * gamma}
    equal(row.get("components"), components, "score.components")
    equal(row.get("scores_weight_1"), scores, "score.fixed_methods")
    return scores


def inference_cache(reader, directory, input_raw, inputs, eos):
    manifest = reader.json(directory / "manifest.json")
    require(manifest.get("schema_version") == "student-sim-cd.inference.v1"
            and manifest.get("config_sha256") == object_sha(manifest.get("config"))
            and manifest.get("inputs_sha256") == sha(input_raw) and manifest.get("sequence_protocol") == SEQUENCE,
            "Inference manifest schema/config/input/EOS protocol differs")
    run_hash = object_sha(manifest)
    candidate_rows = records(reader.read(directory / "candidates.jsonl"), "candidates")
    keyed = index(candidate_rows, ("sample_id",), "candidate cohort")
    require(set(keyed) == {(sid,) for sid in inputs}, "Incomplete candidate cohort")
    pools, copies = {}, {}
    for (sid,), row in keyed.items():
        signed(row, run_hash, CANDIDATE_RECORD_FIELDS)
        require(row["reference_kind"] == "matched", "Matched reference cohort cannot fall back")
        pool = {}
        for candidate in row["candidates"]:
            code, tokens = candidate.get("code"), candidate.get("completion_token_ids")
            identifier = candidate.get("candidate_id")
            require(set(candidate) == CANDIDATE_FIELDS and isinstance(code, str) and identifier == sha(code.encode())
                    and identifier not in pool and isinstance(tokens, list) and tokens
                    and all(type(item) is int and item >= 0 for item in tokens)
                    and type(candidate["token_count"]) is int and candidate["token_count"] == len(tokens)
                    and candidate["eos_included"] is True and tokens[-1] == eos and eos not in tokens[:-1], "Canonical candidate content/token/EOS differs")
            require(isinstance(candidate["sources"], list) and candidate["sources"]
                    and all(isinstance(source, str) for source in candidate["sources"])
                    and len(set(candidate["sources"])) == len(candidate["sources"]), "Invalid candidate source tags")
            if "copy_current" in candidate["sources"]:
                require(sid not in copies and code == inputs[sid]["current_code"], "Copy does not match current input")
                copies[sid] = identifier
            pool[identifier] = candidate
        require(sid in copies, "Copy baseline absent")
        pools[sid] = pool
    score_rows = records(reader.read(directory / "scores.jsonl"), "scores")
    scores, original_scores = {}, {}
    for row in score_rows:
        signed(row, run_hash, SCORE_FIELDS)
        key = row["sample_id"], row["candidate_id"]
        require(key not in scores and key[0] in pools and key[1] in pools[key[0]], "Unknown/duplicate score key")
        candidate = pools[key[0]][key[1]]
        require(row["token_count"] == candidate["token_count"] and row["eos_included"] is True
                and row["reference_kind"] == "matched" and isinstance(row["prompt_token_counts"], dict)
                and set(row["prompt_token_counts"]) == {"11", "01", "10", "00"}
                and all(type(value) is int and value > 0 for value in row["prompt_token_counts"].values()), "Score candidate/prompt protocol differs")
        scores[key] = derived(row)
        original_scores[key] = row
    require(set(scores) == {(sid, cid) for sid, pool in pools.items() for cid in pool}, "Four-condition scoring incomplete")
    return {"manifest": manifest, "run_hash": run_hash, "records": {sid: value for (sid,), value in keyed.items()},
            "pools": pools, "copies": copies, "scores": scores, "raw_scores": original_scores}


def support_cache(cache, pool, inputs, config, eos, prompt_audit, baseline, prepared_hash, generation_hash, protocol_hash):
    metadata = cache["manifest"].get("candidate_support")
    require(isinstance(metadata, dict) and metadata.get("pool") == pool and metadata.get("protocol_sha256") == protocol_hash
            and metadata.get("baseline_manifest_sha256") == baseline["completion_sha"]
            and metadata.get("generation_manifest_sha256") == generation_hash
            and metadata.get("pool_random_count") == 12 and metadata.get("shared_greedy_condition") == "11"
            and metadata.get("shared_copy") is True and metadata.get("generation_performed") is True
            and metadata.get("scoring_performed") is True and metadata.get("labels_read") is False
            and metadata.get("student_execution") is False and metadata.get("baseline_scores_reused") is True
            and metadata.get("baseline_score_records") == 318, "Support pool production provenance differs")
    greedy, choices, codes, coverage = {}, [], {method: {} for method in METHODS}, []
    originals = {}
    for sid in sorted(inputs):
        row, candidates = cache["records"][sid], cache["pools"][sid]
        require(isinstance(row["attempts"], list) and len(row["attempts"]) == 13 and 1 <= len(candidates) <= 14, "Support attempt/candidate count differs")
        expected, original = {}, {}
        counts = {condition: {"random_attempts": 0, "eligible_random_attempts": 0, "unique_random_candidate_ids": set()} for condition in ("11", "01", "10", "00")}
        seen = set()
        for wrapper in row["attempts"]:
            draw = wrapper.get("draw_id")
            condition = "11" if pool == "11_only12" or type(draw) is int and draw <= 3 else "01" if type(draw) is int and draw <= 6 else "10" if type(draw) is int and draw <= 9 else "00"
            require(set(wrapper) == WRAPPER_FIELDS and wrapper["sample_id"] == sid
                    and type(draw) is int and 0 <= draw <= 12 and draw not in seen and wrapper["condition"] == condition
                    and wrapper["prompt_tokens_sha256"] == prompt_audit[sid]["prompt_tokens_sha256"][condition], "Support wrapper allocation/prompt differs")
            seen.add(draw)
            code = raw_attempt(wrapper["raw_generation"], sid, draw, config, eos)
            cid = sha(code.encode()) if code is not None else None
            origin = "greedy:11:0" if draw == 0 else "sampled:" + condition + ":" + str(draw)
            expected[origin] = cid
            if draw <= 3:
                require(wrapper["origin"] == "baseline_cache" and wrapper["slot"] == draw
                        and wrapper["baseline_record_sha256"] == baseline["originals"][sid]["record_sha256"]
                        and wrapper["raw_generation"] == baseline["originals"][sid]["attempts"][draw], "Shared original draw changed from frozen baseline")
                original[draw] = wrapper
            else:
                require(wrapper["origin"] == "new_model_generation" and wrapper["baseline_record_sha256"] is None
                        and wrapper["slot"] == (draw if condition == "11" else (draw - 4) % 3 + 1), "New draw slot/origin differs")
            if draw:
                counts[condition]["random_attempts"] += 1
                if cid is not None:
                    counts[condition]["eligible_random_attempts"] += 1
                    counts[condition]["unique_random_candidate_ids"].add(cid)
            else:
                require(cid is not None, "Original greedy is incomplete")
                greedy[sid] = cid
        require(seen == set(range(13)), "Raw attempt coverage incomplete")
        actual = {}
        for cid, candidate in candidates.items():
            for origin in candidate["sources"]:
                if origin == "copy_current":
                    require(cid == cache["copies"][sid], "Copy mapping differs")
                else:
                    require(origin in expected and cid == expected[origin] and origin not in actual, "Candidate/raw-source mapping differs")
                    actual[origin] = cid
        require(actual == {origin: cid for origin, cid in expected.items() if cid is not None}, "Eligible/truncated draw mapping incomplete")
        for cid, old_candidate in baseline["cache"]["pools"][sid].items():
            require(cid in candidates, "Original candidate removed from new pool")
            for field in ("code", "completion_token_ids", "token_count", "eos_included"):
                equal(candidates[cid][field], old_candidate[field], "baseline.candidate." + field)
            for field in ("l11", "l01", "l10", "l00", "token_count", "eos_included", "prompt_token_counts"):
                require(cache["raw_scores"][sid, cid][field] == baseline["cache"]["raw_scores"][sid, cid][field], "Reused original raw score changed")
        selection = {}
        for method in METHODS:
            cid = cache["copies"][sid] if method == "copy" else min(candidates, key=lambda item: (-cache["scores"][sid, item][method], item))
            codes[method][sid] = candidates[cid]["code"]
            selection[method] = {"candidate_id": cid, "score": None if method == "copy" else cache["scores"][sid, cid][method]}
        choices.append({"pool": pool, "sample_id": sid, "candidate_count": len(candidates), "choices": selection})
        coverage.append({"sample_id": sid, "student_id": inputs[sid]["student_id"], "candidate_count": len(candidates),
            "retained_raw_attempts": 13, "conditions": {condition: {**values, "unique_random_candidate_ids": sorted(values["unique_random_candidate_ids"])} for condition, values in counts.items()},
            "shared_greedy_candidate_id": greedy[sid], "copy_candidate_id": cache["copies"][sid]})
        originals[sid] = original
    cache.update(greedy=greedy, choices=choices, codes=codes, coverage=coverage, originals=originals)


def prompt_baselines(reader, directory, inputs, references, config, eos, protocol, protocol_hash, main_cache, audits):
    manifest = reader.complete(directory)
    fingerprint = reader.json(directory / "checkpoint-manifest.json")
    require(manifest.get("schema_version") == "student-sim-cd.prompt-baselines.v1"
            and manifest.get("status") == "success" and manifest.get("samples") == 70
            and manifest.get("attempts") == 1820 and manifest.get("attempts_per_prompt") == 13
            and manifest.get("random_attempts") == 12 and manifest.get("labels_read") is False
            and manifest.get("student_execution") is False and manifest.get("protocol_sha256") == protocol_hash
            and manifest.get("config") == config and manifest.get("model") == main_cache["manifest"]["model"]
            and manifest.get("runtime_versions") == main_cache["manifest"]["runtime_versions"]
            and manifest.get("backend") == main_cache["manifest"]["candidate_support"]["backend"]
            and manifest.get("sequence_protocol") == SEQUENCE
            and manifest.get("prompts") == protocol["prompt_baseline_texts"], "Strong prompt baseline identity/design differs")
    for field, value in fingerprint.items():
        require(manifest.get(field) == value, "Strong fingerprint/final manifest differs")
    run_hash = object_sha(fingerprint)
    require(manifest.get("run_sha256") == run_hash and fingerprint.get("inputs_sha256") == object_sha(list(inputs.values()))
            and fingerprint.get("references_sha256") == object_sha(references), "Strong input/reference/run identity differs")
    files = manifest.get("files_sha256")
    required_files = {method + ".jsonl" for method in EXTERNALS} | {"raw.jsonl", "scores.jsonl", "summary.json", "checkpoint-manifest.json", "reservations.jsonl"}
    require(isinstance(files, dict) and set(files) == required_files, "Strong completed file inventory differs")
    for name, value in files.items():
        require(sha(reader.read(directory / name)) == digest(value, "strong file"), "Strong completed file changed")
    raw = index(records(reader.read(directory / "raw.jsonl"), "strong raw"), ("sample_id", "method", "attempt"), "strong raw")
    reserved = index(records(reader.read(directory / "reservations.jsonl"), "strong reservations"), ("sample_id", "method", "attempt"), "strong reservations")
    scored = index(records(reader.read(directory / "scores.jsonl"), "strong scores"), ("sample_id", "method", "candidate_id"), "strong scores")
    summaries = index(reader.json(directory / "summary.json"), ("sample_id", "method"), "strong summary")
    require(set(raw) == set(reserved) == {(sid, method, attempt) for sid in inputs for method in PROMPTS for attempt in range(13)}
            and set(summaries) == {(sid, method) for sid in inputs for method in PROMPTS}, "Strong complete attempt/summary cohort differs")
    for row in (*raw.values(), *reserved.values(), *scored.values()):
        signed(row, run_hash)
    predictions = {method: index(records(reader.read(directory / (method + ".jsonl")), "strong predictions"), ("sample_id",), "strong predictions") for method in EXTERNALS}
    for method in EXTERNALS:
        require(set(predictions[method]) == {(sid,) for sid in inputs}, "Strong prediction cohort incomplete")
    codes, all_score_keys, selections = {method: {} for method in EXTERNALS}, set(), []
    for sid in inputs:
        for prompt in PROMPTS:
            summary = summaries[sid, prompt]
            audit = audits[sid, prompt]
            require(summary["prompt_sha256"] == audit["prompt_sha256"] and summary["prompt_token_count"] == audit["prompt_tokens"], "Strong prompt audit differs")
            pool = {sha(inputs[sid]["current_code"].encode()): inputs[sid]["current_code"]}
            sources = {next(iter(pool)): ["copy_current"]}
            greedy, eligibility = None, []
            for attempt in range(13):
                row = raw[sid, prompt, attempt]
                reservation = reserved[sid, prompt, attempt]
                require(row.get("prompt_sha256") == reservation.get("prompt_sha256") == summary["prompt_sha256"], "Strong reservation/current prompt differs")
                code = raw_attempt(row, sid, attempt, config, eos)
                cid = sha(code.encode()) if code is not None else None
                declared = summary["eligible_attempts"][attempt]
                require(declared["attempt"] == attempt and declared["candidate_id"] == cid,
                        "Strong eligibility mapping differs; complete EOS output may not be silently lost")
                if cid is not None:
                    pool[cid] = code
                    sources.setdefault(cid, []).append("greedy:0" if attempt == 0 else "sampled:" + str(attempt))
                    require(declared.get("exclusion_reason") is None and declared.get("extraction") == extract(row["raw_text"])[2], "Strong extraction provenance differs")
                else:
                    require(isinstance(declared.get("exclusion_reason"), str) and declared["exclusion_reason"], "Strong incomplete draw exclusion missing")
                if attempt == 0:
                    require(cid is not None, "Strong greedy is incomplete; no copy substitution")
                    greedy = cid
                eligibility.append(cid)
            require(summary["candidates"] == len(pool) and summary["sources"] == sources, "Strong canonical candidate/source coverage differs")
            for cid in pool:
                key = sid, prompt, cid
                require(key in scored, "Strong candidate score absent")
                item = scored[key]
                require(type(item.get("logp")) in (int, float) and math.isfinite(item["logp"]) and item["logp"] <= 0
                        and item.get("prompt_sha256") == summary["prompt_sha256"] and type(item.get("token_count")) is int
                        and item["token_count"] > 0, "Strong one-branch score invalid")
                digest(item.get("completion_sha256"), "strong completion tokens")
                all_score_keys.add(key)
            best = min(pool, key=lambda cid: (-scored[sid, prompt, cid]["logp"], cid))
            for suffix, cid in (("_greedy", greedy), ("_pool", best)):
                method = prompt + suffix
                equal(predictions[method][sid,], {"sample_id": sid, "method": method, "predicted_code": pool[cid]}, "strong actual prediction")
                codes[method][sid] = pool[cid]
                selections.append({"sample_id": sid, "method": method, "candidate_id": cid, "logp": scored[sid, prompt, cid]["logp"]})
    require(set(scored) == all_score_keys and manifest["scores"] == len(scored), "Strong extra/missing score checkpoint")
    require(manifest["generated_tokens"] == sum(len(row["generated_token_ids"]) for row in raw.values()), "Strong generated token count differs")
    equal(manifest["generation_seconds"], math.fsum(row["generation_seconds"] for row in raw.values()), "strong generation time")
    equal(manifest["scoring_seconds"], math.fsum(row["scoring_seconds"] for row in scored.values()), "strong scoring time")
    return codes, selections, manifest


def generation_checkpoints(reader, prepared, inputs, config, eos, caches):
    """Bind all new pool wrappers to exactly 1260 durable reserved/raw draws."""
    events = records(reader.read(prepared / "draw-checkpoints.jsonl"), "main draw checkpoints")
    schedule = []
    for sid in inputs:
        schedule.extend((sid, "11", draw, draw) for draw in range(4, 13))
        for condition, start in (("01", 4), ("10", 7), ("00", 10)):
            schedule.extend((sid, condition, draw, slot) for slot, draw in enumerate(range(start, start + 3), 1))
    require(len(schedule) == 1260 and len(events) == 2 * len(schedule), "Main generation reservation/raw budget incomplete")
    prepared_hash, observed = sha(reader.read(prepared / "manifest.json")), {}
    for number, (sid, condition, draw, slot) in enumerate(schedule):
        reservation, result = events[2 * number:2 * number + 2]
        identity = {"sample_id": sid, "condition": condition, "draw_id": draw, "slot": slot,
                    "seed": int(object_sha([config["seed"], sid, draw])[:8], 16), "prepared_manifest_sha256": prepared_hash}
        require(reservation == {**identity, "event": "reserved"} and result.get("identity") == identity
                and set(result) == {"event", "identity", "raw_generation", "elapsed_seconds", "record_sha256"}
                and result.get("event") == "raw" and result.get("record_sha256") == object_sha({key: value for key, value in result.items() if key != "record_sha256"})
                and type(result.get("elapsed_seconds")) in (int, float) and math.isfinite(result["elapsed_seconds"])
                and result["elapsed_seconds"] >= 0, "Main generation durable reservation/raw identity differs")
        raw_attempt(result["raw_generation"], sid, draw, config, eos)
        observed[sid, condition, draw] = result["raw_generation"]
    from_pools = {}
    for cache in caches.values():
        for sid, row in cache["records"].items():
            for wrapper in row["attempts"]:
                if wrapper["draw_id"] >= 4:
                    key = sid, wrapper["condition"], wrapper["draw_id"]
                    require(key not in from_pools or from_pools[key] == wrapper["raw_generation"], "Duplicate new source wrappers disagree")
                    from_pools[key] = wrapper["raw_generation"]
    require(from_pools == observed, "Pool raw generations differ from the sealed durable draw source")
    return len(observed)


def edit_events(before, after):
    locations, operations, size = set(), set(), 0
    for operation, start, end, new_start, new_end in difflib.SequenceMatcher(
            a=before.splitlines(keepends=True), b=after.splitlines(keepends=True), autojunk=False).get_opcodes():
        if operation == "equal":
            continue
        if operation == "insert":
            locations.add(("boundary", start))
            operations.add(("insert", start))
        else:
            for position in range(start, end):
                locations.add(("line", position))
                operations.add((operation, position))
        size += end - start + new_end - new_start
    return locations, operations, size


def f1(left, right):
    return 1.0 if not left and not right else 2 * len(left & right) / (len(left) + len(right))


def metrics(current, predicted, target):
    pred_locations, pred_operations, pred_size = edit_events(current, predicted)
    true_locations, true_operations, true_size = edit_events(current, target)
    return {"exact_next_code": float(predicted == target), "true_changed": float(target != current),
        "predicted_changed": float(predicted != current), "edit_location_f1": f1(pred_locations, true_locations),
        "edit_operation_f1": f1(pred_operations, true_operations), "edit_size_absolute_error": abs(pred_size - true_size),
        "text_similarity": difflib.SequenceMatcher(a=predicted, b=target, autojunk=False).ratio(),
        "predicted_edit_size": pred_size, "true_edit_size": true_size, "predicted_unchanged": float(predicted == current),
        "false_edit_on_unchanged": float(target == current and predicted != current)}


@lru_cache(maxsize=4096)
def bootstrap_ci(values, repetitions, seed):
    rng, simulated = random.Random(seed), []
    for _ in range(repetitions):
        simulated.append(statistics.mean(values[rng.randrange(len(values))] for _ in values))
    simulated.sort()
    return simulated[int(0.025 * len(simulated))], simulated[min(len(simulated) - 1, int(0.975 * len(simulated)))]


def summary(rows, repetitions=2000, seed=SEED, delta=False):
    if not rows:
        return {"samples": 0, "students": 0, "metrics": {}, "per_student": []}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["student_id"]].append(row)
    names = sorted(rows[0]["metrics"])
    students = [{"student_id": sid, "samples": len(items), "metrics": {
        name: statistics.mean(row["metrics"][name] for row in items) for name in names}} for sid, items in sorted(grouped.items())]
    output = {}
    for name in names:
        values = tuple(student["metrics"][name] for student in students)
        suffix = "delta" if delta else "mean"
        output[name] = {"student_macro_" + suffix: statistics.mean(values),
            "submission_" + suffix: statistics.mean(row["metrics"][name] for row in rows),
            ("paired_student_bootstrap_95_ci" if delta else "student_bootstrap_95_ci"): list(bootstrap_ci(values, repetitions, seed))}
        if delta:
            output[name]["student_delta_sign_counts"] = {"positive": sum(value > 1e-12 for value in values),
                "tie": sum(abs(value) <= 1e-12 for value in values), "negative": sum(value < -1e-12 for value in values)}
    return {"samples": len(rows), "students": len(students), "metrics": output, "per_student": students}


def difference(left, right):
    require([row["sample_id"] for row in left] == [row["sample_id"] for row in right], "Paired cohort/order differs")
    return [{**first, "metrics": {name: first["metrics"][name] - second["metrics"][name] for name in first["metrics"]}}
            for first, second in zip(left, right)]


def verify(run_dir, labels_path):
    """Return status='verified' only after all in-place independent checks pass."""
    reader = Reader(run_dir)
    root, prepared = reader.root, reader.root / "prepared"
    preparation = reader.complete(prepared)
    protocol_raw = reader.read(prepared / "protocol.json")
    protocol, protocol_hash = parsed(protocol_raw), sha(protocol_raw)
    expected = {"schema_version": "student-sim-cd.candidate-support.protocol.v1", "expected_samples": 70,
        "expected_students": 17, "seed": SEED, "conditions": ["11", "01", "10", "00"],
        "balanced_random_per_condition": 3, "control_random_count": 12, "shared_greedy_condition": "11",
        "canonical_protocol": "cached-python-fence-v1", "new_support_attempts": 1260,
        "new_prompt_baseline_attempts": 1820, "max_new_generation_attempts": 3080,
        "generation_performed": True, "student_execution": False, "test_set_used": False, "parameters_fitted": False,
        "methods": list(METHODS), "fixed_method_parameters": PARAMETERS, "sequence_protocol": SEQUENCE,
        "selection": SELECTION, "bootstrap": {"unit": "student", "paired": True, "repetitions": 2000, "seed": SEED}}
    require(isinstance(protocol, dict) and all(protocol.get(name) == value and type(protocol.get(name)) is type(value)
            for name, value in expected.items()), "Frozen protocol/estimand/complete budget differs")
    require(preparation.get("schema_version") == "student-sim-cd.candidate-support.v1" and preparation.get("stage") == "prepared"
            and preparation.get("protocol") == protocol and preparation.get("protocol_sha256") == protocol_hash
            and preparation.get("samples") == 70 and preparation.get("students") == 17
            and preparation.get("labels_read") is False and preparation.get("student_execution") is False,
            "Prepared protocol/cohort differs")
    inventory = preparation.get("files_sha256")
    require(isinstance(inventory, dict) and set(inventory) == {"protocol.json", "inputs.jsonl", "references.jsonl", "prompt-audit.json"}, "Prepared inventory differs")
    for name, value in inventory.items():
        require(sha(reader.read(prepared / name)) == digest(value, "prepared file"), "Prepared source SHA differs")
    input_raw, reference_raw = reader.read(prepared / "inputs.jsonl"), reader.read(prepared / "references.jsonl")
    input_rows = records(input_raw, "inputs")
    inputs = {sid: row for (sid,), row in index(input_rows, ("sample_id",), "inputs").items()}
    references = {sid: row for (sid,), row in index(records(reference_raw, "references"), ("sample_id",), "references").items()}
    require(len(inputs) == 70 and len({row.get("student_id") for row in inputs.values()}) == 17
            and set(references) == set(inputs) and all(isinstance(row.get("current_code"), str)
                and row.get("schema_version") == "student-sim-cd.progfeed.v1" and "target_code" not in row and "target_results" not in row for row in inputs.values()), "Complete input cohort or future-label exclusion differs")
    seals = protocol.get("baseline_sha256")
    require(isinstance(seals, dict) and set(seals) == BASELINE_FILES, "Frozen baseline SHA inventory differs")
    baseline_root = Reader.path(preparation["baseline_run"], directory=True)
    require(baseline_root.parent == root.parent and baseline_root != root, "Baseline must remain an original sibling")
    for name, value in seals.items():
        require(sha(reader.read(baseline_root / name)) == digest(value, "baseline source"), "Original frozen source SHA differs")
    completion = reader.complete(baseline_root)
    require(completion.get("status") == "success" and completion.get("exit_code") == 0
            and len(completion.get("stages", [])) == 11
            and completion.get("source_root", "").endswith("/releases/" + protocol["baseline_release_sha256"])
            and all(stage.get("status") == "success" and stage.get("exit_code") == 0 for stage in completion.get("stages", [])), "Original source completion differs")
    require(sha(input_raw) == seals["prepared/inputs.jsonl"] and sha(reference_raw) == seals["prepared/references.jsonl"]
            and preparation.get("baseline_manifest_sha256") == seals["manifest.json"], "Input/reference source anchor differs")
    config = protocol.get("generation_config")
    require(isinstance(config, dict) and config == preparation.get("config") and config.get("seed") == SEED
            and config.get("max_new_tokens") == 2048 and config.get("max_context_tokens") == 16384
            and config.get("fence_policy") == "unwrap-single", "Frozen generation/scoring/context settings differ")
    eos = preparation.get("eos_token_id")
    require(type(eos) is int and eos >= 0 and eos in preparation.get("special_token_ids", []), "Prepared EOS identity differs")
    baseline_cache = inference_cache(reader, baseline_root / "prepared/inference", input_raw, inputs, eos)
    require(sum(map(len, baseline_cache["pools"].values())) == 318, "Frozen original candidate count differs")
    old_preparation = reader.json(baseline_root / "prepared/manifest.json")
    old_records_raw = reader.read(baseline_root / "prepared/prepared.jsonl")
    old_preparation_hash = object_sha(old_preparation)
    old_cache_metadata = baseline_cache["manifest"].get("cache_rescore", {})
    require(old_cache_metadata.get("prepared_run_sha256") == old_preparation_hash
            and old_cache_metadata.get("prepared_records_sha256") == sha(old_records_raw), "Original preparation is not bound by the frozen inference manifest")
    original_rows = records(old_records_raw, "original prepared records")
    originals = {sid: row for (sid,), row in index(original_rows, ("sample_id",), "original prepared records").items()}
    require(set(originals) == set(inputs) and old_preparation.get("records_payload_sha256") == object_sha([
            {key: value for key, value in row.items() if key not in {"run_sha256", "record_sha256"}} for row in original_rows]),
            "Original prepared record payload/cohort differs")
    for sid, row in originals.items():
        signed(row, old_preparation_hash)
        require(row.get("attempts") == baseline_cache["records"][sid]["attempts"], "Original raw attempts differ between preparation and frozen scores")
    baseline = {"completion_sha": seals["manifest.json"], "cache": baseline_cache, "originals": originals}
    prompt_audit = {sid: row for (sid,), row in index(reader.json(prepared / "prompt-audit.json"), ("sample_id",), "prompt audit").items()}
    require(set(prompt_audit) == set(inputs), "Full four-condition prompt audit absent")
    generation = reader.complete(prepared, "generation-manifest.json", "GENERATION_SUCCESS")
    require(generation.get("stage") == "generated" and generation.get("prepared_manifest_sha256") == sha(reader.read(prepared / "manifest.json"))
            and generation.get("protocol_sha256") == protocol_hash and generation.get("new_draws") == 1260
            and generation.get("samples") == 70 and generation.get("students") == 17
            and generation.get("labels_read") is False and generation.get("student_execution") is False
            and generation.get("automatic_replacements") == 0, "Generation finalization/budget differs")
    generated_files = generation.get("files_sha256")
    require(isinstance(generated_files, dict) and set(generated_files) == {"draw-checkpoints.jsonl", "source-mappings.jsonl", "pools/balanced12/candidates.jsonl", "pools/11_only12/candidates.jsonl"}, "Generation final inventory differs")
    for name, value in generated_files.items():
        require(sha(reader.read(prepared / name)) == digest(value, "generation file"), "Generation artifact changed")
    caches = {pool: inference_cache(reader, prepared / "pools" / pool / "inference", input_raw, inputs, eos) for pool in POOLS}
    backend = generation.get("backend")
    require(backend in {"HFBackend", "injected_synthetic_backend"}, "Unknown generation backend")
    if backend == "HFBackend":
        require(seals["manifest.json"] == KNOWN_BASELINE and preparation.get("synthetic_tokenizer_injected") is False, "Real execution source/synthetic preparation anchor differs")
    for pool, cache in caches.items():
        require(cache["manifest"]["config"] == config and cache["manifest"]["references_sha256"] == sha(reference_raw)
                and cache["manifest"].get("candidate_support", {}).get("backend") == backend
                and cache["manifest"]["model"] == baseline_cache["manifest"]["model"]
                and cache["manifest"]["runtime_versions"] == baseline_cache["manifest"]["runtime_versions"], "Pool model/runtime/config/reference/backend differs")
        support_cache(cache, pool, inputs, config, eos, prompt_audit, baseline,
            sha(reader.read(prepared / "manifest.json")), sha(reader.read(prepared / "generation-manifest.json")), protocol_hash)
    require(generation_checkpoints(reader, prepared, inputs, config, eos, caches) == 1260, "Complete new main draw source is required")
    overlap = []
    for sid in inputs:
        left, right = caches["balanced12"], caches["11_only12"]
        require(left["originals"][sid] == right["originals"][sid], "Shared original raw wrappers differ")
        common = set(left["pools"][sid]) & set(right["pools"][sid])
        for cid in common:
            for field in ("code", "completion_token_ids", "token_count", "eos_included"):
                require(left["pools"][sid][cid][field] == right["pools"][sid][cid][field], "Shared candidate token/content changed")
            for field in ("l11", "l01", "l10", "l00", "token_count", "eos_included", "prompt_token_counts"):
                require(left["raw_scores"][sid, cid][field] == right["raw_scores"][sid, cid][field], "Shared union score changed")
        overlap.append({"sample_id": sid, "shared_candidate_ids": sorted(common),
            "balanced_only_candidates": len(left["pools"][sid]) - len(common), "control_only_candidates": len(right["pools"][sid]) - len(common)})
    score_final = reader.json(prepared / "SCORE_SUCCESS")
    require(score_final.get("schema_version") == "student-sim-cd.candidate-support.v1"
            and score_final.get("samples") == 70 and score_final.get("students") == 17
            and score_final.get("new_draws") == 1260 and score_final.get("baseline_scores_reused") == 318
            and score_final.get("backend") == backend and score_final.get("labels_read") is False
            and score_final.get("student_execution") is False, "Scoring finalization differs")
    for pool in POOLS:
        require(score_final["pools"][pool]["manifest_sha256"] == sha(reader.read(prepared / "pools" / pool / "inference/manifest.json"))
                and score_final["pools"][pool]["candidates"] == sum(map(len, caches[pool]["pools"].values())), "Final pool manifest/count differs")
    strong_audits = index(reader.json(root / "strong-prompt-audit.json"), ("sample_id", "method"), "strong prompt audit")
    require(set(strong_audits) == {(sid, prompt) for sid in inputs for prompt in PROMPTS}, "Full strong prompt context audit absent")
    external, strong_selections, strong_manifest = prompt_baselines(reader, root / "prompt-baselines", inputs, references,
        config, eos, protocol, protocol_hash, caches["balanced12"], strong_audits)
    analysis = reader.complete(root / "analysis", "analysis.json")
    require(analysis.get("schema_version") == "student-sim-cd.candidate-support-analysis.v1"
            and analysis.get("samples") == 70 and analysis.get("students") == 17 and analysis.get("samples_dropped") == 0
            and analysis.get("cohort_complete") is True and analysis.get("protocol_sha256") == protocol_hash
            and analysis.get("bootstrap_repetitions") == 2000 and analysis.get("bootstrap_seed") == SEED
            and analysis.get("bootstrap_unit") == "student" and analysis.get("parameters_fitted") is False
            and analysis.get("student_execution") is False and analysis.get("labels_used_for_static_evaluation_only") is True,
            "Analysis protocol/cohort/estimand differs")
    require(analysis.get("model_scoring_performed") == (backend == "HFBackend")
            and analysis.get("generation_stage_performed") == (backend == "HFBackend"), "Synthetic/real analysis status differs")
    equal(analysis.get("methods"), list(METHODS), "analysis fixed methods")
    equal(analysis.get("fixed_method_parameters"), PARAMETERS, "analysis fixed coefficients")
    require(analysis.get("primary_metric") == "edit_location_f1"
            and analysis.get("primary_contrast") == "(B-base)_balanced12 - (B-base)_11_only12"
            and analysis.get("selection") == SELECTION and analysis.get("sequence_protocol") == SEQUENCE,
            "Analysis primary contrast/selection/EOS protocol differs")
    for pool, cache in caches.items():
        directory = prepared / "pools" / pool / "inference"
        expected_provenance = {"backend": backend, "run_sha256": cache["run_hash"],
            "source_sha256": {"inputs": sha(input_raw), **{name: sha(reader.read(directory / (name + (".json" if name == "manifest" else ".jsonl")))) for name in ("manifest", "candidates", "scores")}},
            "candidate_support": cache["manifest"]["candidate_support"]}
        equal(analysis.get("cache_provenance", {}).get(pool), expected_provenance, "analysis sealed cache provenance")
    for method in EXTERNALS:
        provenance = analysis.get("external_provenance", {}).get(method, {})
        require(provenance.get("prediction_sha256") == sha(reader.read(root / "prompt-baselines" / (method + ".jsonl")))
                and provenance.get("generation_manifest_sha256") == sha(reader.read(root / "prompt-baselines/manifest.json"))
                and provenance.get("backend") == backend and provenance.get("protocol_sha256") == protocol_hash,
                "Analysis strong prediction source provenance differs")
    selections = [row for pool in POOLS for row in caches[pool]["choices"]]
    equal(analysis.get("selections"), selections, "main fixed argmax selections")
    require(reader.read(root / "analysis/selections.jsonl") == b"".join((canonical(row) + "\n").encode() for row in selections)
            and analysis.get("selections_sha256") == sha(reader.read(root / "analysis/selections.jsonl")), "Main selections file/SHA differs")
    codes = {pool: cache["codes"] for pool, cache in caches.items()}
    codes["external"] = {**external, "original_greedy": {sid: caches["balanced12"]["pools"][sid][caches["balanced12"]["greedy"][sid]]["code"] for sid in inputs}}
    external_selection = [{"method": method, "sample_id": sid, "predicted_code_sha256": sha(values[sid].encode())}
                          for method, values in codes["external"].items() for sid in sorted(inputs)]
    equal(analysis.get("external_selections"), external_selection, "external selections")
    require(reader.read(root / "analysis/external-selections.jsonl") == b"".join((canonical(row) + "\n").encode() for row in external_selection)
            and analysis.get("external_selections_sha256") == sha(reader.read(root / "analysis/external-selections.jsonl")), "External selections file/SHA differs")
    # Future labels are parsed only after every main/strong selection is fixed.
    label_raw = reader.read(labels_path)
    require(sha(label_raw) == protocol.get("labels_sha256"), "Frozen evaluation labels SHA differs")
    labels = {sid: row for (sid,), row in index(records(label_raw, "labels"), ("sample_id",), "labels").items()}
    require(set(labels) == set(inputs) and all(isinstance(row.get("target_code"), str) for row in labels.values()), "Full evaluation label cohort differs")
    equal(analysis.get("source_sha256"), {"inputs": sha(input_raw), "labels": sha(label_raw)}, "analysis source SHA")
    evaluations, all_summaries, groups = {}, {}, {}
    for version, methods in codes.items():
        evaluations[version], all_summaries[version], groups[version] = {}, {}, {}
        for method, predictions in methods.items():
            rows = [{"sample_id": sid, "student_id": inputs[sid]["student_id"], "has_feedback": bool(inputs[sid]["feedback"]),
                "metrics": metrics(inputs[sid]["current_code"], predictions[sid], labels[sid]["target_code"])} for sid in sorted(inputs)]
            evaluations[version][method] = {row["sample_id"]: row for row in rows}
            all_summaries[version][method] = summary(rows)
            groups[version][method] = {name: summary([row for row in rows if condition(row)]) for name, condition in
                (("true_changed", lambda row: row["metrics"]["true_changed"] == 1),
                 ("true_unchanged", lambda row: row["metrics"]["true_changed"] == 0), ("actual_feedback", lambda row: row["has_feedback"]))}
    equal(analysis.get("static_per_sample"), evaluations, "independent difflib per-sample metrics")
    equal(analysis.get("all"), all_summaries, "independent equal-student means/bootstrap")
    equal(analysis.get("subgroups"), groups, "independent complete subgroup metrics")
    contrasts = {pool: summary(difference(list(evaluations[pool]["b"].values()), list(evaluations[pool]["base"].values())), delta=True) for pool in POOLS}
    interaction = summary(difference(
        difference(list(evaluations["balanced12"]["b"].values()), list(evaluations["balanced12"]["base"].values())),
        difference(list(evaluations["11_only12"]["b"].values()), list(evaluations["11_only12"]["base"].values()))), delta=True)
    changes = {method: summary(difference(list(evaluations["balanced12"][method].values()), list(evaluations["11_only12"][method].values())), delta=True) for method in METHODS}
    equal(analysis.get("pool_b_minus_base"), contrasts, "independent B-base each pool")
    equal(analysis.get("paired_pool_interaction"), interaction, "independent paired pool interaction")
    equal(analysis.get("primary"), interaction["metrics"]["edit_location_f1"], "independent primary interaction")
    equal(analysis.get("balanced_minus_control_by_method"), changes, "independent each-method pool difference")
    equal(analysis.get("source_coverage"), {pool: cache["coverage"] for pool, cache in caches.items()}, "independent condition source coverage")
    equal(analysis.get("pool_overlap"), overlap, "independent pool overlap")
    oracle_report = {}
    for pool, cache in caches.items():
        all_rows, generated_rows, details = [], [], []
        for sid in sorted(inputs):
            per_candidate = {cid: metrics(inputs[sid]["current_code"], candidate["code"], labels[sid]["target_code"]) for cid, candidate in cache["pools"][sid].items()}
            generated = {cid for cid, candidate in cache["pools"][sid].items() if any(source != "copy_current" for source in candidate["sources"])}
            best = min(per_candidate, key=lambda cid: (-per_candidate[cid]["edit_location_f1"], cid))
            generated_best = min(generated, key=lambda cid: (-per_candidate[cid]["edit_location_f1"], cid))
            all_rows.append({"sample_id": sid, "student_id": inputs[sid]["student_id"], "metrics": per_candidate[best]})
            generated_rows.append({"sample_id": sid, "student_id": inputs[sid]["student_id"], "metrics": per_candidate[generated_best]})
            details.append({"sample_id": sid, "student_id": inputs[sid]["student_id"], "candidate_count": len(per_candidate),
                "generated_source_candidate_count": len(generated), "exact_observed_next_code_covered": any(value["exact_next_code"] for value in per_candidate.values()),
                "generated_exact_observed_next_code_covered": any(per_candidate[cid]["exact_next_code"] for cid in generated),
                "label_oracle_edit_location_f1": per_candidate[best]["edit_location_f1"],
                "generated_label_oracle_edit_location_f1": per_candidate[generated_best]["edit_location_f1"],
                "oracle_gaps_by_method": {method: per_candidate[best]["edit_location_f1"] - evaluations[pool][method][sid]["metrics"]["edit_location_f1"] for method in METHODS}})
        oracle_report[pool] = {"analysis_kind": "label_oracle_posthoc_diagnostic_not_method", "labels_used_after_choices_frozen": True,
            "feeds_back_into_selection": False, "metric_optimized": "edit_location_f1", "generated_source_definition": "any greedy or random generation source; may also share copy_current",
            "all_candidates": summary(all_rows), "generated_source_candidates": summary(generated_rows), "per_sample": details,
            "exact_observed_next_code_coverage_count": sum(row["exact_observed_next_code_covered"] for row in details),
            "generated_exact_observed_next_code_coverage_count": sum(row["generated_exact_observed_next_code_covered"] for row in details)}
    equal(analysis.get("label_oracle_diagnostics"), oracle_report, "independent fixed-pool posthoc oracle")
    outer_completion = False
    if (root / "SUCCESS").exists():
        outer = reader.complete(root)
        require(outer.get("status") == "success" and outer.get("exit_code") == 0
                and all(stage.get("status") == "success" and stage.get("exit_code") == 0 for stage in outer.get("stages", [])), "Outer completion differs")
        outer_completion = True
    result = {"schema_version": SCHEMA, "status": "verified", "verified": True,
        "analysis_kind": "independent_standard_library_replay" if backend == "HFBackend" else "synthetic_independent_standard_library_replay",
        "samples": 70, "students": 17, "samples_dropped": 0, "main_pools": list(POOLS), "main_methods": list(METHODS),
        "strong_methods": list(EXTERNALS), "raw_main_new_attempts": 1260, "raw_strong_new_attempts": 1820,
        "generation_attempts": 3080, "main_argmax_verified": True, "strong_own_prompt_argmax_verified": True,
        "per_sample_static_metrics_verified": True, "student_means_bootstrap_subgroups_verified": True,
        "oracle_diagnostic_verified": True, "protocol_sha256": protocol_hash, "baseline_completion_sha256": seals["manifest.json"],
        "analysis_sha256": sha(reader.read(root / "analysis/analysis.json")), "verifier_sha256": sha(Path(__file__).read_bytes()),
        "backend": backend, "model_scoring_performed_by_verifier": False, "generation_performed_by_verifier": False,
        "student_execution": False, "private_text_exported": False, "source_sha256": reader.hashes,
        "bootstrap_unit": "student", "bootstrap_repetitions": 2000, "bootstrap_seed": SEED,
        "absolute_numeric_tolerance": 1e-12, "primary": interaction["metrics"]["edit_location_f1"],
        "pool_b_minus_base": contrasts, "all": all_summaries, "strong_selections": strong_selections,
        "outer_completion_verified": outer_completion,
        "phase": "completed_outer_run" if outer_completion else "analysis_child_before_outer_finalization",
        "test_set_not_used": {"protocol_declares": protocol.get("test_set_used") is False,
            "verified_io_scope": "Only named frozen development inputs/references/labels, original/new cached artifacts and analysis files were opened; no test file is read by this verifier.",
            "whole_pipeline_access_audited": False},
        "limitations": ["No model likelihoods are independently recomputed; stored finite raw scores and their provenance are replayed.",
            "Tokenizer decoding and canonical tokenization require the production tokenizer audit; this standard-library verifier checks SHA, integer tokens and single terminal EOS but cannot load that tokenizer.",
            "Static edit-location F1 is not correctness or a student-learning causal effect.",
            "Candidate-label oracle is posthoc evaluation only; no labels enter selection."]}
    def code_free(value):
        if isinstance(value, dict):
            require(not set(value) & FORBIDDEN, "Private student text/token field in verification report")
            for child in value.values():
                code_free(child)
        elif isinstance(value, list):
            for child in value:
                code_free(child)
    code_free(result)
    canonical(result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        output = args.output.absolute()
        require(not output.exists() and not output.is_symlink() and output.resolve() == output,
                "Verification output must be a new canonical file")
        result = verify(args.run_dir, args.labels)
        with output.open("x", encoding="utf-8") as stream:
            stream.write(canonical(result) + "\n")
        print(canonical({"status": result["status"], "verified": True, "verification_sha256": sha(output.read_bytes())}), flush=True)
        return 0
    except (ValueError, OSError, KeyError, TypeError, IndexError) as error:
        parser.exit(2, "error: " + str(error) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
