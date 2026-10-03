#!/usr/bin/env python3
"""Train-only, CPU-only static A preparation; never execute student programs.

Training outcomes are read only after input-only cohort selection. Development
outcomes are deliberately not accepted by preflight or train-predict. The fixed
feedback_numeric model is sealed before any subsequent development evaluation.
All student/code content stays in memory on the machine that holds the data.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import pwd
import re
import signal
import socket
import stat
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from student_sim_cd import inference
from student_sim_cd import progfeed
from student_sim_cd.select_inputs import eligibility_reasons

SERVER_ROOT = Path("/data/zzm110186486/projects/student-sim-cd")
TRAIN_INPUT_SHA = "58beb677a12c41e4cfd8f10e4471aaf0052067b2a3383a583d25c8a170cd54fc"
PROTOCOL_VERSION = "student-sim-cd.a-static.protocol.v1"
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
LABEL_FIELDS = {"schema_version", "sample_id", "target_timestamp", "target_code", "target_results"}
FORBIDDEN_OUTPUT_KEYS = {"current_code", "target_code", "predicted_code", "code", "raw_text",
                         "generated_token_ids", "completion_token_ids", "history", "feedback",
                         "problem_statement", "target_results", "current_results"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stamp():
    return datetime.now(timezone.utc).isoformat()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "Duplicate JSON field")
        result[key] = value
    return result


def decode(raw):
    value = json.loads(raw, object_pairs_hook=unique)
    json.dumps(value, allow_nan=False)
    return value


def code_free(value):
    if isinstance(value, dict):
        for key, item in value.items():
            require(str(key).lower() not in FORBIDDEN_OUTPUT_KEYS, "Private content field in public output")
            code_free(item)
    elif isinstance(value, list):
        for item in value:
            code_free(item)
    elif isinstance(value, float):
        require(math.isfinite(value), "Nonfinite public output")


def write_json(path, value):
    code_free(value)
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode()
    with Path(path).open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return sha(raw)


def write_jsonl(path, rows):
    for row in rows:
        code_free(row)
    raw = b"".join((inference.canonical_json(row) + "\n").encode() for row in rows)
    with Path(path).open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return sha(raw)


def canonical_existing(value, *, directory=False):
    path = Path(value).absolute()
    require(not path.is_symlink() and path.resolve(strict=True) == path,
            "Inputs must be canonical; symlinks and path fallbacks are forbidden")
    info = path.lstat()
    require(stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode),
            "Expected an existing regular directory or file")
    return path


class ReadAudit:
    """Record actual reads, not a guessed collection of neighboring files."""

    def __init__(self):
        self.files = []

    def read(self, value, role, expected_sha=None):
        path = canonical_existing(value)
        raw = path.read_bytes()
        digest = sha(raw)
        if expected_sha is not None:
            require(isinstance(expected_sha, str) and re.fullmatch(r"[0-9a-f]{64}", expected_sha),
                    "Explicit source SHA256 is invalid")
            require(digest == expected_sha, "Frozen source SHA256 differs: " + role)
        self.files.append({"path": str(path), "role": role, "sha256": digest, "bytes": len(raw)})
        return raw

    def verify_unchanged(self):
        for row in self.files:
            require(sha(Path(row["path"]).read_bytes()) == row["sha256"],
                    "A source changed during the static experiment")


def input_rows(raw):
    rows = jsonl_rows(raw)
    require(bool(rows), "Training or prediction inputs must be nonempty")
    ids = set()
    for row in rows:
        inference.validate_input(row, require_problem=False)
        require(row.get("schema_version") == "student-sim-cd.progfeed.v1", "Input schema differs")
        require(row["sample_id"] not in ids, "Duplicate input sample ID")
        ids.add(row["sample_id"])
    return rows


def label_rows(raw, inputs):
    rows = jsonl_rows(raw)
    indexed = {}
    for row in rows:
        require(isinstance(row, dict) and set(row) == LABEL_FIELDS, "Training label schema differs")
        sid = row.get("sample_id")
        require(isinstance(sid, str) and sid and sid not in indexed, "Missing or duplicate training label ID")
        require(row["schema_version"] == "student-sim-cd.progfeed.v1"
                and isinstance(row["target_code"], str) and isinstance(row["target_timestamp"], str),
                "Training outcome code/timestamp/schema is invalid")
        inference._results(row["target_results"], "training label results")
        indexed[sid] = row
    require(set(indexed) == {row["sample_id"] for row in inputs}, "Full training inputs/label IDs differ")
    for row in inputs:
        before = datetime.strptime(row["current_timestamp"], "%Y-%m-%d-%H-%M-%S")
        after = datetime.strptime(indexed[row["sample_id"]]["target_timestamp"], "%Y-%m-%d-%H-%M-%S")
        require(after > before, "Training outcome timestamp is not after the current input")
    return indexed


def jsonl_rows(raw):
    lines = raw.split(b"\n")
    if lines[-1] == b"":
        lines.pop()
    require(bool(lines) and all(line.strip() for line in lines), "Empty JSONL record")
    rows = [decode(line) for line in lines]
    require(all(isinstance(row, dict) for row in rows), "JSONL records must be objects")
    return rows


def scope_pairs(raw):
    value = decode(raw)
    require(isinstance(value, dict) and isinstance(value.get("pairs"), list), "Scope pairs must be explicit")
    pairs = set()
    for row in value["pairs"]:
        require(isinstance(row, dict) and set(row) == {"lab", "source_file"}, "Malformed scope pair")
        require(all(isinstance(row[k], str) and row[k] and "*" not in row[k] for k in row), "Scope wildcards forbidden")
        pair = (row["lab"], row["source_file"])
        require(pair not in pairs, "Duplicate scope pair")
        pairs.add(pair)
    require(bool(pairs), "Empty scope")
    return pairs


def current_file_listing(row, source_root):
    """Enumerate this current submission only; never inspect the next directory."""
    root = canonical_existing(source_root, directory=True)
    parts = [row[k] for k in ("lab", "student_id", "current_timestamp", "source_file")]
    require(all(part not in {".", ".."} and not any(c in part for c in ("/", "\\", "\x00"))
                for part in parts), "Unsafe current submission identity")
    directory = root / "all_labs" / parts[0] / parts[1] / parts[2]
    if not directory.is_dir():
        return "missing_current_submission_directory", {"sample_id": row["sample_id"], "files": None}
    require(directory.resolve(strict=True) == directory and not directory.is_symlink(), "Current directory symlink forbidden")
    files, pending = [], [directory]
    while pending:
        item = pending.pop()
        for child in sorted(item.iterdir()):
            require(not child.is_symlink(), "Current submission symlink forbidden")
            if child.is_dir():
                pending.append(child)
            elif child.is_file() and child.suffix == ".py":
                files.append(child.relative_to(directory).as_posix())
    files.sort()
    record = {"sample_id": row["sample_id"], "files": files}
    reason = None if files == [row["source_file"]] else "current_submission_not_exactly_one_declared_python_file"
    return reason, record


def source_inventory(raw, expected_fingerprint):
    """Validate the original snapshot inventory; no source bodies are opened."""
    value = decode(raw)
    require(isinstance(value, dict) and value.get("schema_version") == "student-sim-cd.progfeed.v1", "Source audit schema differs")
    inventory = value.get("source_files")
    require(isinstance(inventory, list) and inventory, "Frozen source inventory is missing")
    require(value.get("counts", {}).get("missing_or_unreadable_source_files") == 0,
            "Original inventory contains unreadable or missing source files")
    fingerprint = sha(progfeed._json(inventory).encode("utf-8"))
    require(fingerprint == value.get("source_fingerprint_sha256") == expected_fingerprint,
            "Original source fingerprint differs; use the preparation serializer")
    paths, listings = set(), {}
    for entry in inventory:
        require(isinstance(entry, dict) and set(entry) == {"path", "sha256", "bytes"}, "Source inventory record schema differs")
        relative = entry["path"]
        require(isinstance(relative, str) and relative and not Path(relative).is_absolute()
                and all(part not in {".", ".."} for part in Path(relative).parts)
                and relative not in paths, "Source inventory has unsafe or duplicate paths")
        require(isinstance(entry["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
                and type(entry["bytes"]) is int and entry["bytes"] >= 0, "Source inventory byte/hash record invalid")
        paths.add(relative)
        parts = Path(relative).parts
        if len(parts) >= 5 and parts[0] == "all_labs" and relative.endswith(".py"):
            listings.setdefault("/".join(parts[:4]), []).append("/".join(parts[4:]))
    return {key: sorted(items) for key, items in listings.items()}, {
        "mode": "frozen_source_inventory", "physical_raw_directory_checked": False,
        "source_fingerprint_sha256": fingerprint, "source_inventory_entries": len(inventory),
        "original_missing_or_unreadable_source_files": 0,
        "limitation": "Original reader inventoried root-level Python files and CSV-declared files; an unrecorded nested companion is not certified absent. No physical raw files are rechecked."}


def inventory_current_listing(row, inventory):
    parts = [row[k] for k in ("lab", "student_id", "current_timestamp", "source_file")]
    require(all(part not in {".", ".."} and not any(c in part for c in ("/", "\\", "\x00"))
                for part in parts), "Unsafe current submission identity")
    prefix = "all_labs/" + "/".join(parts[:3])
    files = inventory.get(prefix, [])
    reason = None if files == [row["source_file"]] else "current_inventory_not_exactly_one_declared_python_file"
    return reason, {"sample_id": row["sample_id"], "files": files}


def build_train_cohort(rows, pairs, source_root, expected, *, inventory=None):
    """Selection has no labels parameter; future contents cannot select rows."""
    require(len(rows) == expected["train_samples"] and len({r["student_id"] for r in rows}) == expected["train_students"],
            "Full training sample/student counts differ")
    require((source_root is None) != (inventory is None), "One explicit current-file scope source is required")
    if inventory is None:
        canonical_existing(source_root, directory=True)
        require((Path(source_root) / "all_labs").is_dir(), "The fixed raw source lacks all_labs")
    selected, excluded, listings = [], Counter(), []
    scope_count, feedback_count, problem_count = 0, 0, 0
    for row in sorted(rows, key=lambda r: r["sample_id"]):
        reasons = eligibility_reasons(row, pairs, require_history=False)
        scope_count += (row["lab"], row["source_file"]) in pairs
        feedback_count += bool(row["feedback"] and any(item["text"].strip() for item in row["feedback"]))
        problem_count += isinstance(row.get("problem_statement"), str) and bool(row["problem_statement"].strip())
        if not reasons:
            reason, listing = (current_file_listing(row, source_root) if inventory is None
                               else inventory_current_listing(row, inventory))
            listings.append(listing)
            if reason:
                reasons.append(reason)
        if reasons:
            excluded[reasons[0]] += 1
        else:
            selected.append(row)
    history = [r for r in selected if r["history"]]
    report = {"all_train_samples": len(rows), "all_train_students": len({r["student_id"] for r in rows}),
        "inside_lab_file_scope_samples": scope_count, "actual_feedback_samples": feedback_count,
        "nonempty_problem_samples": problem_count, "eligible_samples": len(selected),
        "eligible_students": len({r["student_id"] for r in selected}), "nonempty_history_samples": len(history),
        "nonempty_history_students": len({r["student_id"] for r in history}),
        "empty_history_samples": len(selected) - len(history), "exclusions_by_primary_reason": dict(excluded),
        "eligible_by_lab_file": dict(Counter(r["lab"] + "/" + r["source_file"] for r in selected)),
        "eligible_sample_ids_sha256": sha(inference.canonical_json(sorted(r["sample_id"] for r in selected)).encode()),
        "current_directory_listing_sha256": sha(inference.canonical_json(listings).encode()),
        "current_directories_inspected": len(listings), "labels_used_for_selection": False,
        "future_directories_inspected": False, "count_cap": None,
        "single_file_check": ("current_only_exact_declared_python_file" if inventory is None
                              else "original_inventory_current_only_exact_declared_python_file"),
        "physical_raw_directory_checked": inventory is None}
    required_counts = {"eligible_samples": expected["eligible_samples"], "eligible_students": expected["eligible_students"],
                       "nonempty_history_samples": expected["history_samples"]}
    require(all(report[k] == value for k, value in required_counts.items()), "Complete eligible/history cohort counts differ")
    return selected, history, report


def read_protocol(path, audit):
    raw = audit.read(path, "protocol")
    value = decode(raw)
    require(isinstance(value, dict) and value.get("schema_version") == PROTOCOL_VERSION, "Unsupported static A protocol")
    expected = {"expected_train_samples": 4299, "expected_train_students": 172,
                "expected_eligible_samples": 737, "expected_eligible_students": 140,
                "expected_history_samples": 519, "expected_dev_samples": 70, "expected_dev_students": 17}
    require(all(type(value.get(k)) is int and value[k] == v for k, v in expected.items()), "Fixed complete A cohort differs")
    require(value.get("train_inputs_sha256") == TRAIN_INPUT_SHA, "Fixed training input SHA differs")
    require(all(value.get(key) is False for key in ("student_execution", "grader_execution", "new_generation",
                "new_model_scoring", "test_read", "new_downloads", "dev_selection")),
            "Static A prohibits execution, generation, scoring, downloads and test reads")
    q = value.get("q_model", {})
    require(q.get("primary") == "feedback_numeric" and q.get("folds") == 5
            and q.get("seed") == 20261003 and q.get("l2") == 1.0
            and q.get("selection") == "fixed_not_selected_by_dev_or_oof",
            "Fixed feedback model, five student folds, seed and L2 are required")
    for key in ("train_labels_sha256", "dev_inputs_sha256", "dev_labels_sha256", "scope_sha256",
                "splits_sha256", "source_audit_sha256", "source_inventory_fingerprint_sha256"):
        require(isinstance(value.get(key), str) and re.fullmatch(r"[0-9a-f]{64}", value[key]), "Missing source seal: " + key)
    require(value.get("eligibility_mode") == "frozen_source_inventory_current_submission_only",
            "Original inventory current-only eligibility must be explicitly frozen")
    require(value.get("fixed_method_parameters") == {"base": {"alpha": 0, "beta": 0},
        "cd": {"alpha": 1, "beta": 1}, "b": {"alpha": 0, "beta": 1},
        "history_d0": {"alpha": 1, "beta": 0}}, "Fixed ranker weights differ")
    require(value.get("score_temperature") == 1 and value.get("q_shared_across_rankers") is True
            and value.get("primary_pool") == "balanced12", "Fixed shared-q primary pool differs")
    require(value.get("bootstrap") == {"unit": "student", "paired": True, "repetitions": 2000,
        "seed": 20260929, "interpretation": "seen-dev descriptive"}, "Fixed descriptive bootstrap differs")
    require(value.get("max_cpu_threads") == 1 and value.get("max_wall_seconds") == 7200,
            "Fixed CPU resource cap differs")
    pools = value.get("pool_sources")
    require(isinstance(pools, dict) and set(pools) == {"balanced12", "11_only12", "legacy318"}, "All frozen candidate pools required")
    for name, count in (("balanced12", 861), ("11_only12", 743), ("legacy318", 318)):
        spec = pools[name]
        require(isinstance(spec, dict) and set(spec) == {"directory", "manifest_sha256", "candidates_sha256", "scores_sha256", "expected_candidates"}
                and spec["expected_candidates"] == count, "Frozen pool source/count differs")
        require(isinstance(spec["directory"], str) and not Path(spec["directory"]).is_absolute()
                and all(part not in {".", ".."} for part in Path(spec["directory"]).parts)
                and Path(spec["directory"]).parts[0] == "outputs", "Unsafe pool directory")
        for field in ("manifest_sha256", "candidates_sha256", "scores_sha256"):
            require(isinstance(spec[field], str) and re.fullmatch(r"[0-9a-f]{64}", spec[field]), "Missing frozen pool byte SHA")
    return value, sha(raw)


def expected_counts(protocol):
    return {"train_samples": protocol["expected_train_samples"], "train_students": protocol["expected_train_students"],
            "eligible_samples": protocol["expected_eligible_samples"], "eligible_students": protocol["expected_eligible_students"],
            "history_samples": protocol["expected_history_samples"]}


def prepare_training(args, protocol, audit):
    rows = input_rows(audit.read(args.train_inputs, "train_inputs", protocol["train_inputs_sha256"]))
    pairs = scope_pairs(audit.read(args.scope, "scope", protocol["scope_sha256"]))
    registry = decode(audit.read(args.splits, "student_split_registry", protocol["splits_sha256"]))
    require(registry.get("schema_version") == "student-sim-cd.progfeed.v1" and registry.get("unit") == "student_id"
            and registry.get("seed") == 20260929 and isinstance(registry.get("students"), dict), "Frozen student split registry differs")
    student_splits = registry["students"]
    require(all(student_splits.get(r["student_id"]) == "train" for r in rows), "Training input contains a non-train student")
    mode = protocol.get("eligibility_mode")
    if mode == "frozen_source_inventory_current_submission_only":
        require(args.source_audit is not None and args.source_root is None, "Protocol requires explicit original audit mode; no fallback")
        inventory, source_report = source_inventory(audit.read(args.source_audit, "source_audit", protocol["source_audit_sha256"]),
                                                    protocol["source_inventory_fingerprint_sha256"])
        selected, history, funnel = build_train_cohort(rows, pairs, None, expected_counts(protocol), inventory=inventory)
        funnel["source_scope_audit"] = source_report
    else:
        raise ValueError("A current scope source mode must be explicitly frozen")
    # This is the first training-outcome read. Eligibility is already frozen.
    labels = label_rows(audit.read(args.train_labels, "train_labels", protocol["train_labels_sha256"]), rows)
    selected_labels = [labels[row["sample_id"]] for row in selected]
    history_labels = [labels[row["sample_id"]] for row in history]
    funnel["train_changed_samples"] = sum(row["current_code"] != labels[row["sample_id"]]["target_code"] for row in selected)
    funnel["train_unchanged_samples"] = len(selected) - funnel["train_changed_samples"]
    return selected, selected_labels, history, history_labels, funnel, student_splits


def checked_deadline(value, max_seconds):
    require(type(max_seconds) is int and 1 <= max_seconds <= 7200, "CPU wall cap must be 1..7200 seconds")
    stop = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(stop.utcoffset() is not None and stop.timestamp() > time.time(), "An unexpired timezone-aware deadline is required")
    return min(stop.timestamp(), time.time() + max_seconds)


def validated_pools(protocol, audit, dev_bytes):
    """Read only frozen named cached artifacts; keep their private bodies local."""
    from student_sim_cd import predict
    result, provenance = {}, {}
    for name in ("balanced12", "11_only12", "legacy318"):
        spec = protocol["pool_sources"][name]
        directory = SERVER_ROOT / spec["directory"]
        sources = {"inputs": dev_bytes}
        for key, filename in (("manifest", "manifest.json"), ("candidates", "candidates.jsonl"), ("scores", "scores.jsonl")):
            sources[key] = audit.read(directory / filename, name + "_" + key, spec[key + "_sha256"])
        manifest, run_hash, pools, scores, copies, kinds = predict._validated_run(sources)
        require(len(pools) == protocol["expected_dev_samples"] and len(scores) == spec["expected_candidates"], "Frozen pool coverage differs")
        require(set(kinds.values()) == {"matched"}, "Candidate pool must use frozen matched references")
        result[name] = (pools, scores)
        provenance[name] = {"inference_run_sha256": run_hash, "candidates": len(scores), "samples": len(pools),
                            "reference_kind": "matched", "copy_baseline_samples": len(copies)}
    return result, provenance


def seal_distributions(inputs, pools, scores, q_rows):
    """No labels argument: persist every probability and choice before evaluation."""
    from student_sim_cd.a_static_analysis import GROUPS, METHODS
    from student_sim_cd.scoring import calibrate_groups, softmax
    indexed = {row["sample_id"]: row for row in inputs}
    q = {row["sample_id"]: row for row in q_rows}
    require(len(indexed) == len(inputs) and len(q) == len(q_rows) and set(indexed) == set(pools) == set(q), "Distribution cohort differs")
    require(set(scores) == {(sid, cid) for sid, pool in pools.items() for cid in pool}, "Distribution scores incomplete")
    rows = []
    for sid in sorted(indexed):
        probability = q[sid]["p_changed"]
        require(q[sid]["student_id"] == indexed[sid]["student_id"]
                and type(probability) in (int, float) and math.isfinite(probability) and 0 < probability < 1,
                "Frozen q student/probability differs")
        ids = sorted(pools[sid])
        codes = [pools[sid][cid]["code"] for cid in ids]
        require(ids and len(codes) == len(set(codes)) and all(cid == sha(code.encode()) for cid, code in zip(ids, codes)), "Distribution candidate hashes differ")
        groups = ["unchanged" if code == indexed[sid]["current_code"] else "changed" for code in codes]
        require(set(groups) == set(GROUPS), "A positive-q group has no candidate; no renormalization allowed")
        masses = {"unchanged": 1 - probability, "changed": probability}
        for method in METHODS:
            values = [scores[sid, cid][method] for cid in ids]
            for name, ps in ((method, softmax(values)), ("a_" + method, calibrate_groups(values, groups, masses))):
                require(all(math.isfinite(p) and p >= 0 for p in ps)
                        and math.isclose(math.fsum(ps), 1, abs_tol=1e-10), "Distribution mass is invalid")
                chosen = min(range(len(ids)), key=lambda i: (-ps[i], ids[i]))
                rows.append({"sample_id": sid, "student_id": indexed[sid]["student_id"], "method": name,
                    "q_changed": probability, "candidate_ids": ids, "groups": groups, "probabilities": ps,
                    "argmax_candidate_id": ids[chosen]})
    return rows


def verify_distribution_seal(report, sealed):
    indexed = {(row["sample_id"], row["method"]): row for row in sealed}
    require(len(indexed) == len(sealed) and len(report["distributions"]) == len(sealed), "Sealed distribution coverage differs")
    for row in report["distributions"]:
        expected = indexed[row["sample_id"], row["method"]]
        require(all(row[key] == expected[key] for key in ("student_id", "q_changed", "candidate_ids", "groups", "probabilities")),
                "Evaluation changed a pre-label distribution")
        measured = next(item for item in report["per_sample"][row["method"]] if item["sample_id"] == row["sample_id"])
        require(measured["argmax_candidate_id"] == expected["argmax_candidate_id"], "Evaluation changed a pre-label argmax")


def independent_verify(collection, protocol_path, body_sources=None):
    """The verifier has separate standard-library metric implementations."""
    scripts = str(ROOT / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    from verify_a_static_results import verify
    return verify(collection, protocol_path, body_sources=body_sources)


def controlled_server(args):
    require(sys.platform == "linux" and socket.gethostname().split(".")[0] == "4090-02"
            and pwd.getpwuid(os.getuid()).pw_name == "zzm110186486", "Fixed server host/account required")
    require(Path.cwd() == SERVER_ROOT and Path.cwd().resolve() == SERVER_ROOT, "Use the fixed project working directory")
    require(Path(sys.executable) == SERVER_ROOT / ".venv/bin/python" and Path(sys.prefix) == SERVER_ROOT / ".venv", "Use verified project Python")
    require(ROOT.parent == SERVER_ROOT / "releases" and re.fullmatch(r"[0-9a-f]{64}", ROOT.name), "Use a content-addressed deployed release")
    require(not (ROOT.lstat().st_mode & 0o222) and not (ROOT / ".incomplete").exists(), "Release must be sealed")
    require(sha((ROOT / ".deploy-manifest.json").read_bytes()) == ROOT.name, "Release seal differs")
    require(os.environ.get("STUDENT_SIM_A_STATIC_BOUNDED") == "1", "Use the reviewed bounded CPU launcher")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CPU run must mask every GPU")
    affinity = sorted(os.sched_getaffinity(0))
    require(len(affinity) == 1 and all(os.environ.get(k) == "1" for k in THREAD_VARIABLES), "Exactly one CPU core/thread required")
    paths = {"train_inputs": SERVER_ROOT / "data/prepared/progfeed-v2/inputs.train.jsonl",
             "train_labels": SERVER_ROOT / "data/prepared/progfeed-v2/labels.train.jsonl",
             "dev_inputs": SERVER_ROOT / "data/prepared/progfeed-selected-v1/matched/inputs.dev.jsonl",
             "splits": SERVER_ROOT / "data/prepared/progfeed-v2/splits.json"}
    if args.command == "run":
        paths["dev_labels"] = SERVER_ROOT / "data/prepared/progfeed-selected-v1/matched/labels.dev.jsonl"
    if args.source_root is not None:
        paths["source_root"] = SERVER_ROOT / "data/raw/progfeed"
    if args.source_audit is not None:
        paths["source_audit"] = SERVER_ROOT / "data/prepared/progfeed-v2/audit.json"
    for name, expected in paths.items():
        require(Path(getattr(args, name)).absolute() == expected, "Fixed data path differs: " + name)
    require(Path(args.output).absolute().parent == SERVER_ROOT / "outputs", "Output must be a new server outputs sibling")
    return {"host": "4090-02", "user": "zzm110186486", "cpu_affinity": affinity, "cpu_threads": 1,
            "cuda_visible_devices": "", "process_group": os.getpgrp(), "pid": os.getpid(),
            "release_sha256": ROOT.name, "gpu_used": False, "student_execution": False}


def new_output(value):
    path = Path(value).absolute()
    require(path.resolve() == path and not path.exists() and not path.is_symlink(), "Output must be new and canonical; no overwrite/resume")
    require(path.parent.is_dir(), "Output parent must already exist")
    path.mkdir(mode=0o700)
    return path


def execute(args, *, enforce_server=True):
    end = checked_deadline(args.deadline, args.max_seconds)
    resources = controlled_server(args) if enforce_server else {"synthetic_fixture": True, "gpu_used": False, "cpu_threads": 1}
    audit = ReadAudit()
    protocol, protocol_sha = read_protocol(args.protocol, audit)
    require(args.max_seconds <= protocol["max_wall_seconds"], "Requested CPU cap exceeds frozen protocol")
    require(datetime.fromisoformat(args.deadline.replace("Z", "+00:00")).timestamp()
            <= datetime.fromisoformat(protocol["stop_by"].replace("Z", "+00:00")).timestamp(),
            "Requested deadline exceeds frozen stop-by")
    output = new_output(args.output)
    started = time.monotonic()
    artifacts = {}
    dev_labels_read = False

    def check_time():
        if time.time() >= end:
            raise TimeoutError("Static A aggregate CPU deadline reached")

    def alarm_handler(signum, frame):
        raise TimeoutError("Static A aggregate CPU deadline reached")

    prior_handler = signal.getsignal(signal.SIGALRM) if hasattr(signal, "SIGALRM") else None
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, alarm_handler)
        signal.setitimer(signal.ITIMER_REAL, max(0.001, end - time.time()))
    try:
        check_time()
        selected, labels, history, history_labels, funnel, student_splits = prepare_training(args, protocol, audit)
        artifacts["train-audit.json"] = write_json(output / "train-audit.json", funnel)
        check_time()
        if args.command == "preflight":
            scope = "train-source-and-current-only-cohort-preflight; no model fitted or dev input read"
        else:
            from student_sim_cd import change_q
            # Source code hashes are part of the sealed model/protocol provenance.
            implementation = {"scripts/run_a_static.py": sha(Path(__file__).read_bytes())}
            for module in ("change_q", "a_static_analysis", "predict", "scoring", "evaluate", "inference", "progfeed", "select_inputs"):
                path = ROOT / "src/student_sim_cd" / (module + ".py")
                implementation["src/student_sim_cd/" + module + ".py"] = sha(audit.read(path, "implementation_" + module))
            source_hashes = {row["role"]: row["sha256"] for row in audit.files}
            report, models, oof = change_q.run_train_experiment(selected, labels, folds=5, seed=20261003,
                split="train", source_sha256=source_hashes, student_splits=student_splits)
            check_time()
            history_report, history_models, history_oof = change_q.run_train_experiment(history, history_labels,
                folds=5, seed=20261003, split="train", source_sha256={**source_hashes,
                    "cohort": sha(inference.canonical_json(sorted(r["sample_id"] for r in history)).encode())}, student_splits=student_splits)
            check_time()
            artifacts["train-cv.json"] = write_json(output / "train-cv.json", report)
            artifacts["q-models.json"] = write_json(output / "q-models.json", models)
            artifacts["train-oof.jsonl"] = write_jsonl(output / "train-oof.jsonl", oof)
            artifacts["history-train-cv.json"] = write_json(output / "history-train-cv.json", history_report)
            artifacts["history-q-models.json"] = write_json(output / "history-q-models.json", history_models)
            artifacts["history-train-oof.jsonl"] = write_jsonl(output / "history-train-oof.jsonl", history_oof)
            model_seal = {"q_method": "feedback_numeric", "selection_rule": "fixed before fitting; no OOF/dev model selection",
                "protocol_sha256": protocol_sha, "implementation_sha256": implementation,
                "model_artifact_sha256": {name: digest for name, digest in artifacts.items() if "model" in name},
                "dev_labels_read": False, "test_files_read": False, "training_text_exported": False}
            artifacts["q-fit-seal.json"] = write_json(output / "q-fit-seal.json", model_seal)
            check_time()
            # The fixed fitted model has been written before this first dev read.
            dev_bytes = audit.read(args.dev_inputs, "dev_inputs", protocol["dev_inputs_sha256"])
            dev = input_rows(dev_bytes)
            require(len(dev) == protocol["expected_dev_samples"] and len({r["student_id"] for r in dev}) == protocol["expected_dev_students"], "Frozen dev cohort differs")
            require(not ({r["student_id"] for r in selected} & {r["student_id"] for r in dev}), "Train/dev student leakage")
            require(all(student_splits.get(r["student_id"]) == "dev" for r in dev), "Prediction input contains a non-dev student")
            predictions = change_q.predict(models, dev, method="feedback_numeric", split="holdout")
            sensitivity = change_q.predict(history_models, dev, method="feedback_numeric", split="holdout")
            artifacts["q.dev.jsonl"] = write_jsonl(output / "q.dev.jsonl", predictions)
            artifacts["q-history-sensitivity.dev.jsonl"] = write_jsonl(output / "q-history-sensitivity.dev.jsonl", sensitivity)
            artifacts["q-prediction-seal.json"] = write_json(output / "q-prediction-seal.json", {
                "protocol_sha256": protocol_sha, "q_fit_seal_sha256": artifacts["q-fit-seal.json"],
                "predictions_sha256": {name: artifacts[name] for name in ("q.dev.jsonl", "q-history-sensitivity.dev.jsonl")},
                "dev_inputs_sha256": sha(dev_bytes), "dev_labels_read": False, "test_files_read": False})
            scope = "train-only fitted q and development-input prediction; no dev outcome evaluation"
            if args.command == "run":
                from student_sim_cd import a_static_analysis
                check_time()
                pool_data, pool_provenance = validated_pools(protocol, audit, dev_bytes)
                distributions = {}
                for name, (pools, scores) in pool_data.items():
                    distributions[name] = seal_distributions(dev, pools, scores, predictions)
                    artifacts["distributions." + name + ".pre-label.jsonl"] = write_jsonl(
                        output / ("distributions." + name + ".pre-label.jsonl"), distributions[name])
                pools, scores = pool_data[protocol["primary_pool"]]
                distributions["balanced12-history-sensitivity"] = seal_distributions(dev, pools, scores, sensitivity)
                artifacts["distributions.balanced12-history-sensitivity.pre-label.jsonl"] = write_jsonl(
                    output / "distributions.balanced12-history-sensitivity.pre-label.jsonl",
                    distributions["balanced12-history-sensitivity"])
                artifacts["candidate-distribution-seal.json"] = write_json(output / "candidate-distribution-seal.json", {
                    "protocol_sha256": protocol_sha, "q_prediction_seal_sha256": artifacts["q-prediction-seal.json"],
                    "source_pool_provenance": pool_provenance,
                    "pre_label_distribution_sha256": {name: digest for name, digest in artifacts.items() if name.endswith(".pre-label.jsonl")},
                    "dev_labels_read": False, "test_files_read": False, "all_rankers_share_fixed_q": True,
                    "positive_q_missing_group_policy": "fail_closed_without_renormalization"})
                check_time()
                # Only this statement opens development outcomes. All choices and q are already sealed.
                dev_labels_read = True
                dev_labels = list(label_rows(audit.read(args.dev_labels, "dev_labels", protocol["dev_labels_sha256"]), dev).values())
                for name, (pools, scores) in pool_data.items():
                    report = a_static_analysis.analyze_pool(dev, dev_labels, pools, scores, predictions, name,
                        bootstrap=protocol["bootstrap"]["repetitions"], seed=protocol["bootstrap"]["seed"])
                    verify_distribution_seal(report, distributions[name])
                    report["candidate_distribution_seal_sha256"] = artifacts["candidate-distribution-seal.json"]
                    artifacts["analysis." + name + ".json"] = write_json(output / ("analysis." + name + ".json"), report)
                    check_time()
                pools, scores = pool_data[protocol["primary_pool"]]
                report = a_static_analysis.analyze_pool(dev, dev_labels, pools, scores, sensitivity,
                    "balanced12-history-sensitivity", bootstrap=protocol["bootstrap"]["repetitions"], seed=protocol["bootstrap"]["seed"])
                verify_distribution_seal(report, distributions["balanced12-history-sensitivity"])
                report["candidate_distribution_seal_sha256"] = artifacts["candidate-distribution-seal.json"]
                report["training_cohort"] = "fixed nonempty-history 519 sensitivity; full eligible 737 is primary"
                artifacts["analysis.balanced12-history-sensitivity.json"] = write_json(
                    output / "analysis.balanced12-history-sensitivity.json", report)
                scope = "train-only fitted change/no-change q plus frozen shared-candidate distribution evaluation on seen dev; not executable progress A"
        check_time()
        audit.verify_unchanged()
        manifest = {"schema_version": "student-sim-cd.a-static.run.v1", "status": "success", "scope": scope,
            "started_at": datetime.fromtimestamp(time.time() - (time.monotonic() - started), timezone.utc).isoformat(),
            "finished_at": stamp(), "elapsed_seconds": time.monotonic() - started,
            "protocol_sha256": protocol_sha, "actual_read_files": audit.files, "artifact_sha256": artifacts,
            "resources": resources, "dev_labels_read": dev_labels_read, "test_files_read": False,
            "student_execution": False, "generation_performed": False, "training_text_exported": False,
            "full_progress_A_completed": False, "deadline": args.deadline, "max_seconds": args.max_seconds}
        if args.command == "run":
            calculation = {**manifest, "status": "calculated", "scope": scope,
                           "independent_verification_completed": False}
            calculation_sha = write_json(output / "calculation-manifest.json", calculation)
            check_time()
            verification = independent_verify(output, args.protocol,
                body_sources={"project_root": str(SERVER_ROOT)} if enforce_server else None)
            require(isinstance(verification, dict) and verification.get("status") == "verified",
                    "Independent static A verification did not pass")
            artifacts["calculation-manifest.json"] = calculation_sha
            artifacts["independent-verification.json"] = write_json(output / "independent-verification.json", verification)
            audit.verify_unchanged()
            manifest.update(artifact_sha256=artifacts, independent_verification_completed=True,
                independent_verification_sha256=artifacts["independent-verification.json"],
                independent_body_verification_requested=enforce_server,
                finished_at=stamp(), elapsed_seconds=time.monotonic() - started)
        digest = write_json(output / "manifest.json", manifest)
        check_time()
        with (output / "SUCCESS").open("x") as handle:
            handle.write(digest + "\n")
        return manifest
    except BaseException as error:
        if output.exists() and not (output / "FAILURE.json").exists():
            write_json(output / "FAILURE.json", {"status": "failed", "error_type": type(error).__name__,
                "finished_at": stamp(), "elapsed_seconds": time.monotonic() - started,
                "actual_read_files": audit.files, "dev_labels_read": dev_labels_read, "test_files_read": False,
                "student_execution": False, "generation_performed": False})
        raise
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, prior_handler)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    for command in ("preflight", "train-predict", "run"):
        part = sub.add_parser(command)
        for name in ("train-inputs", "train-labels", "dev-inputs", "scope", "splits", "output"):
            part.add_argument("--" + name, type=Path, required=True)
        source = part.add_mutually_exclusive_group(required=True)
        source.add_argument("--source-root", type=Path)
        source.add_argument("--source-audit", type=Path)
        if command == "run":
            part.add_argument("--dev-labels", type=Path, required=True)
        part.add_argument("--protocol", type=Path, default=ROOT / "configs/a_static_v1.json")
        part.add_argument("--deadline", required=True)
        part.add_argument("--max-seconds", type=int, required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        manifest = execute(args)
    except Exception as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__, "message": str(error)}), file=sys.stderr)
        return 1
    print(json.dumps({"status": manifest["status"], "scope": manifest["scope"], "elapsed_seconds": manifest["elapsed_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
