"""Freeze all input-eligible dev/test samples and prepare past-only references.

Scope JSON contains exact ``pairs: [{lab, source_file}, ...]``. The root output
retains every scoped sample with a task description and delivered feedback;
``--require-history`` optionally restricts this cohort. No count cap or random
subsample is used. Labels are copied by sample_id only, never used for filtering.

The separate matched/ cohort additionally requires a nonempty real history and
a training-input donor: different student, same lab/file, nonempty history, known
current logged score fraction, and donor current time strictly before query time.
Nearest means lexicographic (absolute history-count difference, absolute logged
score-fraction difference, absolute known-result-count difference, donor ID).
The donor's existing history is H*; its current code/feedback is not appended.
Absent matches are reported and excluded only from matched/, never filled with
empty history. Matching is a declared diagnostic heuristic, not a validated
student ability model. Student code and test resources are never executed.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any

from .inference import canonical_json, file_hash, load_inputs, object_hash, read_jsonl


SELECTION_VERSION = "student-sim-cd.selection.v1"
MATCHING_RULE = "lexicographic_history_count_then_logged_score_then_result_count_then_sample_id"


def load_scope(path: Path) -> tuple[dict, set[tuple[str, str]]]:
    specification = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(specification, dict) or not isinstance(specification.get("pairs"), list):
        raise ValueError("scope must contain an explicit pairs list")
    pairs = set()
    for item in specification["pairs"]:
        if not isinstance(item, dict) or set(item) != {"lab", "source_file"}:
            raise ValueError("each scope pair must contain exactly lab and source_file")
        if any(not isinstance(item[key], str) or not item[key] or "*" in item[key]
               for key in ("lab", "source_file")):
            raise ValueError("scope uses nonempty exact lab/file names, not wildcards")
        pair = (item["lab"], item["source_file"])
        if pair in pairs:
            raise ValueError(f"duplicate scope pair {pair}")
        pairs.add(pair)
    if not pairs:
        raise ValueError("scope cannot be empty")
    return specification, pairs


def _time(row: dict) -> datetime:
    value = row.get("current_timestamp")
    if not isinstance(value, str):
        raise ValueError(f"{row['sample_id']}: current_timestamp is required for selection")
    try:
        return datetime.strptime(value, "%Y-%m-%d-%H-%M-%S")
    except ValueError as exc:
        raise ValueError(f"{row['sample_id']}: expected ProgFeed YYYY-MM-DD-HH-MM-SS timestamp") from exc


def eligibility_reasons(row: dict, scope: set[tuple[str, str]], require_history: bool) -> list[str]:
    """All decisions use current inputs only, never label content or model output."""
    reasons = []
    if (row["lab"], row["source_file"]) not in scope:
        reasons.append("outside_declared_lab_file_scope")
    if not any(item["text"].strip() for item in row["feedback"]):
        reasons.append("no_actual_nonempty_feedback")
    if not isinstance(row.get("problem_statement"), str) or not row["problem_statement"].strip():
        reasons.append("missing_problem_statement")
    if require_history and not row["history"]:
        reasons.append("empty_history")
    return reasons


def logged_performance(row: dict) -> dict | None:
    """Current normalized logged score, explicitly not a test pass rate.

    Missing scores are not zero. Conflicting duplicate test keys make the profile
    unavailable; exact duplicates count once. Scores outside [0, max] or zero/absent
    maxima do not define a comparable fraction and are omitted with coverage kept.
    """
    unique = {}
    for result in row["current_results"]:
        key = (result.get("test_name"), result.get("function_name"))
        numeric = (result.get("score"), result.get("max_score"))
        if key in unique and unique[key] != numeric:
            return None
        unique[key] = numeric
    known = []
    for score, maximum in unique.values():
        if (isinstance(score, (int, float)) and not isinstance(score, bool)
                and isinstance(maximum, (int, float)) and not isinstance(maximum, bool)
                and math.isfinite(score) and math.isfinite(maximum) and maximum > 0
                and 0 <= score <= maximum):
            known.append((score, maximum))
    if not known:
        return None
    total = math.fsum(maximum for _, maximum in known)
    return {"logged_score_fraction": math.fsum(score for score, _ in known) / total,
            "known_result_count": len(known), "observed_result_count": len(unique)}


def current_file_reason(row: dict, source_root: Path | None) -> str | None:
    """Inspect only the current submission directory, never future labels/files."""
    if source_root is None:
        return None
    parts = [row["lab"], row["student_id"], row["current_timestamp"], row["source_file"]]
    if any(part in (".", "..") or any(char in part for char in ("/", "\\", "\x00")) for part in parts):
        raise ValueError("unsafe path component in current submission identity")
    directory = source_root / "all_labs" / row["lab"] / row["student_id"] / row["current_timestamp"]
    if source_root.resolve() not in directory.resolve().parents:
        raise ValueError("current submission directory escapes source root")
    if not directory.is_dir():
        return "missing_current_submission_directory"
    files = sorted(path.relative_to(directory).as_posix() for path in directory.rglob("*.py") if path.is_file())
    if files != [row["source_file"]]:
        return "current_submission_not_exactly_one_declared_python_file"
    return None


def build_donor_index(train_rows: list[dict], scope: set[tuple[str, str]], source_root: Path | None = None) -> dict:
    donors = defaultdict(list)
    for row in train_rows:
        pair = (row["lab"], row["source_file"])
        if pair not in scope or not row["history"]:
            continue
        if current_file_reason(row, source_root):
            continue
        performance = logged_performance(row)
        if performance is None:
            continue
        donors[pair].append({"row": row, "time": _time(row), "performance": performance})
    return donors


def match_reference(query: dict, donors: dict, train_sha256: str) -> tuple[dict | None, str | None]:
    if not query["history"]:
        return None, "empty_query_history"
    query_performance = logged_performance(query)
    if query_performance is None:
        return None, "missing_or_conflicting_query_logged_performance"
    query_time = _time(query)
    eligible = [donor for donor in donors.get((query["lab"], query["source_file"]), [])
                if donor["row"]["student_id"] != query["student_id"] and donor["time"] < query_time]
    if not eligible:
        return None, "no_earlier_train_donor_with_history_and_logged_performance"

    def distance(donor: dict) -> tuple:
        return (abs(len(query["history"]) - len(donor["row"]["history"])),
                abs(query_performance["logged_score_fraction"] - donor["performance"]["logged_score_fraction"]),
                abs(query_performance["known_result_count"] - donor["performance"]["known_result_count"]),
                donor["row"]["sample_id"])

    best = min(eligible, key=distance)
    donor_row = best["row"]
    # load_inputs already checks H < donor current; this explicit second check
    # protects the global prediction-time boundary at reference serialization.
    for item in donor_row["history"]:
        if datetime.strptime(item["timestamp"], "%Y-%m-%d-%H-%M-%S") >= query_time:
            raise ValueError("reference history is not strictly earlier than query time")
    return {
        "sample_id": query["sample_id"], "reference_kind": "matched",
        "reference_history": donor_row["history"],
        "provenance": {
            "split": "train", "training_inputs_sha256": train_sha256,
            "donor_sample_id": donor_row["sample_id"], "donor_student_id": donor_row["student_id"],
            "donor_current_timestamp": donor_row["current_timestamp"],
            "query_current_timestamp": query["current_timestamp"],
            "lab": query["lab"], "source_file": query["source_file"],
            "matching_rule": MATCHING_RULE,
            "history_length_query": len(query["history"]), "history_length_donor": len(donor_row["history"]),
            "query_logged_performance": query_performance, "donor_logged_performance": best["performance"],
            "distance": list(distance(best)[:3]), "eligible_donor_count": len(eligible),
            "history_definition": "donor input history only; donor current state not appended",
            "labels_used": False,
        },
    }, None


def _labels_by_id(path: Path, inputs: list[dict]) -> dict:
    """Read IDs for alignment and carry opaque original label records through."""
    labels = {}
    for row in read_jsonl(path):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or sample_id in labels:
            raise ValueError(f"{path}: missing or duplicate label sample_id")
        labels[sample_id] = row
    input_ids = {row["sample_id"] for row in inputs}
    if set(labels) != input_ids:
        raise ValueError(f"{path}: label IDs do not equal input IDs (missing={len(input_ids - labels.keys())}, "
                         f"extra={len(labels.keys() - input_ids)})")
    return labels


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def select_inputs(prepared: Path, output: Path, scope_path: Path,
                  splits: tuple[str, ...] = ("dev", "test"), require_history: bool = False,
                  source_root: Path | None = None) -> dict:
    prepared, output = prepared.resolve(), output.resolve()
    if output == prepared or prepared in output.parents:
        raise ValueError("output must be outside the read-only prepared source directory")
    if output.exists() and any(output.iterdir()):
        raise ValueError("output must be absent or empty; existing results are not overwritten")
    if not splits or len(set(splits)) != len(splits) or set(splits) - {"dev", "test"}:
        raise ValueError("splits must be a nonempty, nonrepeating selection of dev and test")
    specification, scope = load_scope(scope_path)
    train_path = prepared / "inputs.train.jsonl"
    train_rows = load_inputs(train_path, require_problem=False)
    train_sha256 = file_hash(train_path)
    if source_root is not None:
        source_root = source_root.resolve()
        if output == source_root or source_root in output.parents:
            raise ValueError("output must be outside the read-only raw source directory")
        if not (source_root / "all_labs").is_dir():
            raise ValueError("--source-root must contain the official all_labs directory")
    donors = build_donor_index(train_rows, scope, source_root)
    sources = {"inputs.train.jsonl": train_sha256}
    loaded, student_sets = {}, {"train": {row["student_id"] for row in train_rows}}
    for split in splits:
        input_path, label_path = prepared / f"inputs.{split}.jsonl", prepared / f"labels.{split}.jsonl"
        rows = load_inputs(input_path, require_problem=False)
        loaded[split] = (rows, _labels_by_id(label_path, rows))
        student_sets[split] = {row["student_id"] for row in rows}
        sources[input_path.name], sources[label_path.name] = file_hash(input_path), file_hash(label_path)
    names = list(student_sets)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            if student_sets[left] & student_sets[right]:
                raise ValueError(f"student split leakage between {left} and {right}")

    report = {
        "schema_version": SELECTION_VERSION, "implementation_sha256": file_hash(Path(__file__)),
        "source_hashes": sources, "scope": specification, "scope_sha256": file_hash(scope_path),
        "require_actual_feedback": True, "require_problem_statement": True,
        "require_history": require_history, "count_cap": None, "labels_used_for_selection": False,
        "current_file_scope_check": ("exactly_one_declared_python_file_at_current_time" if source_root
                                     else "not_checked_file_level_scope_only"),
        "source_root": str(source_root) if source_root else None,
        "matching_rule": MATCHING_RULE, "reference_source_split": "train",
        "global_time_rule": "donor current timestamp and all donor H timestamps < query current timestamp",
        "reference_cohort": "matched/ excludes unavailable matches; root retains every input-eligible sample",
        "known_limitations": [
            "Logged score fraction is not executed correctness or a validated ability estimate.",
            "Lexicographic nearest neighbors have no tuned distance threshold.",
            "Earlier donors may be reused; report reuse and cluster evaluation by query student.",
            "All cohorts inherit the prepared data's observed-next-submission selection boundary.",
            "Global ordering uses the dataset's timestamps without inventing timezone information.",
        ],
        "splits": {},
    }
    artifacts = {}
    for split, (rows, labels) in loaded.items():
        selected, excluded, matched, references, unmatched = [], [], [], [], []
        for row in rows:
            _time(row)
            reasons = eligibility_reasons(row, scope, require_history)
            if not reasons:
                file_reason = current_file_reason(row, source_root)
                if file_reason:
                    reasons.append(file_reason)
            if reasons:
                excluded.append({"sample_id": row["sample_id"], "reasons": reasons})
                continue
            selected.append(row)
            reference, reason = match_reference(row, donors, train_sha256)
            if reference is None:
                unmatched.append({"sample_id": row["sample_id"], "reason": reason})
            else:
                matched.append(row)
                references.append(reference)
        usage = Counter(reference["provenance"]["donor_sample_id"] for reference in references)
        report["splits"][split] = {
            "source_samples": len(rows), "selected_samples": len(selected), "matched_samples": len(matched),
            "selected_students": len({row["student_id"] for row in selected}),
            "matched_students": len({row["student_id"] for row in matched}),
            "selected_empty_histories": sum(not row["history"] for row in selected),
            "selected_by_lab_file": dict(Counter(f"{row['lab']}/{row['source_file']}" for row in selected)),
            "matched_by_lab_file": dict(Counter(f"{row['lab']}/{row['source_file']}" for row in matched)),
            "excluded_by_primary_reason": dict(Counter(row["reasons"][0] for row in excluded)),
            "unmatched_by_reason": dict(Counter(row["reason"] for row in unmatched)),
            "unique_donor_samples": len(usage), "maximum_donor_reuse": max(usage.values(), default=0),
            "selected_ids_sha256": object_hash([row["sample_id"] for row in selected]),
            "matched_ids_sha256": object_hash([row["sample_id"] for row in matched]),
        }
        artifacts[f"inputs.{split}.jsonl"] = selected
        artifacts[f"labels.{split}.jsonl"] = [labels[row["sample_id"]] for row in selected]
        artifacts[f"excluded.{split}.jsonl"] = excluded
        artifacts[f"unmatched.{split}.jsonl"] = unmatched
        artifacts[f"matched/inputs.{split}.jsonl"] = matched
        artifacts[f"matched/labels.{split}.jsonl"] = [labels[row["sample_id"]] for row in matched]
        artifacts[f"matched/references.{split}.jsonl"] = references
    output.mkdir(parents=True, exist_ok=True)
    (output / "matched").mkdir()
    for filename, rows in artifacts.items():
        _write_jsonl(output / filename, rows)
    report["output_hashes"] = {filename: file_hash(output / filename) for filename in artifacts}
    with (output / "selection.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", type=Path, required=True, help="JSON with explicit lab/source_file pairs")
    parser.add_argument("--splits", choices=("dev", "test"), nargs="+", default=["dev", "test"])
    parser.add_argument("--require-history", action="store_true")
    parser.add_argument("--source-root", type=Path,
                        help="optional raw ProgFeed root; enforce one Python file in CURRENT submission only")
    args = parser.parse_args(argv)
    try:
        report = select_inputs(args.prepared, args.output, args.scope, tuple(args.splits),
                               args.require_history, args.source_root)
        print(json.dumps(report["splits"], ensure_ascii=False, sort_keys=True, indent=2))
        return 0
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
