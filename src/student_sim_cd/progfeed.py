"""Reconstruct ProgFeed submission pairs by reading released files as data only.

No student module or autograder is imported or executed. Run with::

    python -m student_sim_cd.progfeed build --source data/raw/progfeed --output data/prepared/v1
"""

import argparse
import csv
import hashlib
import io
import json
import math
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


SCHEMA_VERSION = "student-sim-cd.progfeed.v1"
DEFAULT_SEED = 20260929
REQUIRED_COLUMNS = {
    "student_id", "lab", "submission_timestamp", "source_file", "test_name",
    "function_name", "score", "max_score", "status", "testcase_mask",
    "ai_feedback_type", "ai_feedback_text",
}


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _identifier(value):
    """Accept one ordinary path component, never an absolute or traversal path."""
    return bool(value) and value not in (".", "..") and not any(
        char in value for char in ("/", "\\", "\x00")
    )


def _timestamp(value):
    if len(value) != 19:
        return None
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d-%H-%M-%S")
        return parsed if parsed.strftime("%Y-%m-%d-%H-%M-%S") == value else None
    except ValueError:
        return None


def _lab_name(value):
    return value.lower().replace("-", "").replace("_", "")


class SourceReader:
    """Read regular, in-repository files and fingerprint exactly the bytes read."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.files = {}

    def read(self, path):
        path = Path(path)
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(self.root)
        except ValueError as error:
            raise ValueError("Source path escapes repository: %s" % path) from error
        if path.is_symlink():
            raise ValueError("Source symlink is not accepted: %s" % path)
        content = resolved.read_bytes()
        self.files[relative.as_posix()] = {
            "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)
        }
        # Preserve line endings; invalid encodings are reported, not silently replaced.
        return content.decode("utf-8-sig")


def _number(value, counts):
    if value is None or not value.strip():
        counts["missing_numeric_fields"] += 1
        return None
    try:
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("non-finite")
        return result
    except ValueError:
        counts["invalid_numeric_fields"] += 1
        return None


def _mask(value, counts):
    if not value or not value.strip():
        counts["rows_missing_testcase_mask"] += 1
        return None
    try:
        result = json.loads(value)
        if not isinstance(result, list) or any(
            type(item) is not int or item not in (0, 1) for item in result
        ):
            raise ValueError("mask must contain integer zero or one")
        if not result:
            counts["rows_empty_testcase_mask"] += 1
        return result
    except (ValueError, TypeError):
        counts["rows_invalid_testcase_mask"] += 1
        return None


def student_splits(students, seed=DEFAULT_SEED):
    """Fixed 80/10/10 student split; no submission can cross a split."""
    students = sorted(set(students))
    random.Random(seed).shuffle(students)
    train_end = int(len(students) * 0.8)
    dev_end = train_end + int(len(students) * 0.1)
    return {
        student: "train" if index < train_end else "dev" if index < dev_end else "test"
        for index, student in enumerate(students)
    }


def _problem_statements(reader, labs, counts):
    """Use all official description texts in the unique matching lab directory.

    This is a lab-level statement, not a guessed function-to-problem mapping.
    PDF-only labs remain explicitly missing until a reviewed extraction is supplied.
    """
    root = reader.root / "autograders"
    directories = list(root.iterdir()) if root.is_dir() else []
    statements, provenance = {}, {}
    for lab in sorted(labs):
        matches = [p for p in directories if p.is_dir()
                   and _lab_name(p.name.split("_autograder", 1)[0]) == _lab_name(lab)]
        paths = sorted(matches[0].rglob("*_desc.txt")) if len(matches) == 1 else []
        sections = []
        for path in paths:
            text = reader.read(path)
            if text.strip():
                sections.append("### %s\n%s" % (path.relative_to(matches[0]).as_posix(), text))
        statements[lab] = "\n\n".join(sections) or None
        provenance[lab] = {
            "scope": "all_official_description_texts_in_lab",
            "paths": [p.relative_to(reader.root).as_posix() for p in paths],
            "description_identifiers": [p.name[:-len("_desc.txt")] for p in paths],
            "available": bool(sections),
            "mapping": "unique_lab_prefix_before_autograder_suffix" if len(matches) == 1 else "unresolved",
        }
        if not sections:
            counts["labs_missing_text_problem_statement"] += 1
    return statements, provenance


def _write_jsonl(path, records):
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(_json(record) + "\n")


def build(source, output, seed=DEFAULT_SEED):
    """Build physical input/label separation and return the static audit report."""
    reader = SourceReader(source)
    output = Path(output).resolve()
    if output == reader.root or reader.root in output.parents:
        raise ValueError("Output must be outside the raw source repository")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be absent or empty; refusing to overwrite")
    csv_path = reader.root / "all_submissions_consolidated.csv"
    # csv.reader needs newline-preserving input for embedded code and feedback strings.
    csv.field_size_limit(sys.maxsize)
    table = csv.DictReader(io.StringIO(reader.read(csv_path), newline=""))
    missing = REQUIRED_COLUMNS - set(table.fieldnames or [])
    if missing:
        raise ValueError("Missing required CSV columns: %s" % ", ".join(sorted(missing)))
    counts = Counter({name: 0 for name in (
        "csv_rows", "unique_csv_rows", "duplicate_csv_rows_removed",
        "rows_missing_testcase_mask", "rows_invalid_testcase_mask", "rows_empty_testcase_mask",
        "rows_with_delivered_feedback", "assigned_feedback_rows_without_delivery",
        "no_feedback_condition_with_delivery", "missing_or_unreadable_source_files",
        "adjacent_file_pairs", "adjacent_file_pairs_missing_code", "unchanged_code_pairs",
        "pairs_with_delivered_feedback", "pairs_missing_text_problem_statement",
        "multi_file_submissions", "pairs_involving_multi_file_submission",
        "file_states_without_observed_next_submission",
    )})
    issues = defaultdict(list)
    submissions = {}
    seen_rows = set()
    arrival_order = defaultdict(list)

    def issue(kind, item):
        counts[kind] += 1
        if len(issues[kind]) < 25:
            issues[kind].append(item)

    def submission(student, lab, timestamp):
        key = (student, lab, timestamp)
        if key not in submissions:
            submissions[key] = {"student_id": student, "lab": lab, "timestamp": timestamp,
                                "rows": [], "files": {}, "expected_files": set()}
        return submissions[key]

    for line_number, row in enumerate(table, 2):
        counts["csv_rows"] += 1
        if None in row or any(row.get(name) is None for name in REQUIRED_COLUMNS):
            issue("malformed_csv_rows", {"line": line_number})
            continue
        signature = _json(row)
        if signature in seen_rows:
            counts["duplicate_csv_rows_removed"] += 1
            continue
        seen_rows.add(signature)
        student, lab, timestamp = (row[name] for name in ("student_id", "lab", "submission_timestamp"))
        if not _identifier(student) or not _identifier(lab) or _timestamp(timestamp) is None:
            issue("invalid_submission_keys", {"line": line_number})
            continue
        counts["unique_csv_rows"] += 1
        item = submission(student, lab, timestamp)
        order = arrival_order[(student, lab)]
        if not order or order[-1] != timestamp:
            order.append(timestamp)
        filename = row["source_file"]
        if filename and not _identifier(filename):
            issue("unsafe_source_file_rows", {"line": line_number})
            continue
        if not filename:
            counts["rows_missing_source_file"] += 1
        else:
            item["expected_files"].add(filename)
        result = {
            "test_name": row["test_name"], "function_name": row["function_name"],
            "status": row["status"], "score": _number(row["score"], counts),
            "max_score": _number(row["max_score"], counts),
            "testcase_mask": _mask(row["testcase_mask"], counts),
        }
        assigned, delivered = row["ai_feedback_type"], row["ai_feedback_text"]
        if assigned in ("tc", "nl"):
            counts["rows_assigned_feedback_condition"] += 1
            if not delivered.strip():
                counts["assigned_feedback_rows_without_delivery"] += 1
        if delivered.strip():
            counts["rows_with_delivered_feedback"] += 1
            if assigned == "no_feedback":
                counts["no_feedback_condition_with_delivery"] += 1
        item["rows"].append({"source_file": filename, "result": result,
                             "assigned_type": assigned, "text": delivered})

    counts["trajectories_requiring_time_sort"] = sum(
        timestamps != sorted(timestamps) for timestamps in arrival_order.values()
    )
    # Include directories absent from the CSV so adjacency never bridges an unlogged attempt.
    labs_root = reader.root / "all_labs"
    if not labs_root.is_dir():
        raise ValueError("Missing all_labs directory")
    for lab_dir in sorted(labs_root.iterdir()):
        if not lab_dir.is_dir() or lab_dir.name.startswith("."):
            continue
        for student_dir in sorted(lab_dir.iterdir()):
            if not student_dir.is_dir() or student_dir.name.startswith("."):
                continue
            for time_dir in sorted(student_dir.iterdir()):
                if not time_dir.is_dir() or time_dir.name.startswith("."):
                    continue
                if _timestamp(time_dir.name) is None:
                    issue("unrecognized_submission_directories", time_dir.relative_to(reader.root).as_posix())
                    continue
                item = submission(student_dir.name, lab_dir.name, time_dir.name)
                if not item["rows"]:
                    counts["submission_directories_without_csv_rows"] += 1
                item["expected_files"].update(p.name for p in time_dir.glob("*.py") if p.is_file())

    trajectories = defaultdict(list)
    for key, item in sorted(submissions.items()):
        student, lab, timestamp = key
        directory = labs_root / lab / student / timestamp
        for filename in sorted(item["expected_files"]):
            path = directory / filename
            try:
                item["files"][filename] = reader.read(path)
                counts["source_files_read"] += 1
            except (OSError, UnicodeError, ValueError) as error:
                issue("missing_or_unreadable_source_files", {
                    "student_id": student, "lab": lab, "timestamp": timestamp,
                    "source_file": filename, "error_type": type(error).__name__,
                })
        if len(item["files"]) > 1:
            counts["multi_file_submissions"] += 1
        if not item["files"]:
            counts["submissions_without_readable_code"] += 1
        # Read logs only to audit delivery provenance, never execute their content.
        log_path = directory / "results.json"
        if log_path.is_file():
            try:
                log = json.loads(reader.read(log_path))
                tests = log.get("tests", []) if isinstance(log, dict) else []
                outputs = "\n".join(str(test.get("output", "")) for test in tests if isinstance(test, dict))
                for row in item["rows"]:
                    if row["text"].strip() and row["text"].strip() not in outputs:
                        counts["csv_feedback_text_not_found_verbatim_in_results"] += 1
                if "🤖 AI Feedback for you" in outputs and not any(row["text"].strip() for row in item["rows"]):
                    counts["results_feedback_marker_without_csv_delivery"] += 1
            except (ValueError, UnicodeError, OSError, AttributeError):
                counts["unreadable_results_json"] += 1
        else:
            counts["submissions_missing_results_json"] += 1
        for filename in item["files"]:
            keys = [(r["result"]["test_name"], r["result"]["function_name"])
                    for r in item["rows"] if r["source_file"] == filename]
            if len(set(keys)) != len(keys):
                counts["files_with_repeated_test_keys"] += 1
        trajectories[(student, lab)].append(item)

    students = sorted({key[0] for key in trajectories})
    splits = student_splits(students, seed)
    labs = sorted({key[1] for key in trajectories})
    statements, statement_provenance = _problem_statements(reader, labs, counts)
    for lab in labs:
        lab_items = [item for item in submissions.values() if item["lab"] == lab]
        functions = sorted({row["result"]["function_name"] for item in lab_items
                            for row in item["rows"] if row["result"]["function_name"]})
        descriptions = set(statement_provenance[lab]["description_identifiers"])
        uncovered = [name for name in functions if name not in descriptions]
        statement_provenance[lab].update({
            "observed_source_files": sorted({name for item in lab_items for name in item["expected_files"]}),
            "observed_function_names": functions,
            "function_names_without_same_named_description": uncovered,
            "coverage_review": "requires_review;nonempty_text_alone_is_not_per_file_coverage_verification",
        })
        # Upstream lab10 ships lab09 descriptions. Fail closed on observable name
        # mismatch instead of feeding the model an unrelated nonempty statement.
        if statements[lab] and uncovered:
            statements[lab] = None
            statement_provenance[lab].update({
                "available": False, "raw_text_available": True,
                "coverage_review": "rejected_observed_function_names_not_covered_by_descriptions",
            })
            counts["labs_with_description_function_mismatch"] += 1
    counts["labs_without_usable_problem_statement"] = sum(text is None for text in statements.values())
    inputs, labels = {s: [] for s in ("train", "dev", "test")}, {s: [] for s in ("train", "dev", "test")}
    censored, excluded = [], []
    by_lab, by_split = defaultdict(Counter), defaultdict(Counter)

    def state(item, filename):
        rows = sorted((r for r in item["rows"] if r["source_file"] == filename),
                      key=lambda r: _json(r["result"]))
        return {
            "timestamp": item["timestamp"], "code": item["files"][filename],
            "results": [r["result"] for r in rows],
            "feedback": [{"test_name": r["result"]["test_name"],
                          "function_name": r["result"]["function_name"],
                          "assigned_type": r["assigned_type"], "text": r["text"]}
                         for r in rows if r["text"].strip()],
        }

    for (student, lab), sequence in sorted(trajectories.items()):
        sequence.sort(key=lambda item: item["timestamp"])
        history = defaultdict(list)
        for index, current in enumerate(sequence):
            next_item = sequence[index + 1] if index + 1 < len(sequence) else None
            for filename in sorted(current["expected_files"]):
                identity = {"student_id": student, "lab": lab, "source_file": filename,
                            "current_timestamp": current["timestamp"]}
                sample_id = "pf_" + hashlib.sha256(_json(identity).encode("utf-8")).hexdigest()[:24]
                if next_item is None:
                    censored.append(dict(identity, sample_id=sample_id, reason="no_observed_next_submission"))
                    counts["file_states_without_observed_next_submission"] += 1
                    continue
                if filename not in current["files"] or filename not in next_item["files"]:
                    if filename not in current["files"]:
                        reason = "current_source_file_unreadable"
                    elif filename not in next_item["expected_files"]:
                        reason = "source_name_absent_in_next_submission_possible_rename_or_deletion"
                        counts["adjacent_file_pairs_with_source_name_change_or_absence"] += 1
                    else:
                        reason = "adjacent_next_source_file_unreadable"
                    excluded.append(dict(identity, sample_id=sample_id, next_timestamp=next_item["timestamp"],
                                         reason=reason))
                    counts["adjacent_file_pairs_missing_code"] += 1
                    continue
                current_state, target_state = state(current, filename), state(next_item, filename)
                record = dict(identity, schema_version=SCHEMA_VERSION, sample_id=sample_id,
                              current_code=current_state["code"], current_results=current_state["results"],
                              feedback=current_state["feedback"], history=list(history[filename]),
                              problem_statement=statements[lab])
                label = {"schema_version": SCHEMA_VERSION, "sample_id": sample_id,
                         "target_timestamp": next_item["timestamp"], "target_code": target_state["code"],
                         "target_results": target_state["results"]}
                split = splits[student]
                inputs[split].append(record)
                labels[split].append(label)
                counts["adjacent_file_pairs"] += 1
                by_lab[lab]["pairs"] += 1
                by_split[split]["pairs"] += 1
                if current_state["code"] == target_state["code"]:
                    counts["unchanged_code_pairs"] += 1
                    by_lab[lab]["unchanged_code_pairs"] += 1
                if current_state["feedback"]:
                    counts["pairs_with_delivered_feedback"] += 1
                    by_lab[lab]["pairs_with_delivered_feedback"] += 1
                if not current_state["results"] or not target_state["results"]:
                    counts["pairs_with_missing_current_or_target_test_rows"] += 1
                if statements[lab] is None:
                    counts["pairs_missing_text_problem_statement"] += 1
                if len(current["files"]) > 1 or len(next_item["files"]) > 1:
                    counts["pairs_involving_multi_file_submission"] += 1
                    by_lab[lab]["multi_file_pairs"] += 1
            # Append only after all current predictions are formed: never include c_t in H.
            for filename in sorted(current["files"]):
                history[filename].append(state(current, filename))

    counts["students"] = len(students)
    counts["labs"] = len(labs)
    counts["submissions"] = len(submissions)
    counts["student_lab_trajectories"] = len(trajectories)
    for student, split in splits.items():
        by_split[split]["students"] += 1
    inventory = [{"path": path, **info} for path, info in sorted(reader.files.items())]
    audit = {
        "schema_version": SCHEMA_VERSION,
        "source_url": "https://github.com/umass-ml4ed/progFeed-dataset-public",
        "source_path": str(reader.root), "seed": seed,
        "source_fingerprint_sha256": hashlib.sha256(_json(inventory).encode("utf-8")).hexdigest(),
        "source_files": inventory, "counts": dict(sorted(counts.items())),
        "by_lab": {k: dict(v) for k, v in sorted(by_lab.items())},
        "by_split": {k: dict(v) for k, v in sorted(by_split.items())},
        "issue_examples": dict(issues), "problem_statements": statement_provenance,
        "scope": {
            "sample_unit": "complete_source_file_between_adjacent_lab_submissions",
            "history": "all_readable_strictly_earlier_submissions_for_same_student_lab_source_file",
            "multi_file_tasks": "file_level_prediction; companion_file_context_not_included",
            "missing_next": "censored_not_unchanged_or_mastered",
            "feedback": "nonempty_ai_feedback_text_is_delivery;assigned_type_is_not_delivery",
            "results": "released_logged_tests_only;score_is_not_testcase_pass_rate",
            "problem_statement": "all_lab_description_texts;missing_text_must_block_model_inference",
            "execution": "static_read_only;no_student_or_autograder_execution",
            "limitations": [
                "Only observed consecutive next submissions are labeled; attrition is not a behavior label.",
                "Multiple source files from one submission are correlated, not independent students.",
                "Multifile predictions omit companion code and are not complete-project simulation.",
                "Missing historical source files produce incomplete observed histories.",
                "CSV/log feedback mismatches need inspection; CSV delivery text is the released authority.",
                "Shared description text is not yet a verified per-file problem mapping.",
            ],
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    for split in inputs:
        _write_jsonl(output / ("inputs.%s.jsonl" % split), inputs[split])
        _write_jsonl(output / ("labels.%s.jsonl" % split), labels[split])
    _write_jsonl(output / "censored.jsonl", censored)
    _write_jsonl(output / "excluded_pairs.jsonl", excluded)
    (output / "splits.json").write_text(_json({
        "schema_version": SCHEMA_VERSION, "seed": seed, "unit": "student_id",
        "ratios": {"train": 0.8, "dev": 0.1, "test": 0.1},
        "algorithm": "sorted_ids_then_python_random_Random_seed_shuffle_floor_train_dev",
        "students": splits,
    }) + "\n", encoding="utf-8")
    (output / "audit.json").write_text(_json(audit) + "\n", encoding="utf-8")
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser("build", help="Build audited, student-disjoint submission pairs")
    command.add_argument("--source", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args(argv)
    try:
        audit = build(args.source, args.output, args.seed)
    except (OSError, ValueError) as error:
        parser.exit(2, "progfeed: %s\n" % error)
    print(_json({"output": str(args.output), "counts": audit["counts"],
                 "source_fingerprint_sha256": audit["source_fingerprint_sha256"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
