"""Static revision baselines and paired evaluation; never execute submissions."""

import argparse
from collections import defaultdict
import difflib
import hashlib
import json
from pathlib import Path
import random
import statistics


def read_records(path):
    records = {}
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            key = item["sample_id"]
            if key in records:
                raise ValueError("Duplicate sample ID at line %d" % number)
            records[key] = item
    return records


def copy_baseline(inputs, output):
    """This entry point has no labels argument: copying cannot inspect the future."""
    records = read_records(inputs)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        for sample_id, item in records.items():
            if "target_code" in item or "target_results" in item:
                raise ValueError("Future label found in inputs")
            prediction = {"sample_id": sample_id, "method": "copy",
                          "predicted_code": item["current_code"]}
            stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    return {"samples": len(records), "method": "copy", "model_used": False}


def edit_events(before, after):
    """Changed original-line indices and insertion boundaries (zero based).

    Locations and operation types are proxies for behavior, not semantic truth.
    Inserting a line at position i is distinct from replacing original line i.
    """
    left, right = before.splitlines(keepends=True), after.splitlines(keepends=True)
    locations, operations, size = set(), set(), 0
    matcher = difflib.SequenceMatcher(a=left, b=right, autojunk=False)
    for op, start, end, new_start, new_end in matcher.get_opcodes():
        if op == "equal":
            continue
        if op == "insert":
            locations.add(("boundary", start))
            operations.add(("insert", start))
        else:
            for index in range(start, end):
                locations.add(("line", index))
                operations.add((op, index))
        size += (end - start) + (new_end - new_start)
    return locations, operations, size


def f1(predicted, actual):
    if not predicted and not actual:
        return 1.0
    return 2 * len(predicted & actual) / (len(predicted) + len(actual))


def pair_metrics(current, predicted, target):
    pred_locations, pred_operations, pred_size = edit_events(current, predicted)
    true_locations, true_operations, true_size = edit_events(current, target)
    return {
        "exact_next_code": float(predicted == target),
        "true_changed": float(target != current),
        "predicted_changed": float(predicted != current),
        "edit_location_f1": f1(pred_locations, true_locations),
        "edit_operation_f1": f1(pred_operations, true_operations),
        "edit_size_absolute_error": abs(pred_size - true_size),
        "text_similarity": difflib.SequenceMatcher(a=predicted, b=target, autojunk=False).ratio(),
    }


def summarize(rows, bootstrap=2000, seed=20260929):
    """Equal-student estimand; uncertainty resamples students, not submissions."""
    if not rows:
        return {"samples": 0, "students": 0, "metrics": {}}
    by_student = defaultdict(list)
    for row in rows:
        by_student[row["student_id"]].append(row["metrics"])
    names = list(rows[0]["metrics"])
    means = [{name: statistics.mean(row[name] for row in student_rows) for name in names}
             for _, student_rows in sorted(by_student.items())]
    rng, sampled = random.Random(seed), {name: [] for name in names}
    if len(means) >= 2:
        for _ in range(bootstrap):
            selected = [means[rng.randrange(len(means))] for _ in means]
            for name in names:
                sampled[name].append(statistics.mean(row[name] for row in selected))
    metrics = {}
    for name in names:
        values = sorted(sampled[name])
        metrics[name] = {
            "student_macro_mean": statistics.mean(row[name] for row in means),
            "submission_mean": statistics.mean(row["metrics"][name] for row in rows),
            "student_bootstrap_95_ci": [values[int(0.025 * len(values))],
                                        values[min(len(values) - 1, int(0.975 * len(values)))]] if values else None,
        }
    return {"samples": len(rows), "students": len(means), "metrics": metrics}


def evaluate(inputs, labels, predictions, output, bootstrap=2000, seed=20260929):
    if bootstrap < 1:
        raise ValueError("Bootstrap repetitions must be positive")
    x, y, p = map(read_records, (inputs, labels, predictions))
    if not x or set(x) != set(y) or set(x) != set(p):
        raise ValueError("Inputs, labels and predictions must have equal nonempty ID sets")
    methods = {row.get("method") for row in p.values()}
    if len(methods) != 1 or not all(isinstance(method, str) and method for method in methods):
        raise ValueError("Each evaluation must contain exactly one named method")
    rows = []
    for sample_id in sorted(x):
        if not all(isinstance(code, str) for code in
                   (x[sample_id]["current_code"], y[sample_id]["target_code"], p[sample_id]["predicted_code"])):
            raise ValueError("Code must be text")
        rows.append({"sample_id": sample_id, "student_id": x[sample_id]["student_id"],
                     "has_feedback": bool(x[sample_id]["feedback"]),
                     "metrics": pair_metrics(x[sample_id]["current_code"], p[sample_id]["predicted_code"],
                                             y[sample_id]["target_code"])})
    report = {
        "schema_version": 1,
        "evaluation_kind": "static_observed_next_revision_no_execution",
        "methods": sorted(methods),
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": seed, "bootstrap_repetitions": bootstrap,
        "source_sha256": {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                          for name, path in (("inputs", inputs), ("labels", labels), ("predictions", predictions))},
        "all": summarize(rows, bootstrap, seed),
        "true_changed": summarize([row for row in rows if row["metrics"]["true_changed"]], bootstrap, seed),
        "true_unchanged": summarize([row for row in rows if not row["metrics"]["true_changed"]], bootstrap, seed),
        "actual_feedback": summarize([row for row in rows if row["has_feedback"]], bootstrap, seed),
        "limitations": ["One observed next submission is not the full behavior distribution.",
                        "Static edit metrics do not measure correctness or feedback uptake.",
                        "Unchanged submissions can inflate aggregate text similarity and edit F1.",
                        "These results do not establish CD effectiveness."],
    }
    output = Path(output)
    if output.exists():
        raise ValueError("Evaluation output already exists")
    output.mkdir(parents=True)
    (output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    with (output / "per_sample.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    copy = sub.add_parser("copy-baseline")
    copy.add_argument("--inputs", required=True)
    copy.add_argument("--output", required=True)
    score = sub.add_parser("evaluate")
    for name in ("inputs", "labels", "predictions", "output"):
        score.add_argument("--" + name, required=True)
    score.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if args.command == "copy-baseline":
        report = copy_baseline(args.inputs, args.output)
    else:
        report = evaluate(args.inputs, args.labels, args.predictions, args.output, args.bootstrap)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
