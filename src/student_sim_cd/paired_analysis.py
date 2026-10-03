"""Complete-set static comparisons with paired, equal-student uncertainty.

This module reads no files, selects no candidates, fits no parameters, and never
executes code supplied as text. Callers must declare whether the supplied scores
are a legacy preserve-format diagnostic or a newly rescored extraction-v1 run.
Bootstrap confidence intervals describe uncertainty; no significance decisions
or method-effectiveness claims are generated, including across multiple pairs.
"""

from collections import defaultdict
import hashlib
import json
import math
import random
import re
import statistics
from typing import Mapping

from .evaluate import edit_events, pair_metrics


SCHEMA_VERSION = "student-sim-cd.paired-analysis.v1"
PROTOCOLS = {"legacy_preserve_diagnostic", "cached-python-fence-v1", "extraction_v1_rescored"}
PROTOCOL_ALIASES = {"extraction_v1_rescored": "cached-python-fence-v1"}
DEFAULT_COMPARISONS = (
    ("b", "base"), ("cd", "base"), ("history_d0", "base"),
    ("b", "copy"), ("cd", "history_d0"),
)
REQUIRED_METRICS = {
    "exact_next_code", "true_changed", "predicted_changed", "edit_location_f1",
    "edit_operation_f1", "edit_size_absolute_error", "text_similarity",
}
OPTIONAL_METRICS = {"predicted_edit_size", "true_edit_size"}
BOUNDED_METRICS = {
    "exact_next_code", "true_changed", "predicted_changed",
    "edit_location_f1", "edit_operation_f1", "text_similarity",
}
BINARY_METRICS = {"exact_next_code", "true_changed", "predicted_changed"}
HIGHER_IS_BETTER = {"exact_next_code", "edit_location_f1", "edit_operation_f1", "text_similarity"}
LOWER_IS_BETTER = {"edit_size_absolute_error"}
METHOD_KINDS = {"shared_pool_reranking", "original_greedy", "copy_current", "other_static"}
KNOWN_KINDS = {
    "base": "shared_pool_reranking", "history_d0": "shared_pool_reranking",
    "cd": "shared_pool_reranking", "b": "shared_pool_reranking", "copy": "copy_current",
}
TIE_TOLERANCE = 1e-12


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _identifier(value, name):
    _require(isinstance(value, str) and bool(value), name + " must be a nonempty string")
    return value


def _records(value, name):
    """Accept record iterables or already indexed records, without losing duplicates."""
    indexed = {}
    is_mapping = isinstance(value, Mapping)
    if is_mapping:
        records = value.items()
    else:
        _require(not isinstance(value, (str, bytes)), name + " must be records, not a path/text")
        try:
            records = ((None, row) for row in value)
        except TypeError as error:
            raise ValueError(name + " must be an iterable of records") from error
    for outer_key, row in records:
        _require(isinstance(row, Mapping), name + " contains a non-object record")
        sample = _identifier(row.get("sample_id"), name + " sample_id")
        _require(not is_mapping or outer_key == sample, name + " index key differs from sample_id")
        _require(sample not in indexed, name + " has duplicate sample_id " + sample)
        indexed[sample] = row
    _require(bool(indexed), name + " is empty")
    return indexed


def _parameters(protocol, expected_samples, expected_students, bootstrap, seed):
    _require(isinstance(protocol, str) and protocol in PROTOCOLS,
             "protocol must explicitly name a supported score protocol")
    for name, value in (("expected_samples", expected_samples), ("expected_students", expected_students),
                        ("bootstrap", bootstrap)):
        _require(type(value) is int and value >= 1, name + " must be a positive integer")
    _require(expected_students <= expected_samples, "Expected students exceed expected samples")
    _require(type(seed) is int, "seed must be an integer")


def _cohort(sample_students, expected_samples, expected_students):
    _require(isinstance(sample_students, Mapping) and bool(sample_students),
             "sample_students must declare the complete sample-to-student mapping")
    cohort = {}
    for sample, student in sample_students.items():
        cohort[_identifier(sample, "cohort sample_id")] = _identifier(student, "cohort student_id")
    _require(len(cohort) == expected_samples, "Cohort does not have the expected complete sample count")
    _require(len(set(cohort.values())) == expected_students, "Cohort does not have the expected student count")
    return cohort


def _same_ids(records, cohort, name):
    missing = sorted(set(cohort) - set(records))
    extra = sorted(set(records) - set(cohort))
    _require(not missing and not extra,
             name + " differs from the complete cohort: missing=" + repr(missing) + "; extra=" + repr(extra))


def _methods(method_records, method_kinds):
    _require(isinstance(method_records, Mapping) and bool(method_records), "Methods must be a nonempty mapping")
    _require(all(isinstance(name, str) and re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name)
                 for name in method_records), "Method names must be simple nonempty identifiers")
    methods = sorted(method_records)
    if method_kinds is None:
        method_kinds = {}
    _require(isinstance(method_kinds, Mapping) and set(method_kinds) <= set(methods),
             "Method kinds must name only supplied methods")
    kinds = {}
    for name in methods:
        kind = method_kinds.get(name, KNOWN_KINDS.get(name, "other_static"))
        _require(isinstance(kind, str) and kind in METHOD_KINDS, "Unknown method kind for " + name)
        if name == "base":
            _require(kind == "shared_pool_reranking", "base is shared-pool reranking, not original greedy")
        elif name in KNOWN_KINDS:
            _require(kind == KNOWN_KINDS[name], "Known method kind cannot change for " + name)
        if name == "greedy" or kind == "original_greedy":
            _require(method_kinds.get(name) == "original_greedy",
                     "Original greedy requires an explicit method_kinds provenance declaration")
        kinds[name] = kind
    return methods, kinds


def _comparisons(value, methods, reference_method):
    _require(reference_method in methods, "Reference method is absent")
    pairs = DEFAULT_COMPARISONS if value is None else value
    _require(not isinstance(pairs, (str, bytes)), "comparisons must contain method/reference pairs")
    result, seen = [], set()
    for pair in pairs:
        _require(isinstance(pair, (list, tuple)) and len(pair) == 2,
                 "Each comparison must contain exactly method and reference")
        method, reference = pair
        _require(method in methods and reference in methods, "Comparison names an absent method")
        _require(method != reference, "Self-comparisons are not informative")
        _require((method, reference) not in seen, "Duplicate comparison")
        seen.add((method, reference))
        result.append((method, reference))
    _require(bool(result), "At least one comparison must be declared")
    return result


def _number(value, name):
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), name + " must be numeric")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(name + " must be finite") from error
    _require(math.isfinite(result), name + " must be finite")
    return result


def _metric_values(row, name):
    metrics = row.get("metrics")
    _require(isinstance(metrics, Mapping) and REQUIRED_METRICS <= set(metrics),
             name + " is missing required static metrics")
    _require(set(metrics) <= REQUIRED_METRICS | OPTIONAL_METRICS,
             name + " contains unsupported metrics")
    values = {key: _number(value, name + " " + key) for key, value in metrics.items()}
    for key in BOUNDED_METRICS:
        _require(0 <= values[key] <= 1, name + " " + key + " lies outside [0, 1]")
    for key in BINARY_METRICS:
        _require(values[key] in (0, 1), name + " " + key + " must be 0 or 1")
    for key in {"edit_size_absolute_error"} | OPTIONAL_METRICS:
        if key in values:
            _require(values[key] >= 0 and values[key].is_integer(), name + " " + key + " must be a nonnegative count")
    return values


def _prediction_records(predictions_by_method, methods, cohort):
    _require(isinstance(predictions_by_method, Mapping) and set(predictions_by_method) == set(methods),
             "Prediction methods must exactly match evaluation methods")
    result = {}
    for method in methods:
        records = _records(predictions_by_method[method], method + " predictions")
        _same_ids(records, cohort, method + " predictions")
        for row in records.values():
            _require(row.get("method") == method, "Prediction method does not match its method mapping")
            _require(isinstance(row.get("predicted_code"), str), "Predicted code must be text")
        result[method] = records
    return result


def _with_edit_sizes(evaluations, cohort, current_codes, predictions_by_method):
    _require(isinstance(current_codes, Mapping), "current_codes must be a mapping of complete source text")
    _same_ids(current_codes, cohort, "current_codes")
    _require(all(isinstance(code, str) for code in current_codes.values()), "Current code must be text")
    predictions = _prediction_records(predictions_by_method, sorted(evaluations), cohort)
    enriched = {}
    for method, records in evaluations.items():
        enriched[method] = {}
        for sample, row in records.items():
            before = current_codes[sample]
            after = predictions[method][sample]["predicted_code"]
            measured_size = edit_events(before, after)[2]
            metrics = dict(row["metrics"])
            if "predicted_edit_size" in metrics:
                _require(metrics["predicted_edit_size"] == measured_size, "Supplied predicted edit size differs from code")
            _require(metrics["predicted_changed"] == float(before != after),
                     "Supplied predicted change flag differs from code")
            metrics["predicted_edit_size"] = measured_size
            enriched[method][sample] = {**row, "metrics": metrics}
    return enriched


def _validate_evaluations(evaluations_by_method, cohort, methods):
    records, metric_names, metadata = {}, None, {}
    for method in methods:
        indexed = _records(evaluations_by_method[method], method + " evaluations")
        _same_ids(indexed, cohort, method + " evaluations")
        checked = {}
        for sample in sorted(cohort):
            row = indexed[sample]
            _require(row.get("student_id") == cohort[sample], "Evaluation student differs from cohort for " + sample)
            _require(type(row.get("has_feedback")) is bool, "has_feedback must be an explicit Boolean")
            _require("method" not in row or row["method"] == method, "Evaluation method differs from mapping")
            metrics = _metric_values(row, method + "/" + sample)
            if metric_names is None:
                metric_names = set(metrics)
            _require(set(metrics) == metric_names, "All methods and samples must have the same metric schema")
            truth = (row["has_feedback"], metrics["true_changed"], metrics.get("true_edit_size"))
            if sample in metadata:
                _require(truth == metadata[sample], "Evaluation truth/feedback differs between methods for " + sample)
            metadata[sample] = truth
            checked[sample] = {"sample_id": sample, "student_id": cohort[sample],
                               "has_feedback": row["has_feedback"], "metrics": metrics}
        records[method] = checked
    return records, sorted(metric_names)


def _ci(values):
    if not values:
        return None
    values = sorted(values)
    return [values[int(0.025 * len(values))], values[min(len(values) - 1, int(0.975 * len(values)))]]


def _direction(metric):
    if metric in HIGHER_IS_BETTER:
        return "higher_is_better"
    if metric in LOWER_IS_BETTER:
        return "lower_is_better"
    return "diagnostic_only"


def _sign_counts(values):
    return {"positive": sum(value > TIE_TOLERANCE for value in values),
            "tie": sum(abs(value) <= TIE_TOLERANCE for value in values),
            "negative": sum(value < -TIE_TOLERANCE for value in values)}


def _wins(values, metric):
    direction = _direction(metric)
    if direction == "diagnostic_only":
        return None
    counts = _sign_counts(values if direction == "higher_is_better" else [-value for value in values])
    return {"wins": counts["positive"], "ties": counts["tie"], "losses": counts["negative"]}


def _summarize_group(records, sample_ids, metric_names, pairs, bootstrap, seed):
    methods = sorted(records)
    if not sample_ids:
        return {"samples": 0, "students": 0, "methods": {}, "comparisons": []}
    by_student = defaultdict(list)
    for sample in sample_ids:
        by_student[records[methods[0]][sample]["student_id"]].append(sample)
    students = sorted(by_student)
    means = {method: [{metric: statistics.mean(records[method][sample]["metrics"][metric]
                                             for sample in by_student[student])
                       for metric in metric_names} for student in students] for method in methods}
    differences = {(method, reference): [{metric: means[method][i][metric] - means[reference][i][metric]
                                          for metric in metric_names} for i in range(len(students))]
                   for method, reference in pairs}
    sampled_means = {method: {metric: [] for metric in metric_names} for method in methods}
    sampled_differences = {pair: {metric: [] for metric in metric_names} for pair in pairs}
    rng = random.Random(seed)
    if len(students) >= 2:
        for _ in range(bootstrap):
            # One shared cluster draw keeps every comparison paired, including
            # methods with very different numbers of submissions per student.
            indices = [rng.randrange(len(students)) for _ in students]
            for method in methods:
                for metric in metric_names:
                    sampled_means[method][metric].append(statistics.mean(means[method][i][metric] for i in indices))
            for pair in pairs:
                for metric in metric_names:
                    sampled_differences[pair][metric].append(statistics.mean(differences[pair][i][metric] for i in indices))
    method_reports = {}
    for method in methods:
        method_reports[method] = {"metrics": {metric: {
            "student_macro_mean": statistics.mean(row[metric] for row in means[method]),
            "submission_mean": statistics.mean(records[method][sample]["metrics"][metric] for sample in sample_ids),
            "student_bootstrap_95_ci": _ci(sampled_means[method][metric]),
            "direction": _direction(metric),
        } for metric in metric_names}}
    comparisons = []
    for method, reference in pairs:
        pair = (method, reference)
        metrics = {}
        for metric in metric_names:
            student_deltas = [row[metric] for row in differences[pair]]
            sample_deltas = [records[method][sample]["metrics"][metric]
                             - records[reference][sample]["metrics"][metric] for sample in sample_ids]
            metrics[metric] = {
                "student_macro_delta": statistics.mean(student_deltas),
                "submission_mean_delta": statistics.mean(sample_deltas),
                "paired_student_bootstrap_95_ci": _ci(sampled_differences[pair][metric]),
                "student_delta_sign_counts": _sign_counts(student_deltas),
                "student_wins_ties_losses": _wins(student_deltas, metric),
                "submission_wins_ties_losses": _wins(sample_deltas, metric),
                "direction": _direction(metric),
            }
        comparisons.append({"method": method, "reference": reference,
            "delta_convention": "method_minus_reference", "metrics": metrics,
            "per_student": [{"student_id": student, "samples": len(by_student[student]),
                             "delta_metrics": differences[pair][i]} for i, student in enumerate(students)]})
    return {"samples": len(sample_ids), "students": len(students),
            "methods": method_reports, "comparisons": comparisons}


def analyze_evaluations(
    evaluations_by_method, *, sample_students, protocol, expected_samples=70,
    expected_students=17, reference_method="base", comparisons=None, bootstrap=2000,
    seed=20261002, method_kinds=None, current_codes=None, predictions_by_method=None,
):
    """Compare complete per-sample static evaluations on a declared cohort.

    ``evaluations_by_method`` maps method names to iterables (or sample-indexed
    mappings) of ``sample_id/student_id/has_feedback/metrics`` records. Optional
    ``current_codes`` and ``predictions_by_method`` together add independently
    measured predicted edit size from text. They never infer true edit size from
    absolute errors. Missing methods, samples or metrics raise ``ValueError``.
    """
    _parameters(protocol, expected_samples, expected_students, bootstrap, seed)
    requested_protocol = protocol
    protocol = PROTOCOL_ALIASES.get(protocol, protocol)
    cohort = _cohort(sample_students, expected_samples, expected_students)
    methods, kinds = _methods(evaluations_by_method, method_kinds)
    pairs = _comparisons(comparisons, methods, reference_method)
    records, metric_names = _validate_evaluations(evaluations_by_method, cohort, methods)
    _require((current_codes is None) == (predictions_by_method is None),
             "current_codes and predictions_by_method must be supplied together")
    if current_codes is not None:
        records = _with_edit_sizes(records, cohort, current_codes, predictions_by_method)
        records, metric_names = _validate_evaluations(records, cohort, methods)
    ids = sorted(cohort)
    first = records[methods[0]]
    groups = {
        "true_changed": [sample for sample in ids if first[sample]["metrics"]["true_changed"] == 1],
        "true_unchanged": [sample for sample in ids if first[sample]["metrics"]["true_changed"] == 0],
        "actual_feedback": [sample for sample in ids if first[sample]["has_feedback"]],
    }
    report = {
        "schema_version": SCHEMA_VERSION, "protocol": protocol,
        "requested_protocol": requested_protocol,
        "protocol_alias_used": requested_protocol != protocol,
        "analysis_kind": ("legacy_format_preserved_diagnostic" if protocol == "legacy_preserve_diagnostic"
                          else "cached_python_fence_v1_rescored_static_comparison"),
        "evaluation_kind": "static_observed_next_revision_no_execution",
        "cohort_complete": True, "samples_dropped": 0,
        "expected_samples": expected_samples, "expected_students": expected_students,
        "sample_ids_sha256": hashlib.sha256(json.dumps(ids, ensure_ascii=False,
                                                     separators=(",", ":")).encode("utf-8")).hexdigest(),
        "seed": seed, "bootstrap_repetitions": bootstrap,
        "bootstrap_unit": "student; all methods share each resampled student draw",
        "estimand": "equal-student mean of within-student paired submission differences",
        "confidence_interval": "descriptive percentile 95%; no significance decision",
        "multiple_comparison_adjustment": "none; comparisons are descriptive and explicitly declared",
        "tie_tolerance": TIE_TOLERANCE, "reference_method": reference_method,
        "declared_comparisons": [{"method": method, "reference": reference} for method, reference in pairs],
        "method_kinds": kinds,
        "available_metrics": metric_names,
        "unavailable_edit_size_metrics": sorted(OPTIONAL_METRICS - set(metric_names)),
        "all": _summarize_group(records, ids, metric_names, pairs, bootstrap, seed),
        "subgroups": {name: _summarize_group(records, selected, metric_names, pairs, bootstrap, seed)
                      for name, selected in groups.items()},
        "limitations": [
            "Shared-pool base is likelihood reranking, not original greedy generation.",
            "Sequence rescoring is not online token contrastive decoding.",
            "Static text edits do not measure program correctness, progress groups or feedback uptake.",
            "One observed next submission is not the student's full behavior distribution.",
            "Small paired differences or confidence intervals do not establish method effectiveness.",
            "True-change subgroups use evaluation labels only and cannot choose candidates or tune extraction.",
            "No multiple-comparison correction or significance claim is made.",
        ],
    }
    if protocol == "legacy_preserve_diagnostic":
        report["limitations"].append("This is the preserved legacy score protocol, not newly extracted/rescored model evidence.")
    return report


def analyze_predictions(
    inputs, labels, predictions_by_method, *, protocol, expected_samples=70,
    expected_students=17, reference_method="base", comparisons=None, bootstrap=2000,
    seed=20261002, method_kinds=None,
):
    """Compute existing static edit metrics and paired statistics from text.

    Inputs contain current_code/student_id/feedback, labels contain target_code,
    and every named prediction contains predicted_code and its matching method.
    Code strings are compared with difflib only; they are never imported or run.
    """
    _parameters(protocol, expected_samples, expected_students, bootstrap, seed)
    x, y = _records(inputs, "inputs"), _records(labels, "labels")
    cohort = _cohort({sample: row.get("student_id") for sample, row in x.items()},
                     expected_samples, expected_students)
    _same_ids(y, cohort, "labels")
    methods, _ = _methods(predictions_by_method, method_kinds)
    predictions = _prediction_records(predictions_by_method, methods, cohort)
    evaluations = {method: [] for method in methods}
    for sample in sorted(cohort):
        _require("target_code" not in x[sample] and "target_results" not in x[sample],
                 "Future labels must not be embedded in inputs")
        before, target = x[sample].get("current_code"), y[sample].get("target_code")
        _require(isinstance(before, str) and isinstance(target, str), "Current/target code must be text")
        _require(isinstance(x[sample].get("feedback"), list), "Inputs must declare a feedback list")
        if "student_id" in y[sample]:
            _require(y[sample]["student_id"] == cohort[sample], "Label student differs from input student")
        true_size = edit_events(before, target)[2]
        for method in methods:
            predicted = predictions[method][sample]["predicted_code"]
            metrics = pair_metrics(before, predicted, target)
            metrics["predicted_edit_size"] = edit_events(before, predicted)[2]
            metrics["true_edit_size"] = true_size
            evaluations[method].append({"sample_id": sample, "student_id": cohort[sample],
                                        "has_feedback": bool(x[sample]["feedback"]), "metrics": metrics})
    return analyze_evaluations(evaluations, sample_students=cohort, protocol=protocol,
        expected_samples=expected_samples, expected_students=expected_students,
        reference_method=reference_method, comparisons=comparisons, bootstrap=bootstrap,
        seed=seed, method_kinds=method_kinds)
