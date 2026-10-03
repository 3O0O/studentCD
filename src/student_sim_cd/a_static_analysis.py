"""Distribution evaluation for A's text-change prototype; never execute code.

This is NOT execution-based progress calibration. q is supplied by a separately
frozen train-only predictor. Candidate scores, extraction and weights are reused.
The returned evidence contains IDs, hashes and numbers, never program text.
"""

from collections import Counter, defaultdict
import hashlib
import math
import random
import statistics

from .evaluate import edit_events, pair_metrics
from .scoring import calibrate_groups, softmax

METHODS = ("base", "cd", "b", "history_d0")
GROUPS = ("unchanged", "changed")
SCHEMA = "student-sim-cd.a-static-analysis.v1"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _index(rows):
    result = {}
    for row in rows:
        sid = row["sample_id"]
        _require(isinstance(sid, str) and sid and sid not in result, "duplicate/invalid sample ID")
        result[sid] = row
    _require(result, "empty cohort")
    return result


def _mean(values):
    return statistics.mean(values) if values else None


def macro(rows, key, subset=None):
    students = defaultdict(list)
    for row in rows:
        if subset is None or bool(row["true_changed"]) == subset:
            value = row[key]
            if value is not None:
                students[row["student_id"]].append(value)
    means = {sid: statistics.mean(values) for sid, values in sorted(students.items())}
    return {"student_macro_mean": _mean(list(means.values())),
            "submission_mean": _mean([row[key] for row in rows
                                      if row[key] is not None and
                                      (subset is None or bool(row["true_changed"]) == subset)]),
            "students": len(means), "per_student": means}


def paired(rows_a, rows_b, key, repetitions=2000, seed=20260929, subset=None):
    a, b = macro(rows_a, key, subset)["per_student"], macro(rows_b, key, subset)["per_student"]
    _require(set(a) == set(b), "paired student coverage mismatch")
    differences = [a[sid] - b[sid] for sid in sorted(a)]
    if not differences:
        return {"students": 0, "difference": None, "student_paired_95_ci": None}
    rng = random.Random(seed)
    draws = sorted(statistics.mean(differences[rng.randrange(len(differences))]
                                  for _ in differences) for _ in range(repetitions))
    return {"students": len(a), "difference": statistics.mean(differences),
            "student_paired_95_ci": [draws[int(.025 * repetitions)],
                                     draws[min(repetitions - 1, int(.975 * repetitions))]],
            "per_student_difference": {sid: a[sid] - b[sid] for sid in sorted(a)}}


def crps(values, probabilities, observed):
    """Exact CRPS for a finite one-dimensional edit-size distribution."""
    _require(len(values) == len(probabilities) and values, "CRPS alignment")
    _require(all(math.isfinite(p) and p >= 0 for p in probabilities)
             and math.isclose(math.fsum(probabilities), 1, abs_tol=1e-10), "CRPS probability mass")
    return (math.fsum(p * abs(value - observed) for value, p in zip(values, probabilities))
            - .5 * math.fsum(pi * pj * abs(vi - vj)
                            for vi, pi in zip(values, probabilities)
                            for vj, pj in zip(values, probabilities)))


def wasserstein_1(predicted, actual):
    """Each argument is (scalar, weight); unequal weighted supports are valid."""
    a, b = defaultdict(float), defaultdict(float)
    for value, weight in predicted:
        a[value] += weight
    for value, weight in actual:
        b[value] += weight
    _require(math.isclose(math.fsum(a.values()), 1, abs_tol=1e-9)
             and math.isclose(math.fsum(b.values()), 1, abs_tol=1e-9), "W1 masses must sum to one")
    support, balance, result = sorted(set(a) | set(b)), 0., 0.
    for index, value in enumerate(support[:-1]):
        balance += a[value] - b[value]
        result += abs(balance) * (support[index + 1] - value)
    return result


def _probability_metrics(current, target, candidates, probabilities, cached=None):
    if cached is None:
        cached = ([pair_metrics(current, code, target) for code in candidates],
                  [edit_events(current, code) for code in candidates],
                  edit_events(current, target))
    measured, events, (actual_locations, _, actual_size) = cached
    universe = [("line", i) for i in range(len(current.splitlines(keepends=True)))]
    universe += [("boundary", i) for i in range(len(current.splitlines(keepends=True)) + 1)]
    change_probability = math.fsum(p for code, p in zip(candidates, probabilities) if code != current)
    truth = float(current != target)
    floor = 1e-12  # Numerical reporting floor only; distribution is not altered.
    loss = -math.log(max(floor, change_probability if truth else 1 - change_probability))
    result = {"expected_" + key: math.fsum(p * row[key] for row, p in zip(measured, probabilities))
              for key in measured[0] if key not in {"true_changed", "predicted_changed"}}
    result.update({"change_probability": change_probability,
                   "change_brier": (change_probability - truth) ** 2,
                   "change_log_loss_floor_1e12": loss,
                   "log_loss_floor_used": float((change_probability if truth else 1 - change_probability) < floor),
                   "change_decision_error": float((change_probability >= .5) != bool(truth)),
                   "expected_edit_size": math.fsum(p * event[2] for p, event in zip(probabilities, events)),
                   "true_edit_size": actual_size,
                   "edit_size_crps": crps([event[2] for event in events], probabilities, actual_size),
                   "location_event_brier": statistics.mean(
                       (math.fsum(p for p, event in zip(probabilities, events) if loc in event[0])
                        - float(loc in actual_locations)) ** 2 for loc in universe),
                   "false_edit_probability": change_probability if not truth else None,
                   "missed_edit_probability": 1 - change_probability if truth else None})
    return result, measured, events


def analyze_pool(input_rows, label_rows, pools, scores, q_rows, pool_name,
                 bootstrap=2000, seed=20260929):
    """Evaluate full distributions and separately labeled argmax diagnostics.

    All choices depend only on inputs, candidates, scores and frozen q. Labels
    are used exclusively for metrics and explicitly named oracle diagnostics.
    """
    inputs, labels, q = _index(input_rows), _index(label_rows), _index(q_rows)
    cohort = set(inputs)
    _require(cohort == set(labels) == set(q) == set(pools), "complete cohort mismatch")
    _require(type(bootstrap) is int and bootstrap > 0, "positive bootstrap count required")
    expected_keys = {(sid, cid) for sid, pool in pools.items() for cid in pool}
    _require(set(scores) == expected_keys, "complete score set required")
    rows_by_method = {name: [] for method in METHODS for name in (method, "a_" + method)}
    distribution_rows, coverage = [], []
    for sid in sorted(cohort):
        current, target = inputs[sid]["current_code"], labels[sid]["target_code"]
        _require(isinstance(current, str) and isinstance(target, str), "code must remain text")
        student = inputs[sid]["student_id"]
        _require(q[sid].get("student_id", student) == student, "q student mismatch")
        probability = q[sid]["p_changed"]
        _require(type(probability) in (int, float) and math.isfinite(probability)
                 and 0 < probability < 1, "frozen q must have two positive finite masses")
        ids = sorted(pools[sid])
        codes = [pools[sid][cid]["code"] for cid in ids]
        _require(ids and len(codes) == len(set(codes)), "unique nonempty pool required")
        _require(all(cid == hashlib.sha256(code.encode("utf-8")).hexdigest()
                     for cid, code in zip(ids, codes)), "candidate body hash mismatch")
        groups = ["unchanged" if code == current else "changed" for code in codes]
        counts = Counter(groups)
        # Missing positive-q groups MUST raise, rather than dropping a query or renormalizing.
        masses = {"unchanged": 1 - probability, "changed": probability}
        coverage.append({"sample_id": sid, "group_counts": {g: counts[g] for g in GROUPS},
                         "missing_q_mass": math.fsum(masses[g] for g in GROUPS if not counts[g]),
                         "copy_singleton": counts["unchanged"] == 1})
        truth_group = "changed" if current != target else "unchanged"
        cached_metrics = ([pair_metrics(current, code, target) for code in codes],
                          [edit_events(current, code) for code in codes],
                          edit_events(current, target))
        for method in METHODS:
            values = [scores[sid, cid][method] for cid in ids]
            distributions = ((method, softmax(values)),
                             ("a_" + method, calibrate_groups(values, groups, masses)))
            for name, ps in distributions:
                measured, candidate_metrics, events = _probability_metrics(current, target, codes, ps, cached_metrics)
                selected = min(range(len(ids)), key=lambda i: (-ps[i], ids[i]))
                inside = [i for i, group in enumerate(groups) if group == truth_group]
                conditional = softmax([values[i] for i in inside]) if inside else []
                conditional_f1 = (math.fsum(p * candidate_metrics[i]["edit_location_f1"]
                                            for p, i in zip(conditional, inside)) if inside else None)
                row = {"sample_id": sid, "student_id": student,
                       "true_changed": float(current != target), **measured,
                       "argmax_candidate_id": ids[selected],
                       "argmax_changed": float(codes[selected] != current),
                       "argmax_edit_location_f1": candidate_metrics[selected]["edit_location_f1"],
                       "observed_group_conditional_f1_DIAGNOSTIC": conditional_f1,
                       "oracle_f1_DIAGNOSTIC": max(m["edit_location_f1"] for m in candidate_metrics)}
                rows_by_method[name].append(row)
                distribution_rows.append({"sample_id": sid, "student_id": student, "method": name,
                                          "q_changed": probability, "candidate_ids": ids, "groups": groups,
                                          "probabilities": ps, "candidate_edit_sizes": [e[2] for e in events],
                                          "candidate_log_scores": values,
                                          "candidate_edit_locations": [sorted(e[0]) for e in events],
                                          "true_edit_locations": sorted(cached_metrics[2][0]),
                                          "current_line_count": len(current.splitlines(keepends=True)),
                                          "candidate_metrics": candidate_metrics})
                if name.startswith("a_"):
                    _require(math.isclose(measured["change_probability"], probability,
                                          rel_tol=0, abs_tol=1e-10), "A group margin differs from q")
    students = {row["student_id"] for row in input_rows}
    numeric_keys = [key for key, value in rows_by_method["base"][0].items()
                    if key not in {"sample_id", "student_id", "true_changed", "argmax_candidate_id"}
                    and (value is None or type(value) in (int, float))]
    summary = {}
    counts_per_student = Counter(row["student_id"] for row in input_rows)
    for name, rows in rows_by_method.items():
        summary[name] = {"all": {key: macro(rows, key) for key in numeric_keys},
                         "true_changed": {key: macro(rows, key, True) for key in numeric_keys},
                         "true_unchanged": {key: macro(rows, key, False) for key in numeric_keys}}
        predicted, actual = [], []
        for distribution in (row for row in distribution_rows if row["method"] == name):
            sid, student = distribution["sample_id"], distribution["student_id"]
            weight = 1 / (len(students) * counts_per_student[student])
            predicted.extend((size, weight * p) for size, p in zip(
                distribution["candidate_edit_sizes"], distribution["probabilities"]))
            actual.append((edit_events(inputs[sid]["current_code"], labels[sid]["target_code"])[2], weight))
        summary[name]["student_weighted_edit_size_wasserstein"] = wasserstein_1(predicted, actual)
    comparisons = {}
    for first, second in (("a_cd", "a_base"), ("a_b", "a_base"),
                          ("a_base", "base"), ("a_history_d0", "a_base")):
        comparisons[first + "_minus_" + second] = {
            key: paired(rows_by_method[first], rows_by_method[second], key, bootstrap, seed)
            for key in ("expected_edit_location_f1", "edit_size_crps", "change_brier")}
        comparisons[first + "_minus_" + second]["changed_conditional_f1_DIAGNOSTIC"] = paired(
            rows_by_method[first], rows_by_method[second], "observed_group_conditional_f1_DIAGNOSTIC",
            bootstrap, seed, True)
    return {"schema_version": SCHEMA, "pool": pool_name, "samples": len(inputs),
            "students": len(students), "candidates": len(scores),
            "grouping": "exact text changed/unchanged; not executable progress",
            "score_temperature": 1, "new_generation": False, "student_execution": False,
            "bootstrap": {"unit": "student", "repetitions": bootstrap, "seed": seed,
                          "kind": "seen-dev descriptive"},
            "q_shared_across_methods": True,
            "primary_comparison": "a_cd_minus_a_base expected_edit_location_f1",
            "coverage": coverage, "summary": summary, "comparisons": comparisons,
            "per_sample": rows_by_method, "distributions": distribution_rows,
            "limitations": ["Group margins equal q mathematically; predictive calibration is separately measured.",
                            "Text changes are not correctness, progress or learning outcomes.",
                            "Expected F1 is a paired behavior proxy, not a proper score of whole-code distributions.",
                            "Observed-group conditional metrics and oracle are label diagnostics, not inference methods.",
                            "Argmax frequencies need not equal q; all probabilities are evaluated before argmax.",
                            "Location-event Brier averages original lines and insertion boundaries; long files dilute sparse events.",
                            "Whole-code NLL is undefined/infinite for targets absent from the finite pool.",
                            "No test split, grader, model loading or student program execution is required."]}
