"""Train-only, student-held-out baselines for exact next-submission change.

This is a cheap observable-feature q baseline, not a semantic knowledge tracer.
Only training labels are read; prediction never takes labels. Source text is
measured as data and is never executed or included in models or predictions.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import math
import random
import re
from typing import Any

from .inference import validate_input

SCHEMA = "student-sim-cd.change-q.v1"
DATA_SCHEMA = "student-sim-cd.progfeed.v1"
SEED = 20261003
FOLDS = 5
L2 = 1.0
METHODS = ("overall_frequency", "task_frequency", "current_numeric", "history_numeric", "feedback_numeric")
DEFAULT_METHOD = "feedback_numeric"
CURRENT_FEATURES = (
    "current_code_log_chars", "current_code_log_lines", "current_result_count_log1p",
    "current_logged_score_fraction", "current_known_result_coverage",
)
HISTORY_FEATURES = CURRENT_FEATURES + (
    "history_count_log1p", "history_last_code_log_chars", "history_mean_code_log_chars",
    "past_transition_exact_change_fraction", "last_to_current_exact_changed",
    "history_last_logged_score_fraction", "history_mean_logged_score_fraction",
)
FEEDBACK_FEATURES = HISTORY_FEATURES + (
    "feedback_nonempty_count_log1p", "feedback_total_chars_log1p",
    "feedback_mean_chars_log1p", "feedback_present",
)
FEATURES = {"current_numeric": CURRENT_FEATURES, "history_numeric": HISTORY_FEATURES,
            "feedback_numeric": FEEDBACK_FEATURES}
LABEL_FIELDS = {"schema_version", "sample_id", "target_timestamp", "target_code", "target_results"}
HASH = re.compile(r"[0-9a-f]{64}\Z")


def _hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric")
    return float(value)


def _future_keys(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError("input object keys must be strings")
            normalized = key.lower().replace("-", "_")
            if normalized.startswith(("target", "future", "next", "label")) or any(
                    part in {"target", "future", "next", "label", "labels"}
                    for part in normalized.split("_")):
                raise ValueError(f"forbidden future/label input field: {key}")
            _future_keys(child)
    elif isinstance(value, list):
        for child in value:
            _future_keys(child)


def _date(value: Any) -> datetime:
    if not isinstance(value, str):
        raise ValueError("ProgFeed timestamp required")
    try:
        return datetime.strptime(value, "%Y-%m-%d-%H-%M-%S")
    except ValueError as exc:
        raise ValueError("timestamp must use canonical ProgFeed YYYY-MM-DD-HH-MM-SS") from exc


def _inputs(rows: list[dict]) -> list[dict]:
    if not isinstance(rows, list) or not rows:
        raise ValueError("nonempty input list required")
    seen = set()
    for row in rows:
        _future_keys(row)
        validate_input(row, require_problem=False)
        if row.get("schema_version") != DATA_SCHEMA:
            raise ValueError("input data schema mismatch")
        current = _date(row.get("current_timestamp"))
        times = [_date(item["timestamp"]) for item in row["history"]]
        if any(a >= b for a, b in zip(times, times[1:] + [current])):
            raise ValueError("history must strictly precede current and retain chronology")
        if row["sample_id"] in seen:
            raise ValueError("duplicate input sample_id")
        seen.add(row["sample_id"])
    return sorted(rows, key=lambda row: row["sample_id"])


def _targets(rows: list[dict], labels: list[dict]) -> dict[str, int]:
    if not isinstance(labels, list):
        raise ValueError("training labels must be a list")
    indexed = {}
    inputs = {row["sample_id"]: row for row in rows}
    for row in labels:
        if not isinstance(row, dict) or set(row) - LABEL_FIELDS:
            raise ValueError("unknown training label fields")
        if row.get("schema_version") != DATA_SCHEMA or not isinstance(row.get("target_code"), str):
            raise ValueError("training label schema/target_code mismatch")
        sid = row.get("sample_id")
        if sid not in inputs or sid in indexed:
            raise ValueError("unknown or duplicate training label sample_id")
        if _date(row.get("target_timestamp")) <= _date(inputs[sid]["current_timestamp"]):
            raise ValueError("target timestamp must follow current (audit only)")
        if not isinstance(row.get("target_results"), list):
            raise ValueError("target_results must be a list; never a feature")
        _hash(row)  # reject nonfinite/non-JSON labels, without using target results
        indexed[sid] = int(row["target_code"] != inputs[sid]["current_code"])
    if set(indexed) != set(inputs):
        raise ValueError("training input/label sample_id sets must match exactly")
    if set(indexed.values()) != {0, 1}:
        raise ValueError("training cohort must contain both changed and unchanged labels")
    return indexed


def _source_hashes(source: dict | None) -> dict:
    if source is None:
        return {}
    if not isinstance(source, dict) or any(not isinstance(k, str) or not k or
                                         not isinstance(v, str) or not HASH.fullmatch(v)
                                         for k, v in source.items()):
        raise ValueError("source_sha256 must map source names to lowercase SHA256")
    return dict(sorted(source.items()))


def _performance(results: list[dict]) -> tuple[float | None, float | None]:
    """Logged weighted score fraction; neither testcase pass rate nor correctness."""
    unique: dict[tuple, tuple] = {}
    conflict = False
    for item in results:
        key = (item.get("test_name"), item.get("function_name"))
        pair = (item.get("score"), item.get("max_score"))
        if key in unique and unique[key] != pair:
            conflict = True
        unique[key] = pair
    valid = [(float(s), float(m)) for s, m in unique.values()
             if isinstance(s, (int, float)) and not isinstance(s, bool)
             and isinstance(m, (int, float)) and not isinstance(m, bool)
             and math.isfinite(s) and math.isfinite(m) and m > 0 and 0 <= s <= m]
    coverage = len(valid) / len(unique) if unique else None
    if conflict or not valid:
        return None, coverage
    # Rescale before summation: several finite 1e308 maxima must not overflow.
    scale = max(m for _, m in valid)
    fraction = math.fsum(s / scale for s, _ in valid) / math.fsum(m / scale for _, m in valid)
    return fraction, coverage


def observable_features(row: dict) -> dict[str, float | None]:
    """Only current/past code sizes, logged scores, and delivered text sizes."""
    code, history = row["current_code"], row["history"]
    fraction, coverage = _performance(row["current_results"])
    old_fractions = [_performance(item["results"])[0] for item in history]
    known = [p for p in old_fractions if p is not None]
    texts = [item["text"] for item in row["feedback"] if item["text"].strip()]
    return {
        "current_code_log_chars": math.log1p(len(code)),
        "current_code_log_lines": math.log1p(len(code.splitlines())),
        "current_result_count_log1p": math.log1p(len(row["current_results"])),
        "current_logged_score_fraction": fraction,
        "current_known_result_coverage": coverage,
        "history_count_log1p": math.log1p(len(history)),
        "history_last_code_log_chars": math.log1p(len(history[-1]["code"])) if history else None,
        "history_mean_code_log_chars": math.fsum(math.log1p(len(h["code"])) for h in history) / len(history) if history else None,
        "past_transition_exact_change_fraction": sum(a["code"] != b["code"] for a, b in zip(history, history[1:])) / (len(history)-1) if len(history) > 1 else None,
        "last_to_current_exact_changed": float(history[-1]["code"] != code) if history else None,
        "history_last_logged_score_fraction": old_fractions[-1] if history else None,
        "history_mean_logged_score_fraction": math.fsum(known) / len(known) if known else None,
        "feedback_nonempty_count_log1p": math.log1p(len(texts)),
        "feedback_total_chars_log1p": math.log1p(sum(len(text) for text in texts)),
        "feedback_mean_chars_log1p": math.log1p(sum(len(text) for text in texts) / len(texts)) if texts else 0.0,
        "feedback_present": float(bool(texts)),
    }


def _weights(rows: list[dict]) -> list[float]:
    counts = Counter(row["student_id"] for row in rows)
    return [1.0 / counts[row["student_id"]] for row in rows]


def _scaler(features: list[dict], names: tuple, weights: list[float]) -> dict:
    means, scales, counts = [], [], []
    for name in names:
        observed = [(row[name], w) for row, w in zip(features, weights) if row[name] is not None]
        total = math.fsum(w for _, w in observed)
        mean = math.fsum(x*w for x, w in observed) / total if observed else 0.0
        variance = math.fsum(w * (x-mean)**2 for x, w in observed) / total if observed else 0.0
        scale = math.sqrt(variance)
        means.append(mean)
        scales.append(scale if scale >= 1e-12 else 1.0)
        counts.append(len(observed))
    return {"mean": means, "scale": scales, "observed_count": counts,
            "missing_imputation": "fold_training_observed_weighted_mean",
            "constant_scale": 1.0}


def _design(features: list[dict], names: tuple, scaler: dict) -> list[list[float]]:
    return [[1.0] + [(row[name]-mean)/scale if row[name] is not None else 0.0
                    for name, mean, scale in zip(names, scaler["mean"], scaler["scale"])]
            + [float(row[name] is None) for name in names] for row in features]


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _softplus(z: float) -> float:
    return max(z, 0.0) + math.log1p(math.exp(-abs(z)))


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float]:
    n = len(rhs)
    a = [list(row) + [value] for row, value in zip(matrix, rhs)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda i: abs(a[i][col]))
        if abs(a[pivot][col]) < 1e-14:
            raise ValueError("singular logistic Hessian")
        a[col], a[pivot] = a[pivot], a[col]
        diagonal = a[col][col]
        a[col] = [v/diagonal for v in a[col]]
        for i in range(n):
            if i == col:
                continue
            factor = a[i][col]
            a[i] = [left-factor*right for left, right in zip(a[i], a[col])]
    return [row[-1] for row in a]


def _logistic(x: list[list[float]], y: list[int], weights: list[float], prior: float) -> tuple[list[float], dict]:
    """Student-unit weighted log loss + (L2/2)*||nonintercept beta||²."""
    d = len(x[0])
    beta = [math.log(prior/(1-prior))] + [0.0]*(d-1)

    def objective(b):
        return math.fsum(w*(_softplus(math.fsum(v*t for v, t in zip(row, b)))
                            - target*math.fsum(v*t for v, t in zip(row, b)))
                         for row, target, w in zip(x, y, weights)) + L2/2*math.fsum(v*v for v in b[1:])

    converged = False
    for iteration in range(1, 61):
        grad = [0.0]*d
        hessian = [[0.0]*d for _ in range(d)]
        for row, target, weight in zip(x, y, weights):
            p = _sigmoid(math.fsum(v*t for v, t in zip(row, beta)))
            error, curvature = weight*(p-target), weight*p*(1-p)
            for j in range(d):
                grad[j] += error*row[j]
                for k in range(j+1):
                    hessian[j][k] += curvature*row[j]*row[k]
        for j in range(d):
            for k in range(j):
                hessian[k][j] = hessian[j][k]
            if j:
                grad[j] += L2*beta[j]
                hessian[j][j] += L2
        if max(abs(v) for v in grad) <= 1e-8:
            converged = True
            break
        step = _solve(hessian, grad)
        old = objective(beta)
        factor = 1.0
        while factor >= 2**-24:
            candidate = [b-factor*s for b, s in zip(beta, step)]
            if objective(candidate) <= old:
                beta = candidate
                break
            factor /= 2
        else:
            raise ValueError("logistic line search did not decrease fixed objective")
    if not converged:
        raise ValueError("fixed logistic solver did not converge; no silent fit acceptance")
    return beta, {"algorithm": "ridge_newton_backtracking", "converged": True,
                  "iterations": iteration, "gradient_max_abs": max(abs(v) for v in grad),
                  "objective": objective(beta), "gradient_tolerance": 1e-8}


def _fit(rows: list[dict], targets: dict, sources: dict, split: str) -> dict:
    labels = [targets[row["sample_id"]] for row in rows]
    if set(labels) != {0, 1}:
        raise ValueError("every training fold must have both classes; no label-based refolding")
    students = sorted({row["student_id"] for row in rows})
    weights = _weights(rows)
    prevalence = (math.fsum(w*y for w, y in zip(weights, labels)) + 0.5) / (len(students)+1)
    task_mass = defaultdict(lambda: [0.0, 0.0])
    for row, target, weight in zip(rows, labels, weights):
        key = json.dumps([row["lab"], row["source_file"]], separators=(",", ":"), ensure_ascii=False)
        task_mass[key][0] += weight*target
        task_mass[key][1] += weight
    methods = {
        "overall_frequency": {"kind": "overall_frequency", "p_changed": prevalence,
                              "changed_student_mass": math.fsum(w*y for w, y in zip(weights, labels)),
                              "student_mass": float(len(students)), "beta_prior": [0.5, 0.5]},
        "task_frequency": {"kind": "task_frequency", "fallback_p_changed": prevalence,
                           "prior_strength_student_units": 5.0,
                           "tasks": {key: {"changed_student_mass": mass[0], "student_mass": mass[1],
                                           "p_changed": (mass[0]+5*prevalence)/(mass[1]+5)}
                                     for key, mass in sorted(task_mass.items())}},
    }
    features = [observable_features(row) for row in rows]
    for method, names in FEATURES.items():
        scaler = _scaler(features, names, weights)
        beta, solver = _logistic(_design(features, names, scaler), labels, weights, prevalence)
        methods[method] = {"kind": "logistic", "feature_names": list(names),
                           "design_names": ["intercept"] + list(names) + [name+"__missing" for name in names],
                           "scaler": scaler, "coefficients": beta, "l2": L2, "solver": solver}
    return {"schema_version": SCHEMA, "fit_split": split, "default_method": DEFAULT_METHOD,
            "l2": L2, "training_student_ids": students, "training_sample_count": len(rows),
            "feature_protocol": "observable-numeric-strictpast-v1", "source_sha256": sources,
            "canonical_training_inputs_sha256": _hash(rows),
            "canonical_training_changed_labels_sha256": _hash([{ "sample_id": r["sample_id"],
                                                                 "changed": targets[r["sample_id"]]} for r in rows]),
            "weighting": "each_training_student_total_weight_1",
            "objective": "sum_i (1/n_student_i)*logloss_i + 0.5*L2*sum_nonintercept_beta_squared",
            "methods": methods}


def _validate_model(model: dict) -> None:
    allowed = {"schema_version", "fit_split", "default_method", "l2", "training_student_ids", "training_sample_count",
               "feature_protocol", "source_sha256", "canonical_training_inputs_sha256", "canonical_training_changed_labels_sha256",
               "weighting", "objective", "methods"}
    if not isinstance(model, dict) or set(model) != allowed or model["schema_version"] != SCHEMA:
        raise ValueError("unsafe or unknown model schema")
    if model["fit_split"] != "train" or _finite(model["l2"], "l2") != L2 or model["default_method"] != DEFAULT_METHOD:
        raise ValueError("model fixed train-only protocol mismatch")
    if model["feature_protocol"] != "observable-numeric-strictpast-v1" or model["weighting"] != "each_training_student_total_weight_1":
        raise ValueError("model feature/weighting protocol mismatch")
    if model["objective"] != "sum_i (1/n_student_i)*logloss_i + 0.5*L2*sum_nonintercept_beta_squared":
        raise ValueError("model fixed objective mismatch")
    students = model["training_student_ids"]
    if not isinstance(students, list) or not students or any(not isinstance(s, str) or not s for s in students) or students != sorted(set(students)):
        raise ValueError("invalid model training student IDs")
    if type(model["training_sample_count"]) is not int or model["training_sample_count"] < len(students):
        raise ValueError("invalid model training sample count")
    _source_hashes(model["source_sha256"])
    for field in ("canonical_training_inputs_sha256", "canonical_training_changed_labels_sha256"):
        if not isinstance(model[field], str) or not HASH.fullmatch(model[field]):
            raise ValueError("model source fingerprint invalid")
    if not isinstance(model["methods"], dict) or set(model["methods"]) != set(METHODS):
        raise ValueError("model methods mismatch")
    overall = model["methods"]["overall_frequency"]
    if set(overall) != {"kind", "p_changed", "changed_student_mass", "student_mass", "beta_prior"} or overall["kind"] != "overall_frequency" or overall["beta_prior"] != [0.5, 0.5]:
        raise ValueError("unsafe frequency model")
    mass = _finite(overall["student_mass"], "student_mass")
    changed = _finite(overall["changed_student_mass"], "changed_mass")
    if mass != len(students) or not 0 < changed < mass or overall["p_changed"] != (changed+0.5)/(mass+1):
        raise ValueError("frequency model inconsistent")
    task = model["methods"]["task_frequency"]
    if set(task) != {"kind", "fallback_p_changed", "prior_strength_student_units", "tasks"} or task["kind"] != "task_frequency" or task["prior_strength_student_units"] != 5.0 or task["fallback_p_changed"] != overall["p_changed"] or not isinstance(task["tasks"], dict):
        raise ValueError("unsafe task frequency model")
    for key, item in task["tasks"].items():
        try:
            decoded = json.loads(key)
        except (ValueError, TypeError) as exc:
            raise ValueError("invalid task key") from exc
        if not isinstance(decoded, list) or len(decoded) != 2 or any(not isinstance(v, str) or not v for v in decoded) or not isinstance(item, dict) or set(item) != {"changed_student_mass", "student_mass", "p_changed"}:
            raise ValueError("invalid task frequency schema")
        m = _finite(item["student_mass"], "task mass")
        c = _finite(item["changed_student_mass"], "task changed")
        if not 0 <= c <= m or not 0 < m <= mass or item["p_changed"] != (c+5*overall["p_changed"])/(m+5):
            raise ValueError("task frequency inconsistent")
    for method, names in FEATURES.items():
        item = model["methods"][method]
        if not isinstance(item, dict) or set(item) != {"kind", "feature_names", "design_names", "scaler", "coefficients", "l2", "solver"} or item["kind"] != "logistic" or _finite(item["l2"], "l2") != L2 or item["feature_names"] != list(names) or item["design_names"] != ["intercept"]+list(names)+[n+"__missing" for n in names]:
            raise ValueError("unsafe logistic feature schema")
        scaler = item["scaler"]
        if not isinstance(scaler, dict) or set(scaler) != {"mean", "scale", "observed_count", "missing_imputation", "constant_scale"} or scaler["missing_imputation"] != "fold_training_observed_weighted_mean" or _finite(scaler["constant_scale"], "constant_scale") != 1.0:
            raise ValueError("unsafe scaler schema")
        for field in ("mean", "scale", "observed_count"):
            if not isinstance(scaler[field], list) or len(scaler[field]) != len(names):
                raise ValueError("scaler dimension mismatch")
        for mean, scale, count in zip(scaler["mean"], scaler["scale"], scaler["observed_count"]):
            _finite(mean, "scaler mean")
            if _finite(scale, "scaler scale") <= 0 or type(count) is not int or not 0 <= count <= model["training_sample_count"]:
                raise ValueError("invalid scaler scale/count")
        if not isinstance(item["coefficients"], list) or len(item["coefficients"]) != 1+2*len(names):
            raise ValueError("coefficient dimension mismatch")
        for v in item["coefficients"]:
            _finite(v, "coefficient")
        solver = item["solver"]
        if not isinstance(solver, dict) or set(solver) != {"algorithm", "converged", "iterations", "gradient_max_abs", "objective", "gradient_tolerance"} or solver["algorithm"] != "ridge_newton_backtracking" or solver["converged"] is not True or type(solver["iterations"]) is not int or not 1 <= solver["iterations"] <= 60 or solver["gradient_tolerance"] != 1e-8:
            raise ValueError("unsafe solver audit schema")
        if not 0 <= _finite(solver["gradient_max_abs"], "gradient") <= 1e-8:
            raise ValueError("unconverged model")
        _finite(solver["objective"], "objective")
    _hash(model)


def _probabilities(model: dict, rows: list[dict], method: str) -> list[float]:
    if method not in METHODS:
        raise ValueError("unknown q method")
    item = model["methods"][method]
    if method == "overall_frequency":
        return [item["p_changed"]]*len(rows)
    if method == "task_frequency":
        return [item["tasks"].get(json.dumps([r["lab"], r["source_file"]], separators=(",", ":"), ensure_ascii=False), {}).get("p_changed", item["fallback_p_changed"]) for r in rows]
    design = _design([observable_features(r) for r in rows], FEATURES[method], item["scaler"])
    return [_sigmoid(math.fsum(x*b for x, b in zip(row, item["coefficients"]))) for row in design]


def predict(model: dict, inputs: list[dict], *, method: str = DEFAULT_METHOD, split: str = "holdout") -> list[dict]:
    """Label-free prediction on students disjoint from the fitted training set."""
    _validate_model(model)
    if split not in {"holdout", "dev", "test", "train"}:
        raise ValueError("unknown prediction split audit")
    rows = _inputs(inputs)
    if set(model["training_student_ids"]) & {r["student_id"] for r in rows}:
        raise ValueError("cross-student leakage: prediction student was in fit cohort")
    probabilities = _probabilities(model, rows, method)
    result = []
    for row, p in zip(rows, probabilities):
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("nonfinite q prediction")
        result.append({"sample_id": row["sample_id"], "student_id": row["student_id"], "p_changed": p})
    return result


def _auc(y: list[int], p: list[float], weights: list[float]) -> float | None:
    positive = math.fsum(w for t, w in zip(y, weights) if t)
    negative = math.fsum(w for t, w in zip(y, weights) if not t)
    if not positive or not negative:
        return None
    ties = defaultdict(lambda: [0.0, 0.0])
    for target, probability, weight in zip(y, p, weights):
        ties[probability][target] += weight
    below, concordance = 0.0, 0.0
    for probability in sorted(ties):
        neg, pos = ties[probability]
        concordance += pos*(below+0.5*neg)
        below += neg
    return concordance/(positive*negative)


def evaluate_probabilities(rows: list[dict], targets: dict[str, int], probabilities: list[float]) -> dict:
    """Static probability metrics; missing-class denominators are explicit."""
    if len(probabilities) != len(rows) or set(targets) != {r["sample_id"] for r in rows}:
        raise ValueError("probability evaluation cohort mismatch")
    if any(not 0 <= _finite(p, "p_changed") <= 1 for p in probabilities) or any(type(v) is not int or v not in {0, 1} for v in targets.values()):
        raise ValueError("invalid probabilities/changed labels")
    groups = defaultdict(list)
    for row, p in zip(rows, probabilities):
        groups[row["student_id"]].append((targets[row["sample_id"]], p))
    per_student = []
    confusion = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    weighted = {key: 0.0 for key in confusion}
    bins = [{"lower": i/5, "upper": (i+1)/5, "upper_inclusive": i == 4,
             "sample_count": 0, "student_ids": set(), "student_weight": 0.0,
             "prediction_sum": 0.0, "changed_sum": 0.0} for i in range(5)]
    for student in sorted(groups):
        pairs = groups[student]
        y, p = zip(*pairs)
        c = {key: 0 for key in confusion}
        for target, probability in pairs:
            key = ("tp" if probability >= 0.5 else "fn") if target else ("fp" if probability >= 0.5 else "tn")
            c[key] += 1
            confusion[key] += 1
            weighted[key] += 1/len(pairs)
            b = bins[min(4, int(probability*5))]
            b["sample_count"] += 1
            b["student_ids"].add(student)
            b["student_weight"] += 1/len(pairs)
            b["prediction_sum"] += probability/len(pairs)
            b["changed_sum"] += target/len(pairs)
        changed, unchanged = sum(y), len(y)-sum(y)
        clipped = [min(1-1e-12, max(1e-12, value)) for value in p]
        per_student.append({"student_id": student, "sample_count": len(pairs), "changed_count": changed,
                            "unchanged_count": unchanged,
                            "logloss": math.fsum(-t*math.log(v)-(1-t)*math.log1p(-v) for t, v in zip(y, clipped))/len(pairs),
                            "brier": math.fsum((v-t)**2 for t, v in pairs)/len(pairs),
                            "auc": _auc(list(y), list(p), [1.0]*len(y)),
                            "balanced_accuracy": 0.5*(c["tp"]/changed+c["tn"]/unchanged) if changed and unchanged else None,
                            "unchanged_false_edit": c["fp"]/unchanged if unchanged else None,
                            "changed_miss": c["fn"]/changed if changed else None, "confusion": c})
    macro = {}
    for key in ("logloss", "brier", "auc", "balanced_accuracy", "unchanged_false_edit", "changed_miss"):
        eligible = [r for r in per_student if r[key] is not None]
        macro[key] = {"value": math.fsum(r[key] for r in eligible)/len(eligible) if eligible else None,
                      "eligible_students": len(eligible), "student_coverage": len(eligible)/len(groups),
                      "eligible_samples": sum(r["sample_count"] for r in eligible)}
    reliability = []
    for b in bins:
        weight = b["student_weight"]
        reliability.append({key: b[key] for key in ("lower", "upper", "upper_inclusive", "sample_count")} | {
            "student_count": len(b["student_ids"]), "student_weight_share": weight/len(groups),
            "mean_p_changed": b["prediction_sum"]/weight if weight else None,
            "observed_changed_fraction": b["changed_sum"]/weight if weight else None})
    y = [targets[r["sample_id"]] for r in rows]
    weights = _weights(rows)
    tn, fp, fn, tp = (weighted[key] for key in ("tn", "fp", "fn", "tp"))
    return {"sample_count": len(rows), "student_count": len(groups), "changed_count": sum(y),
            "unchanged_count": len(y)-sum(y), "threshold": 0.5, "probability_log_clip": 1e-12,
            "student_macro": macro, "per_student": per_student, "submission_confusion": confusion,
            "student_weighted_confusion": weighted,
            "pooled_equal_student_auc": _auc(y, probabilities, weights),
            "pooled_equal_student_balanced_accuracy": 0.5*(tp/(tp+fn)+tn/(tn+fp)) if tp+fn and tn+fp else None,
            "submission_unchanged_false_edit": confusion["fp"]/(confusion["tn"]+confusion["fp"]) if confusion["tn"]+confusion["fp"] else None,
            "submission_changed_miss": confusion["fn"]/(confusion["tp"]+confusion["fn"]) if confusion["tp"]+confusion["fn"] else None,
            "reliability_5_bins": reliability, "reliability_sample_coverage": sum(b["sample_count"] for b in reliability)/len(rows),
            "reliability_ece_equal_student": math.fsum(b["student_weight_share"]*abs(b["mean_p_changed"]-b["observed_changed_fraction"]) for b in reliability if b["sample_count"]),
            "auc_interpretation": "student_macro_auc excludes one-class students with reported coverage; pooled_equal_student_auc is weighted cross-student discrimination, not a macro AUC"}


def run_train_experiment(inputs: list[dict], labels: list[dict], folds: int = FOLDS, seed: int = SEED,
                         *, split: str = "train", source_sha256: dict | None = None,
                         student_splits: dict | None = None) -> tuple[dict, dict, list[dict]]:
    """Five fixed outer student folds; all methods use the identical full cohort."""
    if split != "train":
        raise ValueError("q fitting is train-only; dev/test labels forbidden")
    if type(folds) is not int or folds != FOLDS or type(seed) is not int or seed != SEED:
        raise ValueError("fixed protocol requires folds=5, seed=20261003")
    rows = _inputs(inputs)
    targets = _targets(rows, labels)
    sources = _source_hashes(source_sha256)
    students = sorted({r["student_id"] for r in rows})
    if len(students) < FOLDS:
        raise ValueError("five student-held-out folds need at least five students")
    if student_splits is not None:
        if not isinstance(student_splits, dict) or any(student_splits.get(s) != "train" for s in students):
            raise ValueError("training students must belong exclusively to explicit train registry")
    shuffled = list(students)
    random.Random(seed).shuffle(shuffled)
    assignment = {s: i % folds for i, s in enumerate(shuffled)}
    oof, audits, fold_models = [], [], []
    for fold in range(folds):
        train = [r for r in rows if assignment[r["student_id"]] != fold]
        held = [r for r in rows if assignment[r["student_id"]] == fold]
        model = _fit(train, targets, sources, split)
        _validate_model(model)
        probabilities = {method: predict(model, held, method=method, split="train") for method in METHODS}
        for i, row in enumerate(held):
            values = {method: probabilities[method][i]["p_changed"] for method in METHODS}
            oof.append({"sample_id": row["sample_id"], "student_id": row["student_id"], "fold": fold,
                        "label_changed": targets[row["sample_id"]], "p_changed": values[DEFAULT_METHOD],
                        "p_changed_by_method": values})
        train_ids, held_ids = sorted({r["student_id"] for r in train}), sorted({r["student_id"] for r in held})
        if set(train_ids) & set(held_ids):
            raise ValueError("outer fold student overlap")
        audits.append({"fold": fold, "train_student_ids": train_ids, "heldout_student_ids": held_ids,
                       "train_sample_count": len(train), "heldout_sample_count": len(held),
                       "train_changed_count": sum(targets[r["sample_id"]] for r in train),
                       "heldout_changed_count": sum(targets[r["sample_id"]] for r in held),
                       "heldout_has_both_classes": {targets[r["sample_id"]] for r in held} == {0, 1},
                       "model_sha256": _hash(model)})
        fold_models.append(model)
    oof.sort(key=lambda r: r["sample_id"])
    if [r["sample_id"] for r in oof] != [r["sample_id"] for r in rows]:
        raise ValueError("incomplete OOF cohort; no dropping allowed")
    models = _fit(rows, targets, sources, split)
    _validate_model(models)
    student_classes = {student: {targets[r["sample_id"]] for r in rows if r["student_id"] == student}
                       for student in students}
    report = {"schema_version": SCHEMA, "analysis_kind": "train_only_student_heldout_change_q",
              "target_definition": "changed iff exact target_code != current_code",
              "scope": "caller audited eligible training cohort; module does not filter rows",
              "split": split, "seed": seed, "folds": folds, "l2": L2,
              "default_method": DEFAULT_METHOD, "default_method_predeclared": True,
              "sample_count": len(rows), "student_count": len(students),
              "changed_count": sum(targets.values()), "unchanged_count": len(rows)-sum(targets.values()),
              "class_coverage": {"students_with_changed": sum(1 in v for v in student_classes.values()),
                                 "students_with_unchanged": sum(0 in v for v in student_classes.values()),
                                 "students_with_both": sum(v == {0, 1} for v in student_classes.values()),
                                 "one_class_students": sum(len(v) == 1 for v in student_classes.values()),
                                 "heldout_folds_with_both": sum(a["heldout_has_both_classes"] for a in audits),
                                 "training_folds_with_both": folds,
                                 "rarity_policy": "no stratification or refolding; fail if a training fold lacks a class; evaluation denominators report exclusions"},
              "oof_sample_coverage": 1.0, "oof_student_coverage": 1.0,
              "fold_assignment": dict(sorted(assignment.items())), "fold_audit": audits,
              "fold_models": fold_models, "source_sha256": sources,
              "canonical_inputs_sha256": _hash(rows), "canonical_labels_sha256": _hash(sorted(labels, key=lambda r: r["sample_id"])),
              "oof_sha256": _hash(oof), "final_model_sha256": _hash(models),
              "metrics": {method: evaluate_probabilities(rows, targets, [r["p_changed_by_method"][method] for r in oof]) for method in METHODS},
              "limits": ["cheap numeric/text-size baselines; not semantic q or knowledge tracing",
                         "exact-code changed is a binary revision proxy, not learning gain or correctness",
                         "five outer folds use students only; no target-based refolding, no dev tuning",
                         "logged score fraction is not testcase pass rate",
                         "input schema cannot authenticate its file split; caller must audit source and student registry",
                         "one-class evaluation students remain covered in logloss/Brier and have explicit AUC/BA exclusions",
                         "fixed threshold 0.5 and descriptive five-bin reliability; no fitted calibration"]}
    _hash(report)
    return report, models, oof
