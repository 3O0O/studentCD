"""Complete-cache history interventions with paired, equal-student uncertainty.

All inputs are immutable bytes. Model scores and fixed argmax predictions are
verified before labels are parsed for static evaluation. No files are opened,
models loaded, parameters fitted, or submitted programs parsed or executed.
Permutation seeds are repeated assignments, never independent sample units.
"""

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, fields
import hashlib
import math
import random
import re
import statistics

from . import cache_rescore, history_interventions, inference, paired_analysis, predict
from .evaluate import edit_events, pair_metrics


SCHEMA_VERSION = "student-sim-cd.history-intervention-analysis.v1"
METHODS = ("base", "history_d0", "cd", "b", "copy")
VARIANTS = ("real_history_swapped", "reference_swapped")
DEFAULT_PERMUTATION_SEEDS = (20261002, 20261003, 20261004)
BASELINE_COMPLETION_SHA256 = "9bafd1d5b2dd6a8c3da22dce69c128848fcf9a56411bdb935a93d0ca0b2e9328"
BASELINE_RELEASE_SHA256 = "3cb925a67cee5f5cc00790029a7b20caa0d9405de59f506d9a67f907c91d3a53"
COMPLETION_STAGES = ("prepare", "score", "predict", "greedy", *("evaluate-" + name for name in (*METHODS, "greedy")), "compare")
HISTORY_FIELDS = {"variant", "permutation_seed", "protocol_sha256", "donor_assignment_sha256", "baseline_manifest_sha256", "history_data_sha256"}
ASSIGNMENT_FIELDS = {"schema_version", "sample_id", "query_student_id", "variant", "permutation_seed",
    "donor_sample_id", "donor_student_id", "donor_current_timestamp", "query_current_timestamp", "lab", "source_file",
    "original_reference_donor_sample_id", "original_history_sha256", "donor_history_sha256",
    "real_history_sha256", "reference_history_sha256", "target_history_length", "donor_history_length",
    "absolute_length_difference", "target_history_records", "donor_history_records", "eligible_donor_count",
    "length_measure", "matching_rule", "training_inputs_sha256", "labels_used", "history_definition"}


@dataclass(frozen=True)
class InferenceArtifact:
    """One verified cache and its five predictions, as raw file bytes.

    Baseline additionally needs its outer completion_manifest and success.
    Each intervention needs assignments (canonical JSONL bytes from construction).
    Optional real greedy is supplied separately to analyze_interventions.
    """

    inputs: bytes
    references: bytes
    manifest: bytes
    candidates: bytes
    scores: bytes
    predictions: Mapping[str, bytes]
    assignments: bytes | None = None
    completion_manifest: bytes | None = None
    success: bytes | None = None


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(content):
    return hashlib.sha256(content).hexdigest()


def _digest(value, name):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), name + " must be a SHA256")
    return value


def _artifact(value):
    if isinstance(value, Mapping):
        _require(set(value) <= {field.name for field in fields(InferenceArtifact)}, "Unexpected artifact fields")
        try:
            value = InferenceArtifact(**value)
        except TypeError as error:
            raise ValueError("Incomplete inference artifact") from error
    _require(isinstance(value, InferenceArtifact), "Use InferenceArtifact or its field mapping")
    for name in ("inputs", "references", "manifest", "candidates", "scores"):
        _require(isinstance(getattr(value, name), bytes), name + " must be original file bytes")
    _require(isinstance(value.predictions, Mapping) and set(value.predictions) == set(METHODS), "Exactly five fixed prediction methods required")
    _require(all(isinstance(content, bytes) for content in value.predictions.values()), "Predictions must be file bytes")
    return value


def _index(content, name):
    _require(isinstance(content, bytes), name + " must be bytes")
    return paired_analysis._records(predict._records(content, name), name)


def _references(content, cohort):
    rows = _index(content, "references")
    _require(set(rows) == set(cohort), "References differ from complete cohort")
    for row in rows.values():
        inference._known_fields(row, {"sample_id", "reference_history", "reference_kind", "provenance"}, "reference")
        inference.validate_history(row.get("reference_history"), "reference_history")
        _require(row.get("reference_kind") in {"matched", "shuffled", "empty"}
                 and isinstance(row.get("provenance"), dict) and row["provenance"], "Invalid reference provenance")
        _require(bool(row["reference_history"]) == (row["reference_kind"] != "empty"), "Invalid empty reference protocol")
    return rows


def _cache(value, outer_inputs, expected_candidates):
    artifact = _artifact(value)
    manifest, run_hash, pools, scores, copies, _ = predict._validated_run({
        "inputs": artifact.inputs, "manifest": artifact.manifest,
        "candidates": artifact.candidates, "scores": artifact.scores,
    })
    inputs = _index(artifact.inputs, "cache inputs")
    _require(set(inputs) == set(outer_inputs), "Cache sample set differs from complete cohort")
    references = _references(artifact.references, outer_inputs)
    _require(manifest.get("references_sha256") == _sha(artifact.references), "Reference bytes differ from score manifest")
    _require(len(scores) == expected_candidates, "Cache differs from complete fixed candidate count")
    evidence = manifest.get("cache_rescore", {})
    backend = evidence.get("backend")
    _require(backend in {"HFBackend", "injected_synthetic_backend"}, "Unknown cache scoring backend")
    _require(evidence.get("generation_performed") is False and evidence.get("old_scores_reused") is False,
             "Cache must declare no generation and no reuse of old score files")
    predictions = {}
    for method in METHODS:
        rows = _index(artifact.predictions[method], method + " predictions")
        _require(set(rows) == set(outer_inputs), "Incomplete prediction sample set for " + method)
        for sample, row in rows.items():
            _require(row.get("method") == method and isinstance(row.get("predicted_code"), str), "Prediction method/code mismatch")
            identifier = copies[sample] if method == "copy" else min(pools[sample], key=lambda candidate: (-scores[sample, candidate][method], candidate))
            _require(row["predicted_code"] == pools[sample][identifier]["code"], "Prediction differs from fixed verified argmax: " + method)
            if method == "copy":
                _require(row["predicted_code"] == outer_inputs[sample]["current_code"], "Copy differs from original current code")
        predictions[method] = rows
    candidate_records = _index(artifact.candidates, "candidate records")
    count = manifest.get("config", {}).get("num_generations")
    _require(type(count) is int and count > 0, "Invalid original generation budget")
    for row in candidate_records.values():
        attempts = row["attempts"]
        _require(isinstance(attempts, list) and len(attempts) == count,
                 "Original raw generation attempts are incomplete")
        for index, attempt in enumerate(attempts):
            _require(isinstance(attempt, dict) and type(attempt.get("attempt")) is int and attempt["attempt"] == index
                     and attempt.get("source") == ("greedy" if index == 0 else "sampled")
                     and isinstance(attempt.get("raw_text"), str) and type(attempt.get("eos_reached")) is bool,
                     "Invalid original raw generation provenance")
    return {"artifact": artifact, "manifest": manifest, "run_hash": run_hash,
            "inputs": inputs, "references": references, "pools": pools,
            "candidate_records": candidate_records, "predictions": predictions, "backend": backend}


def _baseline(cache, inputs_bytes, completion_sha256):
    artifact = cache["artifact"]
    _require(artifact.inputs == inputs_bytes, "Baseline inputs must be exact outer evaluation inputs")
    _require(isinstance(artifact.completion_manifest, bytes) and isinstance(artifact.success, bytes), "Baseline needs final completion manifest and SUCCESS bytes")
    _require(_sha(artifact.completion_manifest) == completion_sha256
             and artifact.success == (completion_sha256 + "\n").encode("ascii"), "Baseline completion hard anchor/SUCCESS mismatch")
    completion = predict._json(artifact.completion_manifest)
    stages = completion.get("stages", [])
    _require(completion.get("schema_version") == "student-sim-cd.cached-experiment.v1"
             and completion.get("status") == "success" and type(completion.get("exit_code")) is int and completion["exit_code"] == 0
             and isinstance(stages, list) and len(stages) == len(COMPLETION_STAGES)
             and [row.get("name") for row in stages] == list(COMPLETION_STAGES)
             and all(row.get("status") == "success" and type(row.get("exit_code")) is int and row["exit_code"] == 0 for row in stages)
             and completion.get("generation_performed") is False and completion.get("student_execution") is False,
             "Baseline is not the complete eleven-stage no-generation experiment")
    if cache["backend"] == "HFBackend":
        _require(completion_sha256 == BASELINE_COMPLETION_SHA256
                 and completion.get("source_root", "").endswith("/releases/" + BASELINE_RELEASE_SHA256), "Real baseline must use reviewed existing release/completion")
        _require(completion.get("implementation_sha256", {}).get("src/student_sim_cd/inference.py") == cache["manifest"].get("implementation_sha256")
                 and completion.get("implementation_sha256", {}).get("src/student_sim_cd/scoring.py") == cache["manifest"].get("scoring_sha256"),
                 "Baseline score implementation differs from reviewed completed code")
    _require("history_intervention" not in cache["manifest"], "Original matched baseline cannot be an intervention")
    _require(artifact.assignments is None, "Matched baseline has no permutation assignments")


def _assignment(assignment, original, reference, new_history, variant, permutation_seed, dev_students):
    _require(set(assignment) == ASSIGNMENT_FIELDS
             and assignment.get("schema_version") == history_interventions.ASSIGNMENT_VERSION,
             "Incomplete/unknown donor assignment schema")
    _require(assignment["variant"] == variant and type(assignment["permutation_seed"]) is int
             and assignment["permutation_seed"] == permutation_seed and assignment["labels_used"] is False,
             "Donor assignment variant/seed/label-use differs")
    for field, value in (("sample_id", original["sample_id"]), ("query_student_id", original["student_id"]),
                         ("query_current_timestamp", original["current_timestamp"]), ("lab", original["lab"]),
                         ("source_file", original["source_file"])):
        _require(assignment[field] == value, "Assignment query origin differs: " + field)
    provenance = reference["provenance"]
    train_sha = _digest(provenance.get("training_inputs_sha256"), "Original reference training_inputs_sha256")
    original_donor = provenance.get("donor_sample_id")
    _require(isinstance(original_donor, str) and original_donor,
             "Original reference must identify its matched train donor")
    _require(assignment["training_inputs_sha256"] == train_sha
             and assignment["original_reference_donor_sample_id"] == original_donor,
             "Assignment training source/original matched donor differs")
    _require(isinstance(assignment["donor_student_id"], str) and assignment["donor_student_id"]
             and assignment["donor_student_id"] not in dev_students,
             "Donor student must be outside the entire query cohort")
    _require(isinstance(assignment["donor_sample_id"], str) and assignment["donor_sample_id"]
             and assignment["donor_sample_id"] != original_donor,
             "Intervention must use an alternative train donor")
    cutoff = original["current_timestamp"]
    _require(history_interventions._before(assignment["donor_current_timestamp"], cutoff, "assignment donor cutoff"),
             "Donor current state must strictly precede query")
    for entry in new_history:
        _require(history_interventions._before(entry["timestamp"], cutoff, "intervened history cutoff")
                 and history_interventions._before(entry["timestamp"], assignment["donor_current_timestamp"], "donor own history cutoff"),
                 "Every intervened history timestamp must precede query and donor current state")
    original_history = original["history"] if variant == "real_history_swapped" else reference["reference_history"]
    _require(assignment["original_history_sha256"] == inference.object_hash(original_history)
             and assignment["donor_history_sha256"] == inference.object_hash(new_history)
             and assignment["real_history_sha256"] == inference.object_hash(original["history"])
             and assignment["reference_history_sha256"] == inference.object_hash(reference["reference_history"]),
             "Assignment historical payload fingerprints differ")
    for field in ("target_history_length", "donor_history_length", "absolute_length_difference", "target_history_records", "donor_history_records", "eligible_donor_count"):
        _require(type(assignment[field]) is int and assignment[field] >= 0, "Invalid donor matching count: " + field)
    _require(assignment["target_history_records"] == len(original_history)
             and assignment["donor_history_records"] == len(new_history) and new_history
             and assignment["eligible_donor_count"] > 0
             and assignment["absolute_length_difference"] == abs(assignment["target_history_length"] - assignment["donor_history_length"])
             and assignment["length_measure"] == "tokens"
             and assignment["matching_rule"] == history_interventions.MATCHING_RULE
             and assignment["history_definition"] == "complete donor input history; donor current state not appended",
             "Donor matching must bind complete histories and formal token-length controls")


def _variant(cache, baseline, variant, permutation_seed, protocol_sha256, completion_sha256):
    manifest, artifact = cache["manifest"], cache["artifact"]
    metadata = manifest.get("history_intervention")
    _require(isinstance(metadata, dict) and set(metadata) == HISTORY_FIELDS, "Incomplete history_intervention metadata")
    _require(metadata["variant"] == variant and type(metadata["permutation_seed"]) is int
             and metadata["permutation_seed"] == permutation_seed and metadata["protocol_sha256"] == protocol_sha256
             and metadata["baseline_manifest_sha256"] == completion_sha256, "Intervention identity/seed/protocol/baseline mismatch")
    assignments = _index(artifact.assignments, "donor assignments")
    _require(set(assignments) == set(baseline["inputs"]), "Donor assignments differ from complete cohort")
    _require(metadata["donor_assignment_sha256"] == _sha(artifact.assignments), "Donor assignment bytes/hash mismatch")
    history_content = artifact.inputs if variant == "real_history_swapped" else artifact.references
    _require(metadata["history_data_sha256"] == _sha(history_content), "Intervened history bytes/hash mismatch")
    for field in ("model", "runtime_versions", "config", "config_sha256", "prompt_version", "implementation_sha256", "scoring_sha256", "sequence_protocol"):
        _require(manifest.get(field) == baseline["manifest"].get(field), "Model/scoring protocol differs across versions: " + field)
    _require(cache["backend"] == baseline["backend"] and cache["run_hash"] != baseline["run_hash"], "Scoring backend/run identity differs")
    _require(cache["pools"] == baseline["pools"], "Candidates/tokens/sources differ across fixed-pool versions")
    dev_students = {row["student_id"] for row in baseline["inputs"].values()}
    for sample, row in cache["inputs"].items():
        original = baseline["inputs"][sample]
        ignored = {"history"} if variant == "real_history_swapped" else set()
        _require({key: value for key, value in row.items() if key not in ignored} ==
                 {key: value for key, value in original.items() if key not in ignored}, "Intervention changed original current/problem/results/feedback or identity")
        _require(cache["candidate_records"][sample]["attempts"] == baseline["candidate_records"][sample]["attempts"], "Original generation attempt provenance changed")
        changed_history = row["history"] if variant == "real_history_swapped" else cache["references"][sample]["reference_history"]
        _require(changed_history != original["history"]
                 and changed_history != baseline["references"][sample]["reference_history"], "Intervention must change each history and cannot collapse real/reference histories")
        _assignment(assignments[sample], original, baseline["references"][sample], changed_history,
                    variant, permutation_seed, dev_students)
        if variant == "reference_swapped":
            old_ref, new_ref = baseline["references"][sample], cache["references"][sample]
            _require({key: value for key, value in new_ref.items() if key not in {"reference_history", "provenance"}} ==
                     {key: value for key, value in old_ref.items() if key not in {"reference_history", "provenance"}},
                     "Reference intervention may change only reference history and provenance")
            assignment, provenance = assignments[sample], new_ref["provenance"]
            required_provenance = {"split": "train", "intervention": variant, "permutation_seed": permutation_seed,
                "labels_used": False, "training_inputs_sha256": assignment["training_inputs_sha256"],
                "donor_sample_id": assignment["donor_sample_id"], "donor_student_id": assignment["donor_student_id"],
                "donor_current_timestamp": assignment["donor_current_timestamp"],
                "query_current_timestamp": original["current_timestamp"], "lab": original["lab"], "source_file": original["source_file"],
                "matching_rule": assignment["matching_rule"], "history_length_query": len(original["history"]),
                "history_length_donor": len(changed_history), "history_definition": assignment["history_definition"],
                "original_reference_donor_sample_id": assignment["original_reference_donor_sample_id"],
                "eligible_donor_count": assignment["eligible_donor_count"]}
            _require(provenance == required_provenance, "Intervened reference provenance differs from exact donor/query assignment")
    if variant == "real_history_swapped":
        _require(artifact.references == baseline["artifact"].references, "Real-history intervention changed matched reference")
    else:
        _require(artifact.inputs == baseline["artifact"].inputs, "Reference intervention changed input bytes")


def _metrics(inputs, labels, predictions):
    result = {method: {} for method in predictions}
    for sample, row in inputs.items():
        before, target = row["current_code"], labels[sample]["target_code"]
        for method, indexed in predictions.items():
            after = indexed[sample]["predicted_code"]
            values = pair_metrics(before, after, target)
            values.update(predicted_edit_size=edit_events(before, after)[2],
                          true_edit_size=edit_events(before, target)[2],
                          predicted_unchanged=float(before == after))
            _require(all(type(value) in (int, float) and math.isfinite(value) for value in values.values()), "Nonfinite static evaluation metric")
            result[method][sample] = values
    return result


def _summary(values, sample_values, draws, *, difference=False):
    metric_names = sorted(values[0])
    report = {}
    for metric in metric_names:
        student_values = [row[metric] for row in values]
        boot = [statistics.mean(student_values[index] for index in draw) for draw in draws]
        report[metric] = {
            "student_macro_delta" if difference else "student_macro_mean": statistics.mean(student_values),
            "submission_mean_delta" if difference else "submission_mean": statistics.mean(row[metric] for row in sample_values),
            "paired_student_bootstrap_95_ci": paired_analysis._ci(boot),
            "direction": paired_analysis._direction(metric),
        }
        if difference:
            report[metric]["student_delta_sign_counts"] = paired_analysis._sign_counts(student_values)
    return report


def _group(versions, cohort, sample_ids, permutation_seeds, bootstrap, seed):
    if not sample_ids:
        return {"samples": 0, "students": 0, "versions": {}, "seed_averages": {}, "difference_in_differences": {}}
    clusters = defaultdict(list)
    for sample in sample_ids:
        clusters[cohort[sample]].append(sample)
    students = sorted(clusters)
    metrics = sorted(next(iter(versions["baseline"]["base"].values())))
    student_means = {version: {method: [{metric: statistics.mean(data[method][sample][metric] for sample in clusters[student]) for metric in metrics}
                                      for student in students] for method in METHODS} for version, data in versions.items()}
    rng = random.Random(seed)
    draws = [[rng.randrange(len(students)) for _ in students] for _ in range(bootstrap)] if len(students) >= 2 else []
    reports = {version: {method: {"metrics": _summary(student_means[version][method], [versions[version][method][sample] for sample in sample_ids], draws)} for method in METHODS} for version in versions}
    averages, differences = {}, {}
    for variant in VARIANTS:
        keys = [variant + ":" + str(permutation_seed) for permutation_seed in permutation_seeds]
        means = {method: [{metric: statistics.mean(student_means[key][method][index][metric] for key in keys) for metric in metrics}
                          for index in range(len(students))] for method in METHODS}
        sample_means = {method: [{metric: statistics.mean(versions[key][method][sample][metric] for key in keys) for metric in metrics}
                                for sample in sample_ids] for method in METHODS}
        averages[variant] = {method: {"metrics": _summary(means[method], sample_means[method], draws)} for method in METHODS}
        deltas = [{metric: (student_means["baseline"]["b"][index][metric] - student_means["baseline"]["base"][index][metric]) -
                           (means["b"][index][metric] - means["base"][index][metric]) for metric in metrics} for index in range(len(students))]
        sample_deltas = [{metric: (versions["baseline"]["b"][sample][metric] - versions["baseline"]["base"][sample][metric]) -
                                 (sample_means["b"][index][metric] - sample_means["base"][index][metric]) for metric in metrics} for index, sample in enumerate(sample_ids)]
        per_student = []
        for index, student in enumerate(students):
            per_student.append({"student_id": student, "samples": len(clusters[student]),
                "baseline_b_minus_base": {metric: student_means["baseline"]["b"][index][metric] - student_means["baseline"]["base"][index][metric] for metric in metrics},
                "seed_b_minus_base": {str(permutation_seed): {metric: student_means[variant + ":" + str(permutation_seed)]["b"][index][metric] - student_means[variant + ":" + str(permutation_seed)]["base"][index][metric] for metric in metrics} for permutation_seed in permutation_seeds},
                "seed_mean_b_minus_base": {metric: means["b"][index][metric] - means["base"][index][metric] for metric in metrics},
                "difference_in_differences": deltas[index]})
        differences[variant] = {"method": "b", "reference": "base",
            "delta_convention": "baseline_b_minus_base_minus_seed_mean_intervention_b_minus_base",
            "metrics": _summary(deltas, sample_deltas, draws, difference=True), "per_student": per_student}
    return {"samples": len(sample_ids), "students": len(students), "versions": reports,
            "seed_averages": averages, "difference_in_differences": differences}


def analyze_interventions(inputs_bytes, labels_bytes, baseline, variants, *, protocol_sha256,
                          expected_seeds=DEFAULT_PERMUTATION_SEEDS, expected_samples=70,
                          expected_students=17, expected_candidates=318, bootstrap=2000,
                          seed=20260929, baseline_completion_sha256=BASELINE_COMPLETION_SHA256,
                          fixed_external_greedy=None):
    """Verify all seven fixed-pool versions, then analyze paired student DiD.

    ``variants`` maps exactly VARIANTS to mappings of three integer (or canonical
    decimal string) permutation seeds to InferenceArtifact. ``protocol_sha256``
    is the caller's frozen original protocol-file SHA. Every variant's six-field
    history_intervention metadata must bind it and the actual assignment/history
    bytes. Labels enter only the final static evaluation, after cache validation.
    ``fixed_external_greedy`` may be original greedy prediction JSONL bytes; it
    is verified against original actual attempt 0 and never called a new output.
    """
    for name, value in (("expected_samples", expected_samples), ("expected_students", expected_students),
                        ("expected_candidates", expected_candidates), ("bootstrap", bootstrap)):
        _require(type(value) is int and value > 0, name + " must be a positive integer")
    _require(expected_students <= expected_samples <= expected_candidates and type(seed) is int, "Invalid counts/bootstrap seed")
    _require(isinstance(expected_seeds, (tuple, list)) and len(expected_seeds) == 3
             and all(type(value) is int for value in expected_seeds) and len(set(expected_seeds)) == 3, "Exactly three distinct fixed permutation seeds required")
    permutation_seeds = tuple(sorted(expected_seeds))
    _digest(protocol_sha256, "protocol_sha256")
    _digest(baseline_completion_sha256, "baseline_completion_sha256")
    outer_inputs = _index(inputs_bytes, "original inputs")
    for row in outer_inputs.values():
        inference.validate_input(row)
    cohort = paired_analysis._cohort({sample: row["student_id"] for sample, row in outer_inputs.items()}, expected_samples, expected_students)
    original = _cache(baseline, outer_inputs, expected_candidates)
    _baseline(original, inputs_bytes, baseline_completion_sha256)
    if original["backend"] == "HFBackend":
        _require((expected_samples, expected_students, expected_candidates, bootstrap, seed) == (70, 17, 318, 2000, 20260929)
                 and permutation_seeds == DEFAULT_PERMUTATION_SEEDS, "Formal real analysis requires the complete frozen cohort, seeds and student bootstrap")
    _require(isinstance(variants, Mapping) and set(variants) == set(VARIANTS), "Exactly both history intervention variants required")
    caches = {"baseline": original}
    for variant in VARIANTS:
        _require(isinstance(variants[variant], Mapping), "Each variant must map all fixed seeds")
        normalized = {}
        for key, value in variants[variant].items():
            _require(type(key) is int or (isinstance(key, str) and re.fullmatch(r"-?(?:0|[1-9][0-9]*)", key)), "Invalid seed mapping key")
            integer = int(key)
            _require(integer not in normalized, "Duplicate normalized seed")
            normalized[integer] = value
        _require(set(normalized) == set(permutation_seeds), "Missing or extra fixed permutation seed")
        for permutation_seed in permutation_seeds:
            cache = _cache(normalized[permutation_seed], outer_inputs, expected_candidates)
            _variant(cache, original, variant, permutation_seed, protocol_sha256, baseline_completion_sha256)
            caches[variant + ":" + str(permutation_seed)] = cache
    _require(len({cache["run_hash"] for cache in caches.values()}) == 7, "Versions must have distinct cache identities")
    # Future targets are parsed only after all model-scoring and selection gates.
    labels = _index(labels_bytes, "labels")
    _require(set(labels) == set(cohort), "Labels differ from complete cohort")
    for sample, row in labels.items():
        _require(isinstance(row.get("target_code"), str), "Target code must be text")
        _require("student_id" not in row or row["student_id"] == cohort[sample], "Label student differs from original cohort")
    versions = {name: _metrics(outer_inputs, labels, cache["predictions"]) for name, cache in caches.items()}
    ids = sorted(cohort)
    all_report = _group(versions, cohort, ids, permutation_seeds, bootstrap, seed)
    groups = {"true_changed": [sample for sample in ids if labels[sample]["target_code"] != outer_inputs[sample]["current_code"]],
              "true_unchanged": [sample for sample in ids if labels[sample]["target_code"] == outer_inputs[sample]["current_code"]],
              "actual_feedback": [sample for sample in ids if outer_inputs[sample]["feedback"]]}
    external = None
    if fixed_external_greedy is not None:
        greedy = _index(fixed_external_greedy, "fixed external original greedy")
        _require(set(greedy) == set(cohort) and all(row.get("method") == "greedy" for row in greedy.values()), "External greedy cohort/method mismatch")
        for sample, row in greedy.items():
            attempts = original["candidate_records"][sample]["attempts"]
            _require(attempts and attempts[0].get("attempt") == 0 and attempts[0].get("source") == "greedy"
                     and attempts[0].get("eos_reached") is True, "Actual original greedy provenance missing")
            code, _, _ = cache_rescore.extract_code(attempts[0]["raw_text"])
            _require(row.get("predicted_code") == code and _sha(code.encode("utf-8")) in original["pools"][sample], "External greedy differs from actual original attempt 0")
        external = {"kind": "fixed_original_actual_greedy_external_reference", "prediction_sha256": _sha(fixed_external_greedy),
                    "generated_under_interventions": False, "used_in_primary_contrast": False,
                    "static_per_sample": _metrics(outer_inputs, labels, {"greedy": greedy})["greedy"]}
    actual = original["backend"] == "HFBackend"
    provenance = {name: {"run_sha256": cache["run_hash"], "backend": cache["backend"],
        "source_sha256": {field: _sha(getattr(cache["artifact"], field)) for field in ("inputs", "references", "manifest", "candidates", "scores")},
        "prediction_sha256": {method: _sha(cache["artifact"].predictions[method]) for method in METHODS},
        "history_intervention": cache["manifest"].get("history_intervention")} for name, cache in caches.items()}
    report = {"schema_version": SCHEMA_VERSION, "analysis_kind": ("paired_fixed_candidate_history_intervention" if actual else "synthetic_paired_fixed_candidate_history_intervention"),
        "evaluation_kind": "static_observed_next_revision_no_execution", "cohort_complete": True, "samples_dropped": 0,
        "expected_samples": expected_samples, "expected_students": expected_students, "fixed_candidates": expected_candidates,
        "raw_four_branch_scores_per_version": expected_candidates * 4, "versions": 7, "methods": list(METHODS),
        "permutation_seeds": list(permutation_seeds), "seed_aggregation": "equal seed mean within each query and student; seeds are not independent sample units",
        "bootstrap_seed": seed, "bootstrap_repetitions": bootstrap, "bootstrap_unit": "student, retaining all queries and every method/variant/seed in each shared cluster draw",
        "estimand": "equal-student mean of baseline (B-base) minus seed-average intervention (B-base)",
        "confidence_interval": "descriptive paired student percentile 95%; no causal or significance decision",
        "protocol_sha256": protocol_sha256, "baseline_completion_sha256": baseline_completion_sha256,
        "source_sha256": {"inputs": _sha(inputs_bytes), "labels": _sha(labels_bytes)}, "cache_provenance": provenance,
        "model_scoring_performed": actual, "generation_performed": False, "student_execution": False,
        "labels_used_for_static_evaluation_only": True, "all": all_report,
        "primary": {"variant": "real_history_swapped", "metric": "edit_location_f1", **all_report["difference_in_differences"]["real_history_swapped"]["metrics"]["edit_location_f1"]},
        "secondary": {"variant": "reference_swapped", "metrics": all_report["difference_in_differences"]["reference_swapped"]["metrics"]},
        "subgroups": {name: _group(versions, cohort, selected, permutation_seeds, bootstrap, seed) for name, selected in groups.items()},
        "fixed_external_greedy": external,
        "limitations": ["Development cohort and descriptive uncertainty do not establish method effectiveness or causality.",
            "Swapping may introduce distribution or matching changes; the contrast measures differential sensitivity relative to base.",
            "Fixed cached candidates cannot measure generation under intervened histories.",
            "Static revision metrics do not establish executable correctness, feedback uptake or psychological mechanisms.",
            "Permutation seeds are repeated assignments of the same queries and students, not additional observations."]}
    inference.canonical_json(report)
    return report
