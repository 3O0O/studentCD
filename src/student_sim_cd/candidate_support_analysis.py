"""Code-free static analysis of two equal-count candidate-support pools.

Both pools are independently validated. They are not assumed to be nested:
balanced12 uses three random draws per four prompt conditions; 11_only12 uses
twelve random draws from condition 11. Only the original 11 greedy, its first
three random draws and copy are required to be shared. Fixed choices are made
before reading labels. No model, Torch, network, or student execution is used.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, fields
import hashlib
import json
from pathlib import Path
import random
import re
import statistics

from . import cache_rescore, inference, predict
from . import evaluate, scoring
from .evaluate import edit_events, pair_metrics


SCHEMA_VERSION = "student-sim-cd.candidate-support-analysis.v1"
PROTOCOL_VERSION = "student-sim-cd.candidate-support.protocol.v1"
METHODS = ("base", "history_d0", "cd", "b", "copy")
POOLS = ("balanced12", "11_only12")
EXTERNAL_METHODS = ("student_revision_greedy", "student_revision_pool",
                    "conservative_edit_greedy", "conservative_edit_pool")
BOOTSTRAP_SEED = 20260929
PRIMARY_METRIC = "edit_location_f1"
SEQUENCE_PROTOCOL = "canonical code tokens + exactly one tokenizer EOS; sum logp; no length normalization"
SELECTION = "argmax; ties use lexicographically smallest candidate_id"
METHOD_KINDS = {
    "base": "l11_reranking_within_the_given_candidate_pool",
    "history_d0": "l11_plus_d0_reranking_within_the_given_candidate_pool",
    "cd": "l11_plus_d1_reranking_within_the_given_candidate_pool",
    "b": "l11_plus_gamma_reranking_within_the_given_candidate_pool",
    "copy": "copy_original_current_code",
    "original_greedy": "original_actual_condition11_greedy_draw0",
}
FORBIDDEN_EXPORT_FIELDS = {"code", "current_code", "target_code", "predicted_code", "raw_text",
                           "history", "reference_history", "raw_generation", "completion_token_ids",
                           "generated_token_ids", "inputs_bytes", "training_inputs_bytes"}


@dataclass(frozen=True)
class SupportArtifact:
    """The original inference.v1 manifest/candidates/scores file bytes."""

    manifest: bytes
    candidates: bytes
    scores: bytes


@dataclass(frozen=True)
class ExternalPredictionArtifact:
    """Complete predictions and code-free, explicit production-stage provenance.

    The production stage must verify its own one-branch prompt-pool argmax.
    This analyzer checks declared provenance, byte SHA and complete cohort,
    and does not claim to replay those external likelihood selections.
    """

    predictions: bytes
    provenance: Mapping


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _digest(value, location):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value), location + " must be a full SHA256")
    return value


def _bytes(value, location):
    _require(isinstance(value, bytes), location + " must be original file bytes")
    return value


def _jsonl(rows):
    return "".join(inference.canonical_json(row) + "\n" for row in rows).encode("utf-8")


def _index(data, name):
    rows, indexed = predict._records(_bytes(data, name), name), {}
    for row in rows:
        sid = row.get("sample_id")
        _require(isinstance(sid, str) and sid and sid not in indexed, name + " has missing or duplicate sample IDs")
        indexed[sid] = row
    return indexed


def _same_ids(rows, cohort, name):
    _require(set(rows) == set(cohort), name + " must retain the complete cohort without missing/extra samples")


def _code_free(value):
    if isinstance(value, Mapping):
        _require(not set(value) & FORBIDDEN_EXPORT_FIELDS, "Student text/token fields must not enter code-free analysis")
        for child in value.values():
            _code_free(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _code_free(child)


def _coerce(value, cls):
    if isinstance(value, Mapping):
        _require(set(value) == {field.name for field in fields(cls)}, "Unexpected/incomplete artifact fields")
        try:
            value = cls(**value)
        except TypeError as error:
            raise ValueError("Incomplete artifact") from error
    _require(isinstance(value, cls), "Use " + cls.__name__ + " or its exact field mapping")
    return value


def _expected_draws(pool):
    return {draw: ("11" if pool == "11_only12" or draw <= 3 else
                   "01" if draw <= 6 else "10" if draw <= 9 else "00") for draw in range(1, 13)}


def _cache(value, inputs_bytes, cohort, pool, protocol_sha256):
    artifact = _coerce(value, SupportArtifact)
    sources = {"inputs": inputs_bytes, **{name: _bytes(getattr(artifact, name), pool + " " + name)
                                          for name in ("manifest", "candidates", "scores")}}
    manifest, run_hash, candidates, scores, copies, _ = predict._validated_run(sources)
    metadata = manifest.get("candidate_support")
    _require(isinstance(metadata, dict) and metadata.get("pool") == pool
             and metadata.get("protocol_sha256") == protocol_sha256
             and metadata.get("pool_random_count") == 12
             and metadata.get("shared_greedy_condition") == "11"
             and metadata.get("shared_copy") is True
             and metadata.get("student_execution") is False
             and metadata.get("labels_read") is False
             and metadata.get("scoring_performed") is True
             and metadata.get("baseline_scores_reused") is True
             and metadata.get("baseline_score_records") == 318,
             "Pool manifest differs from the frozen support design")
    for name in ("baseline_manifest_sha256", "generation_manifest_sha256"):
        _digest(metadata.get(name), "candidate_support " + name)
    backend = metadata.get("backend")
    _require(backend in {"HFBackend", "injected_synthetic_backend"}, "Unknown support scoring backend")
    _require(metadata.get("generation_performed") is True, "Support pools must declare their new generation stage")
    _code_free(metadata)
    _same_ids(candidates, cohort, pool + " candidates")
    candidate_records = _index(artifact.candidates, pool + " candidates")
    raw_scores = {(row["sample_id"], row["candidate_id"]): row
                  for row in predict._records(artifact.scores, pool + " scores")}
    draws, originals, coverage = {}, {}, []
    greedy_ids = {}
    for sid in sorted(cohort):
        current = candidates[sid]
        _require(candidate_records[sid].get("reference_kind") == "matched", "The full cohort requires matched references without diagnostic fallback")
        _require(1 <= len(current) <= 14, "A support pool exceeds twelve random draws plus greedy/copy")
        wrappers = candidate_records[sid].get("attempts")
        _require(isinstance(wrappers, list) and len(wrappers) == 13, "All thirteen raw attempt wrappers must be retained")
        expected = {0: "11", **_expected_draws(pool)}
        indexed, expected_origins = {}, {}
        condition_stats = {condition: {"random_attempts": 0, "eligible_random_attempts": 0,
                                       "unique_random_candidate_ids": set()} for condition in inference.CONDITIONS}
        for wrapper in wrappers:
            _require(isinstance(wrapper, dict) and wrapper.get("sample_id") == sid
                     and type(wrapper.get("draw_id")) is int and wrapper["draw_id"] in expected
                     and wrapper["draw_id"] not in indexed
                     and wrapper.get("condition") == expected[wrapper["draw_id"]],
                     "Support attempt condition/draw identity differs from frozen allocation")
            draw, condition = wrapper["draw_id"], wrapper["condition"]
            raw = wrapper.get("raw_generation")
            _require(isinstance(raw, dict) and type(raw.get("attempt")) is int and raw["attempt"] == draw
                     and raw.get("source") == ("greedy" if draw == 0 else "sampled")
                     and isinstance(raw.get("raw_text"), str) and type(raw.get("eos_reached")) is bool,
                     "Raw generation attempt provenance is incomplete")
            origin = "greedy:11:0" if draw == 0 else "sampled:" + condition + ":" + str(draw)
            code = cache_rescore.extract_code(raw["raw_text"])[0]
            identifier = _sha(code.encode("utf-8")) if raw["eos_reached"] else None
            expected_origins[origin] = identifier
            indexed[draw] = wrapper
            if draw:
                condition_stats[condition]["random_attempts"] += 1
                if identifier is not None:
                    condition_stats[condition]["eligible_random_attempts"] += 1
                    condition_stats[condition]["unique_random_candidate_ids"].add(identifier)
            else:
                _require(identifier is not None, "Shared original greedy must be EOS-complete")
                greedy_ids[sid] = identifier
        _require(set(indexed) == set(expected), "Support raw attempt IDs are incomplete")
        seen_origins = {}
        for identifier, candidate in current.items():
            origins = candidate["sources"]
            _require(len(origins) == len(set(origins)), "Candidate contains duplicate source tags")
            for origin in origins:
                if origin == "copy_current":
                    _require(identifier == copies[sid], "Copy source differs from original current code")
                else:
                    _require(origin in expected_origins and expected_origins[origin] == identifier,
                             "Candidate source differs from canonical complete raw generation")
                    _require(origin not in seen_origins, "A raw generation source is mapped twice")
                    seen_origins[origin] = identifier
        _require(seen_origins == {key: value for key, value in expected_origins.items() if value is not None},
                 "Eligible raw draws must all map to the complete candidate pool; truncated draws cannot enter")
        draws[sid] = indexed
        originals[sid] = {draw: indexed[draw] for draw in range(4)}
        coverage.append({"sample_id": sid, "student_id": cohort[sid]["student_id"],
                         "candidate_count": len(current), "retained_raw_attempts": 13,
                         "conditions": {condition: {**statistics_, "unique_random_candidate_ids": sorted(statistics_["unique_random_candidate_ids"])}
                                        for condition, statistics_ in condition_stats.items()},
                         "shared_greedy_candidate_id": greedy_ids[sid], "copy_candidate_id": copies[sid]})
    choices, selected_codes = [], {method: {} for method in METHODS}
    for sid in sorted(cohort):
        selection = {}
        for method in METHODS:
            identifier = copies[sid] if method == "copy" else min(
                candidates[sid], key=lambda candidate: (-scores[sid, candidate][method], candidate))
            selected_codes[method][sid] = candidates[sid][identifier]["code"]
            selection[method] = {"candidate_id": identifier, "score": None if method == "copy" else scores[sid, identifier][method]}
        choices.append({"pool": pool, "sample_id": sid, "candidate_count": len(candidates[sid]), "choices": selection})
    return {"artifact": artifact, "manifest": manifest, "run_hash": run_hash, "metadata": metadata,
            "backend": backend, "pools": candidates, "scores": scores, "raw_scores": raw_scores,
            "copies": copies, "greedy_ids": greedy_ids, "originals": originals,
            "coverage": coverage, "choices": choices, "codes": selected_codes,
            "source_sha256": {name: _sha(content) for name, content in sources.items()}}


def _compare_caches(balanced, control):
    for name in ("model", "runtime_versions", "config", "config_sha256", "prompt_version",
                 "implementation_sha256", "scoring_sha256", "sequence_protocol", "references_sha256"):
        _require(balanced["manifest"].get(name) == control["manifest"].get(name), "Pool model/scoring/input protocol differs: " + name)
    for name in ("baseline_manifest_sha256", "generation_manifest_sha256", "backend"):
        _require(balanced["metadata"][name] == control["metadata"][name], "Pool production provenance differs: " + name)
    _require(balanced["run_hash"] != control["run_hash"], "Different support pools require distinct manifest identities")
    overlap = []
    for sid in balanced["pools"]:
        _require(balanced["originals"][sid] == control["originals"][sid], "Original shared 11 greedy and three random draws changed")
        _require(balanced["copies"][sid] == control["copies"][sid]
                 and balanced["greedy_ids"][sid] == control["greedy_ids"][sid], "Original greedy/copy identity differs across pools")
        shared = set(balanced["pools"][sid]) & set(control["pools"][sid])
        for identifier in shared:
            left, right = balanced["pools"][sid][identifier], control["pools"][sid][identifier]
            for field in ("code", "completion_token_ids", "token_count", "eos_included"):
                _require(left[field] == right[field], "Shared canonical candidate/EOS changed across pools")
            for field in ("l11", "l01", "l10", "l00", "prompt_token_counts", "token_count", "eos_included"):
                _require(balanced["raw_scores"][sid, identifier][field] == control["raw_scores"][sid, identifier][field],
                         "A shared candidate must retain the one union-scoring result")
        overlap.append({"sample_id": sid, "shared_candidate_ids": sorted(shared),
                        "balanced_only_candidates": len(set(balanced["pools"][sid]) - shared),
                        "control_only_candidates": len(set(control["pools"][sid]) - shared)})
    return overlap


def _external(value, method, cohort, protocol_sha256, backend):
    artifact = _coerce(value, ExternalPredictionArtifact)
    provenance = artifact.provenance
    _require(isinstance(provenance, Mapping), "External prediction provenance must be explicit")
    prompt_name = "student_revision" if method.startswith("student_revision_") else "conservative_edit"
    kind = "prompt_greedy" if method.endswith("_greedy") else "prompt_pool_reranking"
    required = {"kind": kind, "prompt_name": prompt_name, "protocol_sha256": protocol_sha256,
                "prediction_sha256": _sha(_bytes(artifact.predictions, method)), "backend": backend,
                "generation_count_per_sample": 13, "random_count_per_sample": 12,
                "labels_used": False, "parameters_fitted": False, "student_execution": False}
    _require(all(provenance.get(key) == value and type(provenance.get(key)) is type(value)
                 for key, value in required.items()), "External prediction prompt/count/hash/provenance differs: " + method)
    _digest(provenance.get("generation_manifest_sha256"), "External generation_manifest_sha256")
    if "source_sha256" in provenance:
        _require(isinstance(provenance["source_sha256"], Mapping), "External source SHA table must be a mapping")
        for value in provenance["source_sha256"].values():
            _digest(value, "External source SHA")
    _code_free(provenance)
    records = _index(artifact.predictions, method)
    _same_ids(records, cohort, method)
    for record in records.values():
        _require(set(record) == {"sample_id", "method", "predicted_code"}
                 and record["method"] == method and isinstance(record["predicted_code"], str),
                 "External prediction has unexpected fields/method/code")
    return {sid: record["predicted_code"] for sid, record in records.items()}, dict(provenance)


def _metrics(current, predicted, target):
    values = pair_metrics(current, predicted, target)
    values.update(predicted_edit_size=edit_events(current, predicted)[2],
                  true_edit_size=edit_events(current, target)[2],
                  predicted_unchanged=float(predicted == current),
                  false_edit_on_unchanged=float(target == current and predicted != current))
    return values


def _ci(values):
    values = sorted(values)
    return [values[int(0.025 * len(values))], values[min(len(values) - 1, int(0.975 * len(values)))]]


def _summary(rows, repetitions, seed, *, delta=False):
    if not rows:
        return {"samples": 0, "students": 0, "metrics": {}, "per_student": []}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["student_id"]].append(row)
    metrics = sorted(rows[0]["metrics"])
    students = [{"student_id": sid, "samples": len(items),
                 "metrics": {name: statistics.mean(row["metrics"][name] for row in items) for name in metrics}}
                for sid, items in sorted(grouped.items())]
    rng = random.Random(seed)
    distributions = {name: [] for name in metrics}
    for _ in range(repetitions):
        selected = [students[rng.randrange(len(students))] for _ in students]
        for name in metrics:
            distributions[name].append(statistics.mean(row["metrics"][name] for row in selected))
    suffix = "delta" if delta else "mean"
    output = {}
    for name in metrics:
        values = [student["metrics"][name] for student in students]
        output[name] = {"student_macro_" + suffix: statistics.mean(values),
                        "submission_" + suffix: statistics.mean(row["metrics"][name] for row in rows),
                        ("paired_student_bootstrap_95_ci" if delta else "student_bootstrap_95_ci"): _ci(distributions[name])}
        if delta:
            output[name]["student_delta_sign_counts"] = {"positive": sum(x > 1e-12 for x in values),
                                                        "tie": sum(abs(x) <= 1e-12 for x in values),
                                                        "negative": sum(x < -1e-12 for x in values)}
    return {"samples": len(rows), "students": len(students), "metrics": output, "per_student": students}


def _difference(left, right):
    _require([row["sample_id"] for row in left] == [row["sample_id"] for row in right], "Paired comparison cohort differs")
    return [{**a, "metrics": {name: a["metrics"][name] - b["metrics"][name] for name in a["metrics"]}}
            for a, b in zip(left, right)]


def _oracle(cache, cohort, labels, evaluations):
    rows, generated_rows, details = [], [], []
    for sid in sorted(cohort):
        current, target = cohort[sid]["current_code"], labels[sid]["target_code"]
        all_metrics = {identifier: _metrics(current, item["code"], target) for identifier, item in cache["pools"][sid].items()}
        generated = {identifier for identifier, item in cache["pools"][sid].items()
                     if any(origin != "copy_current" for origin in item["sources"])}
        _require(bool(generated), "Shared original greedy must provide generated-source coverage")
        best = min(all_metrics, key=lambda identifier: (-all_metrics[identifier][PRIMARY_METRIC], identifier))
        best_generated = min(generated, key=lambda identifier: (-all_metrics[identifier][PRIMARY_METRIC], identifier))
        rows.append({"sample_id": sid, "student_id": cohort[sid]["student_id"], "metrics": all_metrics[best]})
        generated_rows.append({"sample_id": sid, "student_id": cohort[sid]["student_id"], "metrics": all_metrics[best_generated]})
        details.append({"sample_id": sid, "student_id": cohort[sid]["student_id"],
                        "candidate_count": len(all_metrics), "generated_source_candidate_count": len(generated),
                        "exact_observed_next_code_covered": any(values["exact_next_code"] for values in all_metrics.values()),
                        "generated_exact_observed_next_code_covered": any(all_metrics[identifier]["exact_next_code"] for identifier in generated),
                        "label_oracle_edit_location_f1": all_metrics[best][PRIMARY_METRIC],
                        "generated_label_oracle_edit_location_f1": all_metrics[best_generated][PRIMARY_METRIC],
                        "oracle_gaps_by_method": {method: all_metrics[best][PRIMARY_METRIC] -
                                                   evaluations[method][sid]["metrics"][PRIMARY_METRIC] for method in METHODS}})
    return rows, generated_rows, details


def analyze_support(inputs: bytes, labels: bytes, balanced, control, external_predictions: Mapping, *,
                    protocol_sha256: str, expected_samples: int = 70, expected_students: int = 17,
                    bootstrap: int = 2000, seed: int = BOOTSTRAP_SEED) -> dict:
    """Analyze verified balanced12/11_only12 pools and four prompt baselines.

    Selection/provenance validation finishes before labels are parsed. Results
    contain candidate IDs, numeric metrics and hashes, never student code.
    Source membership is the portable production module's explicit tags and
    thirteen raw wrappers, verified against strict cached-code extraction.
    """
    protocol_sha256 = _digest(protocol_sha256, "protocol_sha256")
    _require(type(bootstrap) is int and bootstrap > 0 and type(seed) is int, "Invalid student-bootstrap configuration")
    _require(type(expected_samples) is int and expected_samples > 0 and type(expected_students) is int
             and 0 < expected_students <= expected_samples, "Invalid expected cohort")
    cohort = _index(inputs, "inputs")
    for row in cohort.values():
        inference.validate_input(row)
    _require(len(cohort) == expected_samples and len({row["student_id"] for row in cohort.values()}) == expected_students,
             "The entire expected sample/student cohort must be retained")
    caches = {"balanced12": _cache(balanced, inputs, cohort, "balanced12", protocol_sha256),
              "11_only12": _cache(control, inputs, cohort, "11_only12", protocol_sha256)}
    overlap = _compare_caches(caches["balanced12"], caches["11_only12"])
    _require(isinstance(external_predictions, Mapping) and set(external_predictions) == set(EXTERNAL_METHODS),
             "All four predeclared strong prompt baselines are required")
    external_codes, external_provenance = {}, {}
    for method in EXTERNAL_METHODS:
        external_codes[method], external_provenance[method] = _external(
            external_predictions[method], method, cohort, protocol_sha256, caches["balanced12"]["backend"])
    # Every choice is fixed above this boundary; labels affect only evaluation.
    target = _index(labels, "evaluation labels")
    _same_ids(target, cohort, "evaluation labels")
    _require(all(isinstance(row.get("target_code"), str) for row in target.values()), "Evaluation labels require target code text")
    code_sets = {pool: cache["codes"] for pool, cache in caches.items()}
    code_sets["external"] = {**external_codes, "original_greedy": {
        sid: caches["balanced12"]["pools"][sid][caches["balanced12"]["greedy_ids"][sid]]["code"] for sid in cohort}}
    evaluations, all_summaries, groups = {}, {}, {}
    for version, codes_by_method in code_sets.items():
        evaluations[version], all_summaries[version], groups[version] = {}, {}, {}
        for method, codes in codes_by_method.items():
            rows = [{"sample_id": sid, "student_id": cohort[sid]["student_id"], "has_feedback": bool(cohort[sid]["feedback"]),
                     "metrics": _metrics(cohort[sid]["current_code"], codes[sid], target[sid]["target_code"])} for sid in sorted(cohort)]
            evaluations[version][method] = {row["sample_id"]: row for row in rows}
            all_summaries[version][method] = _summary(rows, bootstrap, seed)
            groups[version][method] = {name: _summary([row for row in rows if condition(row)], bootstrap, seed)
                                     for name, condition in (
                ("true_changed", lambda row: row["metrics"]["true_changed"] == 1),
                ("true_unchanged", lambda row: row["metrics"]["true_changed"] == 0),
                ("actual_feedback", lambda row: row["has_feedback"]))}
    contrasts = {}
    for pool in POOLS:
        b_rows, base_rows = (list(evaluations[pool][method].values()) for method in ("b", "base"))
        contrasts[pool] = _summary(_difference(b_rows, base_rows), bootstrap, seed, delta=True)
    interaction_rows = _difference(
        _difference(list(evaluations["balanced12"]["b"].values()), list(evaluations["balanced12"]["base"].values())),
        _difference(list(evaluations["11_only12"]["b"].values()), list(evaluations["11_only12"]["base"].values())))
    interaction = _summary(interaction_rows, bootstrap, seed, delta=True)
    pool_method_changes = {method: _summary(_difference(list(evaluations["balanced12"][method].values()),
                                                       list(evaluations["11_only12"][method].values())), bootstrap, seed, delta=True)
                           for method in METHODS}
    oracle = {}
    for pool in POOLS:
        oracle_rows, generated_rows, details = _oracle(caches[pool], cohort, target, evaluations[pool])
        oracle[pool] = {"analysis_kind": "label_oracle_posthoc_diagnostic_not_method",
                        "labels_used_after_choices_frozen": True, "feeds_back_into_selection": False,
                        "metric_optimized": PRIMARY_METRIC,
                        "generated_source_definition": "any greedy or random generation source; may also share copy_current",
                        "all_candidates": _summary(oracle_rows, bootstrap, seed),
                        "generated_source_candidates": _summary(generated_rows, bootstrap, seed),
                        "per_sample": details,
                        "exact_observed_next_code_coverage_count": sum(row["exact_observed_next_code_covered"] for row in details),
                        "generated_exact_observed_next_code_coverage_count": sum(row["generated_exact_observed_next_code_covered"] for row in details)}
    selections = [row for pool in POOLS for row in caches[pool]["choices"]]
    external_selections = [{"method": method, "sample_id": sid, "predicted_code_sha256": _sha(codes[sid].encode("utf-8"))}
                           for method, codes in code_sets["external"].items() for sid in sorted(cohort)]
    result = {
        "schema_version": SCHEMA_VERSION, "analysis_kind": ("paired_equal_student_candidate_support" if caches["balanced12"]["backend"] == "HFBackend"
                                                              else "synthetic_paired_equal_student_candidate_support"),
        "protocol_sha256": protocol_sha256, "primary_metric": PRIMARY_METRIC,
        "primary_contrast": "(B-base)_balanced12 - (B-base)_11_only12",
        "primary": interaction["metrics"][PRIMARY_METRIC], "paired_pool_interaction": interaction,
        "pool_b_minus_base": contrasts, "balanced_minus_control_by_method": pool_method_changes,
        "cohort_complete": True, "samples_dropped": 0, "samples": expected_samples, "students": expected_students,
        "pool_candidate_counts": {pool: sum(len(value) for value in cache["pools"].values()) for pool, cache in caches.items()},
        "pool_random_draws_per_sample": 12, "shared_original_greedy_and_copy": True,
        "methods": list(METHODS), "external_methods": list(code_sets["external"]),
        "method_kinds": {**METHOD_KINDS, **{method: external_provenance[method]["kind"] for method in EXTERNAL_METHODS}},
        "fixed_method_parameters": predict.METHOD_PARAMETERS, "selection": SELECTION,
        "sequence_protocol": SEQUENCE_PROTOCOL,
        "model_scoring_performed": caches["balanced12"]["backend"] == "HFBackend",
        "generation_stage_performed": caches["balanced12"]["backend"] == "HFBackend",
        "student_execution": False, "labels_used_for_static_evaluation_only": True,
        "parameters_fitted": False, "bootstrap_unit": "student", "bootstrap_repetitions": bootstrap,
        "bootstrap_seed": seed, "all": all_summaries, "subgroups": groups,
        "static_per_sample": evaluations, "label_oracle_diagnostics": oracle,
        "source_coverage": {pool: cache["coverage"] for pool, cache in caches.items()}, "pool_overlap": overlap,
        "selections": selections, "external_selections": external_selections,
        "selections_sha256": _sha(_jsonl(selections)), "external_selections_sha256": _sha(_jsonl(external_selections)),
        "source_sha256": {"inputs": _sha(inputs), "labels": _sha(labels)},
        "implementation_sha256": {"analysis": inference.file_hash(Path(__file__)),
                                    "evaluate": inference.file_hash(Path(evaluate.__file__)),
                                    "predict": inference.file_hash(Path(predict.__file__)),
                                    "scoring": inference.file_hash(Path(scoring.__file__)),
                                    "canonical_extraction": inference.file_hash(Path(cache_rescore.__file__))},
        "cache_provenance": {pool: {"backend": cache["backend"], "run_sha256": cache["run_hash"],
                                    "source_sha256": cache["source_sha256"], "candidate_support": cache["metadata"]}
                             for pool, cache in caches.items()},
        "external_provenance": external_provenance,
        "external_validation": "Complete prediction bytes/SHA and declared production-stage provenance; external own-prompt likelihood argmax is checked by that production stage, not replayed here.",
        "compute_budget_note": "Main pool has twelve random plus reused original greedy/copy. Each strong prompt baseline has one new greedy plus twelve random/copy. Similar generation count does not imply equal scoring FLOPs or GPU time.",
        "metric_definitions": {
            "edit_location_f1": "F1 of changed original-line indices and insertion boundaries; both empty edit sets score one; not semantic correctness.",
            "edit_operation_f1": "F1 of static operation-type/original-location sets; both empty sets score one.",
            "predicted_edit_size": "Number of original lines removed plus new lines added by SequenceMatcher; per-student mean then equal-student macro and separate submission mean.",
            "false_edit_on_unchanged": "Per-sample indicator(target equals current and prediction differs); unconditional mean uses all samples, true_unchanged subgroup uses only observed unchanged targets.",
            "student_macro": "Mean submissions within each student, then equal weight across distinct students; paired bootstrap resamples those student means.",
        },
        "limitations": ["Development student-clustered static uncertainty is descriptive, not efficacy or causality.",
                        "Modification-location F1 compares original line indices/insertion boundaries, not program correctness.",
                        "A single observed next submission is not the full distribution or the only semantically valid revision.",
                        "Label-oracle coverage is conditional on these pools and cannot be used by a deployed selector.",
                        "Shared original draws and candidate deduplication couple the pools; samples and students are not duplicated."],
    }
    _code_free(result)
    inference.canonical_json(result)
    return result


def _read_artifact(directory):
    directory = Path(directory)
    return SupportArtifact(**{name: (directory / (name + (".json" if name == "manifest" else ".jsonl"))).read_bytes()
                              for name in ("manifest", "candidates", "scores")})


def _frozen_protocol(raw, inputs, labels, balanced, control):
    """Bind the production CLI's actual bytes to its complete frozen contract.

    The pure analysis API accepts explicitly supplied cohort sizes for synthetic
    tests. The production CLI always uses this 70/17 contract and source seals.
    """
    specification = predict._json(raw)
    expected = {
        "schema_version": PROTOCOL_VERSION, "protocol": "candidate-support-v1",
        "expected_samples": 70, "expected_students": 17, "seed": BOOTSTRAP_SEED,
        "conditions": list(inference.CONDITIONS), "balanced_random_per_condition": 3,
        "control_random_count": 12, "shared_greedy_condition": "11",
        "canonical_protocol": "cached-python-fence-v1", "generation_performed": True,
        "student_execution": False, "test_set_used": False, "parameters_fitted": False,
        "new_support_attempts": 1260, "new_prompt_baseline_attempts": 1820,
        "max_new_generation_attempts": 3080, "prompt_baselines": ["student_revision", "conservative_edit"],
        "prompt_baseline_random_count": 12, "prompt_baseline_greedy_count": 1,
        "fixed_method_parameters": predict.METHOD_PARAMETERS, "methods": list(METHODS),
        "sequence_protocol": SEQUENCE_PROTOCOL, "selection": SELECTION,
        "primary_metric": "equal-student macro edit_location_f1",
        "primary_comparison": "(B-base)_balanced12 - (B-base)_11_only12",
        "bootstrap": {"unit": "student", "paired": True, "repetitions": 2000, "seed": BOOTSTRAP_SEED},
    }
    _require(isinstance(specification, dict) and all(specification.get(name) == value
             and type(specification.get(name)) is type(value) for name, value in expected.items()),
             "Use the frozen complete-cohort candidate support protocol")
    _digest(specification.get("baseline_release_sha256"), "Frozen baseline release")
    source_seals = specification.get("baseline_sha256")
    required = {"manifest.json", "SUCCESS", "prepared/inputs.jsonl", "prepared/references.jsonl",
                "prepared/inference/manifest.json", "prepared/inference/candidates.jsonl", "prepared/inference/scores.jsonl"}
    _require(isinstance(source_seals, dict) and required <= set(source_seals), "Frozen baseline source SHA table is incomplete")
    for name, digest in source_seals.items():
        _digest(digest, "Frozen baseline " + name)
    _require(_sha(inputs) == source_seals["prepared/inputs.jsonl"], "Analysis inputs differ from the frozen baseline input bytes")
    _require(_digest(specification.get("labels_sha256"), "Frozen labels") == _sha(labels),
             "Analysis labels differ from the frozen evaluation bytes")
    config = specification.get("generation_config")
    _require(isinstance(config, dict), "Frozen generation/scoring settings are required")
    for pool, artifact in (("balanced12", balanced), ("11_only12", control)):
        manifest = predict._json(artifact.manifest)
        _require(manifest.get("inputs_sha256") == source_seals["prepared/inputs.jsonl"]
                 and manifest.get("references_sha256") == source_seals["prepared/references.jsonl"]
                 and manifest.get("config") == config and manifest.get("sequence_protocol") == SEQUENCE_PROTOCOL,
                 "Pool inputs/references/config/EOS protocol differs from frozen sources: " + pool)
        _require(isinstance(manifest.get("candidate_support"), dict)
                 and manifest["candidate_support"].get("baseline_manifest_sha256") == source_seals["manifest.json"],
                 "Pool baseline completion differs from the frozen source: " + pool)
    return specification


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("balanced-dir", "control-dir", "inputs", "labels", "externals-json", "protocol", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        output = args.output.absolute()
        _require(not output.exists() and not output.is_symlink() and output.resolve() == output,
                 "Output must be a new canonical directory; preserve previous results")
        protocol_raw = args.protocol.read_bytes()
        input_raw, label_raw = args.inputs.read_bytes(), args.labels.read_bytes()
        balanced, control = _read_artifact(args.balanced_dir), _read_artifact(args.control_dir)
        _frozen_protocol(protocol_raw, input_raw, label_raw, balanced, control)
        external_specification = predict._json(args.externals_json.read_bytes())
        _require(isinstance(external_specification, list), "External JSON must be a list of method/path/provenance objects")
        external = {}
        for row in external_specification:
            _require(isinstance(row, dict) and set(row) == {"method", "predictions_path", "provenance"}
                     and row["method"] not in external, "Invalid/duplicate external prediction specification")
            path = Path(row["predictions_path"])
            if not path.is_absolute():
                path = args.externals_json.parent / path
            external[row["method"]] = ExternalPredictionArtifact(path.read_bytes(), row["provenance"])
        result = analyze_support(input_raw, label_raw, balanced, control, external,
                                 protocol_sha256=_sha(protocol_raw))
        output.mkdir()
        with (output / ".incomplete").open("xb") as stream:
            stream.write(b"Preserve incomplete candidate support analysis.\n")
        payloads = {"analysis.json": (inference.canonical_json(result) + "\n").encode("utf-8"),
                    "selections.jsonl": _jsonl(result["selections"]),
                    "external-selections.jsonl": _jsonl(result["external_selections"])}
        for name, content in payloads.items():
            with (output / name).open("xb") as stream:
                stream.write(content)
        (output / ".incomplete").unlink()
        with (output / "SUCCESS").open("x", encoding="ascii") as stream:
            stream.write(_sha(payloads["analysis.json"]) + "\n")
        print(inference.canonical_json({"samples": result["samples"], "students": result["students"],
                                        "analysis_sha256": _sha(payloads["analysis.json"])}), flush=True)
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, "error: " + str(error) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
