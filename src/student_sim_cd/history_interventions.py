"""Build past-only history interventions without labels, models, or execution.

The fixed query cohort is preserved. Each intervention substitutes a complete
training donor history, preserving every record and its internal time order.
Nearest history token length is a diagnostic control for prompt length, not a
claim of matched ability or a causal student effect. Seeded ties can produce
identical assignments; their number is reported rather than treated as repeats.

This module returns records and exact JSONL bytes in memory. It does not read or
write files, load a tokenizer/model, generate candidates, or run student code.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import copy
from datetime import datetime
import hashlib
import json
import re
from typing import Any, Mapping

from .inference import canonical_json, object_hash, validate_history, validate_input


SCHEMA_VERSION = "student-sim-cd.history-interventions.v1"
ASSIGNMENT_VERSION = "student-sim-cd.history-donor-assignment.v1"
DEFAULT_SEEDS = (20261002, 20261003, 20261004)
VARIANTS = ("real_history_swapped", "reference_swapped")
TOKEN_SERIALIZATION = "inference.canonical_json(history); encode(add_special_tokens=False)"
MATCHING_RULE = "absolute_history_length_then_history_record_count_then_seeded_hash_v1"


class DonorUnavailableError(ValueError):
    """No partial intervention is returned when any fixed query has no donor."""

    def __init__(self, failures: list[dict[str, Any]]) -> None:
        self.failures = copy.deepcopy(failures)
        super().__init__("no legal alternative train donor for fixed queries: " +
                         ", ".join(item["sample_id"] for item in failures))


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    """The exact bytes to persist, with one canonical object and LF per row."""
    return "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest(value: Any, location: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{location} must be a lowercase full SHA256")
    return value


def _timestamp(value: Any, location: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{location} requires a string timestamp")
    try:
        return datetime.strptime(value, "%Y-%m-%d-%H-%M-%S")
    except ValueError:
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"{location} has invalid timestamp") from exc


def _before(left: Any, right: Any, location: str) -> bool:
    try:
        return _timestamp(left, location) < _timestamp(right, location)
    except TypeError as exc:
        raise ValueError(f"{location} has incompatible timestamp timezone information") from exc


def _input_rows(rows: Any, location: str, *, problem: bool) -> dict[str, dict]:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{location} must be a nonempty list of input records")
    indexed = {}
    for row in rows:
        validate_input(row, require_problem=problem)
        _timestamp(row.get("current_timestamp"), f"{location}.current_timestamp")
        if row["sample_id"] in indexed:
            raise ValueError(f"{location} contains duplicate sample_id")
        indexed[row["sample_id"]] = row
    return indexed


def _source_bytes(rows: list[dict], data: Any, expected: str) -> None:
    if not isinstance(data, bytes) or _sha(data) != expected:
        raise ValueError("training input bytes do not match frozen training_inputs_sha256")
    try:
        lines = data.decode("utf-8").splitlines()
        if not lines or any(not line.strip() for line in lines):
            raise ValueError("empty training JSONL record")
        parsed = [json.loads(line) for line in lines]
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("training input bytes must contain valid complete JSONL") from exc
    if canonical_json(parsed) != canonical_json(rows):
        raise ValueError("training input records differ from the frozen source bytes")


def _references(rows: Any, queries: dict, train: dict, train_sha256: str) -> dict:
    if not isinstance(rows, list):
        raise ValueError("references must be a complete list")
    indexed = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "sample_id", "reference_history", "reference_kind", "provenance"
        }:
            raise ValueError("reference record has unknown or missing fields")
        sid = row["sample_id"]
        if sid not in queries or sid in indexed:
            raise ValueError("reference IDs must equal the complete query cohort without duplicates")
        if row["reference_kind"] != "matched" or not row["reference_history"]:
            raise ValueError("baseline requires a nonempty matched reference; no empty fallback")
        validate_history(row["reference_history"], "reference_history")
        provenance = row["provenance"]
        if not isinstance(provenance, dict):
            raise ValueError("baseline reference provenance must be an object")
        donor = train.get(provenance.get("donor_sample_id"))
        query = queries[sid]
        if donor is None:
            raise ValueError(f"{sid}: original reference donor is absent from complete train inputs")
        expected = {
            "split": "train", "training_inputs_sha256": train_sha256,
            "donor_student_id": donor["student_id"],
            "donor_current_timestamp": donor["current_timestamp"],
            "query_current_timestamp": query["current_timestamp"],
            "lab": query["lab"], "source_file": query["source_file"], "labels_used": False,
        }
        if any(provenance.get(key) != value for key, value in expected.items()):
            raise ValueError(f"{sid}: original reference provenance differs from frozen train/query")
        if (donor["lab"], donor["source_file"]) != (query["lab"], query["source_file"]):
            raise ValueError(f"{sid}: original reference donor is from another lab/file")
        if (canonical_json(donor["history"]) != canonical_json(row["reference_history"])
                or not _before(donor["current_timestamp"], query["current_timestamp"], sid)):
            raise ValueError(f"{sid}: original reference is not the frozen strictly-past donor history")
        indexed[sid] = row
    if set(indexed) != set(queries):
        raise ValueError("reference IDs must equal the complete query cohort without missing rows")
    return indexed


def _bundle(inputs: list[dict], references: list[dict], assignments: list[dict],
            variant: str, seed: int | None, protocol_sha256: str,
            formal: bool) -> dict:
    input_data, reference_data, assignment_data = (
        jsonl_bytes(inputs), jsonl_bytes(references), jsonl_bytes(assignments)
    )
    input_sha, reference_sha, assignment_sha = map(_sha, (input_data, reference_data, assignment_data))
    return {
        "inputs": inputs, "references": references, "assignments": assignments,
        "inputs_bytes": input_data, "references_bytes": reference_data,
        "assignments_bytes": assignment_data,
        "inputs_sha256": input_sha, "references_sha256": reference_sha,
        "assignments_sha256": assignment_sha, "donor_assignment_sha256": assignment_sha,
        "history_data_sha256": input_sha if variant == "real_history_swapped" else reference_sha,
        "manifest": {
            "schema_version": SCHEMA_VERSION, "variant": variant, "permutation_seed": seed,
            "protocol_sha256": protocol_sha256, "inputs_sha256": input_sha,
            "references_sha256": reference_sha, "donor_assignment_sha256": assignment_sha,
            "history_data_sha256": input_sha if variant == "real_history_swapped" else reference_sha,
            "samples": len(inputs), "students": len({row["student_id"] for row in inputs}),
            "formal_token_matching": formal, "labels_used": False, "generation_performed": False,
        },
    }


def build_interventions(
    inputs: list[dict], references: list[dict], train_inputs: list[dict], *,
    training_inputs_sha256: str, training_inputs_bytes: bytes, protocol_sha256: str,
    token_lengths: Mapping[str, int] | None = None,
    tokenizer_fingerprint_sha256: str | None = None,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    expected_samples: int = 70, expected_students: int = 17,
    length_measure: str = "tokens",
) -> dict:
    """Return baseline and two single-factor variants for every fixed seed.

    ``token_lengths`` maps ``object_hash(history)`` to the length defined by
    TOKEN_SERIALIZATION; the caller must count with the frozen real tokenizer.
    Every nonempty training history, query history, and reference history needs
    a count. ``tokenizer_fingerprint_sha256`` binds that caller-supplied asset.
    This module validates the count table, not the tokenizer implementation.

    A complete raw training JSONL is required and byte-hash checked. For an
    explicitly nonformal feasibility diagnostic only, use
    ``length_measure='utf8_bytes_diagnostic'`` with no token table/fingerprint.
    This mode must never be accepted as formal token-matched model scoring.
    """
    train_sha = _digest(training_inputs_sha256, "training_inputs_sha256")
    protocol_sha = _digest(protocol_sha256, "protocol_sha256")
    if (not isinstance(seeds, tuple) or not seeds or len(set(seeds)) != len(seeds)
            or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)):
        raise ValueError("seeds must be an explicit nonempty tuple of distinct nonnegative integers")
    if any(isinstance(count, bool) or not isinstance(count, int) or count < 1
           for count in (expected_samples, expected_students)):
        raise ValueError("expected cohort counts must be positive integers")
    queries = _input_rows(inputs, "queries", problem=True)
    train = _input_rows(train_inputs, "training inputs", problem=False)
    _source_bytes(train_inputs, training_inputs_bytes, train_sha)
    if len(queries) != expected_samples or len({row["student_id"] for row in inputs}) != expected_students:
        raise ValueError("query cohort must retain exactly the expected samples and students")
    if set(queries) & set(train):
        raise ValueError("query and training sample IDs overlap")
    if {row["student_id"] for row in inputs} & {row["student_id"] for row in train_inputs}:
        raise ValueError("student split leakage between complete train and fixed query cohort")
    if any(not row["history"] for row in inputs):
        raise ValueError("all fixed queries must have a nonempty real history")
    refs = _references(references, queries, train, train_sha)
    # Hash each potentially large student-text payload once, not for every
    # query/donor pair. These are text identities, never executable code.
    input_history_hashes = {row["sample_id"]: object_hash(row["history"])
                            for row in inputs + train_inputs}
    reference_history_hashes = {row["sample_id"]: object_hash(row["reference_history"])
                               for row in references}
    all_histories = {input_history_hashes[row["sample_id"]]: row["history"]
                     for row in inputs + train_inputs if row["history"]}
    all_histories.update({reference_history_hashes[row["sample_id"]]: row["reference_history"]
                         for row in references})
    formal = length_measure == "tokens"
    if formal:
        _digest(tokenizer_fingerprint_sha256, "tokenizer_fingerprint_sha256")
        if not isinstance(token_lengths, Mapping):
            raise ValueError("formal token matching requires the complete tokenizer count table")
        for key, value in token_lengths.items():
            _digest(key, "token length key")
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError("token lengths must be positive integers")
        if set(all_histories) - set(token_lengths):
            raise ValueError("token count table is incomplete for query/reference/train histories")
        lengths = {key: token_lengths[key] for key in all_histories}
    elif length_measure == "utf8_bytes_diagnostic":
        if token_lengths is not None or tokenizer_fingerprint_sha256 is not None:
            raise ValueError("byte diagnostic must not masquerade as tokenizer-matched data")
        lengths = {key: len(canonical_json(history).encode("utf-8"))
                   for key, history in all_histories.items()}
    else:
        raise ValueError("unknown length_measure; explicit tokens or utf8_bytes_diagnostic required")

    donors_by_task = defaultdict(list)
    for donor in train_inputs:
        donors_by_task[(donor["lab"], donor["source_file"])].append(donor)
    eligible_by_query, failures = {}, []
    for sid, query in queries.items():
        reference = refs[sid]
        forbidden = {input_history_hashes[sid], reference_history_hashes[sid]}
        excluded = Counter()
        eligible = []
        task_donors = donors_by_task.get((query["lab"], query["source_file"]), [])
        if len(task_donors) < len(train_inputs):
            excluded["different_lab_file"] = len(train_inputs) - len(task_donors)
        for donor in task_donors:
            if not donor["history"]:
                excluded["empty_history"] += 1
            elif donor["sample_id"] == reference["provenance"]["donor_sample_id"]:
                excluded["original_matched_donor"] += 1
            elif input_history_hashes[donor["sample_id"]] in forbidden:
                excluded["same_original_history_payload"] += 1
            elif not _before(donor["current_timestamp"], query["current_timestamp"], sid):
                excluded["donor_not_strictly_past"] += 1
            elif any(not _before(entry["timestamp"], query["current_timestamp"], sid)
                     for entry in donor["history"]):
                excluded["history_not_strictly_past"] += 1
            else:
                eligible.append(donor)
        eligible_by_query[sid] = eligible
        if not eligible:
            failures.append({"sample_id": sid, "reason": "no_legal_alternative_train_history",
                             "excluded_train_records": dict(sorted(excluded.items()))})
    if failures:
        raise DonorUnavailableError(failures)

    baseline = _bundle(copy.deepcopy(inputs), copy.deepcopy(references), [], "baseline", None, protocol_sha, formal)
    variants, summaries = {}, {}
    for variant in VARIANTS:
        variants[variant], summaries[variant] = {}, {}
        for seed in seeds:
            out_inputs, out_references = copy.deepcopy(inputs), copy.deepcopy(references)
            output_queries = {row["sample_id"]: row for row in out_inputs}
            output_refs = {row["sample_id"]: row for row in out_references}
            assignments = []
            for sid, query in queries.items():
                reference = refs[sid]
                target = query["history"] if variant == "real_history_swapped" else reference["reference_history"]
                target_hash = input_history_hashes[sid] if variant == "real_history_swapped" else reference_history_hashes[sid]

                def rank(donor: dict) -> tuple[int, int, str, str]:
                    donor_hash = input_history_hashes[donor["sample_id"]]
                    return (abs(lengths[donor_hash] - lengths[target_hash]),
                            abs(len(donor["history"]) - len(target)),
                            object_hash({"seed": seed, "variant": variant, "query_sample_id": sid,
                                         "donor_sample_id": donor["sample_id"]}), donor["sample_id"])

                donor = min(eligible_by_query[sid], key=rank)
                donor_hash = input_history_hashes[donor["sample_id"]]
                assignment = {
                    "schema_version": ASSIGNMENT_VERSION, "sample_id": sid,
                    "query_student_id": query["student_id"], "variant": variant, "permutation_seed": seed,
                    "donor_sample_id": donor["sample_id"], "donor_student_id": donor["student_id"],
                    "donor_current_timestamp": donor["current_timestamp"],
                    "query_current_timestamp": query["current_timestamp"],
                    "lab": query["lab"], "source_file": query["source_file"],
                    "original_reference_donor_sample_id": reference["provenance"]["donor_sample_id"],
                    "original_history_sha256": target_hash, "donor_history_sha256": donor_hash,
                    "real_history_sha256": input_history_hashes[sid],
                    "reference_history_sha256": reference_history_hashes[sid],
                    "target_history_length": lengths[target_hash], "donor_history_length": lengths[donor_hash],
                    "absolute_length_difference": rank(donor)[0],
                    "target_history_records": len(target), "donor_history_records": len(donor["history"]),
                    "eligible_donor_count": len(eligible_by_query[sid]),
                    "length_measure": length_measure, "matching_rule": MATCHING_RULE,
                    "training_inputs_sha256": train_sha, "labels_used": False,
                    "history_definition": "complete donor input history; donor current state not appended",
                }
                assignments.append(assignment)
                if variant == "real_history_swapped":
                    output_queries[sid]["history"] = copy.deepcopy(donor["history"])
                    validate_input(output_queries[sid])
                else:
                    output_refs[sid]["reference_history"] = copy.deepcopy(donor["history"])
                    output_refs[sid]["reference_kind"] = "matched"
                    output_refs[sid]["provenance"] = {
                        "split": "train", "training_inputs_sha256": train_sha,
                        "donor_sample_id": donor["sample_id"], "donor_student_id": donor["student_id"],
                        "donor_current_timestamp": donor["current_timestamp"],
                        "query_current_timestamp": query["current_timestamp"],
                        "lab": query["lab"], "source_file": query["source_file"],
                        "matching_rule": MATCHING_RULE, "intervention": variant,
                        "permutation_seed": seed, "labels_used": False,
                        "history_length_query": len(query["history"]), "history_length_donor": len(donor["history"]),
                        "history_definition": assignment["history_definition"],
                        "original_reference_donor_sample_id": assignment["original_reference_donor_sample_id"],
                        "eligible_donor_count": assignment["eligible_donor_count"],
                    }
            bundle = _bundle(out_inputs, out_references, assignments, variant, seed, protocol_sha, formal)
            variants[variant][str(seed)] = bundle
            use = Counter(item["donor_sample_id"] for item in assignments)
            summaries[variant][str(seed)] = {
                "inputs_sha256": bundle["inputs_sha256"], "references_sha256": bundle["references_sha256"],
                "donor_assignment_sha256": bundle["donor_assignment_sha256"],
                "assigned_history_sha256": object_hash([item["donor_history_sha256"] for item in assignments]),
                "unique_donor_samples": len(use), "maximum_donor_reuse": max(use.values()),
                "donor_reuse_counts": dict(sorted(use.items())),
                "absolute_length_differences": [item["absolute_length_difference"] for item in assignments],
            }

    return {
        "schema_version": SCHEMA_VERSION, "baseline": baseline, "variants": variants,
        "manifest": {
            "schema_version": SCHEMA_VERSION, "protocol_sha256": protocol_sha,
            "training_inputs_sha256": train_sha, "training_records_sha256": object_hash(train_inputs),
            "query_records_sha256": object_hash(inputs), "reference_records_sha256": object_hash(references),
            "samples": expected_samples, "students": expected_students, "seeds": list(seeds),
            "variants": list(VARIANTS), "length_measure": length_measure, "formal_token_matching": formal,
            "token_serialization": TOKEN_SERIALIZATION if formal else None,
            "tokenizer_fingerprint_sha256": tokenizer_fingerprint_sha256,
            "token_count_table_sha256": object_hash(lengths), "matching_rule": MATCHING_RULE,
            "labels_used": False, "generation_performed": False, "partial_cohort_allowed": False,
            "history_internal_order_preserved": True,
            "assignment_summaries": summaries,
            "unique_assignments_across_seeds": {
                variant: len({summary["assigned_history_sha256"] for summary in summaries[variant].values()})
                for variant in VARIANTS
            },
            "limitations": [
                "Nearest token length of serialized history is not exact complete-prompt token matching.",
                "Task and length matching do not equate student ability, solution stage, or semantics.",
                "Donors may be reused; analyze paired effects by query student.",
                "Seeded tie-breaking can yield identical assignments, which are not independent repetitions.",
                "No score, candidate, label, future submission, or model output informs assignment.",
            ],
        },
    }
