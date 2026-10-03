"""Synthetic fixed-cache fixtures: no model, GPU, server or student execution."""

import copy
from dataclasses import asdict
import hashlib
import json
import math
import sys
import unittest
from unittest.mock import patch

from student_sim_cd import inference, intervention_analysis as analysis, scoring
from tests.test_inference import sample


def jsonl(rows):
    return "".join(inference.canonical_json(row) + "\n" for row in rows).encode("utf-8")


def sha(content):
    return hashlib.sha256(content).hexdigest()


def signed(record, run_hash):
    record = dict(record, run_sha256=run_hash)
    record["record_sha256"] = inference.object_hash(record)
    return record


class InterventionAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.inputs = []
        for index, name in enumerate(("a", "b", "c", "d")):
            row = sample()
            row.update(sample_id=name, student_id="student1" if index < 3 else "student2")
            self.inputs.append(row)
        self.references = [{"sample_id": row["sample_id"], "reference_kind": "matched",
                            "reference_history": [dict(copy.deepcopy(row["history"][0]), code="print(8)\n")],
                            "provenance": {"split": "train", "training_inputs_sha256": "5" * 64,
                                "donor_sample_id": "original-ref-" + row["sample_id"],
                                "donor_student_id": "train-original-" + row["sample_id"],
                                "donor_current_timestamp": "2020-01-01T10:01:00", "query_current_timestamp": row["current_timestamp"],
                                "lab": row["lab"], "source_file": row["source_file"], "labels_used": False}} for row in self.inputs]
        self.labels = [{"sample_id": row["sample_id"], "student_id": row["student_id"],
                        "target_code": "print(1)\n"} for row in self.inputs]
        self.config = asdict(inference.InferenceConfig(model_path="synthetic-model", num_generations=4))
        completion = {"schema_version": "student-sim-cd.cached-experiment.v1", "status": "success", "exit_code": 0,
            "source_root": "/synthetic/release", "generation_performed": False, "student_execution": False,
            "stages": [{"name": name, "status": "success", "exit_code": 0} for name in analysis.COMPLETION_STAGES]}
        self.completion = jsonl([completion])
        self.completion_sha = sha(self.completion)
        self.protocol_sha = "1" * 64
        self.baseline = self.bundle(self.inputs, self.references, {"a", "b", "c"})
        self.baseline["completion_manifest"] = self.completion
        self.baseline["success"] = (self.completion_sha + "\n").encode("ascii")
        self.variants = {variant: {} for variant in analysis.VARIANTS}
        for variant in analysis.VARIANTS:
            for index, permutation_seed in enumerate(analysis.DEFAULT_PERMUTATION_SEEDS):
                inputs, refs = copy.deepcopy(self.inputs), copy.deepcopy(self.references)
                if variant == "real_history_swapped":
                    for row in inputs:
                        row["history"][0]["code"] = "print(" + str(20 + index) + ")\n"
                    selected = {"a", "b", "c"} if index == 2 else set()
                else:
                    for row in refs:
                        row["reference_history"][0]["code"] = "print(" + str(30 + index) + ")\n"
                    selected = {"a", "b", "c"}
                assigned_rows = [self.assignment(row, ref, inputs[position]["history"] if variant == "real_history_swapped" else refs[position]["reference_history"], variant, permutation_seed)
                                 for position, (row, ref) in enumerate(zip(self.inputs, self.references))]
                if variant == "reference_swapped":
                    for ref, assignment, original in zip(refs, assigned_rows, self.inputs):
                        ref["provenance"] = {"split": "train", "intervention": variant, "permutation_seed": permutation_seed,
                            "labels_used": False, "training_inputs_sha256": assignment["training_inputs_sha256"],
                            "donor_sample_id": assignment["donor_sample_id"], "donor_student_id": assignment["donor_student_id"],
                            "donor_current_timestamp": assignment["donor_current_timestamp"],
                            "query_current_timestamp": original["current_timestamp"], "lab": original["lab"], "source_file": original["source_file"],
                            "matching_rule": assignment["matching_rule"], "history_length_query": len(original["history"]),
                            "history_length_donor": len(ref["reference_history"]), "history_definition": assignment["history_definition"],
                            "original_reference_donor_sample_id": assignment["original_reference_donor_sample_id"],
                            "eligible_donor_count": assignment["eligible_donor_count"]}
                assignments = jsonl(assigned_rows)
                metadata = {"variant": variant, "permutation_seed": permutation_seed,
                    "protocol_sha256": self.protocol_sha, "donor_assignment_sha256": sha(assignments),
                    "baseline_manifest_sha256": self.completion_sha,
                    "history_data_sha256": sha(jsonl(inputs) if variant == "real_history_swapped" else jsonl(refs))}
                bundle = self.bundle(inputs, refs, selected, metadata)
                bundle["assignments"] = assignments
                self.variants[variant][permutation_seed] = bundle
        self.kwargs = {"protocol_sha256": self.protocol_sha, "expected_samples": 4, "expected_students": 2,
                       "expected_candidates": 8, "bootstrap": 200, "seed": 20260929,
                       "baseline_completion_sha256": self.completion_sha}

    def assignment(self, original, reference, history, variant, permutation_seed):
        target = original["history"] if variant == "real_history_swapped" else reference["reference_history"]
        return {"schema_version": analysis.history_interventions.ASSIGNMENT_VERSION,
            "sample_id": original["sample_id"], "query_student_id": original["student_id"],
            "variant": variant, "permutation_seed": permutation_seed,
            "donor_sample_id": "train-donor-" + original["sample_id"], "donor_student_id": "train-student-" + original["sample_id"],
            "donor_current_timestamp": "2020-01-01T10:01:00", "query_current_timestamp": original["current_timestamp"],
            "lab": original["lab"], "source_file": original["source_file"],
            "original_reference_donor_sample_id": reference["provenance"]["donor_sample_id"],
            "original_history_sha256": inference.object_hash(target), "donor_history_sha256": inference.object_hash(history),
            "real_history_sha256": inference.object_hash(original["history"]),
            "reference_history_sha256": inference.object_hash(reference["reference_history"]),
            "target_history_length": 10, "donor_history_length": 11, "absolute_length_difference": 1,
            "target_history_records": len(target), "donor_history_records": len(history), "eligible_donor_count": 3,
            "length_measure": "tokens", "matching_rule": analysis.history_interventions.MATCHING_RULE,
            "training_inputs_sha256": "5" * 64, "labels_used": False,
            "history_definition": "complete donor input history; donor current state not appended"}

    def bundle(self, inputs, references, b_alt_samples, metadata=None):
        inputs_bytes, references_bytes = jsonl(inputs), jsonl(references)
        manifest = {"schema_version": inference.SCHEMA_VERSION, "prompt_version": inference.PROMPT_VERSION,
            "config": self.config, "config_sha256": inference.object_hash(self.config),
            "inputs_sha256": sha(inputs_bytes), "references_sha256": sha(references_bytes),
            "implementation_sha256": "2" * 64, "scoring_sha256": "3" * 64,
            "sequence_protocol": "synthetic code + EOS sumlogp", "model": {"sha256": "4" * 64},
            "runtime_versions": {"torch": "synthetic-not-installed"},
            "cache_rescore": {"backend": "injected_synthetic_backend", "generation_performed": False, "old_scores_reused": False}}
        if metadata:
            manifest["history_intervention"] = metadata
        run_hash = inference.object_hash(manifest)
        candidates, scores = [], []
        predictions = {method: [] for method in analysis.METHODS}
        for row in inputs:
            sample_id = row["sample_id"]
            candidate_list = []
            for code, origins in ((row["current_code"], ["copy_current"]), ("print(1)\n", ["greedy:0", "sampled:1", "sampled:2", "sampled:3"])):
                identifier = sha(code.encode("utf-8"))
                tokens = list(code.encode("utf-8")) + [256]
                candidate_list.append({"candidate_id": identifier, "code": code, "sources": origins,
                                       "transform": "none", "completion_token_ids": tokens,
                                       "token_count": len(tokens), "eos_included": True})
                raw = scoring.LogProbs(-1, -1, -1, -1) if "copy_current" in origins else (
                    scoring.LogProbs(-2, -6, -4, -4) if sample_id in b_alt_samples else scoring.LogProbs(-2, -2, -2, -2))
                scores.append(signed({"sample_id": sample_id, "candidate_id": identifier, **asdict(raw),
                    "token_count": len(tokens), "eos_included": True, "prompt_token_counts": {name: 1 for name in inference.CONDITIONS},
                    "reference_kind": "matched", "components": scoring.components(raw), "scores_weight_1": scoring.method_scores(raw)}, run_hash))
            attempts = [{"attempt": index, "source": "greedy" if index == 0 else "sampled", "seed": index,
                         "raw_text": "print(1)\n", "eos_reached": True, "finish_reason": "eos",
                         "generated_token_ids": list(b"print(1)\n") + [256], "candidate_id": sha(b"print(1)\n"), "transform": "none"} for index in range(4)]
            candidates.append(signed({"sample_id": sample_id, "candidates": candidate_list, "attempts": attempts,
                "reference_kind": "matched", "reference_provenance": {"split": "train", "rule": "synthetic"}}, run_hash))
            for method in analysis.METHODS:
                alt = method in {"b", "cd"} and sample_id in b_alt_samples
                predictions[method].append({"sample_id": sample_id, "method": method,
                    "predicted_code": "print(1)\n" if alt else row["current_code"]})
        return {"inputs": inputs_bytes, "references": references_bytes, "manifest": jsonl([manifest]),
                "candidates": jsonl(candidates), "scores": jsonl(scores),
                "predictions": {method: jsonl(rows) for method, rows in predictions.items()}}

    def report(self, baseline=None, variants=None, **kwargs):
        return analysis.analyze_interventions(jsonl(self.inputs), jsonl(self.labels),
            self.baseline if baseline is None else baseline, self.variants if variants is None else variants,
            **{**self.kwargs, **kwargs})

    def mutate_records(self, artifact, field, transform):
        rows = [json.loads(line) for line in artifact[field].decode().splitlines()]
        transform(rows)
        for row in rows:
            if "record_sha256" in row:
                row["record_sha256"] = inference.object_hash({key: value for key, value in row.items() if key != "record_sha256"})
        artifact[field] = jsonl(rows)

    def mutate_manifest(self, artifact, transform):
        manifest = json.loads(artifact["manifest"])
        transform(manifest)
        if "config" in manifest:
            manifest["config_sha256"] = inference.object_hash(manifest["config"])
        run_hash = inference.object_hash(manifest)
        artifact["manifest"] = jsonl([manifest])
        for field in ("candidates", "scores"):
            self.mutate_records(artifact, field, lambda rows: [row.update(run_sha256=run_hash) for row in rows])

    def test_seed_average_then_equal_student_difference_in_differences(self):
        report = self.report()
        self.assertAlmostEqual(report["primary"]["student_macro_delta"], 1 / 3)
        self.assertAlmostEqual(report["primary"]["submission_mean_delta"], 0.5)
        ci = report["primary"]["paired_student_bootstrap_95_ci"]
        self.assertAlmostEqual(ci[0], 0)
        self.assertAlmostEqual(ci[1], 2 / 3)
        self.assertEqual(report["secondary"]["metrics"]["edit_location_f1"]["student_macro_delta"], 0)
        students = report["all"]["difference_in_differences"]["real_history_swapped"]["per_student"]
        self.assertEqual(len(students), 2)
        self.assertEqual(students[0]["samples"], 3)
        self.assertAlmostEqual(students[0]["seed_mean_b_minus_base"]["edit_location_f1"], 1 / 3)
        self.assertFalse(report["model_scoring_performed"])
        self.assertTrue(report["analysis_kind"].startswith("synthetic_"))
        self.assertFalse(report["generation_performed"])
        json.dumps(report, allow_nan=False)

    def test_diagnostics_and_all_five_controls_are_retained(self):
        report = self.report()
        self.assertEqual(set(report["all"]["versions"]["baseline"]), set(analysis.METHODS))
        metrics = report["all"]["difference_in_differences"]["real_history_swapped"]["metrics"]
        for name in ("predicted_edit_size", "true_edit_size", "edit_size_absolute_error", "predicted_unchanged", "predicted_changed"):
            self.assertIn(name, metrics)
        self.assertEqual(metrics["predicted_unchanged"]["direction"], "diagnostic_only")
        self.assertEqual(report["subgroups"]["true_unchanged"]["samples"], 0)
        self.assertEqual(report["expected_students"], 2)
        self.assertEqual(report["versions"], 7)
        self.assertIn("not independent", report["seed_aggregation"])

    def test_identical_paired_effect_has_zero_width_interval(self):
        baseline = self.bundle(self.inputs, self.references, {"a", "b", "c", "d"})
        baseline.update(completion_manifest=self.completion, success=self.baseline["success"])
        variants = copy.deepcopy(self.variants)
        for value in variants["real_history_swapped"].values():
            self.mutate_records(value, "scores", lambda rows: [self.force_b_copy(row) for row in rows])
            for method in ("b", "cd"):
                value["predictions"][method] = jsonl([{ "sample_id": row["sample_id"], "method": method, "predicted_code": row["current_code"]} for row in self.inputs])
        primary = self.report(baseline=baseline, variants=variants)["primary"]
        self.assertEqual(primary["student_macro_delta"], 1)
        self.assertEqual(primary["paired_student_bootstrap_95_ci"], [1, 1])

    @staticmethod
    def force_b_copy(row):
        if row["candidate_id"] == sha(b"print(1)\n"):
            raw = scoring.LogProbs(-2, -2, -2, -2)
            row.update(**asdict(raw), components=scoring.components(raw), scores_weight_1=scoring.method_scores(raw))

    def test_seed_keys_and_input_order_are_reproducible(self):
        variants = {variant: {str(seed): bundle for seed, bundle in reversed(list(values.items()))} for variant, values in reversed(list(self.variants.items()))}
        self.assertEqual(self.report(), self.report(variants=variants))
        self.assertEqual(self.report(), self.report())

    def test_missing_extra_or_duplicate_seed_rejects_without_drop(self):
        for operation in ("missing", "extra", "duplicate"):
            variants = copy.deepcopy(self.variants)
            values = variants["real_history_swapped"]
            if operation == "missing":
                values.pop(20261002)
            elif operation == "extra":
                values[9] = values[20261002]
            else:
                values["20261002"] = values[20261002]
            with self.subTest(operation=operation), self.assertRaisesRegex(ValueError, "seed"):
                self.report(variants=variants)

    def test_missing_variant_or_prediction_method_rejects(self):
        with self.assertRaisesRegex(ValueError, "both history"):
            self.report(variants={"real_history_swapped": self.variants["real_history_swapped"]})
        variants = copy.deepcopy(self.variants)
        del variants["reference_swapped"][20261002]["predictions"]["history_d0"]
        with self.assertRaisesRegex(ValueError, "five fixed"):
            self.report(variants=variants)

    def test_missing_sample_score_or_assignment_rejects(self):
        for field in ("candidates", "scores", "assignments"):
            variants = copy.deepcopy(self.variants)
            bundle = variants["real_history_swapped"][20261002]
            self.mutate_records(bundle, field, lambda rows: rows.pop())
            if field == "assignments":
                self.mutate_manifest(bundle, lambda m: m["history_intervention"].update(donor_assignment_sha256=sha(bundle["assignments"])))
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.report(variants=variants)

    def test_nonfinite_positive_or_forged_derived_score_rejects(self):
        for value in (math.nan, math.inf, -math.inf, 1, True):
            variants = copy.deepcopy(self.variants)
            artifact = variants["real_history_swapped"][20261002]
            rows = [json.loads(line) for line in artifact["scores"].decode().splitlines()]
            rows[0]["l11"] = value
            artifact["scores"] = (json.dumps(rows[0],allow_nan=True) + "\n").encode() + jsonl(rows[1:])
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.report(variants=variants)
        variants = copy.deepcopy(self.variants)
        self.mutate_records(variants["real_history_swapped"][20261002], "scores", lambda rows: rows[0]["scores_weight_1"].update(b=100))
        with self.assertRaisesRegex(ValueError, "four raw"):
            self.report(variants=variants)

    def test_prediction_argmax_or_method_mismatch_rejects(self):
        for field, value in (("method", "base"), ("predicted_code", "print(1)\n")):
            variants = copy.deepcopy(self.variants)
            artifact = variants["real_history_swapped"][20261002]
            rows = [json.loads(line) for line in artifact["predictions"]["b"].decode().splitlines()]
            rows[0][field] = value
            artifact["predictions"]["b"] = jsonl(rows)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.report(variants=variants)

    def test_metadata_hashes_and_identity_cannot_change(self):
        for field, value in (("variant", "reference_swapped"), ("permutation_seed", 99),
                             ("protocol_sha256", "9" * 64), ("donor_assignment_sha256", "9" * 64),
                             ("baseline_manifest_sha256", "9" * 64), ("history_data_sha256", "9" * 64)):
            variants = copy.deepcopy(self.variants)
            self.mutate_manifest(variants["real_history_swapped"][20261002], lambda m: m["history_intervention"].update({field: value}))
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.report(variants=variants)

    def test_assignment_origin_and_temporal_semantics_not_just_file_hash(self):
        for field, value in (("variant", "reference_swapped"), ("permutation_seed", 9),
                             ("training_inputs_sha256", "9" * 64), ("query_student_id", "wrong-query"),
                             ("donor_student_id", "student1"), ("donor_student_id", "student2"),
                             ("donor_current_timestamp", "2020-01-01T10:02:00"),
                             ("donor_history_sha256", "9" * 64), ("labels_used", True),
                             ("original_reference_donor_sample_id", "wrong-original"),
                             ("donor_sample_id", "original-ref-a"), ("length_measure", "bytes"),
                             ("donor_history_records", 2), ("eligible_donor_count", 0)):
            variants = copy.deepcopy(self.variants)
            artifact = variants["real_history_swapped"][20261002]
            self.mutate_records(artifact, "assignments", lambda rows: rows[0].update({field: value}))
            self.mutate_manifest(artifact, lambda m: m["history_intervention"].update(donor_assignment_sha256=sha(artifact["assignments"])))
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.report(variants=variants)

    def test_reference_future_history_is_rejected_with_coherent_hashes(self):
        variants = copy.deepcopy(self.variants)
        artifact = variants["reference_swapped"][20261002]
        self.mutate_records(artifact, "references", lambda rows: rows[0]["reference_history"][0].update(timestamp=self.inputs[0]["current_timestamp"]))
        refs = [json.loads(line) for line in artifact["references"].decode().splitlines()]
        self.mutate_records(artifact, "assignments", lambda rows: rows[0].update(donor_history_sha256=inference.object_hash(refs[0]["reference_history"])))
        def bind(manifest):
            manifest["references_sha256"] = sha(artifact["references"])
            manifest["history_intervention"].update(history_data_sha256=sha(artifact["references"]), donor_assignment_sha256=sha(artifact["assignments"]))
        self.mutate_manifest(artifact, bind)
        with self.assertRaisesRegex(ValueError, "Every intervened history timestamp"):
            self.report(variants=variants)

    def test_reference_kind_and_provenance_origin_are_frozen(self):
        for field, value in (("reference_kind", "shuffled"), ("provenance", {"split": "train", "labels_used": False})):
            variants = copy.deepcopy(self.variants)
            artifact = variants["reference_swapped"][20261002]
            self.mutate_records(artifact, "references", lambda rows: rows[0].update({field: value}))
            def bind(manifest):
                manifest["references_sha256"] = sha(artifact["references"])
                manifest["history_intervention"]["history_data_sha256"] = sha(artifact["references"])
            self.mutate_manifest(artifact, bind)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "Reference intervention|reference provenance"):
                self.report(variants=variants)

    def test_input_context_and_candidate_tokens_are_frozen(self):
        variants = copy.deepcopy(self.variants)
        artifact = variants["real_history_swapped"][20261002]
        self.mutate_records(artifact, "inputs", lambda rows: rows[0].update(problem_statement="changed problem"))
        def bind_changed_inputs(manifest):
            manifest.update(inputs_sha256=sha(artifact["inputs"]))
            manifest["history_intervention"].update(history_data_sha256=sha(artifact["inputs"]))
        self.mutate_manifest(artifact, bind_changed_inputs)
        with self.assertRaisesRegex(ValueError, "current/problem"):
            self.report(variants=variants)
        variants = copy.deepcopy(self.variants)
        artifact = variants["real_history_swapped"][20261002]
        self.mutate_records(artifact, "candidates", lambda rows: rows[0]["candidates"][1]["completion_token_ids"].__setitem__(0, 42))
        with self.assertRaisesRegex(ValueError, "Candidates/tokens"):
            self.report(variants=variants)

    def test_backend_or_model_or_raw_attempt_mismatch_rejects(self):
        for mutation in (
            lambda m: m["cache_rescore"].update(backend="unknown"),
            lambda m: m["model"].update(sha256="9" * 64),
            lambda m: m["config"].update(seed=77),
        ):
            variants = copy.deepcopy(self.variants)
            self.mutate_manifest(variants["real_history_swapped"][20261002], mutation)
            with self.assertRaises(ValueError):
                self.report(variants=variants)
        variants = copy.deepcopy(self.variants)
        self.mutate_records(variants["real_history_swapped"][20261002], "candidates", lambda rows: rows[0]["attempts"].pop())
        with self.assertRaisesRegex(ValueError, "raw generation"):
            self.report(variants=variants)

    def test_baseline_hard_anchor_and_success_are_independently_required(self):
        for field, value in (("success", b"wrong\n"), ("completion_manifest", b"{}\n")):
            baseline = copy.deepcopy(self.baseline)
            baseline[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "hard anchor"):
                self.report(baseline=baseline)

    def test_labels_are_not_passed_to_scoring_validator(self):
        from student_sim_cd import predict
        validator = predict._validated_run
        passed = []
        def checked(sources):
            self.assertEqual(set(sources), {"inputs", "manifest", "candidates", "scores"})
            self.assertNotIn(b'"target_code"', sources["inputs"])
            passed.append(True)
            return validator(sources)
        with patch.object(predict, "_validated_run", side_effect=checked):
            self.report()
        self.assertEqual(len(passed), 7)

    def test_future_label_inside_variant_input_rejects(self):
        variants = copy.deepcopy(self.variants)
        artifact = variants["real_history_swapped"][20261002]
        self.mutate_records(artifact, "inputs", lambda rows: rows[0]["history"][0].update(target_code="secret"))
        self.mutate_manifest(artifact, lambda m: m.update(inputs_sha256=sha(artifact["inputs"])))
        with self.assertRaisesRegex(ValueError, "target_code"):
            self.report(variants=variants)

    def test_fixed_actual_greedy_is_external_and_never_intervention_generation(self):
        greedy = jsonl([{"sample_id": row["sample_id"], "method": "greedy", "predicted_code": "print(1)\n"} for row in self.inputs])
        report = self.report(fixed_external_greedy=greedy)
        self.assertFalse(report["fixed_external_greedy"]["generated_under_interventions"])
        self.assertFalse(report["fixed_external_greedy"]["used_in_primary_contrast"])
        self.assertNotIn("greedy", report["all"]["versions"]["baseline"])
        self.assertEqual(report["primary"], self.report()["primary"])
        rows = [json.loads(line) for line in greedy.decode().splitlines()]
        rows[0]["predicted_code"] = "print(0)\n"
        with self.assertRaisesRegex(ValueError, "actual original"):
            self.report(fixed_external_greedy=jsonl(rows))

    def test_counts_and_fixed_seed_parameters_are_not_relaxed(self):
        for kwargs in ({"expected_samples": 70, "expected_students": 17}, {"expected_candidates": 318},
                       {"expected_seeds": (1, 2)}, {"expected_seeds": (1, 1, 2)}, {"bootstrap": True},
                       {"seed": True}, {"protocol_sha256": "unknown"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.report(**kwargs)

    def test_no_torch_import_or_student_code_execution(self):
        before = set(sys.modules)
        report = self.report()
        self.assertNotIn("torch", set(sys.modules) - before)
        self.assertFalse(report["student_execution"])


if __name__ == "__main__":
    unittest.main()
