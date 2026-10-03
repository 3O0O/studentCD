"""Synthetic two-pool/static-analysis tests; never load or execute a model/code."""

import copy
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from student_sim_cd import candidate_support_analysis as analysis, inference, predict, scoring
from tests.test_inference import sample


def jsonl(rows):
    return "".join(inference.canonical_json(row) + "\n" for row in rows).encode("utf-8")


def sha(content):
    return hashlib.sha256(content).hexdigest()


def signed(record, run_hash):
    record = dict(record, run_sha256=run_hash)
    record["record_sha256"] = inference.object_hash(record)
    return record


CURRENT = "a = 0\nb = 0\nc = 0\n"
GOOD = "a = 1\nb = 0\nc = 0\n"
BAD = "a = 0\nb = 0\nc = 2\n"


class SupportFixture:
    def __init__(self, names=("a", "b", "c", "d"), students=None, protocol_sha="1" * 64):
        self.protocol_sha = protocol_sha
        students = students or ["student1" if i < 3 else "student2" for i in range(len(names))]
        self.inputs = []
        for name, student in zip(names, students):
            item = sample()
            item.update(sample_id=name, student_id=student, current_code=CURRENT)
            self.inputs.append(item)
        self.labels = [{"sample_id": name, "target_code": CURRENT if name == "c" else GOOD} for name in names]
        self.input_bytes = jsonl(self.inputs)
        self.artifacts = {pool: self.artifact(pool) for pool in analysis.POOLS}
        self.external = {}
        for method in analysis.EXTERNAL_METHODS:
            content = jsonl([{"sample_id": name, "method": method, "predicted_code": GOOD} for name in names])
            self.external[method] = analysis.ExternalPredictionArtifact(content, {
                "kind": "prompt_greedy" if method.endswith("_greedy") else "prompt_pool_reranking",
                "prompt_name": "student_revision" if method.startswith("student_revision_") else "conservative_edit",
                "protocol_sha256": protocol_sha, "prediction_sha256": sha(content),
                "generation_manifest_sha256": "9" * 64, "backend": "injected_synthetic_backend",
                "generation_count_per_sample": 13, "random_count_per_sample": 12,
                "labels_used": False, "parameters_fitted": False, "student_execution": False,
                "source_sha256": {"selected_predictions": sha(content)},
            })

    def artifact(self, pool):
        config = asdict(inference.InferenceConfig(model_path="synthetic-model", num_generations=4))
        manifest = {
            "schema_version": inference.SCHEMA_VERSION, "prompt_version": inference.PROMPT_VERSION,
            "config": config, "config_sha256": inference.object_hash(config),
            "inputs_sha256": sha(self.input_bytes), "references_sha256": "8" * 64,
            "implementation_sha256": "2" * 64, "scoring_sha256": "3" * 64,
            "sequence_protocol": analysis.SEQUENCE_PROTOCOL, "model": {"sha256": "4" * 64},
            "runtime_versions": {"backend": "synthetic-not-installed"},
            "candidate_support": {
                "pool": pool, "protocol_sha256": self.protocol_sha,
                "baseline_manifest_sha256": "5" * 64, "generation_manifest_sha256": "6" * 64,
                "pool_random_count": 12, "shared_greedy_condition": "11", "shared_copy": True,
                "backend": "injected_synthetic_backend", "generation_performed": True,
                "student_execution": False, "labels_read": False, "scoring_performed": True,
                "baseline_scores_reused": True, "baseline_score_records": 318,
            },
        }
        run_hash = inference.object_hash(manifest)
        candidates, scores = [], []
        for item in self.inputs:
            sid = item["sample_id"]
            by_code = {CURRENT: {"copy_current"}}
            wrappers = []
            for draw in range(13):
                condition = "11" if pool == "11_only12" or draw <= 3 else "01" if draw <= 6 else "10" if draw <= 9 else "00"
                if draw == 0:
                    code = BAD
                elif draw <= 3:
                    code = "a = 0\nb = 0\nc = " + str(draw + 2) + "\n"
                elif pool == "balanced12":
                    code = "a = " + str(draw - 3) + "\nb = 0\nc = 0\n"
                else:
                    code = "a = 0\nb = " + str(draw) + "\nc = 0\n"
                origin = "greedy:11:0" if draw == 0 else "sampled:" + condition + ":" + str(draw)
                by_code.setdefault(code, set()).add(origin)
                wrappers.append({
                    "sample_id": sid, "condition": condition, "draw_id": draw,
                    "slot": draw, "origin": "reused" if draw <= 3 else "new",
                    "baseline_record_sha256": "7" * 64 if draw <= 3 else None,
                    "prompt_tokens_sha256": "8" * 64,
                    "raw_generation": {"attempt": draw, "source": "greedy" if draw == 0 else "sampled",
                        "seed": 100 + draw, "raw_text": code, "generated_token_ids": list(code.encode()) + [256],
                        "eos_reached": True, "finish_reason": "eos"},
                })
            candidate_list = []
            for code, origins in sorted(by_code.items()):
                identifier = sha(code.encode())
                tokens = list(code.encode()) + [256]
                candidate_list.append({"candidate_id": identifier, "code": code, "sources": sorted(origins),
                    "transform": "none", "completion_token_ids": tokens, "token_count": len(tokens), "eos_included": True})
                raw = scoring.LogProbs(-1, -1, -1, -1) if code == BAD else scoring.LogProbs(-30, -30, -30, -30)
                if code == CURRENT:
                    raw = scoring.LogProbs(-4, -4, -4, -4)
                if code == GOOD and sid in {"a", "b", "c"}:
                    raw = scoring.LogProbs(-2, -20, -2, -2)
                scores.append(signed({"sample_id": sid, "candidate_id": identifier,
                    **asdict(raw), "token_count": len(tokens), "eos_included": True,
                    "prompt_token_counts": {"11": 100, "01": 90, "10": 80, "00": 70},
                    "reference_kind": "matched", "components": scoring.components(raw),
                    "scores_weight_1": scoring.method_scores(raw)}, run_hash))
            candidates.append(signed({"sample_id": sid, "candidates": candidate_list, "attempts": wrappers,
                "reference_kind": "matched", "reference_provenance": {"split": "train", "rule": "synthetic"}}, run_hash))
        return analysis.SupportArtifact(jsonl([manifest]), jsonl(candidates), jsonl(scores))

    def report(self, **overrides):
        options = {"protocol_sha256": self.protocol_sha, "expected_samples": len(self.inputs),
            "expected_students": len({row["student_id"] for row in self.inputs}), "bootstrap": 20}
        arguments = {"inputs": self.input_bytes, "labels": jsonl(self.labels),
            "balanced": self.artifacts["balanced12"], "control": self.artifacts["11_only12"],
            "external_predictions": self.external, **options}
        arguments.update(overrides)
        return analysis.analyze_support(**arguments)

    def frozen_protocol(self):
        return {
            "schema_version": analysis.PROTOCOL_VERSION, "protocol": "candidate-support-v1",
            "expected_samples": 70, "expected_students": 17, "seed": analysis.BOOTSTRAP_SEED,
            "conditions": list(inference.CONDITIONS), "balanced_random_per_condition": 3,
            "control_random_count": 12, "shared_greedy_condition": "11",
            "canonical_protocol": "cached-python-fence-v1", "generation_performed": True,
            "student_execution": False, "test_set_used": False, "parameters_fitted": False,
            "new_support_attempts": 1260, "new_prompt_baseline_attempts": 1820,
            "max_new_generation_attempts": 3080, "prompt_baselines": ["student_revision", "conservative_edit"],
            "prompt_baseline_random_count": 12, "prompt_baseline_greedy_count": 1,
            "fixed_method_parameters": predict.METHOD_PARAMETERS, "methods": list(analysis.METHODS),
            "sequence_protocol": analysis.SEQUENCE_PROTOCOL, "selection": analysis.SELECTION,
            "primary_metric": "equal-student macro edit_location_f1",
            "primary_comparison": "(B-base)_balanced12 - (B-base)_11_only12",
            "bootstrap": {"unit": "student", "paired": True, "repetitions": 2000, "seed": analysis.BOOTSTRAP_SEED},
            "baseline_release_sha256": "3" * 64,
            "baseline_sha256": {"manifest.json": "5" * 64, "SUCCESS": "6" * 64,
                "prepared/inputs.jsonl": sha(self.input_bytes), "prepared/references.jsonl": "8" * 64,
                "prepared/inference/manifest.json": "0" * 64,
                "prepared/inference/candidates.jsonl": "0" * 64, "prepared/inference/scores.jsonl": "0" * 64},
            "labels_sha256": sha(jsonl(self.labels)),
            "generation_config": json.loads(self.artifacts["balanced12"].manifest)["config"],
        }

    @staticmethod
    def mutate(artifact, field, operation):
        values = {name: getattr(artifact, name) for name in ("manifest", "candidates", "scores")}
        records = [json.loads(line) for line in values[field].decode().split("\n") if line]
        operation(records)
        for row in records:
            if "record_sha256" in row:
                row["record_sha256"] = inference.object_hash({key: value for key, value in row.items() if key != "record_sha256"})
        values[field] = jsonl(records)
        if field == "manifest":
            run_hash = inference.object_hash(records[0])
            for record_field in ("candidates", "scores"):
                values[record_field] = SupportFixture.mutate(analysis.SupportArtifact(**values), record_field,
                    lambda rows: [row.update(run_sha256=run_hash) for row in rows]).__getattribute__(record_field)
        return analysis.SupportArtifact(**values)


class CandidateSupportAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.fixture = SupportFixture()

    def test_independent_non_nested_pools_and_all_declared_controls(self):
        result = self.fixture.report()
        self.assertEqual(result["samples"], 4)
        self.assertEqual(result["students"], 2)
        self.assertTrue(result["cohort_complete"])
        self.assertEqual(result["samples_dropped"], 0)
        self.assertFalse(result["model_scoring_performed"])
        self.assertFalse(result["generation_stage_performed"])
        self.assertTrue(result["analysis_kind"].startswith("synthetic_"))
        self.assertEqual(result["pool_candidate_counts"], {"balanced12": 56, "11_only12": 56})
        self.assertEqual(set(result["all"]["balanced12"]), set(analysis.METHODS))
        self.assertEqual(set(result["all"]["external"]), set(analysis.EXTERNAL_METHODS) | {"original_greedy"})
        self.assertIn("reranking", result["method_kinds"]["base"])
        self.assertEqual(result["method_kinds"]["original_greedy"], "original_actual_condition11_greedy_draw0")
        for row in result["pool_overlap"]:
            self.assertEqual(len(row["shared_candidate_ids"]), 5)
            self.assertEqual(row["balanced_only_candidates"], 9)
            self.assertEqual(row["control_only_candidates"], 9)
        for row in result["source_coverage"]["balanced12"]:
            self.assertEqual([row["conditions"][condition]["random_attempts"] for condition in inference.CONDITIONS], [3, 3, 3, 3])
        for row in result["source_coverage"]["11_only12"]:
            self.assertEqual([row["conditions"][condition]["random_attempts"] for condition in inference.CONDITIONS], [12, 0, 0, 0])

    def test_student_paired_interaction_and_2000_student_bootstrap(self):
        result = self.fixture.report(bootstrap=2000)
        primary = result["primary"]
        self.assertAlmostEqual(primary["student_macro_delta"], 1 / 3)
        self.assertAlmostEqual(primary["submission_delta"], 0.5)
        self.assertEqual(primary["paired_student_bootstrap_95_ci"], [0, 2 / 3])
        self.assertEqual(primary["student_delta_sign_counts"], {"positive": 1, "tie": 1, "negative": 0})
        self.assertEqual(result["pool_b_minus_base"]["11_only12"]["metrics"]["edit_location_f1"]["student_macro_delta"], 0)
        self.assertEqual(result["bootstrap_repetitions"], 2000)
        self.assertEqual(result["bootstrap_seed"], 20260929)
        self.assertEqual(result, self.fixture.report(bootstrap=2000))

    def test_edit_amount_and_unchanged_subgroup_denominators(self):
        result = self.fixture.report()
        changed = result["subgroups"]["balanced12"]["b"]["true_changed"]
        unchanged = result["subgroups"]["balanced12"]["b"]["true_unchanged"]
        self.assertEqual((changed["samples"], unchanged["samples"]), (3, 1))
        self.assertEqual(unchanged["metrics"]["false_edit_on_unchanged"]["submission_mean"], 1)
        self.assertEqual(result["all"]["balanced12"]["b"]["metrics"]["false_edit_on_unchanged"]["submission_mean"], 0.25)
        self.assertEqual(result["all"]["balanced12"]["copy"]["metrics"]["predicted_edit_size"]["student_macro_mean"], 0)
        self.assertIn("edit_size_absolute_error", changed["metrics"])
        self.assertIn("text_similarity", changed["metrics"])

    def test_oracle_is_diagnostic_and_never_changes_selection(self):
        before = self.fixture.report()
        labels = jsonl([{"sample_id": row["sample_id"], "target_code": CURRENT} for row in self.fixture.inputs])
        after = self.fixture.report(labels=labels)
        self.assertEqual(before["selections"], after["selections"])
        self.assertEqual(before["external_selections"], after["external_selections"])
        self.assertNotEqual(before["primary"], after["primary"])
        for pool in analysis.POOLS:
            self.assertFalse(before["label_oracle_diagnostics"][pool]["feeds_back_into_selection"])
            self.assertEqual(before["label_oracle_diagnostics"][pool]["analysis_kind"], "label_oracle_posthoc_diagnostic_not_method")
        self.assertEqual(before["label_oracle_diagnostics"]["balanced12"]["exact_observed_next_code_coverage_count"], 4)
        self.assertEqual(before["label_oracle_diagnostics"]["balanced12"]["generated_exact_observed_next_code_coverage_count"], 3)
        self.assertEqual(before["label_oracle_diagnostics"]["11_only12"]["exact_observed_next_code_coverage_count"], 1)
        self.assertEqual(before["label_oracle_diagnostics"]["11_only12"]["generated_exact_observed_next_code_coverage_count"], 0)

    def test_ties_use_smallest_content_identifier(self):
        def tie(rows):
            raw = scoring.LogProbs(-1, -1, -1, -1)
            for row in rows:
                row.update(**asdict(raw), components=scoring.components(raw), scores_weight_1=scoring.method_scores(raw))
        balanced = SupportFixture.mutate(self.fixture.artifacts["balanced12"], "scores", tie)
        control = SupportFixture.mutate(self.fixture.artifacts["11_only12"], "scores", tie)
        result = self.fixture.report(balanced=balanced, control=control)
        for pool, artifact in (("balanced12", balanced), ("11_only12", control)):
            records = json.loads(artifact.candidates.decode().split("\n")[0])
            identifier = min(candidate["candidate_id"] for candidate in records["candidates"])
            selection = next(row for row in result["selections"] if row["pool"] == pool)
            for method in predict.METHOD_PARAMETERS:
                self.assertEqual(selection["choices"][method]["candidate_id"], identifier)
            self.assertEqual(selection["choices"]["copy"]["candidate_id"], sha(CURRENT.encode()))

    def test_all_70_samples_and_17_students_are_kept(self):
        fixture = SupportFixture(["q" + str(i) for i in range(70)], ["dev" + str(i % 17) for i in range(70)])
        report = fixture.report(expected_samples=70, expected_students=17, bootstrap=1)
        self.assertEqual((report["samples"], report["students"]), (70, 17))
        self.assertEqual(len(report["selections"]), 140)
        self.assertEqual(len(report["external_selections"]), 350)
        self.assertEqual({len(rows) for version in report["static_per_sample"].values() for rows in version.values()}, {70})
        with self.assertRaisesRegex(ValueError, "entire expected"):
            fixture.report(expected_samples=69, expected_students=17)
        with self.assertRaisesRegex(ValueError, "entire expected"):
            fixture.report(expected_samples=70, expected_students=16)

    def test_complete_labels_and_external_predictions_are_required(self):
        with self.assertRaisesRegex(ValueError, "complete cohort"):
            self.fixture.report(labels=jsonl(self.fixture.labels[:-1]))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.fixture.report(labels=jsonl(self.fixture.labels + self.fixture.labels[:1]))
        missing = dict(self.fixture.external)
        missing.pop(analysis.EXTERNAL_METHODS[0])
        with self.assertRaisesRegex(ValueError, "four predeclared"):
            self.fixture.report(external_predictions=missing)
        wrong = copy.deepcopy(self.fixture.external)
        method = analysis.EXTERNAL_METHODS[0]
        records = predict._records(wrong[method].predictions, "synthetic")[:-1]
        content = jsonl(records)
        wrong[method] = analysis.ExternalPredictionArtifact(content, {**wrong[method].provenance, "prediction_sha256": sha(content)})
        with self.assertRaisesRegex(ValueError, "complete cohort"):
            self.fixture.report(external_predictions=wrong)

    def test_external_provenance_hash_and_label_feedback_are_rejected(self):
        method = analysis.EXTERNAL_METHODS[0]
        for key, value in (("prediction_sha256", "0" * 64), ("labels_used", True),
                           ("parameters_fitted", True), ("generation_count_per_sample", 4),
                           ("backend", "HFBackend"), ("protocol_sha256", "0" * 64)):
            wrong = copy.deepcopy(self.fixture.external)
            item = wrong[method]
            wrong[method] = analysis.ExternalPredictionArtifact(item.predictions, {**item.provenance, key: value})
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "provenance"):
                self.fixture.report(external_predictions=wrong)

    def test_wrong_draw_condition_origin_or_missing_wrapper_rejects(self):
        for operation in (lambda rows: rows[0]["attempts"][4].update(condition="11"),
                          lambda rows: rows[0]["attempts"].pop(),
                          lambda rows: rows[0]["attempts"][4]["raw_generation"].update(attempt=1),
                          lambda rows: rows[0]["candidates"][0]["sources"].append("sampled:11:99")):
            wrong = SupportFixture.mutate(self.fixture.artifacts["balanced12"], "candidates", operation)
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                self.fixture.report(balanced=wrong)

    def test_truncated_random_draw_is_preserved_and_explicitly_excluded(self):
        balanced = self.fixture.artifacts["balanced12"]
        identifier = sha(GOOD.encode())
        def truncate(rows):
            rows[0]["attempts"][4]["raw_generation"].update(eos_reached=False, finish_reason="length_limit")
            rows[0]["candidates"] = [item for item in rows[0]["candidates"] if item["candidate_id"] != identifier]
        truncated = SupportFixture.mutate(balanced, "candidates", truncate)
        truncated = SupportFixture.mutate(truncated, "scores", lambda rows: rows.__setitem__(slice(None),
            [item for item in rows if (item["sample_id"], item["candidate_id"]) != ("a", identifier)]))
        result = self.fixture.report(balanced=truncated)
        row = next(row for row in result["source_coverage"]["balanced12"] if row["sample_id"] == "a")
        self.assertEqual(row["retained_raw_attempts"], 13)
        self.assertEqual(row["conditions"]["01"]["random_attempts"], 3)
        self.assertEqual(row["conditions"]["01"]["eligible_random_attempts"], 2)
        unexcluded = SupportFixture.mutate(balanced, "candidates", lambda rows:
            rows[0]["attempts"][4]["raw_generation"].update(eos_reached=False))
        with self.assertRaisesRegex(ValueError, "canonical complete"):
            self.fixture.report(balanced=unexcluded)

    def test_shared_original_wrapper_and_union_score_drift_reject(self):
        changed = SupportFixture.mutate(self.fixture.artifacts["balanced12"], "candidates", lambda rows:
            rows[0]["attempts"][1]["raw_generation"].update(seed=999))
        with self.assertRaisesRegex(ValueError, "Original shared"):
            self.fixture.report(balanced=changed)
        def drift(rows):
            item = next(row for row in rows if row["sample_id"] == "a" and row["candidate_id"] == sha(BAD.encode()))
            raw = scoring.LogProbs(-1.1, -1, -1, -1)
            item.update(**asdict(raw), components=scoring.components(raw), scores_weight_1=scoring.method_scores(raw))
        changed = SupportFixture.mutate(self.fixture.artifacts["balanced12"], "scores", drift)
        with self.assertRaisesRegex(ValueError, "one union-scoring"):
            self.fixture.report(balanced=changed)

    def test_shared_canonical_tokens_and_model_protocol_drift_reject(self):
        def change_token(rows):
            candidate = next(item for item in rows[0]["candidates"] if item["candidate_id"] == sha(BAD.encode()))
            candidate["completion_token_ids"][-1] = 255
        changed = SupportFixture.mutate(self.fixture.artifacts["balanced12"], "candidates", change_token)
        with self.assertRaisesRegex(ValueError, "canonical candidate/EOS"):
            self.fixture.report(balanced=changed)
        changed = SupportFixture.mutate(self.fixture.artifacts["balanced12"], "manifest", lambda rows:
            rows[0]["model"].update(sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "model/scoring"):
            self.fixture.report(balanced=changed)

    def test_pool_protocol_and_source_hash_are_required(self):
        for key, value in (("protocol_sha256", "0" * 64), ("pool_random_count", 3),
                           ("pool", "old318"), ("shared_copy", False),
                           ("generation_manifest_sha256", "short"), ("student_execution", True),
                           ("labels_read", True), ("scoring_performed", False), ("baseline_score_records", 317)):
            changed = SupportFixture.mutate(self.fixture.artifacts["balanced12"], "manifest", lambda rows:
                rows[0]["candidate_support"].update({key: value}))
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.fixture.report(balanced=changed)

    def test_derived_score_corruption_cannot_be_rehashed_into_success(self):
        changed = SupportFixture.mutate(self.fixture.artifacts["balanced12"], "scores", lambda rows:
            rows[0]["scores_weight_1"].update(b=123))
        with self.assertRaisesRegex(ValueError, "four raw"):
            self.fixture.report(balanced=changed)

    def test_output_contains_hashes_numbers_and_ids_but_no_student_text(self):
        report = self.fixture.report()
        content = inference.canonical_json(report)
        for text in (CURRENT, GOOD, BAD, "Check the required integer.", "Print one integer."):
            self.assertNotIn(text, content)
            self.assertNotIn(json.dumps(text)[1:-1], content)
        analysis._code_free(report)
        self.assertEqual(report["selections_sha256"], sha(jsonl(report["selections"])))
        self.assertEqual(report["external_selections_sha256"], sha(jsonl(report["external_selections"])))
        poisoned = copy.deepcopy(self.fixture.external)
        method = analysis.EXTERNAL_METHODS[0]
        item = poisoned[method]
        poisoned[method] = analysis.ExternalPredictionArtifact(item.predictions, {**item.provenance, "history": []})
        with self.assertRaisesRegex(ValueError, "text/token"):
            self.fixture.report(external_predictions=poisoned)

    def test_production_protocol_binds_labels_inputs_and_original_completion(self):
        protocol = self.fixture.frozen_protocol()
        arguments = (self.fixture.input_bytes, jsonl(self.fixture.labels),
                     self.fixture.artifacts["balanced12"], self.fixture.artifacts["11_only12"])
        self.assertEqual(analysis._frozen_protocol(jsonl([protocol]), *arguments), protocol)
        for key, value in (("seed", 1), ("bootstrap", {"unit": "submission", "paired": True, "repetitions": 2000, "seed": 20260929}),
                           ("balanced_random_per_condition", 4), ("methods", ["b", "base"]), ("test_set_used", True)):
            wrong = {**protocol, key: value}
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "frozen complete"):
                analysis._frozen_protocol(jsonl([wrong]), *arguments)
        wrong = {**protocol, "labels_sha256": "0" * 64}
        with self.assertRaisesRegex(ValueError, "evaluation bytes"):
            analysis._frozen_protocol(jsonl([wrong]), *arguments)
        wrong = copy.deepcopy(protocol)
        wrong["baseline_sha256"]["prepared/inputs.jsonl"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "baseline input bytes"):
            analysis._frozen_protocol(jsonl([wrong]), *arguments)
        changed = SupportFixture.mutate(self.fixture.artifacts["balanced12"], "manifest", lambda rows:
            rows[0]["candidate_support"].update(baseline_manifest_sha256="0" * 64))
        with self.assertRaisesRegex(ValueError, "baseline completion"):
            analysis._frozen_protocol(jsonl([protocol]), arguments[0], arguments[1], changed, arguments[3])

    def test_cli_new_output_and_exclusive_code_free_artifacts(self):
        # Parsing/cohort analysis is tested above. This test covers real output
        # bytes/hash/exclusive writes using the already-verified synthetic report.
        report = self.fixture.report()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory).resolve()
            protocol = self.fixture.frozen_protocol()
            (directory / "protocol.json").write_bytes(jsonl([protocol]))
            for pool in analysis.POOLS:
                path = directory / pool
                path.mkdir()
                artifact = self.fixture.artifacts[pool]
                for name in ("manifest", "candidates", "scores"):
                    (path / (name + (".json" if name == "manifest" else ".jsonl"))).write_bytes(getattr(artifact, name))
            (directory / "inputs.jsonl").write_bytes(self.fixture.input_bytes)
            (directory / "labels.jsonl").write_bytes(jsonl(self.fixture.labels))
            specs = []
            for method, item in self.fixture.external.items():
                name = method + ".jsonl"
                (directory / name).write_bytes(item.predictions)
                specs.append({"method": method, "predictions_path": name, "provenance": item.provenance})
            (directory / "externals.json").write_bytes(jsonl([specs]))
            output = directory / "output"
            argv = ["--balanced-dir", str(directory / "balanced12"), "--control-dir", str(directory / "11_only12"),
                "--inputs", str(directory / "inputs.jsonl"), "--labels", str(directory / "labels.jsonl"),
                "--protocol", str(directory / "protocol.json"), "--externals-json", str(directory / "externals.json"),
                "--output", str(output)]
            with patch.object(analysis, "analyze_support", return_value=report) as call, redirect_stdout(io.StringIO()):
                self.assertEqual(analysis.main(argv), 0)
            self.assertNotIn("expected_samples", call.call_args.kwargs)
            self.assertEqual(call.call_args.kwargs["protocol_sha256"], sha((directory / "protocol.json").read_bytes()))
            self.assertEqual((output / "SUCCESS").read_text().strip(), sha((output / "analysis.json").read_bytes()))
            self.assertEqual((output / "selections.jsonl").read_bytes(), jsonl(report["selections"]))
            self.assertEqual((output / "external-selections.jsonl").read_bytes(), jsonl(report["external_selections"]))
            self.assertEqual(set(path.name for path in output.iterdir()), {"analysis.json", "selections.jsonl", "external-selections.jsonl", "SUCCESS"})
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                analysis.main(argv)
            self.assertEqual(error.exception.code, 2)
            self.assertEqual((output / "analysis.json").read_bytes(), (inference.canonical_json(report) + "\n").encode())


if __name__ == "__main__":
    unittest.main()
