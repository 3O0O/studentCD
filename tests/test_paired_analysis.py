"""Synthetic checks of pairing, complete cohorts and protocol provenance."""

import copy
import json
import math
import unittest

from student_sim_cd import paired_analysis as analysis


def row(sample, student, value=0.0, changed=1.0, error=1):
    return {"sample_id": sample, "student_id": student, "has_feedback": True,
            "metrics": {"exact_next_code": 0.0, "true_changed": changed,
                        "predicted_changed": 1.0, "edit_location_f1": value,
                        "edit_operation_f1": value, "edit_size_absolute_error": error,
                        "text_similarity": value}}


class PairedAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.cohort = {"a": "student1", "b": "student1", "c": "student1", "d": "student2"}
        self.data = {"base": [row(sample, student) for sample, student in self.cohort.items()],
                     "b": [row(sample, student, 1.0 if student == "student1" else 0.0)
                           for sample, student in self.cohort.items()]}
        self.kwargs = {"sample_students": self.cohort, "protocol": "legacy_preserve_diagnostic",
                       "expected_samples": 4, "expected_students": 2,
                       "comparisons": [("b", "base")], "bootstrap": 400, "seed": 101}

    def report(self, data=None, **kwargs):
        return analysis.analyze_evaluations(self.data if data is None else data, **{**self.kwargs, **kwargs})

    def paired_metric(self, report, metric="edit_location_f1"):
        return report["all"]["comparisons"][0]["metrics"][metric]

    def test_equal_student_mean_differs_from_submission_mean(self):
        report = self.report()
        metric = self.paired_metric(report)
        self.assertEqual(metric["student_macro_delta"], 0.5)
        self.assertEqual(metric["submission_mean_delta"], 0.75)
        self.assertEqual(metric["paired_student_bootstrap_95_ci"], [0.0, 1.0])
        self.assertEqual(metric["student_wins_ties_losses"], {"wins": 1, "ties": 1, "losses": 0})
        self.assertEqual(metric["submission_wins_ties_losses"], {"wins": 3, "ties": 1, "losses": 0})
        self.assertEqual(report["all"]["samples"], 4)
        self.assertEqual(report["all"]["students"], 2)
        self.assertEqual(report["samples_dropped"], 0)

    def test_bootstrap_resamples_paired_student_differences(self):
        data = copy.deepcopy(self.data)
        for method in data:
            for record in data[method]:
                baseline = 0.1 if record["student_id"] == "student1" else 0.7
                record["metrics"]["edit_location_f1"] = baseline + (0.2 if method == "b" else 0)
        metric = self.paired_metric(self.report(data))
        self.assertAlmostEqual(metric["student_macro_delta"], 0.2)
        for bound in metric["paired_student_bootstrap_95_ci"]:
            self.assertAlmostEqual(bound, 0.2)

    def test_lower_error_counts_as_a_win_without_inverting_reported_delta(self):
        data = copy.deepcopy(self.data)
        for record in data["base"]:
            record["metrics"]["edit_size_absolute_error"] = 5
        for record in data["b"]:
            record["metrics"]["edit_size_absolute_error"] = 2
        metric = self.paired_metric(self.report(data), "edit_size_absolute_error")
        self.assertEqual(metric["student_macro_delta"], -3)
        self.assertEqual(metric["student_wins_ties_losses"], {"wins": 2, "ties": 0, "losses": 0})

    def test_edit_rate_is_a_diagnostic_without_a_better_direction(self):
        metric = self.paired_metric(self.report(), "predicted_changed")
        self.assertEqual(metric["direction"], "diagnostic_only")
        self.assertIsNone(metric["student_wins_ties_losses"])

    def test_seed_and_input_order_are_reproducible(self):
        reversed_data = {method: list(reversed(rows)) for method, rows in reversed(list(self.data.items()))}
        self.assertEqual(self.report(), self.report(reversed_data))
        json.dumps(self.report(), allow_nan=False)

    def test_missing_prediction_cannot_be_dropped(self):
        data = copy.deepcopy(self.data)
        data["b"].pop()
        with self.assertRaisesRegex(ValueError, "complete cohort"):
            self.report(data)

    def test_equal_but_incomplete_method_sets_still_fail_cohort(self):
        data = {method: rows[:-1] for method, rows in self.data.items()}
        with self.assertRaisesRegex(ValueError, "complete cohort"):
            self.report(data)

    def test_extra_or_replaced_id_fails_even_if_counts_match(self):
        data = copy.deepcopy(self.data)
        data["b"][-1]["sample_id"] = "replacement"
        with self.assertRaisesRegex(ValueError, "extra="):
            self.report(data)

    def test_duplicate_ids_are_rejected(self):
        data = copy.deepcopy(self.data)
        data["b"].append(data["b"][0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.report(data)

    def test_mapping_key_cannot_hide_a_mismatched_sample(self):
        data = copy.deepcopy(self.data)
        data["b"] = {"wrong": data["b"][0]}
        with self.assertRaisesRegex(ValueError, "index key"):
            self.report(data)

    def test_method_student_assignment_cannot_change(self):
        data = copy.deepcopy(self.data)
        data["b"][0]["student_id"] = "student2"
        with self.assertRaisesRegex(ValueError, "student differs"):
            self.report(data)

    def test_truth_and_feedback_must_agree_between_methods(self):
        for field in ("true_changed", "has_feedback"):
            data = copy.deepcopy(self.data)
            if field == "has_feedback":
                data["b"][0][field] = False
            else:
                data["b"][0]["metrics"][field] = 0.0
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "truth/feedback"):
                self.report(data)

    def test_nonfinite_invalid_or_missing_metrics_fail(self):
        for value in (math.nan, math.inf, -math.inf, True, "0.1", 1.1):
            data = copy.deepcopy(self.data)
            data["b"][0]["metrics"]["edit_location_f1"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.report(data)
        data = copy.deepcopy(self.data)
        del data["b"][0]["metrics"]["edit_location_f1"]
        with self.assertRaisesRegex(ValueError, "missing required"):
            self.report(data)

    def test_optional_metric_schema_must_be_complete(self):
        data = copy.deepcopy(self.data)
        data["b"][0]["metrics"]["predicted_edit_size"] = 2
        with self.assertRaisesRegex(ValueError, "same metric schema"):
            self.report(data)

    def test_protocol_and_expected_counts_are_explicit(self):
        for protocol in ("new_scores_probably", {}, []):
            with self.subTest(protocol=protocol), self.assertRaisesRegex(ValueError, "protocol"):
                self.report(protocol=protocol)
        with self.assertRaisesRegex(ValueError, "sample count"):
            self.report(expected_samples=70, expected_students=17)
        with self.assertRaisesRegex(ValueError, "student count"):
            self.report(expected_students=3)
        for name in ("bootstrap", "expected_samples", "expected_students"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.report(**{name: True})

    def test_malformed_method_metadata_has_a_clear_validation_error(self):
        for data, kinds in (({**self.data, 1: []}, None), (self.data, {"b": []}),
                            (self.data, {"copy": "copy_current"})):
            with self.subTest(kinds=kinds), self.assertRaises(ValueError):
                self.report(data, method_kinds=kinds)

    def test_all_default_comparisons_are_preserved_and_named(self):
        methods = ("base", "b", "cd", "history_d0", "copy")
        data = {method: copy.deepcopy(self.data["base"]) for method in methods}
        report = self.report(data, comparisons=None)
        actual = [(pair["method"], pair["reference"]) for pair in report["all"]["comparisons"]]
        self.assertEqual(tuple(actual), analysis.DEFAULT_COMPARISONS)
        self.assertEqual(report["multiple_comparison_adjustment"],
                         "none; comparisons are descriptive and explicitly declared")
        with self.assertRaisesRegex(ValueError, "absent method"):
            self.report(comparisons=None)

    def test_comparison_list_cannot_drop_duplicate_or_absent_pairs(self):
        for pairs in ([], [("b", "missing")], [("b", "b")], [("b", "base"), ("b", "base")]):
            with self.subTest(pairs=pairs), self.assertRaises(ValueError):
                self.report(comparisons=pairs)

    def test_base_cannot_be_mislabelled_as_original_greedy(self):
        with self.assertRaisesRegex(ValueError, "not original greedy"):
            self.report(method_kinds={"base": "original_greedy"})
        data = {**self.data, "greedy": copy.deepcopy(self.data["base"])}
        with self.assertRaisesRegex(ValueError, "provenance"):
            self.report(data)
        report = self.report(data, method_kinds={"greedy": "original_greedy"})
        self.assertEqual(report["method_kinds"]["base"], "shared_pool_reranking")
        self.assertEqual(report["method_kinds"]["greedy"], "original_greedy")

    def test_subgroups_preserve_overall_and_single_student_has_no_ci(self):
        data = copy.deepcopy(self.data)
        for method in data:
            for record in data[method]:
                record["metrics"]["true_changed"] = float(record["student_id"] == "student1")
        report = self.report(data)
        self.assertEqual(report["all"]["samples"], 4)
        self.assertEqual(report["subgroups"]["true_changed"]["samples"], 3)
        self.assertEqual(report["subgroups"]["true_unchanged"]["samples"], 1)
        self.assertIsNone(report["subgroups"]["true_unchanged"]["comparisons"][0]
                          ["metrics"]["edit_location_f1"]["paired_student_bootstrap_95_ci"])

    def test_legacy_report_never_claims_new_rescoring_or_guesses_true_size(self):
        report = self.report()
        self.assertEqual(report["analysis_kind"], "legacy_format_preserved_diagnostic")
        self.assertEqual(report["unavailable_edit_size_metrics"], ["predicted_edit_size", "true_edit_size"])
        self.assertTrue(any("not newly" in text for text in report["limitations"]))

    def test_frozen_protocol_name_and_explicit_alias_are_unambiguous(self):
        report = self.report(protocol="cached-python-fence-v1")
        self.assertEqual(report["protocol"], "cached-python-fence-v1")
        self.assertFalse(report["protocol_alias_used"])
        alias = self.report(protocol="extraction_v1_rescored")
        self.assertEqual(alias["protocol"], "cached-python-fence-v1")
        self.assertEqual(alias["requested_protocol"], "extraction_v1_rescored")
        self.assertTrue(alias["protocol_alias_used"])

    def test_prediction_diagnostics_measure_size_without_mutating_or_guessing(self):
        original = copy.deepcopy(self.data)
        current = {sample: "x = 0\n" for sample in self.cohort}
        predictions = {method: [{"sample_id": sample, "method": method, "predicted_code": "x = 1\n"}
                                for sample in self.cohort] for method in self.data}
        report = self.report(current_codes=current, predictions_by_method=predictions)
        self.assertEqual(report["all"]["methods"]["b"]["metrics"]["predicted_edit_size"]["student_macro_mean"], 2)
        self.assertEqual(report["unavailable_edit_size_metrics"], ["true_edit_size"])
        self.assertEqual(self.data, original)
        with self.assertRaisesRegex(ValueError, "together"):
            self.report(current_codes=current)
        predictions["b"][0]["predicted_code"] = "x = 0\n"
        with self.assertRaisesRegex(ValueError, "change flag"):
            self.report(current_codes=current, predictions_by_method=predictions)


class PredictionAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.inputs = [{"sample_id": "a", "student_id": "s1", "current_code": "x = 0\n", "feedback": []},
                       {"sample_id": "b", "student_id": "s2", "current_code": "y = 0\n", "feedback": [{"text": "hint"}]}]
        self.labels = [{"sample_id": "a", "target_code": "x = 1\n"},
                       {"sample_id": "b", "target_code": "y = 0\n"}]
        self.predictions = {"base": [{"sample_id": "a", "method": "base", "predicted_code": "x = 0\n"},
                                     {"sample_id": "b", "method": "base", "predicted_code": "y = 1\n"}],
                            "b": [{"sample_id": "a", "method": "b", "predicted_code": "x = 1\n"},
                                  {"sample_id": "b", "method": "b", "predicted_code": "y = 0\n"}]}
        self.kwargs = {"protocol": "extraction_v1_rescored", "expected_samples": 2,
                       "expected_students": 2, "comparisons": [("b", "base")], "bootstrap": 100}

    def report(self, **kwargs):
        return analysis.analyze_predictions(self.inputs, self.labels, self.predictions, **{**self.kwargs, **kwargs})

    def test_text_predictions_have_true_and_predicted_sizes(self):
        report = self.report()
        self.assertEqual(report["analysis_kind"], "cached_python_fence_v1_rescored_static_comparison")
        self.assertEqual(report["unavailable_edit_size_metrics"], [])
        self.assertEqual(report["all"]["methods"]["b"]["metrics"]["true_edit_size"]["student_macro_mean"], 1)
        pair = report["all"]["comparisons"][0]["metrics"]
        self.assertEqual(pair["exact_next_code"]["student_macro_delta"], 1)
        self.assertEqual(pair["edit_size_absolute_error"]["student_macro_delta"], -2)

    def test_future_labels_in_inputs_are_rejected(self):
        self.inputs[0]["target_code"] = "secret future"
        with self.assertRaisesRegex(ValueError, "Future labels"):
            self.report()

    def test_prediction_method_and_completeness_are_checked(self):
        self.predictions["b"][0]["method"] = "greedy"
        with self.assertRaisesRegex(ValueError, "Prediction method"):
            self.report()
        self.predictions["b"][0]["method"] = "b"
        self.predictions["b"].pop()
        with self.assertRaisesRegex(ValueError, "complete cohort"):
            self.report()

    def test_duplicate_labels_and_wrong_student_are_rejected(self):
        self.labels.append(self.labels[0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.report()
        self.labels.pop()
        self.labels[0]["student_id"] = "someone_else"
        with self.assertRaisesRegex(ValueError, "Label student"):
            self.report()

    def test_code_is_text_only_and_is_not_executed(self):
        dangerous = "raise RuntimeError(\"This is a text fixture, never run it\")\n"
        self.predictions["b"][0]["predicted_code"] = dangerous
        report = self.report()
        self.assertEqual(report["evaluation_kind"], "static_observed_next_revision_no_execution")
        self.predictions["b"][0]["predicted_code"] = 2
        with self.assertRaisesRegex(ValueError, "code must be text"):
            self.report()


if __name__ == "__main__":
    unittest.main()
