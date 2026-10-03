"""Synthetic-only checks: no real training/dev/test student text is accessed."""
from copy import deepcopy
import hashlib
import json
import math
import unittest

from student_sim_cd import change_q as q


def cohort(students=10, repeats=2):
    inputs, labels = [], []
    for s in range(students):
        for n in range(repeats):
            sid = f"sample-{s:02d}-{n:02d}"
            code = "synthetic-current-" + "x"*(s+n)
            row = {
                "schema_version": q.DATA_SCHEMA, "sample_id": sid, "student_id": f"s{s:02d}",
                "lab": "lab02", "source_file": "to_do_list.py", "current_timestamp": "2020-01-03-10-00-00",
                "current_code": code, "problem_statement": "Synthetic task, never executed.",
                "history": [{"timestamp": "2020-01-01-10-00-00", "code": "synthetic-past",
                             "results": [], "feedback": []}],
                "feedback": [{"test_name": "a", "function_name": "f", "assigned_type": "nl", "text": "Synthetic feedback"}],
                "current_results": [{"test_name": "a", "function_name": "f", "status": "partial",
                                     "score": (s % 3)/2, "max_score": 1.0, "testcase_mask": None}],
            }
            inputs.append(row)
            labels.append({"schema_version": q.DATA_SCHEMA, "sample_id": sid,
                           "target_timestamp": "2020-01-04-10-00-00", "target_code": code+"changed" if n % 2 else code,
                           "target_results": []})
    return inputs, labels


class ChangeQTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs, cls.labels = cohort()
        cls.result = q.run_train_experiment(cls.inputs, cls.labels, source_sha256={"synthetic_inputs": "a"*64})

    def test_full_cohort_deterministic_and_no_body_output(self):
        report, models, oof = self.result
        again = q.run_train_experiment(list(reversed(self.inputs)), list(reversed(self.labels)), source_sha256={"synthetic_inputs": "a"*64})
        self.assertEqual(self.result, again)
        self.assertEqual((report["sample_count"], report["student_count"], len(oof)), (20, 10, 20))
        self.assertEqual(report["oof_sample_coverage"], 1.0)
        self.assertEqual(report["oof_student_coverage"], 1.0)
        self.assertEqual(set(models["methods"]), set(q.METHODS))
        self.assertTrue(report["default_method_predeclared"])
        serialized = json.dumps(self.result, allow_nan=False)
        for body in ("synthetic-current-", "synthetic-past", "Synthetic feedback", "Synthetic task"):
            self.assertNotIn(body, serialized)
        self.assertTrue(all(set(row) == {"sample_id", "student_id", "fold", "label_changed", "p_changed", "p_changed_by_method"} for row in oof))

    def test_disjoint_folds_and_exact_oof_replay(self):
        report, _, oof = self.result
        held_students = []
        for audit, model in zip(report["fold_audit"], report["fold_models"]):
            self.assertFalse(set(audit["train_student_ids"]) & set(audit["heldout_student_ids"]))
            self.assertEqual(model["training_student_ids"], audit["train_student_ids"])
            held_students.extend(audit["heldout_student_ids"])
            rows = [row for row in self.inputs if row["student_id"] in audit["heldout_student_ids"]]
            predictions = {method: q.predict(model, rows, method=method, split="train") for method in q.METHODS}
            for i, row in enumerate(sorted(rows, key=lambda r: r["sample_id"])):
                actual = next(v for v in oof if v["sample_id"] == row["sample_id"])
                for method in q.METHODS:
                    self.assertEqual(actual["p_changed_by_method"][method], predictions[method][i]["p_changed"])
        self.assertEqual(sorted(held_students), sorted({r["student_id"] for r in self.inputs}))

    def test_scaler_fits_fold_train_only_and_weighted_observed(self):
        report, _, _ = self.result
        for audit, model in zip(report["fold_audit"], report["fold_models"]):
            rows = [r for r in self.inputs if r["student_id"] in audit["train_student_ids"]]
            features = [q.observable_features(r) for r in rows]
            scaler = model["methods"]["current_numeric"]["scaler"]
            expected = math.fsum(f["current_code_log_chars"] for f in features)/len(features)
            self.assertAlmostEqual(scaler["mean"][0], expected, places=14)
            self.assertEqual(scaler["observed_count"][0], len(rows))
            self.assertEqual(scaler["missing_imputation"], "fold_training_observed_weighted_mean")
        # Change only the held-out fold's observable text, then the training scaler stays identical.
        audit = report["fold_audit"][0]
        mutated = deepcopy(self.inputs)
        for row in mutated:
            if row["student_id"] in audit["heldout_student_ids"]:
                row["current_code"] = "x"*10000
        targets = {r["sample_id"]: int(label["target_code"] != r["current_code"])
                   for r, label in zip(self.inputs, self.labels)}
        train = sorted([r for r in mutated if r["student_id"] in audit["train_student_ids"]], key=lambda r: r["sample_id"])
        rebuilt = q._fit(train, targets, {"synthetic_inputs": "a"*64}, "train")
        self.assertEqual(rebuilt, report["fold_models"][0])

    def test_equal_student_frequency_not_submission_frequency(self):
        rows, labels = cohort()
        # Add eight changed observations for one student: their total fit weight is still one.
        for i in range(8):
            row = deepcopy(rows[1]); row["sample_id"] = f"extra-{i}"
            label = deepcopy(labels[1]); label["sample_id"] = row["sample_id"]
            rows.append(row); labels.append(label)
        report, model, _ = q.run_train_experiment(rows, labels)
        expected_changed_mass = 9/10 + 9*0.5
        overall = model["methods"]["overall_frequency"]
        self.assertAlmostEqual(overall["changed_student_mass"], expected_changed_mass)
        self.assertAlmostEqual(overall["p_changed"], (expected_changed_mass+0.5)/11)
        self.assertNotAlmostEqual(overall["p_changed"], (18+0.5)/29)
        self.assertEqual(report["sample_count"], 28)

    def test_missing_scores_and_history_have_explicit_masks(self):
        rows, labels = cohort()
        for row in rows:
            row["history"] = []
            row["current_results"] = []
            row["feedback"] = []
        report, model, _ = q.run_train_experiment(rows, labels)
        item = model["methods"]["feedback_numeric"]
        index = item["feature_names"].index("current_logged_score_fraction")
        self.assertEqual(item["scaler"]["observed_count"][index], 0)
        self.assertEqual(item["scaler"]["mean"][index], 0.0)
        self.assertEqual(item["scaler"]["scale"][index], 1.0)
        self.assertIn("current_logged_score_fraction__missing", item["design_names"])
        self.assertEqual(report["oof_sample_coverage"], 1.0)

    def test_extreme_scores_finite_and_not_pass_rate(self):
        rows, labels = cohort()
        for row in rows:
            row["current_results"] = [
                {"test_name": str(i), "function_name": "f", "score": 1e308 if i == 0 else 0,
                 "max_score": 1e308, "testcase_mask": "111", "status": "partial"} for i in range(3)]
        features = q.observable_features(rows[0])
        self.assertAlmostEqual(features["current_logged_score_fraction"], 1/3)
        report, model, oof = q.run_train_experiment(rows, labels)
        json.dumps((report, model, oof), allow_nan=False)
        self.assertTrue(all(math.isfinite(r["p_changed"]) for r in oof))
        conflict = deepcopy(rows[0])
        conflict["current_results"].append(dict(conflict["current_results"][0], score=0))
        self.assertIsNone(q.observable_features(conflict)["current_logged_score_fraction"])

    def test_future_fields_recursive_and_nonchronological_rejected(self):
        variants = []
        row = deepcopy(self.inputs[0]); row["target_code"] = "forbidden"; variants.append(row)
        row = deepcopy(self.inputs[0]); row["history"][0]["future_results"] = []; variants.append(row)
        row = deepcopy(self.inputs[0]); row["current_results"][0]["testcase_mask"] = {"nextCode": "forbidden"}; variants.append(row)
        row = deepcopy(self.inputs[0]); row["history"][0]["timestamp"] = row["current_timestamp"]; variants.append(row)
        row = deepcopy(self.inputs[0]); row.pop("current_timestamp"); variants.append(row)
        for row in variants:
            with self.subTest(row=row["sample_id"]):
                with self.assertRaises(ValueError):
                    q.run_train_experiment([row]+self.inputs[1:], self.labels)

    def test_label_id_and_future_timestamp_guards(self):
        for labels in (self.labels[:-1], self.labels+[deepcopy(self.labels[0])]):
            with self.assertRaises(ValueError):
                q.run_train_experiment(self.inputs, labels)
        labels = deepcopy(self.labels); labels[0]["sample_id"] = "unknown"
        with self.assertRaises(ValueError):
            q.run_train_experiment(self.inputs, labels)
        labels = deepcopy(self.labels); labels[0]["target_timestamp"] = self.inputs[0]["current_timestamp"]
        with self.assertRaisesRegex(ValueError, "target timestamp"):
            q.run_train_experiment(self.inputs, labels)

    def test_split_registry_fixed_protocol_and_class_guards(self):
        for split in ("dev", "test", "unknown"):
            with self.assertRaisesRegex(ValueError, "train-only"):
                q.run_train_experiment(self.inputs, self.labels, split=split)
        for kwargs in ({"seed": 1}, {"folds": 4}, {"source_sha256": {"inputs": "bad"}},
                       {"student_splits": {r["student_id"]: "dev" for r in self.inputs}}):
            with self.assertRaises(ValueError):
                q.run_train_experiment(self.inputs, self.labels, **kwargs)
        labels = deepcopy(self.labels)
        for row, label in zip(self.inputs, labels):
            label["target_code"] = row["current_code"]
        with self.assertRaisesRegex(ValueError, "both changed and unchanged"):
            q.run_train_experiment(self.inputs, labels)
        # A globally rare class confined to one student makes its outer training fold invalid.
        labels[0]["target_code"] += "changed"
        with self.assertRaisesRegex(ValueError, "every training fold"):
            q.run_train_experiment(self.inputs, labels)

    def test_one_class_students_retained_with_metric_coverage(self):
        rows, labels = cohort(students=10, repeats=1)
        for s, label in enumerate(labels):
            if s % 2:
                label["target_code"] += "changed"
        report, _, oof = q.run_train_experiment(rows, labels)
        self.assertEqual(len(oof), 10)
        self.assertEqual(report["class_coverage"]["students_with_both"], 0)
        self.assertEqual(report["class_coverage"]["one_class_students"], 10)
        for metrics in report["metrics"].values():
            self.assertEqual(metrics["student_macro"]["logloss"]["eligible_students"], 10)
            self.assertEqual(metrics["student_macro"]["auc"]["eligible_students"], 0)
            self.assertIsNone(metrics["student_macro"]["auc"]["value"])
            self.assertEqual(metrics["student_macro"]["unchanged_false_edit"]["eligible_students"], 5)
            self.assertEqual(metrics["student_macro"]["changed_miss"]["eligible_students"], 5)
            self.assertEqual(metrics["reliability_sample_coverage"], 1.0)
            self.assertAlmostEqual(sum(b["student_weight_share"] for b in metrics["reliability_5_bins"]), 1.0)

    def test_metrics_hand_calculation_ties_and_reliability(self):
        rows, _ = cohort(students=2)
        targets = {r["sample_id"]: i % 2 for i, r in enumerate(rows)}
        metrics = q.evaluate_probabilities(rows, targets, [0.5]*4)
        self.assertAlmostEqual(metrics["student_macro"]["logloss"]["value"], math.log(2))
        self.assertEqual(metrics["student_macro"]["brier"]["value"], 0.25)
        self.assertEqual(metrics["student_macro"]["auc"]["value"], 0.5)
        self.assertEqual(metrics["pooled_equal_student_auc"], 0.5)
        self.assertEqual(metrics["submission_confusion"], {"tn": 0, "fp": 2, "fn": 0, "tp": 2})
        self.assertEqual(metrics["student_macro"]["unchanged_false_edit"]["value"], 1.0)
        self.assertEqual(metrics["student_macro"]["changed_miss"]["value"], 0.0)
        self.assertEqual(metrics["reliability_ece_equal_student"], 0.0)
        extreme = q.evaluate_probabilities(rows, targets, [0.0, 1.0, 0.0, 1.0])
        self.assertTrue(math.isfinite(extreme["student_macro"]["logloss"]["value"]))
        self.assertEqual(extreme["student_macro"]["brier"]["value"], 0.0)
        self.assertEqual(extreme["student_macro"]["auc"]["value"], 1.0)

    def test_predict_label_free_disjoint_unknown_task_fallback(self):
        _, model, _ = self.result
        rows, _ = cohort(students=1)
        for row in rows:
            row["student_id"] = "new-heldout-student"
            row["lab"] = "unseen-lab"
        predictions = q.predict(model, rows, method="task_frequency", split="dev")
        self.assertEqual([p["p_changed"] for p in predictions], [model["methods"]["overall_frequency"]["p_changed"]]*2)
        self.assertTrue(all(set(p) == {"sample_id", "student_id", "p_changed"} for p in predictions))
        with self.assertRaisesRegex(ValueError, "cross-student leakage"):
            q.predict(model, self.inputs)
        with self.assertRaises(ValueError):
            q.predict(model, rows, method="auto_selected_from_dev")
        with self.assertRaises(TypeError):
            q.predict(model, rows, labels=[])

    def test_serialized_model_schema_rejects_poison_and_nonfinite(self):
        _, model, _ = self.result
        rows, _ = cohort(students=1)
        for row in rows:
            row["student_id"] = "heldout"
        self.assertEqual(q.predict(model, rows), q.predict(json.loads(json.dumps(model)), rows))
        variants = []
        changed = deepcopy(model); changed["target_code"] = "forbidden"; variants.append(changed)
        changed = deepcopy(model); changed["methods"]["feedback_numeric"]["coefficients"][1] = float("nan"); variants.append(changed)
        changed = deepcopy(model); changed["methods"]["feedback_numeric"]["scaler"]["scale"][0] = 0; variants.append(changed)
        changed = deepcopy(model); changed["methods"]["feedback_numeric"]["feature_names"][0] = "student_id"; variants.append(changed)
        changed = deepcopy(model); changed["objective"] = "untrusted text"; variants.append(changed)
        changed = deepcopy(model); changed["methods"]["feedback_numeric"]["solver"]["converged"] = False; variants.append(changed)
        changed = deepcopy(model); changed["l2"] = True; variants.append(changed)
        for changed in variants:
            with self.assertRaises(ValueError):
                q.predict(changed, rows)

    def test_sources_recorded_without_claiming_file_authentication(self):
        report, models, _ = self.result
        self.assertEqual(report["source_sha256"], {"synthetic_inputs": "a"*64})
        canonical = json.dumps(sorted(self.inputs, key=lambda r: r["sample_id"]), ensure_ascii=False,
                               sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        self.assertEqual(report["canonical_inputs_sha256"], hashlib.sha256(canonical).hexdigest())
        self.assertEqual(report["final_model_sha256"], q._hash(models))
        self.assertIn("caller must audit source and student registry", " ".join(report["limits"]))

    def test_large_synthetic_fixed_cohort_converges_and_covers_every_student(self):
        rows, labels = cohort(students=140, repeats=5)
        # A heterogeneous observable signal and asymmetric labels, still exclusively synthetic.
        for i, (row, label) in enumerate(zip(rows, labels)):
            student, repetition = divmod(i, 5)
            row["lab"] = "lab02" if student % 2 else "lab04"
            row["source_file"] = "synthetic_a.py" if student % 2 else "synthetic_b.py"
            row["feedback"][0]["text"] = "synthetic-feedback-" + "x"*(student % 31)
            if (student + repetition) % 5:
                label["target_code"] = row["current_code"] + "change"
            else:
                label["target_code"] = row["current_code"]
        # Match production-sized 737/140 without touching production data.
        for i in range(37):
            row = deepcopy(rows[i*5]); row["sample_id"] += "-extra"
            label = deepcopy(labels[i*5]); label["sample_id"] = row["sample_id"]
            rows.append(row); labels.append(label)
        report, model, oof = q.run_train_experiment(rows, labels)
        self.assertEqual((report["sample_count"], report["student_count"], len(oof)), (737, 140, 737))
        self.assertEqual(sum(a["heldout_sample_count"] for a in report["fold_audit"]), 737)
        for fold_model in report["fold_models"] + [model]:
            for method in q.FEATURES:
                self.assertTrue(fold_model["methods"][method]["solver"]["converged"])
        self.assertTrue(all(math.isfinite(p) for r in oof for p in r["p_changed_by_method"].values()))


if __name__ == "__main__":
    unittest.main()
