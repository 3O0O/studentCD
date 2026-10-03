"""Independent A verifier checks against real synthetic fitting and analysis.

All programs are inert strings; these tests never execute student code, open
project train/test data, load a language model, or contact the server.
"""

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

from tests.test_run_a_static import runner
from student_sim_cd import a_static_analysis, change_q

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_a_static_results", ROOT / "scripts/verify_a_static_results.py")
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def sha(body):
    return hashlib.sha256(body.encode()).hexdigest()


def as_json(value):
    # On-disk JSON turns edit-event tuples into arrays. Match the real receipt.
    return json.loads(json.dumps(value))


def input_row(student, index):
    return {"schema_version": "student-sim-cd.progfeed.v1", "sample_id": "sample-" + str(index),
        "student_id": student, "lab": "lab02", "source_file": "x.py",
        "current_timestamp": "2020-01-02-00-00-%02d" % (index % 60),
        "current_code": "a = 1\nb = 0\nprint(a + b)\n", "current_results": [],
        "history": [], "feedback": [{"test_name": "case", "function_name": "f",
            "assigned_type": "nl", "text": "Please inspect the boundary"}], "problem_statement": "Task"}


def label_row(current, code):
    return {"schema_version": "student-sim-cd.progfeed.v1", "sample_id": current["sample_id"],
            "target_timestamp": "2020-01-03-00-00-00", "target_code": code, "target_results": []}


class IndependentStaticAVerifierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads((ROOT / "configs/a_static_v1.json").read_text())
        rows, labels, pools, scores, q = [], [], {}, {}, []
        for index in range(70):
            current = input_row("dev-" + str(index % 17), index)
            sid, code = current["sample_id"], current["current_code"]
            target = (code if index % 4 == 0 else
                "a = 3\nb = 0\nprint(a + b)\n" if index % 4 == 1 else
                "a = 1\nprint(a + b)\n" if index % 4 == 2 else
                code + "print('next')\n")
            rows.append(current); labels.append(label_row(current, target))
            # Candidate creation sees current input only, not the target above.
            candidates = (code, "a = 1\nb = 2\nprint(a + b)\n", code + "# tentative revision\n")
            pools[sid] = {sha(candidate): {"code": candidate} for candidate in candidates}
            for position, cid in enumerate(sorted(pools[sid])):
                l00, l01, l10, l11 = -2.1-position, -2.3+position*.2, -1.4-position*.7, -.5-position*.8
                d0, d1 = l10-l00, l11-l01
                scores[sid, cid] = {"base": l11, "cd": l11+d1, "b": l11+d1-d0, "history_d0": l11+d0}
            q.append({"sample_id": sid, "student_id": current["student_id"], "p_changed": .15+.7*(index % 9)/8})
        cls.inputs, cls.labels, cls.pools, cls.scores, cls.q = rows, labels, pools, scores, q
        cls.sealed = as_json(runner.seal_distributions(rows, pools, scores, q))
        cls.report = as_json(a_static_analysis.analyze_pool(rows, labels, pools, scores, q, "synthetic-70",
            bootstrap=cls.protocol["bootstrap"]["repetitions"], seed=cls.protocol["bootstrap"]["seed"]))
        cls.body = ({r["sample_id"]: r for r in rows}, {r["sample_id"]: r for r in labels},
                    {sid: {cid: c["code"] for cid, c in pool.items()} for sid, pool in pools.items()}, scores)
        training = [input_row("train-" + str(student), 2*student+offset)
                    for student in range(10) for offset in (0, 1)]
        outcomes = [label_row(r, r["current_code"] if index % 2 else r["current_code"]+"# changed\n")
                    for index, r in enumerate(training)]
        cls.q_report, cls.models, cls.oof = change_q.run_train_experiment(training, outcomes)

    def test_independent_numeric_and_body_analysis_agrees_with_real_analysis_70_17(self):
        result = verifier.verify_analysis(self.report, self.sealed, self.protocol, self.body)
        self.assertEqual(result, {"samples": 70, "students": 17, "distributions": 560,
                                  "candidate_metrics_body_checked": True})
        result = verifier.verify_analysis(self.report, self.sealed, self.protocol)
        self.assertFalse(result["candidate_metrics_body_checked"])

    def test_numeric_summary_tamper_is_rejected(self):
        altered = deepcopy(self.report)
        altered["summary"]["a_cd"]["all"]["expected_edit_location_f1"]["student_macro_mean"] += .01
        with self.assertRaisesRegex(ValueError, "Mismatch"):
            verifier.verify_analysis(altered, self.sealed, self.protocol)

    def test_frozen_q_alignment_and_complete_pool_count_are_verified(self):
        result = verifier.verify_analysis(self.report, self.sealed, self.protocol, q_rows=self.q)
        self.assertEqual(result["samples"], 70)
        altered_q = deepcopy(self.q)
        altered_q[0]["p_changed"] = .99
        with self.assertRaisesRegex(ValueError, "Frozen q artifact alignment"):
            verifier.verify_analysis(self.report, self.sealed, self.protocol, q_rows=altered_q)
        altered_q = deepcopy(self.q)
        altered_q[0]["student_id"] = "other-student"
        with self.assertRaisesRegex(ValueError, "Frozen q sample/student alignment"):
            verifier.verify_analysis(self.report, self.sealed, self.protocol, q_rows=altered_q)
        altered = deepcopy(self.report)
        altered["candidates"] += 1
        with self.assertRaisesRegex(ValueError, "candidate count"):
            verifier.verify_analysis(altered, self.sealed, self.protocol)

    def test_sealed_probability_and_joint_tamper_are_rejected(self):
        altered = deepcopy(self.report)
        altered["distributions"][0]["probabilities"] = [.2, .3, .5]
        with self.assertRaisesRegex(ValueError, "Pre-label distribution altered"):
            verifier.verify_analysis(altered, self.sealed, self.protocol)
        sealed = deepcopy(self.sealed)
        sealed[0]["probabilities"] = [.2, .3, .5]
        with self.assertRaisesRegex(ValueError, "independent score normalization"):
            verifier.verify_analysis(altered, sealed, self.protocol)

    def test_argmax_or_body_truth_tamper_is_rejected(self):
        altered = deepcopy(self.report)
        altered["per_sample"]["base"][0]["argmax_candidate_id"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "Argmax"):
            verifier.verify_analysis(altered, self.sealed, self.protocol)
        current, target, pools, scores = self.body
        target = deepcopy(target)
        sid = self.report["distributions"][0]["sample_id"]
        target[sid]["target_code"] = current[sid]["current_code"] + "# false future truth\n"
        with self.assertRaisesRegex(ValueError, "body true changed|independent difflib"):
            verifier.verify_analysis(self.report, self.sealed, self.protocol, (current, target, pools, scores))

    def test_candidate_body_metric_tamper_is_rejected_independently(self):
        altered = deepcopy(self.report)
        altered["distributions"][0]["candidate_metrics"][0]["edit_operation_f1"] += .1
        with self.assertRaisesRegex(ValueError, "independent difflib edit_operation_f1"):
            verifier.verify_analysis(altered, self.sealed, self.protocol, self.body)

    def test_verify_q_accepts_real_20_10_fitting_and_rejects_oof_tamper(self):
        comparisons = verifier.verify_q(self.q_report, self.oof, self.protocol)
        self.assertIn("feedback_numeric_minus_overall_frequency", comparisons)
        altered = deepcopy(self.oof)
        altered[0]["p_changed_by_method"]["current_numeric"] = .99
        with self.assertRaisesRegex(ValueError, "q current_numeric|q macro"):
            verifier.verify_q(self.q_report, altered, self.protocol)

    def test_q_label_fold_and_macro_denominator_tamper_are_rejected(self):
        altered = deepcopy(self.oof)
        altered[0]["label_changed"] = 1-altered[0]["label_changed"]
        with self.assertRaises(ValueError):
            verifier.verify_q(self.q_report, altered, self.protocol)
        altered = deepcopy(self.q_report)
        altered["fold_assignment"][self.oof[0]["student_id"]] = 99
        with self.assertRaisesRegex(ValueError, "fixed folds"):
            verifier.verify_q(altered, self.oof, self.protocol)
        altered = deepcopy(self.q_report)
        altered["metrics"]["feedback_numeric"]["student_macro"]["brier"]["eligible_students"] -= 1
        with self.assertRaisesRegex(ValueError, "macro denominator"):
            verifier.verify_q(altered, self.oof, self.protocol)

    def test_one_class_student_auc_is_missing_and_cannot_be_reported_as_measured(self):
        training = [input_row("single-class-" + str(student), 2*student+offset)
                    for student in range(10) for offset in (0, 1)]
        outcomes = [label_row(r, r["current_code"] if index // 2 < 5 else r["current_code"]+"# changed\n")
                    for index, r in enumerate(training)]
        report, _, oof = change_q.run_train_experiment(training, outcomes)
        self.assertIsNone(report["metrics"]["feedback_numeric"]["student_macro"]["auc"]["value"])
        verifier.verify_q(report, oof, self.protocol)
        altered = deepcopy(report)
        altered["metrics"]["feedback_numeric"]["student_macro"]["auc"]["value"] = .99
        with self.assertRaisesRegex(ValueError, "q macro|Missing value"):
            verifier.verify_q(altered, oof, self.protocol)


if __name__ == "__main__":
    unittest.main()
