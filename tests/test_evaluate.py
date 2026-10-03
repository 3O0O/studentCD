import json
from pathlib import Path
import tempfile
import unittest

from student_sim_cd.evaluate import copy_baseline, edit_events, evaluate, pair_metrics, summarize


class EvaluationTests(unittest.TestCase):
    def test_copy_does_not_get_edit_credit_for_a_changed_target(self):
        result = pair_metrics("a=1\n", "a=1\n", "a=2\n")
        self.assertEqual(result["edit_location_f1"], 0)
        self.assertEqual(result["exact_next_code"], 0)

    def test_unchanged_target_is_separate_from_changed(self):
        self.assertEqual(pair_metrics("x", "x", "x")["edit_operation_f1"], 1)

    def test_newline_only_change_uses_consistent_raw_text_definition(self):
        result = pair_metrics("x=1", "x=1", "x=1\n")
        self.assertEqual(result["true_changed"], 1)
        self.assertEqual(result["edit_location_f1"], 0)
        self.assertGreater(result["edit_size_absolute_error"], 0)

    def test_insertion_boundary_is_not_replacement(self):
        inserted, _, _ = edit_events("a\nb", "c\na\nb")
        replaced, _, _ = edit_events("a\nb", "c\nb")
        self.assertFalse(inserted & replaced)

    def test_student_weighting_does_not_treat_repeats_as_people(self):
        rows = [{"student_id": "a", "metrics": {"value": 1}}] * 9
        rows += [{"student_id": "b", "metrics": {"value": 0}}]
        result = summarize(rows, bootstrap=20)
        self.assertEqual(result["metrics"]["value"]["student_macro_mean"], 0.5)
        self.assertEqual(result["metrics"]["value"]["submission_mean"], 0.9)

    def test_copy_then_evaluate_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            x, y, p = [root / name for name in ("input.jsonl", "label.jsonl", "pred.jsonl")]
            x.write_text(json.dumps({"sample_id": "one", "student_id": "s", "current_code": "a",
                                     "feedback": []}) + "\n")
            y.write_text(json.dumps({"sample_id": "one", "target_code": "b"}) + "\n")
            copy_baseline(x, p)
            result = evaluate(x, y, p, root / "result", bootstrap=10)
            self.assertEqual(result["all"]["samples"], 1)
            self.assertEqual(result["true_unchanged"]["samples"], 0)
            with self.assertRaises(ValueError):
                evaluate(x, y, p, root / "result", bootstrap=10)

    def test_mismatched_ids_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            paths = [root / str(i) for i in range(3)]
            for index, path in enumerate(paths):
                path.write_text(json.dumps({"sample_id": str(index)}) + "\n")
            with self.assertRaises(ValueError):
                evaluate(*paths, root / "result")

    def test_different_methods_cannot_be_mixed_into_one_metric(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            paths = [root / str(i) for i in range(3)]
            inputs = [{"sample_id": key, "student_id": key, "current_code": "x", "feedback": []}
                      for key in ("a", "b")]
            labels = [{"sample_id": key, "target_code": "x"} for key in ("a", "b")]
            predictions = [{"sample_id": key, "method": method, "predicted_code": "x"}
                           for key, method in (("a", "base"), ("b", "cd"))]
            for path, rows in zip(paths, (inputs, labels, predictions)):
                path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            with self.assertRaisesRegex(ValueError, "exactly one"):
                evaluate(*paths, root / "result")


if __name__ == "__main__":
    unittest.main()
