import copy
import json
from pathlib import Path
import tempfile
import unittest

from student_sim_cd.inference import canonical_json, load_references, read_jsonl
from student_sim_cd.select_inputs import (
    build_donor_index, eligibility_reasons, logged_performance, match_reference, select_inputs,
)


def row(sample_id, student, day=3, score=0.5, history=True):
    return {
        "schema_version": "student-sim-cd.progfeed.v1", "sample_id": sample_id,
        "student_id": student, "lab": "lab02", "source_file": "to_do_list.py",
        "current_timestamp": f"2020-01-{day:02}-10-00-00", "current_code": "x = 1\n",
        "history": ([{"timestamp": f"2020-01-{day - 1:02}-10-00-00", "code": "x = 0\n",
                      "results": [], "feedback": []}] if history else []),
        "feedback": [{"test_name": "test_a", "function_name": "a",
                      "assigned_type": "nl", "text": "Check your loop."}],
        "current_results": [{"test_name": "test_a", "function_name": "a",
                             "status": "partial", "score": score, "max_score": 1,
                             "testcase_mask": None}],
        "problem_statement": "Maintain a to-do list.",
    }


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prepared = self.root / "prepared"
        self.prepared.mkdir()
        self.scope = self.root / "scope.json"
        self.scope.write_text(canonical_json({"pairs": [{"lab": "lab02", "source_file": "to_do_list.py"}]}))
        self.pairs = {("lab02", "to_do_list.py")}
        self.train = [row("donor-a", "student-train", day=2)]
        self.write("inputs.train.jsonl", self.train)
        self.dev = [row("dev-a", "student-dev")]
        self.test = [row("test-a", "student-test")]
        self.write_split("dev", self.dev)
        self.write_split("test", self.test)

    def write(self, name, rows):
        (self.prepared / name).write_text("".join(canonical_json(item) + "\n" for item in rows), encoding="utf-8")

    def write_split(self, split, rows):
        self.write(f"inputs.{split}.jsonl", rows)
        # Deliberately reverse the labels: file order is not an alignment contract.
        self.write(f"labels.{split}.jsonl", [{"sample_id": item["sample_id"], "target_code": "HIDDEN_LABEL_" + item["sample_id"]}
                                            for item in reversed(rows)])

    def test_all_eligible_rows_retained_and_labels_only_aligned_by_id(self):
        rows = [row("dev-a", "student-dev"), row("dev-b", "student-dev", history=False)]
        self.write_split("dev", rows)
        output = self.root / "selection"
        report = select_inputs(self.prepared, output, self.scope)
        self.assertEqual(report["splits"]["dev"]["selected_samples"], 2)
        self.assertEqual(report["splits"]["dev"]["matched_samples"], 1)
        self.assertEqual([item["sample_id"] for item in read_jsonl(output / "inputs.dev.jsonl")], ["dev-a", "dev-b"])
        self.assertEqual([item["sample_id"] for item in read_jsonl(output / "labels.dev.jsonl")], ["dev-a", "dev-b"])
        references = load_references(output / "matched/references.dev.jsonl", {"dev-a"})
        self.assertNotIn("HIDDEN_LABEL", canonical_json(references))
        self.assertEqual(references["dev-a"]["reference_history"], self.train[0]["history"])
        self.assertFalse(report["labels_used_for_selection"])
        self.assertFalse((self.prepared / "labels.train.jsonl").exists())

    def test_label_content_changes_do_not_change_selection_or_reference(self):
        first, second = self.root / "first", self.root / "second"
        select_inputs(self.prepared, first, self.scope)
        self.write("labels.dev.jsonl", [{"sample_id": "dev-a", "target_code": "",
                                       "target_results": [{"score": 999999}]}])
        select_inputs(self.prepared, second, self.scope)
        for name in ("inputs.dev.jsonl", "matched/inputs.dev.jsonl", "matched/references.dev.jsonl"):
            self.assertEqual((first / name).read_bytes(), (second / name).read_bytes())

    def test_filters_are_declared_current_information_only(self):
        missing_feedback = row("no-f", "student-dev")
        missing_feedback["feedback"][0]["text"] = "  "
        missing_problem = row("no-p", "student-dev")
        missing_problem["problem_statement"] = None
        outside = row("other", "student-dev")
        outside["source_file"] = "other.py"
        self.write_split("dev", [row("good", "student-dev"), missing_feedback, missing_problem, outside,
                                 row("no-h", "student-dev", history=False)])
        report = select_inputs(self.prepared, self.root / "filtered", self.scope, require_history=True)
        self.assertEqual(report["splits"]["dev"]["selected_samples"], 1)
        self.assertEqual(report["splits"]["dev"]["excluded_by_primary_reason"], {
            "no_actual_nonempty_feedback": 1, "missing_problem_statement": 1,
            "outside_declared_lab_file_scope": 1, "empty_history": 1,
        })

    def test_reference_global_time_and_different_student(self):
        query = row("query", "student-q", day=3)
        donors = [row("future", "future-student", day=4), row("equal", "equal-student", day=3),
                  row("same-student", "student-q", day=2), row("past", "past-student", day=2)]
        reference, reason = match_reference(query, build_donor_index(donors, self.pairs), "train-hash")
        self.assertIsNone(reason)
        self.assertEqual(reference["provenance"]["donor_sample_id"], "past")
        self.assertEqual(reference["provenance"]["eligible_donor_count"], 1)

    def test_reference_nearest_and_tiebreak_are_deterministic(self):
        query = row("query", "student-q", day=5, score=0.75)
        donors = [row("b", "student-b", day=3, score=0.75), row("a", "student-a", day=3, score=0.75),
                  row("worse", "student-c", day=3, score=0)]
        for order in (donors, list(reversed(donors))):
            reference, _ = match_reference(query, build_donor_index(order, self.pairs), "hash")
            self.assertEqual(reference["provenance"]["donor_sample_id"], "a")
            self.assertEqual(reference["provenance"]["distance"], [0, 0, 0])

    def test_missing_match_is_reported_without_fallback(self):
        self.write("inputs.train.jsonl", [row("future", "student-train", day=5)])
        output = self.root / "unmatched"
        report = select_inputs(self.prepared, output, self.scope)
        self.assertEqual(report["splits"]["dev"]["selected_samples"], 1)
        self.assertEqual(report["splits"]["dev"]["matched_samples"], 0)
        self.assertEqual(read_jsonl(output / "matched/references.dev.jsonl"), [])
        self.assertIn("no_earlier_train", read_jsonl(output / "unmatched.dev.jsonl")[0]["reason"])

    def test_missing_and_conflicting_performance_are_not_zero(self):
        query = row("query", "student-q")
        query["current_results"][0]["score"] = None
        self.assertIsNone(logged_performance(query))
        query = row("query", "student-q")
        duplicate = copy.deepcopy(query["current_results"][0])
        query["current_results"].append(duplicate)
        self.assertEqual(logged_performance(query)["known_result_count"], 1)
        duplicate["score"] = 0
        self.assertIsNone(logged_performance(query))
        reference, reason = match_reference(query, {}, "hash")
        self.assertIsNone(reference)
        self.assertIn("logged_performance", reason)

    def test_current_file_check_does_not_inspect_next_submission(self):
        raw = self.root / "raw"
        for item in self.train + self.dev + self.test:
            directory = raw / "all_labs" / item["lab"] / item["student_id"] / item["current_timestamp"]
            directory.mkdir(parents=True)
            (directory / item["source_file"]).write_text(item["current_code"])
            future = directory.parent / "2020-01-20-10-00-00"
            future.mkdir()
            (future / "extra.py").write_text("future must not influence filtering")
            (future / item["source_file"]).write_text("future")
        report = select_inputs(self.prepared, self.root / "single", self.scope, source_root=raw)
        self.assertEqual(report["splits"]["dev"]["selected_samples"], 1)
        with self.assertRaisesRegex(ValueError, "read-only raw source"):
            select_inputs(self.prepared, raw / "selection", self.scope, source_root=raw)
        current = raw / "all_labs/lab02/student-dev/2020-01-03-10-00-00"
        (current / "extra.py").write_text("current companion file")
        report = select_inputs(self.prepared, self.root / "multi", self.scope, source_root=raw)
        self.assertEqual(report["splits"]["dev"]["selected_samples"], 0)
        self.assertEqual(report["splits"]["dev"]["excluded_by_primary_reason"],
                         {"current_submission_not_exactly_one_declared_python_file": 1})

    def test_labels_must_align_and_students_must_not_cross_splits(self):
        self.write("labels.dev.jsonl", [{"sample_id": "wrong", "target_code": "x"}])
        with self.assertRaisesRegex(ValueError, "label IDs"):
            select_inputs(self.prepared, self.root / "badlabels", self.scope)
        self.write_split("dev", [row("dev-a", "student-train")])
        with self.assertRaisesRegex(ValueError, "student split leakage"):
            select_inputs(self.prepared, self.root / "badstudents", self.scope)

    def test_output_cannot_overwrite_or_modify_prepared_source(self):
        with self.assertRaisesRegex(ValueError, "read-only"):
            select_inputs(self.prepared, self.prepared / "new", self.scope)
        output = self.root / "output"
        select_inputs(self.prepared, output, self.scope)
        with self.assertRaisesRegex(ValueError, "not overwritten"):
            select_inputs(self.prepared, output, self.scope)


if __name__ == "__main__":
    unittest.main()
