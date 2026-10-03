"""Synthetic fixtures only: these tests do not execute a student's program."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from student_sim_cd.progfeed import REQUIRED_COLUMNS, build, student_splits


class ProgFeedTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "raw"
        self.source.mkdir()
        (self.source / "all_labs").mkdir()
        descriptions = self.source / "autograders" / "lab01_autograder_github"
        descriptions.mkdir(parents=True)
        (descriptions / "f_desc.txt").write_text("Return the sum of the two arguments.", encoding="utf-8")
        self.rows = []

    def add(self, timestamp, code="def f(a, b):\n    return a + b\n", student="student_1",
            source_file="answer.py", feedback="", assigned="no_feedback", test_name="f",
            mask="[1, 0]", write_code=True, lab="lab01"):
        directory = self.source / "all_labs" / lab / student / timestamp
        directory.mkdir(parents=True, exist_ok=True)
        if write_code:
            (directory / source_file).write_bytes(code.encode("utf-8"))
        (directory / "results.json").write_text(json.dumps({
            "tests": [{"name": test_name, "output": "🤖 AI Feedback for you " + feedback if feedback else ""}]
        }), encoding="utf-8")
        row = {name: "" for name in REQUIRED_COLUMNS}
        row.update(student_id=student, lab=lab, submission_timestamp=timestamp,
                   source_file=source_file, test_name=test_name, function_name=test_name,
                   score="1", max_score="2", status="failed", testcase_mask=mask,
                   ai_feedback_type=assigned, ai_feedback_text=feedback)
        self.rows.append(row)
        return row

    def run_build(self, output="built"):
        with (self.source / "all_submissions_consolidated.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted(REQUIRED_COLUMNS))
            writer.writeheader()
            writer.writerows(self.rows)
        destination = self.root / output
        audit = build(self.source, destination)
        inputs, labels = [], []
        for split in ("train", "dev", "test"):
            inputs.extend(json.loads(line) for line in (destination / ("inputs.%s.jsonl" % split)).read_text().splitlines())
            labels.extend(json.loads(line) for line in (destination / ("labels.%s.jsonl" % split)).read_text().splitlines())
        return audit, inputs, labels

    def test_aggregate_rows_sort_time_and_keep_strictly_past_history(self):
        self.add("2025-01-01-10-02-00", code="THIRD_FUTURE_CODE", feedback="FUTURE_FEEDBACK", assigned="nl")
        self.add("2025-01-01-10-00-00", code="FIRST", feedback="Try addition", assigned="nl")
        second = self.add("2025-01-01-10-01-00", code="SECOND\r\n", assigned="nl")
        self.add("2025-01-01-10-01-00", code="SECOND\r\n", test_name="g", assigned="nl")["function_name"] = "f"
        self.rows.append(dict(second))
        audit, inputs, labels = self.run_build()
        self.assertEqual(2, len(inputs))
        self.assertEqual(2, len(inputs[1]["current_results"]))
        self.assertEqual([], inputs[0]["history"])
        self.assertEqual("FIRST", inputs[1]["history"][0]["code"])
        self.assertEqual("SECOND\r\n", inputs[1]["current_code"])
        self.assertEqual("THIRD_FUTURE_CODE", labels[1]["target_code"])
        self.assertNotIn("THIRD_FUTURE_CODE", json.dumps(inputs))
        self.assertNotIn("FUTURE_FEEDBACK", json.dumps(inputs))
        self.assertNotIn("target_timestamp", json.dumps(inputs))
        self.assertEqual([], inputs[1]["feedback"])
        self.assertEqual("Try addition", inputs[0]["feedback"][0]["text"])
        self.assertIn("sum", inputs[0]["problem_statement"])
        self.assertEqual(1, audit["counts"]["duplicate_csv_rows_removed"])
        self.assertEqual(1, audit["counts"]["trajectories_requiring_time_sort"])
        self.assertEqual(1, audit["counts"]["file_states_without_observed_next_submission"])

    def test_missing_code_does_not_bridge_adjacent_submission(self):
        self.add("2025-01-01-10-00-00", code="A")
        self.add("2025-01-01-10-01-00", write_code=False)
        self.add("2025-01-01-10-02-00", code="B")
        audit, inputs, labels = self.run_build()
        self.assertEqual([], inputs)
        self.assertEqual([], labels)
        self.assertEqual(2, audit["counts"]["adjacent_file_pairs_missing_code"])
        self.assertEqual(1, audit["counts"]["missing_or_unreadable_source_files"])

    def test_directory_absent_from_csv_is_still_next_submission(self):
        self.add("2025-01-01-10-00-00", code="A")
        self.add("2025-01-01-10-01-00", code="B")
        self.rows.pop()
        self.add("2025-01-01-10-02-00", code="C")
        audit, inputs, labels = self.run_build()
        self.assertEqual(2, len(inputs))
        self.assertEqual("B", labels[0]["target_code"])
        self.assertEqual([], labels[0]["target_results"])
        self.assertEqual(1, audit["counts"]["submission_directories_without_csv_rows"])

    def test_filename_change_is_not_guessed_or_called_unreadable_code(self):
        self.add("2025-01-01-10-00-00", source_file="hello.py.py", code="FIRST")
        self.add("2025-01-01-10-01-00", source_file="hello.py", code="SECOND")
        audit, inputs, _ = self.run_build()
        self.assertEqual([], inputs)
        self.assertEqual(0, audit["counts"]["missing_or_unreadable_source_files"])
        self.assertEqual(1, audit["counts"]["adjacent_file_pairs_with_source_name_change_or_absence"])
        excluded = json.loads((self.root / "built/excluded_pairs.jsonl").read_text())
        self.assertIn("possible_rename_or_deletion", excluded["reason"])

    def test_duplicates_multifile_and_masks_are_audited_not_filtered(self):
        for time in ("2025-01-01-10-00-00", "2025-01-01-10-01-00"):
            self.add(time, source_file="one.py", code="SAME", mask="")
            self.add(time, source_file="two.py", code="ALSO_SAME", mask="[true]")
        audit, inputs, labels = self.run_build()
        self.assertEqual(2, len(inputs))
        self.assertEqual(2, audit["counts"]["unchanged_code_pairs"])
        self.assertEqual(2, audit["counts"]["multi_file_submissions"])
        self.assertEqual(2, audit["counts"]["rows_missing_testcase_mask"])
        self.assertEqual(2, audit["counts"]["rows_invalid_testcase_mask"])
        self.assertTrue(all(r["current_results"][0]["testcase_mask"] is None for r in inputs))

    def test_empty_submitted_source_is_an_observed_state(self):
        self.add("2025-01-01-10-00-00", code="")
        self.add("2025-01-01-10-01-00", code="print('hello')\n")
        _, inputs, labels = self.run_build()
        self.assertEqual(1, len(inputs))
        self.assertEqual("", inputs[0]["current_code"])
        self.assertEqual("print('hello')\n", labels[0]["target_code"])

    def test_student_split_is_fixed_and_student_disjoint(self):
        students = ["s%02d" % i for i in range(20)]
        first = student_splits(students, 41)
        self.assertEqual(first, student_splits(reversed(students), 41))
        self.assertEqual(16, list(first.values()).count("train"))
        self.assertEqual(2, list(first.values()).count("dev"))
        self.assertEqual(2, list(first.values()).count("test"))
        for student in students:
            self.add("2025-01-01-10-00-00", student=student)
            self.add("2025-01-01-10-01-00", student=student)
        self.run_build()
        seen = {}
        for split in ("train", "dev", "test"):
            path = self.root / "built" / ("inputs.%s.jsonl" % split)
            for line in path.read_text().splitlines():
                student = json.loads(line)["student_id"]
                self.assertNotIn(student, seen)
                seen[student] = split
        self.assertEqual(20, len(seen))

    def test_repeated_build_is_byte_deterministic_and_cannot_overwrite(self):
        self.add("2025-01-01-10-00-00")
        self.add("2025-01-01-10-01-00")
        self.run_build("first")
        self.run_build("second")
        for first in (self.root / "first").iterdir():
            self.assertEqual(first.read_bytes(), (self.root / "second" / first.name).read_bytes())
        with self.assertRaisesRegex(ValueError, "refusing to overwrite"):
            build(self.source, self.root / "first")

    def test_missing_problem_statement_is_explicit(self):
        self.add("2025-01-01-10-00-00", lab="lab99")
        self.add("2025-01-01-10-01-00", lab="lab99")
        audit, inputs, _ = self.run_build()
        self.assertIsNone(inputs[0]["problem_statement"])
        self.assertEqual(1, audit["counts"]["pairs_missing_text_problem_statement"])

    def test_nonempty_description_for_a_different_task_is_rejected(self):
        self.add("2025-01-01-10-00-00", test_name="thermostat")
        self.add("2025-01-01-10-01-00", test_name="thermostat")
        audit, inputs, _ = self.run_build()
        self.assertIsNone(inputs[0]["problem_statement"])
        self.assertEqual(1, audit["counts"]["labs_with_description_function_mismatch"])
        self.assertTrue(audit["problem_statements"]["lab01"]["raw_text_available"])

    def test_source_symlink_escape_is_not_read(self):
        self.add("2025-01-01-10-00-00", write_code=False)
        self.add("2025-01-01-10-01-00")
        outside = self.root / "outside.py"
        outside.write_text("THIS_MUST_NOT_ENTER_INPUTS", encoding="utf-8")
        path = self.source / "all_labs/lab01/student_1/2025-01-01-10-00-00/answer.py"
        path.symlink_to(outside)
        audit, inputs, _ = self.run_build()
        self.assertEqual([], inputs)
        self.assertEqual(1, audit["counts"]["missing_or_unreadable_source_files"])


if __name__ == "__main__":
    unittest.main()
