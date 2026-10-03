import copy
import hashlib
import unittest

from student_sim_cd.history_interventions import (
    DEFAULT_SEEDS, DonorUnavailableError, build_interventions, jsonl_bytes,
)
from student_sim_cd.inference import canonical_json, object_hash


def row(sid, student, code, *, day=10, lab="lab02", source_file="to_do_list.py", history_code=None):
    return {
        "schema_version": "student-sim-cd.progfeed.v1", "sample_id": sid,
        "student_id": student, "lab": lab, "source_file": source_file,
        "current_timestamp": f"2020-01-{day:02}-10-00-00", "current_code": code,
        "history": [{"timestamp": "2020-01-01-10-00-00", "code": history_code or code + "_past",
                     "results": [], "feedback": []}],
        "feedback": [{"test_name": "test", "function_name": "f", "assigned_type": "nl",
                      "text": "Check the loop."}],
        "current_results": [{"test_name": "test", "function_name": "f", "status": "partial",
                             "score": 0.5, "max_score": 1, "testcase_mask": None}],
        "problem_statement": "Maintain a to-do list.",
    }


class HistoryInterventionTests(unittest.TestCase):
    def setUp(self):
        self.queries = [row("q1", "dev1", "current1"), row("q2", "dev2", "current2")]
        self.train = [row("original", "train-original", "original", day=4),
                      row("near", "train-near", "whole near program\n", day=4),
                      row("far", "train-far", "far", day=4)]
        self.protocol = "1" * 64
        self.fingerprint = "2" * 64
        self.reset_sources()

    def reset_sources(self):
        self.train_bytes = jsonl_bytes(self.train)
        self.train_sha = hashlib.sha256(self.train_bytes).hexdigest()
        original = self.train[0]
        self.refs = [{
            "sample_id": query["sample_id"], "reference_kind": "matched",
            "reference_history": copy.deepcopy(original["history"]),
            "provenance": {
                "split": "train", "training_inputs_sha256": self.train_sha,
                "donor_sample_id": original["sample_id"], "donor_student_id": original["student_id"],
                "donor_current_timestamp": original["current_timestamp"],
                "query_current_timestamp": query["current_timestamp"],
                "lab": query["lab"], "source_file": query["source_file"], "labels_used": False,
            },
        } for query in self.queries]
        self.counts = {object_hash(item["history"]): 100 for item in self.queries + self.train}
        self.counts[object_hash(self.train[-1]["history"])] = 500

    def build(self, **kwargs):
        options = {
            "training_inputs_sha256": self.train_sha, "training_inputs_bytes": self.train_bytes,
            "protocol_sha256": self.protocol, "token_lengths": self.counts,
            "tokenizer_fingerprint_sha256": self.fingerprint,
            "expected_samples": len(self.queries),
            "expected_students": len({item["student_id"] for item in self.queries}),
        }
        options.update(kwargs)
        return build_interventions(self.queries, self.refs, self.train, **options)

    def test_complete_single_factor_variants_keep_current_information_and_original_sources(self):
        before = copy.deepcopy((self.queries, self.refs, self.train))
        result = self.build()
        self.assertEqual((self.queries, self.refs, self.train), before)
        self.assertEqual(result["baseline"]["inputs"], self.queries)
        self.assertEqual(result["baseline"]["references"], self.refs)
        self.assertEqual(result["manifest"]["seeds"], list(DEFAULT_SEEDS))
        for seed in DEFAULT_SEEDS:
            real = result["variants"]["real_history_swapped"][str(seed)]
            ref = result["variants"]["reference_swapped"][str(seed)]
            self.assertEqual(real["references"], self.refs)
            self.assertEqual(ref["inputs"], self.queries)
            for original, output in zip(self.queries, real["inputs"]):
                self.assertEqual({k: v for k, v in original.items() if k != "history"},
                                 {k: v for k, v in output.items() if k != "history"})
                self.assertEqual(output["history"], self.train[1]["history"])
            for output in ref["references"]:
                self.assertEqual(output["reference_history"], self.train[1]["history"])
                self.assertFalse(output["provenance"]["labels_used"])
            for bundle in (real, ref):
                self.assertEqual(len(bundle["assignments"]), 2)
                self.assertEqual(bundle["assignments_bytes"], jsonl_bytes(bundle["assignments"]))
                for field in ("inputs", "references", "assignments"):
                    self.assertEqual(bundle[field + "_sha256"],
                                     hashlib.sha256(bundle[field + "_bytes"]).hexdigest())
                self.assertEqual(bundle["donor_assignment_sha256"], bundle["assignments_sha256"])
            self.assertEqual(real["history_data_sha256"], real["inputs_sha256"])
            self.assertEqual(ref["history_data_sha256"], ref["references_sha256"])

    def test_full_70_samples_and_17_students_retained_without_subsampling(self):
        self.queries = [row(f"q{i}", f"dev{i % 17}", f"current{i}", day=20) for i in range(70)]
        self.reset_sources()
        result = self.build(expected_samples=70, expected_students=17)
        for variants in result["variants"].values():
            for bundle in variants.values():
                self.assertEqual(bundle["manifest"]["samples"], 70)
                self.assertEqual(bundle["manifest"]["students"], 17)
                self.assertEqual([item["sample_id"] for item in bundle["inputs"]],
                                 [item["sample_id"] for item in self.queries])
                self.assertEqual(len(bundle["assignments"]), 70)
        with self.assertRaisesRegex(ValueError, "exactly the expected"):
            self.build(expected_samples=69, expected_students=17)

    def test_token_nearest_can_differ_from_byte_nearest_and_keeps_full_history(self):
        self.train[1]["history"] = [
            {"timestamp": "2020-01-01-10-00-00", "code": "x" * 5000, "results": [], "feedback": []},
            {"timestamp": "2020-01-02-10-00-00", "code": "y" * 5000, "results": [], "feedback": []},
        ]
        self.reset_sources()
        self.counts[object_hash(self.train[1]["history"])] = 101
        self.counts[object_hash(self.train[2]["history"])] = 200
        result = self.build()
        for variants in result["variants"].values():
            for bundle in variants.values():
                self.assertEqual({a["donor_sample_id"] for a in bundle["assignments"]}, {"near"})
                self.assertEqual({a["absolute_length_difference"] for a in bundle["assignments"]}, {1})
        out = result["variants"]["real_history_swapped"][str(DEFAULT_SEEDS[0])]["inputs"][0]
        self.assertEqual(out["history"], self.train[1]["history"])
        self.assertEqual([item["timestamp"] for item in out["history"]],
                         ["2020-01-01-10-00-00", "2020-01-02-10-00-00"])

    def test_no_labels_or_current_logged_performance_are_used_to_assign(self):
        first = self.build()
        for item in self.queries + self.train:
            item["current_results"][0]["score"] = None
            item["current_code"] = "a different current program"
            item["feedback"][0]["text"] = "different feedback"
        self.reset_sources()
        second = self.build()
        for variant in first["variants"]:
            for seed in first["variants"][variant]:
                self.assertEqual([a["donor_sample_id"] for a in first["variants"][variant][seed]["assignments"]],
                                 [a["donor_sample_id"] for a in second["variants"][variant][seed]["assignments"]])
        self.queries[0]["target_code"] = "LABEL MUST NOT ENTER"
        with self.assertRaisesRegex(ValueError, "forbidden or unknown fields"):
            self.build()

    def test_seeded_ties_are_repeatable_and_input_donor_order_independent(self):
        self.counts[object_hash(self.train[-1]["history"])] = 100
        first = self.build()
        second = self.build()
        self.assertEqual(first, second)
        self.train[1:] = list(reversed(self.train[1:]))
        self.reset_sources()
        self.counts = {key: 100 for key in self.counts}
        reversed_result = self.build()
        for variant in first["variants"]:
            for seed in first["variants"][variant]:
                self.assertEqual([a["donor_sample_id"] for a in first["variants"][variant][seed]["assignments"]],
                                 [a["donor_sample_id"] for a in reversed_result["variants"][variant][seed]["assignments"]])

    def test_identical_nearest_assignments_are_reported_not_claimed_as_independent(self):
        result = self.build()
        self.assertEqual(result["manifest"]["unique_assignments_across_seeds"],
                         {"real_history_swapped": 1, "reference_swapped": 1})
        self.assertEqual(result["manifest"]["assignment_summaries"]["real_history_swapped"]
                         [str(DEFAULT_SEEDS[0])]["maximum_donor_reuse"], 2)

    def test_original_donor_and_equal_original_histories_excluded_even_if_shorter(self):
        self.train.extend([
            row("same-ref", "train-same-ref", "other", day=4,
                history_code=self.train[0]["history"][0]["code"]),
            row("same-real", "train-same-real", "other", day=4,
                history_code=self.queries[0]["history"][0]["code"]),
        ])
        self.reset_sources()
        self.counts = {key: 100 for key in self.counts}
        self.counts[object_hash(self.train[2]["history"])] = 500
        result = self.build()
        for variants in result["variants"].values():
            for bundle in variants.values():
                a = bundle["assignments"][0]
                self.assertEqual(a["donor_sample_id"], "near")
                self.assertNotIn(a["donor_history_sha256"],
                                 (a["real_history_sha256"], a["reference_history_sha256"]))

    def test_future_equal_time_wrong_task_and_empty_donors_never_used(self):
        self.train.extend([row("future", "train-future", "future", day=11),
                           row("equal", "train-equal", "equal", day=10),
                           row("wrong-task", "train-wrong", "wrong", day=4, lab="lab09"),
                           row("empty", "train-empty", "empty", day=4)])
        self.train[-1]["history"] = []
        self.reset_sources()
        self.counts[object_hash(self.train[2]["history"])] = 500
        result = self.build()
        for variants in result["variants"].values():
            for bundle in variants.values():
                self.assertEqual({a["donor_sample_id"] for a in bundle["assignments"]}, {"near"})

    def test_any_missing_legal_donor_stops_whole_cohort_and_reports_ids(self):
        self.queries[1]["current_timestamp"] = "2020-01-03-10-00-00"
        self.train[0]["current_timestamp"] = "2020-01-02-10-00-00"
        self.reset_sources()
        before = copy.deepcopy((self.queries, self.refs, self.train))
        with self.assertRaises(DonorUnavailableError) as captured:
            self.build()
        self.assertEqual((self.queries, self.refs, self.train), before)
        self.assertEqual([failure["sample_id"] for failure in captured.exception.failures], ["q2"])
        self.assertEqual(captured.exception.failures[0]["excluded_train_records"],
                         {"donor_not_strictly_past": 2, "original_matched_donor": 1})

    def test_train_students_and_sample_ids_must_be_disjoint_from_entire_query_cohort(self):
        self.train[1]["student_id"] = "dev2"
        self.reset_sources()
        with self.assertRaisesRegex(ValueError, "student split leakage"):
            self.build()
        self.train[1]["student_id"] = "train-near"
        self.train[1]["sample_id"] = "q2"
        self.reset_sources()
        with self.assertRaisesRegex(ValueError, "sample IDs overlap"):
            self.build()

    def test_incomplete_counts_or_missing_fingerprint_cannot_fall_back_to_bytes(self):
        missing = dict(self.counts)
        missing.pop(object_hash(self.train[-1]["history"]))
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.build(token_lengths=missing)
        with self.assertRaisesRegex(ValueError, "complete tokenizer count"):
            self.build(token_lengths=None)
        with self.assertRaisesRegex(ValueError, "tokenizer_fingerprint"):
            self.build(tokenizer_fingerprint_sha256=None)
        for value in (0, -1, True, 1.5):
            bad = dict(self.counts)
            bad[next(iter(bad))] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "positive integers"):
                self.build(token_lengths=bad)

    def test_explicit_byte_diagnostic_is_marked_nonformal(self):
        result = self.build(length_measure="utf8_bytes_diagnostic", token_lengths=None,
                            tokenizer_fingerprint_sha256=None)
        self.assertFalse(result["manifest"]["formal_token_matching"])
        self.assertIsNone(result["manifest"]["token_serialization"])
        for variants in result["variants"].values():
            for bundle in variants.values():
                self.assertFalse(bundle["manifest"]["formal_token_matching"])
        with self.assertRaisesRegex(ValueError, "masquerade"):
            self.build(length_measure="utf8_bytes_diagnostic")

    def test_complete_frozen_train_bytes_and_matching_original_reference_required(self):
        with self.assertRaisesRegex(ValueError, "frozen training_inputs_sha256"):
            self.build(training_inputs_bytes=self.train_bytes + b" ")
        changed = copy.deepcopy(self.train)
        changed[1]["history"][0]["code"] = "forged donor"
        original = self.train
        self.train = changed
        with self.assertRaisesRegex(ValueError, "differ from the frozen source bytes"):
            self.build()
        self.train = original
        self.refs[0]["reference_history"][0]["code"] = "forged reference"
        with self.assertRaisesRegex(ValueError, "frozen strictly-past donor"):
            self.build()

    def test_reference_provenance_missing_rows_empty_or_mismatched_train_are_rejected(self):
        original = copy.deepcopy(self.refs)
        self.refs.pop()
        with self.assertRaisesRegex(ValueError, "missing rows"):
            self.build()
        self.refs = copy.deepcopy(original)
        self.refs[0]["reference_history"] = []
        with self.assertRaisesRegex(ValueError, "no empty fallback"):
            self.build()
        self.refs = copy.deepcopy(original)
        self.refs[0]["provenance"]["training_inputs_sha256"] = "3" * 64
        with self.assertRaisesRegex(ValueError, "provenance differs"):
            self.build()

    def test_invalid_internal_timeline_and_timezone_comparison_are_not_repaired(self):
        self.train[1]["history"].append(copy.deepcopy(self.train[1]["history"][0]))
        self.reset_sources()
        with self.assertRaisesRegex(ValueError, "strictly increase and precede"):
            self.build()
        self.train[1]["history"].pop()
        self.train[1]["current_timestamp"] = "2020-01-04T10:00:00Z"
        self.train[1]["history"][0]["timestamp"] = "2020-01-01T10:00:00Z"
        self.reset_sources()
        with self.assertRaisesRegex(ValueError, "incompatible timestamp timezone"):
            self.build()

    def test_bad_seeds_cohort_duplicates_and_nonformal_modes_are_rejected(self):
        for seeds in ([1, 2], (), (1, 1), (True,), (-1,)):
            with self.subTest(seeds=seeds), self.assertRaisesRegex(ValueError, "seeds"):
                self.build(seeds=seeds)
        with self.assertRaisesRegex(ValueError, "unknown length_measure"):
            self.build(length_measure="bytes")
        self.queries.append(copy.deepcopy(self.queries[0]))
        with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
            self.build()


if __name__ == "__main__":
    unittest.main()
