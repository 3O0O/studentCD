import hashlib
import math
import unittest

from student_sim_cd.a_static_analysis import analyze_pool, crps, wasserstein_1
from student_sim_cd.scoring import MissingGroupError


class AStaticAnalysisTests(unittest.TestCase):
    def fixture(self, extra=0):
        inputs = [{"sample_id": "s", "student_id": "u", "current_code": "x=0\n"}]
        labels = [{"sample_id": "s", "target_code": "x=1\n"}]
        codes = ["x=0\n", "x=1\n", "wrong=2\n"] + [f"other{i}=3\n" for i in range(extra)]
        pool = {hashlib.sha256(code.encode()).hexdigest(): {"code": code} for code in codes}
        scores = {("s", cid): {method: -1. for method in ("base", "cd", "b", "history_d0")} for cid in pool}
        return inputs, labels, {"s": pool}, scores, [{"sample_id": "s", "student_id": "u", "p_changed": .8}]

    def test_group_marginals_same_q_and_argmax_not_q(self):
        report = analyze_pool(*self.fixture(extra=8), pool_name="synthetic", bootstrap=20)
        a = report["per_sample"]["a_base"][0]
        self.assertAlmostEqual(a["change_probability"], .8)
        self.assertEqual(a["argmax_changed"], 0)  # .2 copy vs .08 each changed.
        self.assertEqual(report["comparisons"]["a_cd_minus_a_base"]["change_brier"]["difference"], 0)

    def test_missing_group_fails_without_drop(self):
        x, y, p, s, q = self.fixture()
        p["s"] = {cid: c for cid, c in p["s"].items() if c["code"] == x[0]["current_code"]}
        s = {key: value for key, value in s.items() if key[1] in p["s"]}
        with self.assertRaises(MissingGroupError):
            analyze_pool(x, y, p, s, q, "synthetic", bootstrap=20)

    def test_same_location_wrong_content_does_not_claim_correctness(self):
        report = analyze_pool(*self.fixture(), pool_name="synthetic", bootstrap=20)
        row = report["per_sample"]["a_base"][0]
        self.assertAlmostEqual(row["expected_edit_location_f1"], .8)
        self.assertAlmostEqual(row["expected_exact_next_code"], .4)

    def test_crps_known_distribution_and_dirac(self):
        self.assertEqual(crps([0, 2], [.5, .5], 0), .5)
        self.assertEqual(crps([2], [1.], 5), 3)
        with self.assertRaises(ValueError):
            crps([1], [.7], 0)

    def test_weighted_wasserstein(self):
        self.assertEqual(wasserstein_1([(0, .5), (2, .5)], [(0, 1.)]), 1)
        self.assertEqual(wasserstein_1([(0, .5), (2, .5)], [(0, .5), (2, .5)]), 0)

    def test_nonfinite_and_nonpositive_q_rejected(self):
        for invalid in (0, 1, math.nan, True):
            x, y, p, s, q = self.fixture()
            q[0]["p_changed"] = invalid
            with self.assertRaises(ValueError):
                analyze_pool(x, y, p, s, q, "synthetic", bootstrap=20)

    def test_candidate_hash_and_score_coverage_enforced(self):
        x, y, p, s, q = self.fixture()
        s.pop(next(iter(s)))
        with self.assertRaises(ValueError):
            analyze_pool(x, y, p, s, q, "synthetic", bootstrap=20)

    def test_output_has_no_body_and_is_deterministic(self):
        first = analyze_pool(*self.fixture(), pool_name="synthetic", bootstrap=20)
        second = analyze_pool(*self.fixture(), pool_name="synthetic", bootstrap=20)
        self.assertEqual(first, second)
        self.assertNotIn("x=0", repr(first))
        self.assertNotIn("x=1", repr(first))


if __name__ == "__main__":
    unittest.main()
