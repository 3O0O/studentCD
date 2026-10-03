import math
import unittest

from student_sim_cd.scoring import (
    LogProbs, MissingGroupError, calibrate_groups, components, method_scores, score, softmax,
)


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.logp = LogProbs(-7.0, -10.0, -12.0, -14.0)

    def test_definitions_and_cd_degeneracy(self):
        self.assertEqual(components(self.logp), {"base": -7, "d0": 2, "d1": 3, "gamma": 1})
        self.assertEqual(score(self.logp), -7)
        for weight in (-2, 0, 0.3, 2):
            self.assertAlmostEqual(
                score(self.logp, weight, weight),
                (1 + weight) * self.logp.l11 - weight * self.logp.l01,
            )
        self.assertEqual(method_scores(self.logp),
                         {"base": -7, "history_d0": -5, "cd": -4, "b": -6, "joint": -6})

    def test_identical_histories_remove_all_contrast(self):
        logp = LogProbs(-3, -3, -5, -5)
        self.assertTrue(all(value == -3 for value in method_scores(logp).values()))

    def test_no_feedback_effect_removes_interaction(self):
        logp = LogProbs(-3, -7, -3, -7)
        self.assertEqual(components(logp)["gamma"], 0)
        self.assertEqual(score(logp, beta=100), logp.l11)

    def test_group_masses_and_conditional_odds(self):
        scores, groups = [-1000, -1001, 3000], ["same", "same", "improved"]
        probabilities = calibrate_groups(scores, groups, {"same": 0.7, "improved": 0.3})
        self.assertAlmostEqual(sum(probabilities[:2]), 0.7)
        self.assertAlmostEqual(probabilities[2], 0.3)
        self.assertAlmostEqual(probabilities[0] / probabilities[1], math.e)

    def test_natural_group_mass_is_identity(self):
        scores, groups = [3, -2, 1], ["a", "b", "a"]
        original = softmax(scores)
        q = {"a": original[0] + original[2], "b": original[1]}
        calibrated = calibrate_groups(scores, groups, q)
        for left, right in zip(original, calibrated):
            self.assertAlmostEqual(left, right)

    def test_missing_positive_group_does_not_renormalize(self):
        with self.assertRaises(MissingGroupError) as raised:
            calibrate_groups([1], ["a"], {"a": 0.6, "b": 0.4})
        self.assertEqual(raised.exception.missing_groups, ("b",))
        self.assertEqual(calibrate_groups([1], ["a"], {"a": 1, "b": 0}), [1])

    def test_nonfinite_inputs_and_overflow_rejected(self):
        for invalid in (math.nan, math.inf, -math.inf, True):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    LogProbs(invalid, -1, -1, -1)
                with self.assertRaises(ValueError):
                    softmax([0, invalid])
                with self.assertRaises(ValueError):
                    score(self.logp, beta=invalid)
                with self.assertRaises(ValueError):
                    calibrate_groups([1], ["a"], {"a": invalid})
        with self.assertRaises(ValueError):
            score(LogProbs(-1, -1e308, -1, -1), beta=1e308)

    def test_invalid_q_and_candidate_alignment(self):
        for q in ({"a": 0.8}, {"a": -0.1, "b": 1.1}, {"b": 1}):
            with self.assertRaises(ValueError):
                calibrate_groups([1], ["a"], q)
        with self.assertRaises(ValueError):
            calibrate_groups([1, 2], ["a"], {"a": 1})
        with self.assertRaises(ValueError):
            softmax([])
        with self.assertRaises(ValueError):
            LogProbs(0.1, -1, -1, -1)


if __name__ == "__main__":
    unittest.main()
