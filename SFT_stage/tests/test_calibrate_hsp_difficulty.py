import unittest

from SFT_stage.calibrate_hsp_difficulty import (
    aggregate_accuracy,
    difficulty_bin,
    difficulty_histogram,
    extract_last_boxed,
    filter_band,
    make_scorer,
    select_to_target,
    stratify_balanced,
)


def problems(rates):
    return [{"id": f"p{i}", "solve_rate": rate} for i, rate in enumerate(rates)]


class CalibrationLogicTest(unittest.TestCase):
    def test_aggregate_accuracy_is_mean_solve_rate(self):
        self.assertAlmostEqual(aggregate_accuracy(problems([0.0, 0.5, 1.0])), 0.5)
        self.assertEqual(aggregate_accuracy([]), 0.0)

    def test_histogram_buckets(self):
        hist = difficulty_histogram(problems([0.0, 0.2, 0.5, 0.75, 0.9, 1.0]))
        self.assertEqual(hist["0.0"], 1)
        self.assertEqual(hist["(0,0.25]"], 1)
        self.assertEqual(hist["(0.25,0.5]"], 1)
        self.assertEqual(hist["(0.5,0.75]"], 1)
        self.assertEqual(hist["(0.75,1.0)"], 1)
        self.assertEqual(hist["1.0"], 1)

    def test_filter_band_keeps_inclusive_range(self):
        kept = filter_band(problems([0.0, 0.25, 0.5, 0.75, 1.0]), 0.25, 0.75)
        self.assertEqual([r["solve_rate"] for r in kept], [0.25, 0.5, 0.75])

    def test_select_to_target_drops_unsolved_by_default(self):
        kept = select_to_target(problems([0.0, 0.0, 1.0, 1.0]), target_accuracy=1.0)
        self.assertTrue(all(r["solve_rate"] > 0.0 for r in kept))

    def test_select_to_target_raises_accuracy_when_too_hard(self):
        # mean of [0,0,0.5,1,1] = 0.5; dropping zeros -> [0.5,1,1] mean 0.833
        # target 0.75 should drop until mean within tolerance
        kept = select_to_target(problems([0.0, 0.0, 0.5, 1.0, 1.0]), target_accuracy=0.75, tolerance=0.05)
        self.assertGreaterEqual(aggregate_accuracy(kept), 0.70)
        self.assertLessEqual(aggregate_accuracy(kept), 0.80)

    def test_select_to_target_lowers_accuracy_when_too_easy(self):
        # all easy -> must drop easiest to approach 0.70
        kept = select_to_target(
            problems([0.6, 0.7, 0.8, 1.0, 1.0, 1.0]), target_accuracy=0.70, tolerance=0.03
        )
        self.assertLessEqual(abs(aggregate_accuracy(kept) - 0.70), 0.05)

    def test_select_to_target_can_keep_unsolved_when_requested(self):
        kept = select_to_target(
            problems([0.0, 0.5, 1.0]), target_accuracy=0.5, tolerance=0.5, drop_unsolved=False
        )
        self.assertTrue(any(r["solve_rate"] == 0.0 for r in kept))

    def test_stratify_keeps_easy_and_hard_balanced(self):
        # 10 easy, 4 hard(0.0), 2 medium -> balanced keeps min-bin count from each bin
        recs = problems([1.0] * 10 + [0.0] * 4 + [0.5] * 2)
        kept = stratify_balanced(recs, seed=0)
        hist = difficulty_histogram(kept)
        # smallest non-empty bin has 2 -> each present bin capped at 2
        self.assertEqual(hist["1.0"], 2)
        self.assertEqual(hist["0.0"], 2)
        self.assertEqual(hist["(0.25,0.5]"], 2)
        # crucially: BOTH easy and hard survive
        self.assertGreater(hist["1.0"], 0)
        self.assertGreater(hist["0.0"], 0)

    def test_stratify_respects_max_per_bin(self):
        recs = problems([1.0] * 10 + [0.0] * 10)
        kept = stratify_balanced(recs, max_per_bin=3, seed=0)
        hist = difficulty_histogram(kept)
        self.assertEqual(hist["1.0"], 3)
        self.assertEqual(hist["0.0"], 3)

    def test_difficulty_bin_boundaries(self):
        self.assertEqual(difficulty_bin(0.0), "0.0")
        self.assertEqual(difficulty_bin(0.25), "(0,0.25]")
        self.assertEqual(difficulty_bin(1.0), "1.0")

    def test_extract_last_boxed(self):
        self.assertEqual(extract_last_boxed("answer \\boxed{42}."), "42")
        self.assertEqual(extract_last_boxed("\\boxed{1} then \\boxed{2}"), "2")
        self.assertIsNone(extract_last_boxed("no boxed answer here"))

    def test_scorer_matches_boxed_answer(self):
        is_correct, name = make_scorer()
        self.assertIn(name, {"math_verify", "string_fallback"})
        self.assertTrue(is_correct("So \\boxed{42}.", "42"))
        self.assertFalse(is_correct("So \\boxed{41}.", "42"))
        self.assertFalse(is_correct("no answer", "42"))


if __name__ == "__main__":
    unittest.main()
