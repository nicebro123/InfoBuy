import unittest

from eval.summarize_hsp_results import summarize_items, summarize_paired_calibration


class SummarizeHSPResultsTest(unittest.TestCase):
    def test_action_and_cost_groups_are_aggregated(self):
        items = [
            {
                "interaction_policy": "hsp",
                "collection_mode": "independent",
                "data_role": "train",
                "score": 1,
                "ask_count": 0,
                "verify_count": 0,
                "accept_count": 0,
                "teacher_tokens_used": 0,
                "events": [],
            },
            {
                "interaction_policy": "hsp",
                "collection_mode": "force_verify_after_draft",
                "data_role": "train",
                "score": 1,
                "ask_count": 0,
                "verify_count": 1,
                "accept_count": 1,
                "teacher_tokens_used": 20,
                "events": [{"action": "verify", "feedback_truncated": True}],
            },
            {
                "interaction_policy": "hsp",
                "collection_mode": "policy",
                "data_role": "test",
                "score": 0,
                "ask_count": 1,
                "verify_count": 0,
                "accept_count": 0,
                "teacher_tokens_used": 10,
                "events": [{"action": "ask", "error": "timeout"}],
            },
        ]
        summary = summarize_items(items)
        self.assertEqual(summary["examples"], 3)
        self.assertAlmostEqual(summary["mean_score"], 2 / 3)
        self.assertEqual(summary["teacher_tokens_total"], 30)
        self.assertEqual(summary["action_totals"]["ask_count"], 1)
        self.assertEqual(summary["action_totals"]["verify_count"], 1)
        self.assertEqual(summary["action_totals"]["invalid_protocol_count"], 0)
        self.assertEqual(summary["groups"]["no_interaction"]["count"], 1)
        self.assertEqual(summary["groups"]["any_interaction"]["count"], 2)
        self.assertEqual(summary["feedback_truncated_count"], 1)
        self.assertEqual(summary["event_error_count"], 1)
        self.assertEqual(summary["collection_mode_counts"]["force_verify_after_draft"], 1)
        self.assertEqual(summary["data_role_counts"], {"test": 1, "train": 2})

    def test_rejects_non_hsp_input(self):
        with self.assertRaisesRegex(ValueError, "No HSP"):
            summarize_items([{"interaction_policy": "relay_call", "score": 1}])

    def test_paired_calibration_reports_rescue_and_implicit_adoption(self):
        items = [
            {
                "question": "p",
                "sample_index": 0,
                "interaction_policy": "hsp",
                "collection_mode": "independent",
                "score": 0,
                "teacher_tokens_used": 0,
            },
            {
                "question": "p",
                "sample_index": 0,
                "interaction_policy": "hsp",
                "collection_mode": "force_verify_after_draft",
                "score": 1,
                "verify_count": 1,
                "teacher_tokens_used": 96,
                "events": [{"action": "verify", "accepted": False, "implicit_adoption_without_accept": True}],
            },
        ]
        report = summarize_paired_calibration(items)
        comparison = report["comparisons_vs_independent"]["force_verify_after_draft"]
        self.assertEqual(comparison["paired_examples"], 1)
        self.assertEqual(comparison["rescued_failures"], 1)
        self.assertAlmostEqual(comparison["mean_score_delta_vs_independent"], 1.0)
        self.assertEqual(report["verify_trust"]["implicit_adoptions_without_accept"], 1)


if __name__ == "__main__":
    unittest.main()
