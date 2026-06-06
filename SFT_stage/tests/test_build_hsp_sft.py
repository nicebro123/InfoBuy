import random
import unittest

from SFT_stage.build_hsp_sft import SAMPLE_WEIGHTS, build_examples, normalize_record


class BuildHSPDataTest(unittest.TestCase):
    def test_normalize_removes_relay_call_markers(self):
        record = {
            "id": "x",
            "question": "Compute 1 + 1.",
            "answer": "First add. <call></call> The result is \\boxed{2}.",
        }
        normalized = normalize_record(record, 0)
        self.assertIsNotNone(normalized)
        self.assertNotIn("<call>", normalized["gold_solution"])
        self.assertEqual(normalized["gold_answer"], "2")

    def test_normalize_strips_hsp_markers_from_source_prompt_and_solution(self):
        normalized = normalize_record(
            {
                "question": "<ASK> Compute 1 + 1. <ENVIRONMENT_NOTICE>bad</ENVIRONMENT_NOTICE>",
                "answer": "<ACCEPT> \\boxed{2}",
            },
            0,
        )
        self.assertNotIn("<ASK>", normalized["question"])
        self.assertNotIn("<ENVIRONMENT_NOTICE>", normalized["question"])
        self.assertNotIn("<ACCEPT>", normalized["gold_solution"])

    def test_normalize_removes_sentence_punctuation_from_boxed_answer(self):
        normalized = normalize_record(
            {
                "question": "Compute in base 5.",
                "gold_solution": "Therefore the answer is \\boxed{30_5.}",
            },
            0,
        )
        self.assertEqual(normalized["gold_answer"], "30_5")
        self.assertTrue(normalized["gold_solution"].endswith("\\boxed{30_5}."))

    def test_emit_all_protocol_types(self):
        records = [
            {
                "id": "x",
                "question": "Compute 2 + 3.",
                "gold_solution": "Add the values. The final answer is \\boxed{5}.",
                "gold_answer": "5",
            }
        ]
        examples, skipped = build_examples(records, random.Random(0), 1, emit_all_types=True)
        self.assertEqual(skipped, 0)
        self.assertEqual({item["sample_type"] for item in examples}, set(SAMPLE_WEIGHTS))
        for item in examples:
            self.assertEqual(item["generation_mode"], "synthetic_protocol_seed")
            self.assertTrue(all(segment["loss"] is (segment["source"] == "student") for segment in item["segments"]))

    def test_provenance_is_retained_in_generated_protocol_examples(self):
        records = [
            {
                "id": "tracked",
                "question": "Compute 1 + 2.",
                "gold_solution": "Add. \\boxed{3}.",
                "source_dataset": "AI-MO/NuminaMath-CoT",
                "source_category": "synthetic_math",
                "source_row_index": 17,
            }
        ]
        examples, skipped = build_examples(records, random.Random(0), 1, emit_all_types=False)
        self.assertEqual(skipped, 0)
        self.assertEqual(examples[0]["source_dataset"], "AI-MO/NuminaMath-CoT")
        self.assertEqual(examples[0]["source_category"], "synthetic_math")
        self.assertEqual(examples[0]["source_row_index"], 17)


if __name__ == "__main__":
    unittest.main()
