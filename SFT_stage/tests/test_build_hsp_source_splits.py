import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path

from SFT_stage import build_hsp_source_splits as source_splits
from SFT_stage.build_hsp_source_splits import (
    deduplicate_source_records,
    deterministic_split,
    find_contamination,
    load_heldout_snapshot,
    normalize_question,
    prepare_records,
)


class BuildHSPSourceSplitsTest(unittest.TestCase):
    def test_normalize_removes_instruction_suffix(self):
        first = normalize_question("Compute $1+1$.\\nPlease reason step by step, and put your final answer within \\\\boxed{}.")
        second = normalize_question(" compute $1 + 1$. ")
        self.assertEqual(first, second)

    def test_filters_exact_and_near_duplicates(self):
        records = [
            {"id": "exact", "question": "Find the value of x if x + 2 = 4."},
            {"id": "near", "question": "Evaluate the expression (1+2+3+4+5+6+7+8+9+10)."},
            {"id": "clean", "question": "Solve y^2 = 49 for positive y."},
        ]
        heldout = {
            "test": [
                "Find the value of x if x + 2 = 4.",
                "Evaluate the expression (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10).",
            ]
        }
        clean, removed = find_contamination(records, heldout, 0.85)
        self.assertEqual([record["id"] for record in clean], ["clean"])
        self.assertEqual({item["reason"] for item in removed}, {"exact_match"})

    def test_checks_lower_overlap_candidate_with_lower_union_threshold(self):
        question = "".join(chr(0x1000 + index) for index in range(100))
        long_non_match = question + "".join(chr(0x2000 + index) for index in range(100))
        actual_near_match = question[:-4]
        clean, removed = find_contamination(
            [{"id": "target", "question": question}],
            {"test": [long_non_match, actual_near_match]},
            0.85,
        )
        self.assertEqual(clean, [])
        self.assertEqual(removed[0]["reason"], "near_duplicate")

    def test_deduplicates_normalized_questions_before_split(self):
        records = [
            {"id": "b", "question": "Compute $x + 1$."},
            {"id": "a", "question": "compute $x+1$."},
            {"id": "c", "question": "Compute $y+1$."},
        ]
        unique, removed = deduplicate_source_records(records)
        self.assertEqual([record["id"] for record in unique], ["a", "c"])
        self.assertEqual(removed, [{"id": "b", "reason": "internal_normalized_duplicate", "duplicate_of": "a"}])

    def test_prepares_rl_answer_field_from_solution(self):
        prepared, removed = prepare_records(
            [{"id": "p", "question": "Compute 1+1.", "gold_solution": "Thus \\boxed{2}."}]
        )
        self.assertEqual(removed, [])
        self.assertEqual(prepared[0]["gold_answer"], "2")

    def test_retries_transient_heldout_viewer_gateway_failure(self):
        outcomes = [
            urllib.error.HTTPError("url", 502, "gateway", {}, None),
            io.BytesIO(json.dumps({"rows": []}).encode()),
        ]
        delays = []

        def opener(_url, timeout=60):
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        original_sleep = source_splits.time.sleep
        source_splits.time.sleep = delays.append
        try:
            payload = source_splits._get_json("url", opener=opener, max_retries=1)
        finally:
            source_splits.time.sleep = original_sleep
        self.assertEqual(payload, {"rows": []})
        self.assertEqual(delays, [2])

    def test_loads_saved_heldout_snapshot_for_reproducible_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "heldout.json"
            path.write_text(json.dumps({"questions": {"test": ["Compute 1+1."]}}), encoding="utf-8")
            self.assertEqual(load_heldout_snapshot(path), {"test": ["Compute 1+1."]})

    def test_deterministic_split_is_disjoint(self):
        records = [{"id": f"p{index}", "question": str(index)} for index in range(5)]
        train, validation = deterministic_split(records, 3, 2, seed=42)
        self.assertEqual(len(train), 3)
        self.assertEqual(len(validation), 2)
        self.assertFalse({item["id"] for item in train} & {item["id"] for item in validation})
        self.assertEqual((train, validation), deterministic_split(records, 3, 2, seed=42))


if __name__ == "__main__":
    unittest.main()
