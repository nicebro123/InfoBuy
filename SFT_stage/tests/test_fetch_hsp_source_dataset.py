import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path

from SFT_stage import fetch_hsp_source_dataset as source_dataset
from SFT_stage.fetch_hsp_source_dataset import fetch_records


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class FetchHSPSourceDatasetTest(unittest.TestCase):
    def test_filters_category_deduplicates_and_preserves_provenance(self):
        pages = [
            {
                "num_rows_total": 4,
                "rows": [
                    {"row_idx": 0, "row": {"source": "math", "problem": "skip", "solution": "skip"}},
                    {"row_idx": 1, "row": {"source": "synthetic_math", "problem": "P 1", "solution": "\\boxed{1}"}},
                ],
            },
            {
                "num_rows_total": 4,
                "rows": [
                    {"row_idx": 2, "row": {"source": "synthetic_math", "problem": "P   1", "solution": "\\boxed{1}"}},
                    {"row_idx": 3, "row": {"source": "synthetic_math", "problem": "P 2", "solution": "\\boxed{2}"}},
                ],
            },
        ]

        def opener(_url, timeout=60):
            self.assertEqual(timeout, 60)
            return Response(json.dumps(pages.pop(0)).encode())

        records, scanned = fetch_records("synthetic_math", max_records=2, page_size=2, opener=opener)
        self.assertEqual(scanned, 4)
        self.assertEqual([record["question"] for record in records], ["P 1", "P 2"])
        self.assertEqual(records[0]["source_dataset"], "AI-MO/NuminaMath-CoT")
        self.assertEqual(records[0]["source_category"], "synthetic_math")
        self.assertEqual(records[1]["source_row_index"], 3)

    def test_deduplicates_normalized_question_variants(self):
        rows = {
            "num_rows_total": 2,
            "rows": [
                {"row_idx": 0, "row": {"source": "synthetic_math", "problem": "Compute $x + 1$.", "solution": "a"}},
                {"row_idx": 1, "row": {"source": "synthetic_math", "problem": "compute $x+1$.", "solution": "b"}},
            ],
        }

        def opener(_url, timeout=60):
            return Response(json.dumps(rows).encode())

        with self.assertRaises(ValueError):
            fetch_records("synthetic_math", max_records=2, page_size=2, opener=opener)

    def test_rejects_invalid_page_size(self):
        with self.assertRaises(ValueError):
            fetch_records("synthetic_math", max_records=1, page_size=101)

    def test_shuffled_pages_avoids_prefix_only_selection(self):
        requested_offsets = []

        def opener(url, timeout=60):
            requested_offsets.append(int(url.split("offset=")[1].split("&")[0]))
            return Response(
                json.dumps(
                    {
                        "num_rows_total": 300,
                        "rows": [
                            {
                                "row_idx": requested_offsets[-1],
                                "row": {
                                    "source": "synthetic_math",
                                    "problem": f"P {requested_offsets[-1]}",
                                    "solution": "\\boxed{1}",
                                },
                            }
                        ],
                    }
                ).encode()
            )

        records, _ = fetch_records(
            "synthetic_math",
            max_records=2,
            page_size=100,
            opener=opener,
            sampling_mode="shuffled_pages",
            seed=0,
            total_rows=300,
        )
        self.assertEqual(len(records), 2)
        self.assertNotEqual(requested_offsets, [0, 100])

    def test_retries_rate_limited_viewer_request(self):
        outcomes = [
            urllib.error.HTTPError("url", 429, "rate limited", {}, None),
            Response(json.dumps({"rows": []}).encode()),
        ]
        delays = []

        def opener(_url, timeout=60):
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        original_sleep = source_dataset.time.sleep
        source_dataset.time.sleep = delays.append
        try:
            payload = source_dataset._get_json({}, opener=opener, max_retries=1)
        finally:
            source_dataset.time.sleep = original_sleep
        self.assertEqual(payload, {"rows": []})
        self.assertEqual(delays, [2])

    def test_resume_continues_after_committed_page(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "pool.checkpoint.json"
            first_pages = [
                {
                    "num_rows_total": 2,
                    "rows": [{"row_idx": 0, "row": {"source": "synthetic_math", "problem": "P 0", "solution": "S 0"}}],
                },
                urllib.error.URLError("interrupted"),
            ]

            def interrupted_opener(_url, timeout=60):
                result = first_pages.pop(0)
                if isinstance(result, Exception):
                    raise result
                return Response(json.dumps(result).encode())

            with self.assertRaises(urllib.error.URLError):
                fetch_records(
                    "synthetic_math",
                    max_records=2,
                    page_size=1,
                    opener=interrupted_opener,
                    max_retries=0,
                    checkpoint_path=checkpoint_path,
                )

            requested_offsets = []

            def resumed_opener(url, timeout=60):
                requested_offsets.append(int(url.split("offset=")[1].split("&")[0]))
                return Response(
                    json.dumps(
                        {
                            "num_rows_total": 2,
                            "rows": [
                                {
                                    "row_idx": 1,
                                    "row": {"source": "synthetic_math", "problem": "P 1", "solution": "S 1"},
                                }
                            ],
                        }
                    ).encode()
                )

            records, scanned = fetch_records(
                "synthetic_math",
                max_records=2,
                page_size=1,
                opener=resumed_opener,
                max_retries=0,
                checkpoint_path=checkpoint_path,
                resume=True,
            )
            self.assertEqual(requested_offsets, [1])
            self.assertEqual([record["id"] for record in records], [
                "numinamath_cot_synthetic_math_0000000",
                "numinamath_cot_synthetic_math_0000001",
            ])
            self.assertEqual(scanned, 2)
            with checkpoint_path.open("r", encoding="utf-8") as source:
                self.assertEqual(json.load(source)["status"], "complete")


if __name__ == "__main__":
    unittest.main()
