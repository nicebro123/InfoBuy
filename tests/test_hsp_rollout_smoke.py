import json
import tempfile
import unittest
from pathlib import Path

from scripts.hsp_rollout_smoke import result_dataset_key, validate_result_file


class HSPRolloutSmokeTest(unittest.TestCase):
    def test_result_key_distinguishes_local_json_paths(self):
        first = result_dataset_key("local_json", "/tmp/a/train.jsonl", "smoke")
        second = result_dataset_key("local_json", "/tmp/b/train.jsonl", "smoke")
        self.assertNotEqual(first, second)
        self.assertIn("train", first)

    def test_validates_forced_ask_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "interaction_policy": "hsp",
                            "collection_error": None,
                            "ask_count": 1,
                            "verify_count": 0,
                            "accept_count": 0,
                            "teacher_tokens_used": 3,
                            "segments": [
                                {"source": "user", "text": "Q", "loss": False},
                                {"source": "student", "text": "<ASK>48</ASK>", "loss": True},
                                {
                                    "source": "teacher",
                                    "text": "\n<TEACHER_HELP>\nhint\n</TEACHER_HELP>\n",
                                    "loss": False,
                                },
                                {"source": "student", "text": "final \\boxed{1}", "loss": True},
                            ],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            report = validate_result_file(path, "force_ask_first")
            self.assertEqual(report["ask_count"], 1)


if __name__ == "__main__":
    unittest.main()
