import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


math_verify = types.ModuleType("math_verify")
math_verify.parse = lambda value: value
math_verify.verify = lambda left, right: left == right
datasets = types.ModuleType("datasets")
datasets.load_dataset = lambda *args, **kwargs: None
sys.modules.setdefault("math_verify", math_verify)
sys.modules.setdefault("pandas", types.ModuleType("pandas"))
sys.modules.setdefault("datasets", datasets)

MODULE_PATH = Path(__file__).parents[1] / "datasets_loader.py"
SPEC = importlib.util.spec_from_file_location("datasets_loader_local_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class LocalJsonDatasetHandlerTest(unittest.TestCase):
    def test_reads_training_jsonl_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            path.write_text(
                json.dumps({"question": "1+1", "gold_answer": "2"}) + "\n"
                + json.dumps({"problem": "2+2", "answer": "4"}) + "\n",
                encoding="utf-8",
            )
            handler = MODULE.get_dataset_handler("local_json", str(path))
            questions, answers = handler.load_data()
        self.assertEqual(questions, ["1+1", "2+2"])
        self.assertEqual(answers, ["2", "4"])

    def test_requires_path(self):
        with self.assertRaisesRegex(ValueError, "requires --name"):
            MODULE.get_dataset_handler("local_json")

    def test_extracts_last_balanced_boxed_answer(self):
        handler = MODULE.MathDatasetHandler()
        response = r"Intermediate \boxed{1}. Final \boxed{\frac{1}{2}}."
        self.assertEqual(handler.extract_answer(response), r"\frac{1}{2}")

    def test_multiple_choice_uses_alternative_capture_group(self):
        handler = MODULE.MmluProDatasetHandler()
        self.assertEqual(handler.extract_answer("Correct Answer: B."), "B")

    def test_missing_answer_is_false(self):
        handler = MODULE.MathDatasetHandler()
        self.assertFalse(handler.compare_answer("No boxed final answer.", "2"))


if __name__ == "__main__":
    unittest.main()
