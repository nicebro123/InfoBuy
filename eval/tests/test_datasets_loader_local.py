import importlib.util
import importlib.machinery
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path


def ensure_module(name, **attrs):
    try:
        __import__(name)
    except ImportError:
        module = types.ModuleType(name)
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules.setdefault(name, module)


ensure_module(
    "math_verify",
    parse=lambda value: value,
    verify=lambda left, right: left == right,
)
ensure_module("pandas")
ensure_module("datasets", load_dataset=lambda *args, **kwargs: None)

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
