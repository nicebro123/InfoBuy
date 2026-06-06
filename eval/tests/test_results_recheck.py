import importlib.util
import os
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace


tqdm_module = types.ModuleType("tqdm")
tqdm_module.tqdm = lambda values, **kwargs: values
sys.modules.setdefault("tqdm", tqdm_module)

MODULE_PATH = Path(__file__).parents[1] / "results_recheck.py"
SPEC = importlib.util.spec_from_file_location("results_recheck_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ResultsRecheckTest(unittest.TestCase):
    def test_missing_api_key_fails_explicitly(self):
        prior = os.environ.pop("OPENAI_API_KEY", None)
        try:
            with self.assertRaisesRegex(RuntimeError, "OPENAI_API_KEY"):
                MODULE.create_judge_client()
        finally:
            if prior is not None:
                os.environ["OPENAI_API_KEY"] = prior

    def test_output_tag_selects_matching_result_file(self):
        args = SimpleNamespace(
            interaction_policy="hsp",
            collection_mode="policy",
            output_tag="round 1",
            fix_number=None,
        )
        self.assertEqual(MODULE.result_filename("math", args), "results_math_round_1_hsp.json")

    def test_invalid_judge_response_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unexpected judge response"):
            MODULE.normalize_judgement("I cannot determine that.")

    def test_main_persists_corrected_per_item_results_without_overwriting_raw_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            storage = Path(directory)
            source_dir = storage / "evaluation" / "student_teacher"
            source_dir.mkdir(parents=True)
            source = source_dir / "results_math_hsp.json"
            source.write_text(
                json.dumps([{"answer": "42", "response": "\\boxed{42}", "score": 0}]),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                model_name="student",
                larger_model="teacher",
                fix_number=None,
                interaction_policy="hsp",
                collection_mode="policy",
                output_tag=None,
                skip_llm_recheck=False,
                judge_model="judge",
                overwrite_recheck=False,
            )
            old_storage = os.environ.get("STORAGE_PATH")
            old_parse_args = MODULE.parse_args
            old_datasets = MODULE.DATASETS_TO_CHECK
            old_create_client = MODULE.create_judge_client
            old_process = MODULE.process_example
            try:
                os.environ["STORAGE_PATH"] = str(storage)
                MODULE.parse_args = lambda: args
                MODULE.DATASETS_TO_CHECK = ["math"]
                MODULE.create_judge_client = lambda: object()
                MODULE.process_example = lambda client, model, answer, response: "Yes"
                MODULE.main()
            finally:
                MODULE.parse_args = old_parse_args
                MODULE.DATASETS_TO_CHECK = old_datasets
                MODULE.create_judge_client = old_create_client
                MODULE.process_example = old_process
                if old_storage is None:
                    os.environ.pop("STORAGE_PATH", None)
                else:
                    os.environ["STORAGE_PATH"] = old_storage

            raw = json.loads(source.read_text(encoding="utf-8"))
            rechecked = json.loads(
                (source_dir / "results_math_hsp_rechecked.json").read_text(encoding="utf-8")
            )
            summary = json.loads(
                (source_dir / "results_math_hsp_rechecked_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(raw[0]["score"], 0)
            self.assertEqual(rechecked[0]["deterministic_score"], 0)
            self.assertEqual(rechecked[0]["score"], 1)
            self.assertEqual(rechecked[0]["llm_recheck_status"], "completed")
            self.assertEqual(summary["score"], 100.0)
            self.assertEqual(summary["per_item_results"], str(source_dir / "results_math_hsp_rechecked.json"))
            self.assertFalse((storage / "scores_recheck.jsonl").exists())

    def test_rechecked_sidecars_require_explicit_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "results_math_hsp.json"
            source.write_text("[]", encoding="utf-8")
            existing = MODULE.rechecked_results_path(source)
            existing.write_text("old", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "overwrite_recheck"):
                MODULE.ensure_recheck_outputs_available(source, overwrite=False)

            output = MODULE.write_rechecked_results(source, [{"score": 1}], overwrite=True)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))[0]["score"], 1)


if __name__ == "__main__":
    unittest.main()
