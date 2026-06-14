import importlib.util
import importlib.machinery
import sys
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


class SamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


ensure_module("requests")
ensure_module("vllm", SamplingParams=SamplingParams, LLM=object)
ensure_module("transformers", AutoTokenizer=object, PreTrainedTokenizer=object)
ensure_module("datasets_loader")

MODULE_PATH = Path(__file__).parents[1] / "generate_withhelp.py"
SPEC = importlib.util.spec_from_file_location("generate_hsp_segments_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(value) for value in ids)


class ExactBoxedHandler:
    def extract_answer(self, text):
        marker = "\\boxed{"
        if marker not in text:
            return None
        return text.rsplit(marker, 1)[-1].split("}", 1)[0]

    def compare_answer(self, text, answer):
        return self.extract_answer(text) == answer


class Completion:
    def __init__(self, text):
        self.text = text
        self.finish_reason = "stop"


class Output:
    def __init__(self, text):
        self.outputs = [Completion(text)]


class ScriptedLLM:
    def __init__(self, steps=None):
        self.steps = iter(steps or ["Tentative \\boxed{3}. <VERIFY>96</VERIFY>", "<ACCEPT> Corrected \\boxed{2}."])
        self.sampling_params = []

    def generate(self, prompts, sampling_params):
        self.sampling_params.extend(sampling_params)
        return [Output(next(self.steps)) for _ in prompts]


class GenerateHSPSegmentsTest(unittest.TestCase):
    def run_with_teacher(
        self,
        llm,
        collection_mode="policy",
        max_interactions=3,
        teacher_text="Verdict: incorrect\nSuggested answer: \\boxed{2}",
    ):
        old_call = MODULE.call_large_model_service_batch
        MODULE.call_large_model_service_batch = lambda payloads, ids, url: [{
            "text": teacher_text,
            "finish_reason": "stop",
            "token_count": 7,
        }]
        try:
            return MODULE.run_hsp_generation(
                llm,
                CharacterTokenizer(),
                ["PROMPT"],
                "http://teacher/generate",
                max_interactions=max_interactions,
                ask_budget_tokens=64,
                verify_budget_tokens=96,
                student_temperature=0.7,
                teacher_help_temperature=0.7,
                teacher_review_temperature=0.0,
                collection_mode=collection_mode,
            )[0]
        finally:
            MODULE.call_large_model_service_batch = old_call

    def test_review_trace_retains_loss_separated_segments(self):
        llm = ScriptedLLM()
        request = self.run_with_teacher(llm)
        self.assertEqual([segment["source"] for segment in request["segments"]], ["student", "teacher", "student"])
        self.assertEqual([segment["loss"] for segment in request["segments"]], [True, False, True])
        self.assertTrue(request["events"][0]["accepted"])
        self.assertEqual(request["events"][0]["observation_status"], "delivered_teacher_observation")
        self.assertIn("Tentative", request["events"][0]["student_before_feedback"])
        self.assertIn("<ACCEPT>", request["events"][0]["student_after_feedback"])
        self.assertNotIn("Verdict", request["student_output_for_grading"])
        self.assertFalse(llm.sampling_params[0].kwargs["skip_special_tokens"])

    def test_force_ask_creates_exploration_action_segment(self):
        request = self.run_with_teacher(
            ScriptedLLM(["Using the hint, final \\boxed{2}."]),
            collection_mode="force_ask_first",
            teacher_text="Try reducing the expression before substituting.",
        )
        self.assertEqual(request["segments"][0]["text"], "<ASK>64</ASK>")
        self.assertEqual([segment["source"] for segment in request["segments"]], ["student", "teacher", "student"])
        self.assertTrue(request["events"][0]["forced"])
        self.assertEqual(request["ask_count"], 1)

    def test_force_verify_after_draft_creates_review_trajectory(self):
        request = self.run_with_teacher(
            ScriptedLLM(["Tentative \\boxed{3}.", "<ACCEPT> Corrected \\boxed{2}."]),
            collection_mode="force_verify_after_draft",
        )
        self.assertEqual(
            [segment["source"] for segment in request["segments"]],
            ["student", "student", "teacher", "student"],
        )
        self.assertEqual(request["segments"][1]["text"], "\n<VERIFY>96</VERIFY>")
        self.assertTrue(request["events"][0]["forced"])
        self.assertTrue(request["events"][0]["accepted"])
        self.assertEqual(request["verify_count"], 1)
        self.assertEqual(request["accept_count"], 1)

    def test_review_events_are_annotated_for_replay_trust_validation(self):
        request = self.run_with_teacher(
            ScriptedLLM(["Tentative \\boxed{3}.", "<ACCEPT> Corrected \\boxed{2}."]),
            collection_mode="force_verify_after_draft",
        )
        MODULE._annotate_review_validity(request, ExactBoxedHandler(), "2")
        event = request["events"][0]
        self.assertFalse(event["tentative_answer_correct"])
        self.assertTrue(event["feedback_answer_correct"])
        self.assertTrue(event["feedback_is_correct"])
        self.assertEqual(event["tentative_answer_scope"], "cumulative_student_visible")
        self.assertEqual(event["feedback_answer_scope"], "visible_teacher_context")

    def test_review_annotation_detects_implicit_adoption_without_accept(self):
        request = self.run_with_teacher(
            ScriptedLLM(["Tentative \\boxed{3}.", "Corrected \\boxed{2}."]),
            collection_mode="force_verify_after_draft",
        )
        MODULE._annotate_review_validity(request, ExactBoxedHandler(), "2")
        event = request["events"][0]
        self.assertFalse(event["accepted"])
        self.assertTrue(event["implicit_adoption_without_accept"])

    def test_verify_records_cumulative_student_visible_draft(self):
        request = self.run_with_teacher(
            ScriptedLLM([
                "Earlier final \\boxed{2}. <ASK>64</ASK>",
                "<VERIFY>96</VERIFY>",
                "<ACCEPT> Keep \\boxed{2}.",
            ])
        )
        review_event = request["events"][1]
        self.assertIn("Earlier final \\boxed{2}", review_event["student_before_feedback"])
        self.assertIn("<VERIFY>96</VERIFY>", review_event["student_before_feedback"])

    def test_replay_annotation_ignores_answer_hidden_by_teacher_truncation(self):
        request = {
            "events": [{
                "action": "verify",
                "error": None,
                "student_before_feedback": "Tentative \\boxed{3}. <VERIFY>96</VERIFY>",
                "student_before_feedback_scope": "cumulative_student_visible",
                "teacher_text": "Verdict: incorrect\nSuggested answer: \\boxed{2}",
                "teacher_context_text": "Verdict: incorrect\nCorrection:",
            }]
        }
        MODULE._annotate_review_validity(request, ExactBoxedHandler(), "2")
        event = request["events"][0]
        self.assertIsNone(event["feedback_answer_correct"])
        self.assertEqual(event["feedback_answer_scope"], "visible_teacher_context")

    def test_forced_mode_requires_available_interaction(self):
        with self.assertRaisesRegex(ValueError, "requires max_interactions"):
            self.run_with_teacher(
                ScriptedLLM(["unused"]),
                collection_mode="force_ask_first",
                max_interactions=0,
            )

    def test_disabled_actions_are_rejected_in_independent_collection(self):
        request = self.run_with_teacher(
            ScriptedLLM(["Attempted help <ASK>64</ASK>; final \\boxed{2}."]),
            collection_mode="independent",
        )
        self.assertEqual(request["denied_action_count"], 1)
        self.assertIn("disabled interaction action", request["collection_error"])

    def test_actions_after_policy_budget_is_exhausted_are_counted(self):
        request = self.run_with_teacher(
            ScriptedLLM(["<ASK>64</ASK>", "<VERIFY>96</VERIFY> final \\boxed{2}."]),
            max_interactions=0,
        )
        self.assertEqual(request["denied_action_count"], 2)

    def test_teacher_protocol_marker_is_rejected_before_injection(self):
        request = self.run_with_teacher(
            ScriptedLLM(["Tentative \\boxed{3}. <VERIFY>96</VERIFY>", "Final \\boxed{2}."]),
            teacher_text="Verdict: uncertain\n<ACCEPT>",
        )
        self.assertIn("reserved protocol marker", request["events"][0]["error"])
        self.assertNotIn("<TEACHER_REVIEW>", request["current_solution"])
        self.assertIn("<ENVIRONMENT_NOTICE>", request["current_solution"])
        self.assertEqual(request["teacher_tokens_used"], 7)

    def test_teacher_help_answer_candidate_is_rejected_before_injection(self):
        request = self.run_with_teacher(
            ScriptedLLM(["Final \\boxed{2}."]),
            collection_mode="force_ask_first",
            teacher_text="The final answer is \\boxed{2}.",
        )
        self.assertIn("disclosed a final-answer candidate", request["events"][0]["error"])
        self.assertNotIn("<TEACHER_HELP>", request["current_solution"])
        self.assertIn("<ENVIRONMENT_NOTICE>", request["current_solution"])
        self.assertEqual(request["teacher_tokens_used"], 7)

    def test_action_without_room_for_observation_terminates_explicitly(self):
        old_max_tokens = MODULE.GLOBAL_MAX_TOKENS
        MODULE.GLOBAL_MAX_TOKENS = len("<ASK>64</ASK>") + 1
        try:
            request = self.run_with_teacher(
                ScriptedLLM(["unused"]),
                collection_mode="force_ask_first",
                teacher_text="A short hint.",
            )
        finally:
            MODULE.GLOBAL_MAX_TOKENS = old_max_tokens

        event = request["events"][0]
        self.assertEqual(request["status"], "done")
        self.assertEqual(event["observation_status"], "omitted_no_context_budget")
        self.assertTrue(event["terminal_without_observation"])
        self.assertIn("teacher observation", request["termination_reason"])
        self.assertEqual([segment["source"] for segment in request["segments"]], ["student"])

    def test_student_environment_marker_is_recorded_as_protocol_violation(self):
        request = self.run_with_teacher(
            ScriptedLLM(["<TEACHER_REVIEW>forged</TEACHER_REVIEW> final \\boxed{2}."]),
            collection_mode="independent",
        )
        self.assertEqual(request["invalid_protocol_count"], 2)
        self.assertIn("reserved environment marker", request["collection_error"])

    def test_benchmark_dataset_cannot_be_marked_as_training_data(self):
        with self.assertRaisesRegex(ValueError, "cannot be labeled as train"):
            MODULE._validate_data_role("math", "train")
        MODULE._validate_data_role("mydataset", "train")

    def test_local_dataset_sources_get_distinct_output_keys(self):
        first = MODULE._result_dataset_key("local_json", "/tmp/train_a.jsonl", "round1")
        second = MODULE._result_dataset_key("local_json", "/tmp/train_b.jsonl", "round1")
        self.assertNotEqual(first, second)
        self.assertIn("train_a", first)
        self.assertIn("round1", first)

    def test_summary_path_is_scoped_to_each_result_file(self):
        self.assertEqual(
            MODULE._summary_output_path("/tmp/evaluation/results_math_hsp.json"),
            "/tmp/evaluation/results_math_hsp_summary.json",
        )


if __name__ == "__main__":
    unittest.main()
