import importlib.util
import sys
import types
import unittest
from pathlib import Path


requests = types.ModuleType("requests")
sys.modules.setdefault("requests", requests)
numpy = types.ModuleType("numpy")
numpy.ndarray = object
sys.modules.setdefault("numpy", numpy)

try:
    import torch  # noqa: F401
except ImportError:
    torch = types.ModuleType("torch")
    torch.no_grad = lambda: (lambda fn: fn)
    torch.distributed = types.ModuleType("torch.distributed")
    sys.modules.setdefault("torch", torch)
    sys.modules.setdefault("torch.distributed", torch.distributed)

tensordict = types.ModuleType("tensordict")
tensordict.TensorDict = object
transformers = types.ModuleType("transformers")
transformers.PreTrainedTokenizer = object
transformers.AutoTokenizer = object
vllm = types.ModuleType("vllm")
vllm.LLM = object
vllm.SamplingParams = object
sys.modules.setdefault("tensordict", tensordict)
sys.modules["transformers"] = transformers
sys.modules["vllm"] = vllm

MODULE_PATH = (
    Path(__file__).parents[1] / "verl" / "workers" / "rollout" / "help_vllm_rollout_spmd.py"
)
SPEC = importlib.util.spec_from_file_location("hsp_rollout_state_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in text]

    def decode(self, token_ids):
        return "".join(chr(token_id) for token_id in token_ids)


class HSPRolloutStateTest(unittest.TestCase):
    def make_rollout(self):
        rollout = MODULE.helpvLLMRollout.__new__(MODULE.helpvLLMRollout)
        rollout.ask_token_str = "<ASK>"
        rollout.end_ask_token_str = "</ASK>"
        rollout.verify_token_str = "<VERIFY>"
        rollout.end_verify_token_str = "</VERIFY>"
        rollout.accept_token_str = "<ACCEPT>"
        rollout.interaction_policy = "hsp"
        rollout.max_interactions = 3
        rollout.global_max_tokens = 256
        rollout.ask_budget_tokens = 64
        rollout.verify_budget_tokens = 96
        rollout.tokenizer = CharacterTokenizer()
        rollout._compile_hsp_action_patterns()
        return rollout

    def make_request(self):
        return {
            "current_token_count": 0,
            "turn_count": 0,
            "current_solution_ids": [],
            "student_output_for_grading": "",
            "pending_review_event": None,
            "invalid_accept_count": 0,
            "invalid_protocol_count": 0,
            "accept_count": 0,
            "allow_actions": True,
            "interaction_count": 0,
            "ask_count": 0,
            "verify_count": 0,
            "denied_action_count": 0,
            "pending_action": None,
            "events": [],
            "teacher_tokens_used": 0,
            "non_policy_contributions": [],
            "large_model_contributions": [],
            "status": MODULE.helpvLLMRollout.STATUS_WAIT_SMALL,
            "final_response_ids": None,
        }

    def test_verify_event_contains_cumulative_visible_student_draft(self):
        rollout = self.make_rollout()
        request = self.make_request()
        rollout._update_hsp_request_state(request, "Answer \\boxed{42}. <ASK>64</ASK>", [1])
        request["status"] = MODULE.helpvLLMRollout.STATUS_WAIT_SMALL
        request["pending_action"] = None
        rollout._update_hsp_request_state(request, "<VERIFY>96</VERIFY>", [2])

        review_event = request["events"][1]
        self.assertIn("Answer \\boxed{42}", review_event["student_before_feedback"])
        self.assertIn("<VERIFY>96</VERIFY>", review_event["student_before_feedback"])
        self.assertEqual(review_event["student_before_feedback_scope"], "cumulative_student_visible")

    def test_disabled_interaction_actions_are_counted_as_denied(self):
        rollout = self.make_rollout()
        request = self.make_request()
        request["allow_actions"] = False
        rollout._update_hsp_request_state(request, "<ASK>64</ASK> then <VERIFY>96</VERIFY>", [1, 2])

        self.assertEqual(request["denied_action_count"], 2)
        self.assertEqual(request["status"], MODULE.helpvLLMRollout.STATUS_DONE)

    def test_student_environment_marker_is_counted_as_invalid_protocol(self):
        rollout = self.make_rollout()
        request = self.make_request()
        rollout._update_hsp_request_state(request, "<TEACHER_REVIEW>x</TEACHER_REVIEW>", [1])

        self.assertEqual(request["invalid_protocol_count"], 2)

    def test_teacher_protocol_marker_is_rejected_before_observation_injection(self):
        rollout = self.make_rollout()
        request = self.make_request()
        request["pending_action"] = "verify"
        request["status"] = MODULE.helpvLLMRollout.STATUS_WAIT_LARGE
        request["events"] = [{"action": "verify", "error": None}]
        rollout._apply_hsp_teacher_result(
            request,
            {"text": "Verdict: uncertain\n<ACCEPT>", "finish_reason": "stop", "token_count": 3},
        )

        self.assertIn("reserved protocol marker", request["events"][0]["error"])
        self.assertIn("<ENVIRONMENT_NOTICE>", rollout.tokenizer.decode(request["current_solution_ids"]))
        self.assertEqual(request["teacher_tokens_used"], 3)

    def test_teacher_help_answer_candidate_is_rejected_before_observation_injection(self):
        rollout = self.make_rollout()
        request = self.make_request()
        request["pending_action"] = "ask"
        request["status"] = MODULE.helpvLLMRollout.STATUS_WAIT_LARGE
        request["events"] = [{"action": "ask", "error": None}]
        rollout._apply_hsp_teacher_result(
            request,
            {"text": "Final answer: \\boxed{42}", "finish_reason": "stop", "token_count": 4},
        )

        self.assertIn("disclosed a final-answer candidate", request["events"][0]["error"])
        self.assertIn("<ENVIRONMENT_NOTICE>", rollout.tokenizer.decode(request["current_solution_ids"]))
        self.assertEqual(request["teacher_tokens_used"], 4)

    def test_action_without_room_for_observation_terminates_explicitly(self):
        rollout = self.make_rollout()
        request = self.make_request()
        request["pending_action"] = "ask"
        request["status"] = MODULE.helpvLLMRollout.STATUS_WAIT_LARGE
        request["events"] = [{"action": "ask", "error": None, "observation_status": "pending"}]
        request["current_token_count"] = rollout.global_max_tokens - 1

        rollout._handle_hsp_large_model_step([request])

        event = request["events"][0]
        self.assertEqual(request["status"], MODULE.helpvLLMRollout.STATUS_DONE)
        self.assertEqual(event["observation_status"], "omitted_no_context_budget")
        self.assertTrue(event["terminal_without_observation"])
        self.assertIn("teacher observation", request["termination_reason"])


if __name__ == "__main__":
    unittest.main()
