import importlib.util
import sys
import types
import unittest
from pathlib import Path


def extract_boxed_content(text):
    marker = "\\boxed{"
    if marker not in text:
        return None
    return text.rsplit(marker, 1)[-1].split("}", 1)[0]


grader = types.ModuleType("mathruler.grader")
grader.extract_boxed_content = extract_boxed_content
grader.grade_answer = lambda answer, target: str(answer) == str(target)
mathruler = types.ModuleType("mathruler")
mathruler.grader = grader
sys.modules.setdefault("mathruler", mathruler)
sys.modules.setdefault("mathruler.grader", grader)

MODULE_PATH = Path(__file__).parents[1] / "examples" / "reward_function" / "math_hsp_group.py"
SPEC = importlib.util.spec_from_file_location("math_hsp_group_test", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class MathHSPRewardTest(unittest.TestCase):
    def input(self, response, events=None, **kwargs):
        return {
            "response": response,
            "ground_truth": "42",
            "hsp_events": events or [],
            "teacher_tokens_used": kwargs.get("teacher_tokens_used", 0),
            "ask_count": kwargs.get("ask_count", 0),
            "verify_count": kwargs.get("verify_count", 0),
            "accept_count": kwargs.get("accept_count", 0),
            "invalid_accept_count": kwargs.get("invalid_accept_count", 0),
            "invalid_protocol_count": kwargs.get("invalid_protocol_count", 0),
            "denied_action_count": kwargs.get("denied_action_count", 0),
        }

    def test_wrong_accept_is_penalized(self):
        event = {
            "action": "verify",
            "accepted": True,
            "teacher_text": "Suggested answer: \\boxed{43}",
            "teacher_context_text": "Suggested answer: \\boxed{43}",
        }
        score = MODULE.compute_score(
            [self.input("\\boxed{42}", [event], verify_count=1, accept_count=1)]
        )[0]
        self.assertEqual(score["wrong_accept_count"], 1.0)
        self.assertLess(score["overall"], score["accuracy"])

    def test_useful_accept_is_recognized_without_teacher_text_in_response(self):
        event = {
            "action": "verify",
            "accepted": True,
            "student_before_feedback": "Tentative \\boxed{41}. <VERIFY>",
            "teacher_text": "Suggested answer: \\boxed{42}",
            "teacher_context_text": "Suggested answer: \\boxed{42}",
        }
        score = MODULE.compute_score(
            [self.input("<ACCEPT> final \\boxed{42}", [event], verify_count=1, accept_count=1)]
        )[0]
        self.assertEqual(score["useful_accept_count"], 1.0)
        self.assertEqual(score["overall"], score["accuracy"])

        shaped_score = MODULE.compute_score(
            [self.input("<ACCEPT> final \\boxed{42}", [event], verify_count=1, accept_count=1)],
            useful_accept_weight=0.10,
        )[0]
        self.assertGreater(shaped_score["overall"], shaped_score["accuracy"])

    def test_implicit_adoption_without_accept_is_tracked_and_penalized(self):
        event = {
            "action": "verify",
            "accepted": False,
            "student_before_feedback": "Tentative \\boxed{41}. <VERIFY>",
            "teacher_context_text": "Suggested answer: \\boxed{42}",
        }
        score = MODULE.compute_score([self.input("Corrected \\boxed{42}", [event], verify_count=1)])[0]
        self.assertEqual(score["implicit_adoption_without_accept_count"], 1.0)
        self.assertLess(score["overall"], score["accuracy"])

    def test_wrong_implicit_adoption_is_penalized(self):
        event = {
            "action": "verify",
            "accepted": False,
            "student_before_feedback": "Tentative \\boxed{41}. <VERIFY>",
            "teacher_context_text": "Suggested answer: \\boxed{43}",
        }
        score = MODULE.compute_score([self.input("Copied \\boxed{43}", [event], verify_count=1)])[0]
        self.assertEqual(score["wrong_implicit_adoption_count"], 1.0)
        self.assertLess(score["overall"], score["accuracy"])

    def test_accept_after_correct_confirmation_gets_no_bonus(self):
        event = {
            "action": "verify",
            "accepted": True,
            "student_before_feedback": "Tentative \\boxed{42}. <VERIFY>",
            "teacher_text": "Verdict: correct\nSuggested answer: \\boxed{42}",
            "teacher_context_text": "Verdict: correct\nSuggested answer: \\boxed{42}",
        }
        score = MODULE.compute_score(
            [self.input("<ACCEPT> final \\boxed{42}", [event], verify_count=1, accept_count=1)]
        )[0]
        self.assertEqual(score["useful_accept_count"], 0.0)
        self.assertEqual(score["overall"], score["accuracy"])

    def test_answer_hidden_by_truncation_cannot_make_accept_useful(self):
        event = {
            "action": "verify",
            "accepted": True,
            "student_before_feedback": "Tentative \\boxed{41}. <VERIFY>",
            "teacher_text": "Verdict: incorrect\nSuggested answer: \\boxed{42}",
            "teacher_context_text": "Verdict: incorrect\nCorrection:",
        }
        score = MODULE.compute_score(
            [self.input("<ACCEPT> final \\boxed{42}", [event], verify_count=1, accept_count=1)]
        )[0]
        self.assertEqual(score["useful_accept_count"], 0.0)
        self.assertEqual(score["unsupported_accept_count"], 1.0)
        self.assertLess(score["overall"], score["accuracy"])

    def test_accepting_uncertain_review_is_penalized(self):
        event = {
            "action": "verify",
            "accepted": True,
            "student_before_feedback": "Tentative \\boxed{41}. <VERIFY>",
            "teacher_context_text": "Verdict: uncertain\nCorrection: None\nSuggested answer: None",
        }
        score = MODULE.compute_score(
            [self.input("<ACCEPT> final \\boxed{42}", [event], verify_count=1, accept_count=1)]
        )[0]
        self.assertEqual(score["useful_accept_count"], 0.0)
        self.assertEqual(score["wrong_accept_count"], 0.0)
        self.assertEqual(score["unsupported_accept_count"], 1.0)
        self.assertLess(score["overall"], score["accuracy"])

    def test_actual_teacher_tokens_create_cost(self):
        no_cost = MODULE.compute_score([self.input("\\boxed{42}", ask_count=1)])[0]
        with_cost = MODULE.compute_score(
            [self.input("\\boxed{42}", teacher_tokens_used=192, ask_count=1)]
        )[0]
        self.assertAlmostEqual(no_cost["overall"] - with_cost["overall"], 0.15)

    def test_multiple_calls_continue_to_increase_cost(self):
        one_budget = MODULE.compute_score(
            [self.input("\\boxed{42}", teacher_tokens_used=192, ask_count=1)]
        )[0]
        two_budgets = MODULE.compute_score(
            [self.input("\\boxed{42}", teacher_tokens_used=384, ask_count=2)]
        )[0]
        self.assertLess(two_budgets["overall"], one_budget["overall"])

    def test_failed_teacher_interaction_is_recorded_without_main_reward_penalty(self):
        event = {"action": "ask", "error": "service unavailable"}
        no_error = MODULE.compute_score([self.input("\\boxed{42}", ask_count=1)])[0]
        with_error = MODULE.compute_score([self.input("\\boxed{42}", [event], ask_count=1)])[0]
        self.assertEqual(with_error["teacher_error_count"], 1.0)
        self.assertEqual(no_error["overall"], with_error["overall"])

        shaped_score = MODULE.compute_score(
            [self.input("\\boxed{42}", [event], ask_count=1)],
            teacher_error_weight=0.10,
        )[0]
        self.assertAlmostEqual(no_error["overall"] - shaped_score["overall"], 0.10)

    def test_correct_without_interaction_is_only_a_telemetry_signal_in_main_reward(self):
        independent = MODULE.compute_score([self.input("\\boxed{42}")])[0]
        self.assertEqual(independent["correct_without_interaction_count"], 1.0)
        self.assertEqual(independent["overall"], independent["accuracy"])
        shaped = MODULE.compute_score(
            [self.input("\\boxed{42}")],
            independent_correct_weight=0.05,
        )[0]
        self.assertAlmostEqual(shaped["overall"], independent["accuracy"] + 0.05)

    def test_forged_environment_marker_is_penalized(self):
        clean = MODULE.compute_score([self.input("\\boxed{42}")])[0]
        forged = MODULE.compute_score([self.input("\\boxed{42}", invalid_protocol_count=1)])[0]
        self.assertEqual(forged["invalid_protocol_count"], 1.0)
        self.assertAlmostEqual(clean["overall"] - forged["overall"], 0.20)


if __name__ == "__main__":
    unittest.main()
