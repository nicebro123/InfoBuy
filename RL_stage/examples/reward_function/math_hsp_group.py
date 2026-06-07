"""Outcome reward for student-controlled HSP interactions.

The response passed by the reward manager contains student policy tokens only.
Teacher observations and their actual generation cost arrive through metadata.
"""

from __future__ import annotations

from typing import Any

from mathruler.grader import extract_boxed_content, grade_answer


def visible_teacher_feedback(event: dict[str, Any]) -> str:
    """Return only the teacher observation that was actually exposed to the policy."""
    return event.get("teacher_context_text") or ""


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    if answer is None:
        return 0.0
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def feedback_correctness(event: dict[str, Any], ground_truth: str) -> bool | None:
    if event.get("action") != "verify" or event.get("error"):
        return None

    feedback = visible_teacher_feedback(event)
    proposed_answer = extract_boxed_content(feedback)
    if proposed_answer is not None:
        return bool(grade_answer(proposed_answer, ground_truth))

    tentative = event.get("student_before_feedback") or ""
    tentative_answer = extract_boxed_content(tentative)
    tentative_correct = tentative_answer is not None and grade_answer(tentative_answer, ground_truth)
    verdict_line = feedback.lower()
    if "verdict: correct" in verdict_line:
        return bool(tentative_correct)
    if "verdict: incorrect" in verdict_line:
        return not bool(tentative_correct)
    return None


def tentative_answer_correctness(event: dict[str, Any], ground_truth: str) -> bool | None:
    tentative_answer = extract_boxed_content(event.get("student_before_feedback") or "")
    if tentative_answer is None:
        return None
    return bool(grade_answer(tentative_answer, ground_truth))


def corrective_answer_correctness(event: dict[str, Any], ground_truth: str) -> bool | None:
    proposed_answer = extract_boxed_content(visible_teacher_feedback(event))
    if proposed_answer is None:
        return None
    return bool(grade_answer(proposed_answer, ground_truth))


def adopted_teacher_answer_without_accept(event: dict[str, Any], response: str) -> bool:
    if bool(event.get("accepted", False)):
        return False
    proposed_answer = extract_boxed_content(visible_teacher_feedback(event))
    final_answer = extract_boxed_content(response)
    tentative_answer = extract_boxed_content(event.get("student_before_feedback") or "")
    if proposed_answer is None or final_answer is None:
        return False
    if tentative_answer is not None and grade_answer(tentative_answer, proposed_answer):
        return False
    return bool(grade_answer(final_answer, proposed_answer))


def compute_score(
    reward_inputs: list[dict[str, Any]],
    teacher_token_budget: float = 192.0,
    teacher_cost_weight: float = 0.15,
    useful_accept_weight: float = 0.0,
    resist_bad_review_weight: float = 0.0,
    wrong_accept_weight: float = 0.50,
    wrong_reject_weight: float = 0.50,
    implicit_adoption_weight: float = 0.05,
    wrong_implicit_adoption_weight: float = 0.50,
    unsupported_accept_weight: float = 0.10,
    invalid_accept_weight: float = 0.10,
    invalid_protocol_weight: float = 0.20,
    denied_action_weight: float = 0.05,
    teacher_error_weight: float = 0.0,
    independent_correct_weight: float = 0.0,
) -> list[dict[str, float]]:
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use reward_type=batch for the HSP reward function.")
    if teacher_token_budget <= 0:
        raise ValueError("teacher_token_budget must be positive.")

    scores = []
    for reward_input in reward_inputs:
        response = reward_input["response"]
        ground_truth = reward_input["ground_truth"]
        raw_events = reward_input.get("hsp_events", [])
        events = list(raw_events) if raw_events is not None else []
        accuracy = accuracy_reward(response, ground_truth)
        teacher_tokens = float(reward_input.get("teacher_tokens_used", 0))
        cost_ratio = teacher_tokens / teacher_token_budget
        teacher_errors = sum(1 for event in events if event.get("error"))

        useful_accepts = 0
        wrong_accepts = 0
        wrong_rejects = 0
        implicit_adoptions_without_accept = 0
        wrong_implicit_adoptions = 0
        unsupported_accepts = 0
        resisted_bad_reviews = 0
        for event in events:
            correctness = feedback_correctness(event, ground_truth)
            tentative_correct = tentative_answer_correctness(event, ground_truth)
            corrective_answer_correct = corrective_answer_correctness(event, ground_truth)
            accepted = bool(event.get("accepted", False))
            implicitly_adopted = adopted_teacher_answer_without_accept(event, response)
            if (
                accepted
                and corrective_answer_correct is True
                and tentative_correct is not True
                and accuracy > 0.5
            ):
                useful_accepts += 1
            elif accepted and (correctness is False or corrective_answer_correct is False):
                wrong_accepts += 1
            elif accepted and corrective_answer_correct is not True:
                unsupported_accepts += 1
            elif implicitly_adopted and corrective_answer_correct is False:
                wrong_implicit_adoptions += 1
            elif (
                implicitly_adopted
                and corrective_answer_correct is True
                and tentative_correct is not True
                and accuracy > 0.5
            ):
                implicit_adoptions_without_accept += 1
            elif (
                not accepted
                and not implicitly_adopted
                and corrective_answer_correct is True
                and tentative_correct is not True
                and accuracy < 0.5
            ):
                wrong_rejects += 1
            elif not accepted and correctness is False and accuracy > 0.5:
                resisted_bad_reviews += 1

        interaction_count = int(reward_input.get("ask_count", 0)) + int(reward_input.get("verify_count", 0))
        invalid_accepts = int(reward_input.get("invalid_accept_count", 0))
        invalid_protocols = int(reward_input.get("invalid_protocol_count", 0))
        denied_actions = int(reward_input.get("denied_action_count", 0))
        correct_without_interaction = float(accuracy > 0.5 and interaction_count == 0)
        independent_bonus = independent_correct_weight * correct_without_interaction
        overall = (
            accuracy
            + useful_accept_weight * useful_accepts
            + resist_bad_review_weight * resisted_bad_reviews
            + independent_bonus
            - teacher_cost_weight * cost_ratio
            - wrong_accept_weight * wrong_accepts
            - wrong_reject_weight * wrong_rejects
            - implicit_adoption_weight * implicit_adoptions_without_accept
            - wrong_implicit_adoption_weight * wrong_implicit_adoptions
            - unsupported_accept_weight * unsupported_accepts
            - invalid_accept_weight * invalid_accepts
            - invalid_protocol_weight * invalid_protocols
            - denied_action_weight * denied_actions
            - teacher_error_weight * teacher_errors
        )
        scores.append({
            "overall": float(overall),
            "accuracy": float(accuracy),
            "teacher_token_ratio": float(cost_ratio),
            "teacher_tokens_used": teacher_tokens,
            "ask_count": float(reward_input.get("ask_count", 0)),
            "verify_count": float(reward_input.get("verify_count", 0)),
            "accept_count": float(reward_input.get("accept_count", 0)),
            "useful_accept_count": float(useful_accepts),
            "wrong_accept_count": float(wrong_accepts),
            "wrong_reject_count": float(wrong_rejects),
            "implicit_adoption_without_accept_count": float(implicit_adoptions_without_accept),
            "wrong_implicit_adoption_count": float(wrong_implicit_adoptions),
            "unsupported_accept_count": float(unsupported_accepts),
            "resisted_bad_review_count": float(resisted_bad_reviews),
            "correct_without_interaction_count": correct_without_interaction,
            "invalid_accept_count": float(invalid_accepts),
            "invalid_protocol_count": float(invalid_protocols),
            "denied_action_count": float(denied_actions),
            "teacher_error_count": float(teacher_errors),
        })
    return scores
