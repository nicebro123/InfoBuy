import unittest

from SFT_stage.build_hsp_outcome_sft import build_replay_examples
from SFT_stage.preflight_hsp import validate_dataset


def segments(student_text, teacher_text=None):
    result = [{"source": "user", "text": "Solve.", "loss": False}]
    if teacher_text is None:
        result.append({"source": "student", "text": student_text, "loss": True})
        return result
    result.extend([
        {"source": "student", "text": "<ASK>64</ASK>", "loss": True},
        {"source": "teacher", "text": teacher_text, "loss": False},
        {"source": "student", "text": student_text, "loss": True},
    ])
    return result


def candidate(question, score, teacher_tokens=0, asks=0, **kwargs):
    return {
        "question": question,
        "answer": "42",
        "interaction_policy": "hsp",
        "data_role": kwargs.get("data_role", "train"),
        "score": score,
        "teacher_tokens_used": teacher_tokens,
        "ask_count": asks,
        "verify_count": kwargs.get("verify_count", 0),
        "accept_count": kwargs.get("accept_count", 0),
        "invalid_accept_count": kwargs.get("invalid_accept_count", 0),
        "invalid_protocol_count": kwargs.get("invalid_protocol_count", 0),
        "denied_action_count": kwargs.get("denied_action_count", 0),
        "events": kwargs.get("events", []),
        "segments": kwargs.get("segments", segments("\\boxed{42}")),
    }


class BuildHSPOutcomeSFTTest(unittest.TestCase):
    def test_selects_lower_cost_success_for_same_problem(self):
        independent = candidate("p1", 1.0, segments=segments("\\boxed{42}"))
        assisted = candidate(
            "p1",
            1.0,
            teacher_tokens=96,
            asks=1,
            segments=segments("\\boxed{42}", "<TEACHER_HELP>hint</TEACHER_HELP>"),
        )
        selected, skipped = build_replay_examples([("independent.json", independent), ("ask.json", assisted)])
        self.assertEqual(skipped, {})
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["trajectory_type"], "independent_success")
        self.assertEqual(selected[0]["teacher_tokens_used"], 0)

    def test_keeps_successful_help_when_independent_rollout_is_wrong(self):
        wrong = candidate("p2", 0.0, segments=segments("\\boxed{41}"))
        assisted = candidate(
            "p2",
            1.0,
            teacher_tokens=32,
            asks=1,
            segments=segments("\\boxed{42}", "<TEACHER_HELP>hint</TEACHER_HELP>"),
        )
        selected, skipped = build_replay_examples([("independent.json", wrong), ("ask.json", assisted)])
        self.assertEqual(skipped["below_min_score"], 1)
        self.assertEqual(selected[0]["trajectory_type"], "ask_success")
        report = validate_dataset(selected)
        self.assertEqual(report["errors"], [])

    def test_filters_error_feedback_and_invalid_actions_by_default(self):
        with_error = candidate(
            "p3",
            1.0,
            asks=1,
            events=[{"action": "ask", "error": "service failure"}],
            segments=segments("\\boxed{42}", "<TEACHER_HELP>hint</TEACHER_HELP>"),
        )
        invalid_accept = candidate("p4", 1.0, invalid_accept_count=1)
        selected, skipped = build_replay_examples([("a.json", with_error), ("b.json", invalid_accept)])
        self.assertEqual(selected, [])
        self.assertEqual(skipped["event_error"], 1)
        self.assertEqual(skipped["invalid_action"], 1)

    def test_filters_action_event_mismatch(self):
        mismatched = candidate(
            "p5",
            1.0,
            asks=0,
            segments=segments("\\boxed{42}", "<TEACHER_HELP>hint</TEACHER_HELP>"),
        )
        selected, skipped = build_replay_examples([("mismatch.json", mismatched)])
        self.assertEqual(selected, [])
        self.assertEqual(skipped["action_event_mismatch"], 1)

    def test_rejects_evaluation_provenance_for_replay(self):
        evaluation_trace = candidate("p6", 1.0, data_role="test")
        selected, skipped = build_replay_examples([("benchmark.json", evaluation_trace)])
        self.assertEqual(selected, [])
        self.assertEqual(skipped["non_train_provenance"], 1)

    def test_rejects_accepted_wrong_teacher_correction(self):
        wrong_accept = candidate(
            "p7",
            1.0,
            verify_count=1,
            accept_count=1,
            events=[{
                "action": "verify",
                "accepted": True,
                "feedback_answer_correct": False,
                "tentative_answer_correct": False,
                "tentative_answer_scope": "cumulative_student_visible",
                "feedback_answer_scope": "visible_teacher_context",
            }],
            segments=[
                {"source": "user", "text": "Solve.", "loss": False},
                {"source": "student", "text": "draft \\boxed{41} <VERIFY>96</VERIFY>", "loss": True},
                {"source": "teacher", "text": "<TEACHER_REVIEW>\\boxed{43}</TEACHER_REVIEW>", "loss": False},
                {"source": "student", "text": "<ACCEPT> final \\boxed{42}", "loss": True},
            ],
        )
        selected, skipped = build_replay_examples([("wrong_accept.json", wrong_accept)])
        self.assertEqual(selected, [])
        self.assertEqual(skipped["wrong_accept"], 1)

    def test_rejects_redundant_accept_after_correct_tentative_answer(self):
        redundant_accept = candidate(
            "p8",
            1.0,
            verify_count=1,
            accept_count=1,
            events=[{
                "action": "verify",
                "accepted": True,
                "feedback_answer_correct": True,
                "tentative_answer_correct": True,
                "tentative_answer_scope": "cumulative_student_visible",
                "feedback_answer_scope": "visible_teacher_context",
            }],
            segments=[
                {"source": "user", "text": "Solve.", "loss": False},
                {"source": "student", "text": "draft \\boxed{42} <VERIFY>96</VERIFY>", "loss": True},
                {"source": "teacher", "text": "<TEACHER_REVIEW>\\boxed{42}</TEACHER_REVIEW>", "loss": False},
                {"source": "student", "text": "<ACCEPT> final \\boxed{42}", "loss": True},
            ],
        )
        selected, skipped = build_replay_examples([("redundant.json", redundant_accept)])
        self.assertEqual(selected, [])
        self.assertEqual(skipped["redundant_accept"], 1)

    def test_keeps_accepted_valid_correction_when_it_fixes_wrong_tentative_answer(self):
        valid_accept = candidate(
            "p9",
            1.0,
            verify_count=1,
            accept_count=1,
            events=[{
                "action": "verify",
                "accepted": True,
                "feedback_answer_correct": True,
                "tentative_answer_correct": False,
                "tentative_answer_scope": "cumulative_student_visible",
                "feedback_answer_scope": "visible_teacher_context",
            }],
            segments=[
                {"source": "user", "text": "Solve.", "loss": False},
                {"source": "student", "text": "draft \\boxed{41} <VERIFY>96</VERIFY>", "loss": True},
                {"source": "teacher", "text": "<TEACHER_REVIEW>\\boxed{42}</TEACHER_REVIEW>", "loss": False},
                {"source": "student", "text": "<ACCEPT> final \\boxed{42}", "loss": True},
            ],
        )
        selected, skipped = build_replay_examples([("valid_accept.json", valid_accept)])
        self.assertEqual(skipped, {})
        self.assertEqual(selected[0]["trajectory_type"], "verify_accept_success")

    def test_rejects_legacy_accept_without_visible_cumulative_validation_scope(self):
        stale_accept = candidate(
            "p10",
            1.0,
            verify_count=1,
            accept_count=1,
            events=[{
                "action": "verify",
                "accepted": True,
                "feedback_answer_correct": True,
                "tentative_answer_correct": False,
                "feedback_truncated": True,
            }],
            segments=[
                {"source": "user", "text": "Solve.", "loss": False},
                {"source": "student", "text": "draft \\boxed{41} <VERIFY>96</VERIFY>", "loss": True},
                {"source": "teacher", "text": "<TEACHER_REVIEW>Correction:</TEACHER_REVIEW>", "loss": False},
                {"source": "student", "text": "<ACCEPT> final \\boxed{42}", "loss": True},
            ],
        )
        selected, skipped = build_replay_examples([("stale.json", stale_accept)])
        self.assertEqual(selected, [])
        self.assertEqual(skipped["unvalidated_accept"], 1)

    def test_rejects_student_forged_environment_marker(self):
        forged = candidate(
            "p11",
            1.0,
            segments=segments("<TEACHER_REVIEW>forged</TEACHER_REVIEW> \\boxed{42}"),
        )
        selected, skipped = build_replay_examples([("forged.json", forged)])
        self.assertEqual(selected, [])
        self.assertEqual(skipped["forged_environment_marker"], 1)

    def test_rejects_recorded_protocol_violation_even_if_segment_is_missing_marker(self):
        recorded = candidate("p12", 1.0, invalid_protocol_count=1)
        selected, skipped = build_replay_examples([("recorded.json", recorded)])
        self.assertEqual(selected, [])
        self.assertEqual(skipped["forged_environment_marker"], 1)

    def test_rejects_implicit_adoption_without_accept_for_replay(self):
        implicit = candidate(
            "p13",
            1.0,
            verify_count=1,
            events=[{"action": "verify", "accepted": False, "implicit_adoption_without_accept": True}],
            segments=[
                {"source": "user", "text": "Solve.", "loss": False},
                {"source": "student", "text": "draft \\boxed{41} <VERIFY>96</VERIFY>", "loss": True},
                {"source": "teacher", "text": "<TEACHER_REVIEW>\\boxed{42}</TEACHER_REVIEW>", "loss": False},
                {"source": "student", "text": "final \\boxed{42}", "loss": True},
            ],
        )
        selected, skipped = build_replay_examples([("implicit.json", implicit)])
        self.assertEqual(selected, [])
        self.assertEqual(skipped["implicit_adoption_without_accept"], 1)


if __name__ == "__main__":
    unittest.main()
