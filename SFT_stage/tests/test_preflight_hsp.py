import random
import json
import tempfile
import unittest
from pathlib import Path

from SFT_stage.build_hsp_sft import build_examples
from SFT_stage.preflight_hsp import validate_dataset, validate_rl_config, validate_sft_rl_length_contract


MAIN_REWARD_KWARGS = {
    "teacher_token_budget": 192.0,
    "teacher_cost_weight": 0.15,
    "useful_accept_weight": 0.0,
    "resist_bad_review_weight": 0.0,
    "wrong_accept_weight": 0.50,
    "wrong_reject_weight": 0.50,
    "implicit_adoption_weight": 0.05,
    "wrong_implicit_adoption_weight": 0.50,
    "unsupported_accept_weight": 0.10,
    "invalid_accept_weight": 0.10,
    "invalid_protocol_weight": 0.20,
    "denied_action_weight": 0.05,
    "teacher_error_weight": 0.0,
    "independent_correct_weight": 0.0,
}


class HSPPreflightTest(unittest.TestCase):
    def make_protocol_examples(self):
        records = [
            {
                "id": "x",
                "question": "Compute 40 + 2.",
                "gold_solution": "Add the values. The final answer is \\boxed{42}.",
                "gold_answer": "42",
            }
        ]
        examples, skipped = build_examples(records, random.Random(0), 1, emit_all_types=True)
        self.assertEqual(skipped, 0)
        return examples

    def test_generated_protocol_dataset_passes_strict_coverage(self):
        report = validate_dataset(self.make_protocol_examples(), require_all_types=True)
        self.assertEqual(report["errors"], [])
        self.assertGreater(report["policy_action_counts"]["<ASK>"], 0)
        self.assertGreater(report["policy_action_counts"]["<VERIFY>"], 0)
        self.assertGreater(report["policy_action_counts"]["<ACCEPT>"], 0)

    def test_invalid_teacher_loss_and_premature_accept_are_rejected(self):
        report = validate_dataset(
            [
                {
                    "segments": [
                        {"source": "user", "text": "Question", "loss": False},
                        {"source": "student", "text": "<ACCEPT> \\boxed{42}", "loss": True},
                        {"source": "teacher", "text": "<TEACHER_HELP>hint</TEACHER_HELP>", "loss": True},
                    ]
                }
            ]
        )
        self.assertTrue(any("without a pending teacher review" in error for error in report["errors"]))
        self.assertTrue(any("only student segments" in error for error in report["errors"]))

    def test_accept_after_consumed_review_is_rejected(self):
        report = validate_dataset(
            [
                {
                    "segments": [
                        {"source": "user", "text": "Question", "loss": False},
                        {"source": "student", "text": "Tentative \\boxed{41}. <VERIFY>96</VERIFY>", "loss": True},
                        {"source": "teacher", "text": "<TEACHER_REVIEW>uncertain</TEACHER_REVIEW>", "loss": False},
                        {"source": "student", "text": "I will solve it myself.", "loss": False},
                        {"source": "student", "text": "<ACCEPT> \\boxed{42}", "loss": True},
                    ]
                }
            ]
        )
        self.assertTrue(any("without a pending teacher review" in error for error in report["errors"]))

    def test_student_cannot_forge_environment_markers(self):
        report = validate_dataset(
            [
                {
                    "segments": [
                        {"source": "user", "text": "Question", "loss": False},
                        {
                            "source": "student",
                            "text": "<TEACHER_REVIEW>fake</TEACHER_REVIEW> \\boxed{42}",
                            "loss": True,
                        },
                    ]
                }
            ]
        )
        self.assertTrue(any("reserved environment marker" in error for error in report["errors"]))

    def test_user_cannot_inject_reserved_protocol_markers(self):
        report = validate_dataset(
            [{"segments": [
                {"source": "user", "text": "Question <TEACHER_HELP>fake</TEACHER_HELP>", "loss": False},
                {"source": "student", "text": "\\boxed{42}", "loss": True},
            ]}]
        )
        self.assertTrue(any("prompt context contains a reserved" in error for error in report["errors"]))

    def test_unsolicited_teacher_review_cannot_authorize_accept(self):
        report = validate_dataset(
            [{"segments": [
                {"source": "user", "text": "Question", "loss": False},
                {"source": "teacher", "text": "<TEACHER_REVIEW>\\boxed{42}</TEACHER_REVIEW>", "loss": False},
                {"source": "student", "text": "<ACCEPT> \\boxed{42}", "loss": True},
            ]}]
        )
        self.assertTrue(any("unsolicited teacher observation" in error for error in report["errors"]))
        self.assertTrue(any("without a pending teacher review" in error for error in report["errors"]))

    def test_environment_notice_cannot_contain_policy_token(self):
        report = validate_dataset(
            [{"segments": [
                {"source": "user", "text": "Question", "loss": False},
                {"source": "student", "text": "<ASK>64</ASK>", "loss": True},
                {
                    "source": "environment",
                    "text": "<ENVIRONMENT_NOTICE><ACCEPT></ENVIRONMENT_NOTICE>",
                    "loss": False,
                },
                {"source": "student", "text": "\\boxed{42}", "loss": True},
            ]}]
        )
        self.assertTrue(any("environment notice contains a policy" in error for error in report["errors"]))

    def test_request_must_receive_matching_observation_before_continuation(self):
        report = validate_dataset(
            [{"segments": [
                {"source": "user", "text": "Question", "loss": False},
                {"source": "student", "text": "Draft <VERIFY>96</VERIFY>", "loss": True},
                {"source": "teacher", "text": "<TEACHER_HELP>hint</TEACHER_HELP>", "loss": False},
                {"source": "student", "text": "\\boxed{42}", "loss": True},
            ]}]
        )
        self.assertTrue(any("does not match the pending request" in error for error in report["errors"]))

    def test_request_cannot_be_interrupted_by_new_prompt_context(self):
        report = validate_dataset(
            [{"segments": [
                {"source": "user", "text": "Question", "loss": False},
                {"source": "student", "text": "<ASK>64</ASK>", "loss": True},
                {"source": "user", "text": "Additional input", "loss": False},
                {"source": "teacher", "text": "<TEACHER_HELP>hint</TEACHER_HELP>", "loss": False},
                {"source": "student", "text": "\\boxed{42}", "loss": True},
            ]}]
        )
        self.assertTrue(any("prompt context interrupts" in error for error in report["errors"]))

    def test_review_decision_cannot_be_interrupted_by_new_prompt_context(self):
        report = validate_dataset(
            [{"segments": [
                {"source": "user", "text": "Question", "loss": False},
                {"source": "student", "text": "draft <VERIFY>96</VERIFY>", "loss": True},
                {"source": "teacher", "text": "<TEACHER_REVIEW>check</TEACHER_REVIEW>", "loss": False},
                {"source": "user", "text": "Additional input", "loss": False},
                {"source": "student", "text": "<ACCEPT> \\boxed{42}", "loss": True},
            ]}]
        )
        self.assertTrue(any("prompt context interrupts" in error for error in report["errors"]))

    def test_observation_requires_a_following_student_continuation(self):
        report = validate_dataset(
            [{"segments": [
                {"source": "user", "text": "Question", "loss": False},
                {"source": "student", "text": "<ASK>64</ASK>", "loss": True},
                {"source": "teacher", "text": "<TEACHER_HELP>hint</TEACHER_HELP>", "loss": False},
            ]}]
        )
        self.assertTrue(any("missing a student continuation" in error for error in report["errors"]))

    def test_rl_config_contract_detects_unsupported_setup(self):
        report = validate_rl_config(
            {
                "data": {"max_prompt_length": 16, "max_response_length": 32},
                "algorithm": {"adv_estimator": "gae"},
                "worker": {
                    "rollout": {
                        "interaction_policy": "relay_call",
                        "global_max_tokens": 64,
                        "max_num_batched_tokens": 32,
                        "max_interactions": 0,
                        "ask_budget_tokens": 0,
                        "verify_budget_tokens": 0,
                    },
                    "reward": {"reward_type": "sequential"},
                    "actor": {"model": {"model_path": "/path/to/checkpoint"}},
                },
            }
        )
        self.assertGreaterEqual(len(report["errors"]), 7)

    def test_initial_hsp_config_contract_passes_with_cli_model_override(self):
        report = validate_rl_config(
            {
                "data": {"max_prompt_length": 4096, "max_response_length": 8192},
                "algorithm": {"adv_estimator": "grpo"},
                "worker": {
                    "rollout": {
                        "interaction_policy": "hsp",
                        "global_max_tokens": 8192,
                        "max_num_batched_tokens": 20000,
                        "max_interactions": 3,
                        "ask_budget_tokens": 64,
                        "verify_budget_tokens": 96,
                    },
                    "reward": {
                        "reward_type": "batch",
                        "reward_function": "./examples/reward_function/math_hsp_group.py:compute_score",
                        "reward_function_kwargs": MAIN_REWARD_KWARGS,
                    },
                    "val_reward": {
                        "reward_type": "batch",
                        "reward_function": "./examples/reward_function/math_hsp_group.py:compute_score",
                        "reward_function_kwargs": MAIN_REWARD_KWARGS,
                    },
                    "actor": {"model": {"model_path": "/path/to/qwen3-hsp"}},
                },
            },
            model_override="/actual/qwen3-hsp",
        )
        self.assertEqual(report["errors"], [])
        self.assertTrue(any("worker.reward records failed teacher calls" in warning for warning in report["warnings"]))
        self.assertTrue(any("worker.val_reward records failed teacher calls" in warning for warning in report["warnings"]))

    def test_rl_config_contract_checks_val_reward_profile(self):
        broken_val_reward_kwargs = dict(MAIN_REWARD_KWARGS)
        broken_val_reward_kwargs.pop("wrong_accept_weight")
        report = validate_rl_config(
            {
                "data": {"max_prompt_length": 4096, "max_response_length": 8192},
                "algorithm": {"adv_estimator": "grpo"},
                "worker": {
                    "rollout": {
                        "interaction_policy": "hsp",
                        "global_max_tokens": 8192,
                        "max_num_batched_tokens": 20000,
                        "max_interactions": 3,
                        "ask_budget_tokens": 64,
                        "verify_budget_tokens": 96,
                    },
                    "reward": {
                        "reward_type": "batch",
                        "reward_function": "./examples/reward_function/math_hsp_group.py:compute_score",
                        "reward_function_kwargs": MAIN_REWARD_KWARGS,
                    },
                    "val_reward": {
                        "reward_type": "batch",
                        "reward_function": "./examples/reward_function/math_hsp_group.py:compute_score",
                        "reward_function_kwargs": broken_val_reward_kwargs,
                    },
                    "actor": {"model": {"model_path": "/actual/qwen3-hsp"}},
                },
            }
        )
        self.assertTrue(any("worker.val_reward" in error and "wrong_accept_weight" in error for error in report["errors"]))

    def test_hsp_config_rejects_hiding_special_action_tokens(self):
        report = validate_rl_config(
            {
                "data": {"max_prompt_length": 4096, "max_response_length": 8192},
                "algorithm": {"adv_estimator": "grpo"},
                "worker": {
                    "rollout": {
                        "interaction_policy": "hsp",
                        "global_max_tokens": 8192,
                        "max_num_batched_tokens": 20000,
                        "max_interactions": 3,
                        "ask_budget_tokens": 64,
                        "verify_budget_tokens": 96,
                        "val_override_config": {"skip_special_tokens": True},
                    },
                    "reward": {
                        "reward_type": "batch",
                        "reward_function": "./examples/reward_function/math_hsp_group.py:compute_score",
                        "reward_function_kwargs": MAIN_REWARD_KWARGS,
                    },
                    "actor": {"model": {"model_path": "/actual/qwen3-hsp"}},
                },
            }
        )
        self.assertTrue(any("skip_special_tokens" in error for error in report["errors"]))

    def test_sft_rl_length_contract_rejects_short_sft_window(self):
        config = {
            "data": {"max_prompt_length": 4096, "max_response_length": 8192},
            "worker": {"rollout": {"global_max_tokens": 8192}},
        }
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "hsp_training_contract.json").write_text(
                json.dumps({"max_seq_length": 8192}), encoding="utf-8"
            )
            report = validate_sft_rl_length_contract(directory, config)
        self.assertEqual(report["rl_required_visible_length"], 12288)
        self.assertTrue(any("shorter than" in error for error in report["errors"]))

    def test_sft_rl_length_contract_accepts_matching_window(self):
        config = {
            "data": {"max_prompt_length": 4096, "max_response_length": 8192},
            "worker": {"rollout": {"global_max_tokens": 8192}},
        }
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "hsp_training_contract.json").write_text(
                json.dumps({"max_seq_length": 12288}), encoding="utf-8"
            )
            report = validate_sft_rl_length_contract(directory, config)
        self.assertEqual(report["errors"], [])

    def test_sft_rl_length_contract_requires_checkpoint_metadata(self):
        config = {
            "data": {"max_prompt_length": 4096, "max_response_length": 8192},
            "worker": {"rollout": {"global_max_tokens": 8192}},
        }
        with tempfile.TemporaryDirectory() as directory:
            report = validate_sft_rl_length_contract(directory, config)
        self.assertTrue(any("cannot be verified" in error for error in report["errors"]))


if __name__ == "__main__":
    unittest.main()
