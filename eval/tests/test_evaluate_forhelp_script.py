import os
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "evaluate_forhelp.bash"
COLLECT_SCRIPT_PATH = Path(__file__).parents[1] / "collect_hsp_candidates.bash"


class EvaluateForHelpScriptTest(unittest.TestCase):
    def test_missing_required_arguments_prints_usage(self):
        result = subprocess.run(
            ["bash", str(SCRIPT_PATH)],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("Usage:", result.stderr)

    def test_empty_gpu_queue_fails_without_starting_jobs(self):
        env = dict(os.environ)
        env.pop("SKIP_LLM_RECHECK", None)
        env.pop("OPENAI_API_KEY", None)
        result = subprocess.run(
            ["bash", str(SCRIPT_PATH), "model", "teacher", "7778", "", "hsp", "1", "policy"],
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("At least one GPU id", result.stderr)

    def test_recheck_failure_is_propagated(self):
        with tempfile.TemporaryDirectory() as directory:
            bin_dir = Path(directory)
            python_stub = bin_dir / "python"
            python_stub.write_text(
                "#!/bin/sh\n"
                "case \"$*\" in\n"
                "  *eval.results_recheck*) exit 7 ;;\n"
                "  *) exit 0 ;;\n"
                "esac\n",
                encoding="utf-8",
            )
            sleep_stub = bin_dir / "sleep"
            sleep_stub.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            python_stub.chmod(0o755)
            sleep_stub.chmod(0o755)
            env = dict(os.environ)
            env["PATH"] = f"{bin_dir}:{env['PATH']}"
            env.pop("SKIP_LLM_RECHECK", None)
            env["OPENAI_API_KEY"] = "test-key"
            result = subprocess.run(
                ["bash", str(SCRIPT_PATH), "model", "teacher", "7778", "0", "hsp", "1", "policy"],
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(result.returncode, 7)
        self.assertNotIn("All tasks have finished", result.stdout)

    def test_task_and_example_limits_are_forwarded(self):
        with tempfile.TemporaryDirectory() as directory:
            bin_dir = Path(directory)
            call_log = bin_dir / "calls.log"
            python_stub = bin_dir / "python"
            python_stub.write_text(
                "#!/bin/sh\n"
                "printf '%s\\n' \"$*\" >> \"$CALL_LOG\"\n"
                "exit 0\n",
                encoding="utf-8",
            )
            sleep_stub = bin_dir / "sleep"
            sleep_stub.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            python_stub.chmod(0o755)
            sleep_stub.chmod(0o755)
            env = dict(os.environ)
            env["PATH"] = f"{bin_dir}:{env['PATH']}"
            env["CALL_LOG"] = str(call_log)
            env["SKIP_LLM_RECHECK"] = "1"
            env["EVAL_TASKS"] = "math"
            env["MAX_EXAMPLES"] = "2"
            env["OUTPUT_TAG"] = "smoke"
            result = subprocess.run(
                ["bash", str(SCRIPT_PATH), "model", "teacher", "7778", "0", "hsp", "1", "policy"],
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )

            log_text = call_log.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0)
        self.assertIn("--dataset math", log_text)
        self.assertIn("--max_examples 2", log_text)
        self.assertIn("--output_tag smoke", log_text)
        self.assertNotIn("--dataset gsm8k", log_text)

    def test_hsp_generator_controls_are_forwarded(self):
        with tempfile.TemporaryDirectory() as directory:
            bin_dir = Path(directory)
            call_log = bin_dir / "calls.log"
            python_stub = bin_dir / "python"
            python_stub.write_text(
                "#!/bin/sh\n"
                "printf '%s\\n' \"$*\" >> \"$CALL_LOG\"\n"
                "exit 0\n",
                encoding="utf-8",
            )
            sleep_stub = bin_dir / "sleep"
            sleep_stub.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            python_stub.chmod(0o755)
            sleep_stub.chmod(0o755)
            env = dict(os.environ)
            env["PATH"] = f"{bin_dir}:{env['PATH']}"
            env["CALL_LOG"] = str(call_log)
            env["SKIP_LLM_RECHECK"] = "1"
            env["EVAL_TASKS"] = "math"
            env["MAX_INTERACTIONS"] = "0"
            env["ASK_BUDGET_TOKENS"] = "32"
            env["VERIFY_BUDGET_TOKENS"] = "48"
            env["STUDENT_TEMPERATURE"] = "0.2"
            env["TEACHER_HELP_TEMPERATURE"] = "0.4"
            env["TEACHER_REVIEW_TEMPERATURE"] = "0.0"
            result = subprocess.run(
                ["bash", str(SCRIPT_PATH), "model", "teacher", "7778", "0", "hsp", "1", "independent"],
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )

            log_text = call_log.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0)
        self.assertIn("--max_interactions 0", log_text)
        self.assertIn("--ask_budget_tokens 32", log_text)
        self.assertIn("--verify_budget_tokens 48", log_text)
        self.assertIn("--student_temperature 0.2", log_text)
        self.assertIn("--teacher_help_temperature 0.4", log_text)
        self.assertIn("--teacher_review_temperature 0.0", log_text)

    def test_collection_script_missing_required_arguments_prints_usage(self):
        result = subprocess.run(
            ["bash", str(COLLECT_SCRIPT_PATH)],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("Usage:", result.stderr)


if __name__ == "__main__":
    unittest.main()
