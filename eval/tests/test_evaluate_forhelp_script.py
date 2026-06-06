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
