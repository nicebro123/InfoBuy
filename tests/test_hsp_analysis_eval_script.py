import os
import subprocess
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "run_hsp_analysis_eval.sh"


class HSPAnalysisEvalScriptTest(unittest.TestCase):
    def test_fixed_budget_suite_expands_three_budget_evals(self):
        with tempfile.TemporaryDirectory() as directory:
            bin_dir = Path(directory)
            call_log = bin_dir / "calls.log"
            bash_stub = bin_dir / "bash_stub"
            bash_stub.write_text(
                "#!/usr/bin/env bash\n"
                "printf '%s\\n' \"$*\" >> \"$CALL_LOG\"\n"
                "printf 'env EVAL_TASKS=%s OUTPUT_TAG=%s ASK=%s VERIFY=%s\\n' "
                "\"${EVAL_TASKS:-}\" \"${OUTPUT_TAG:-}\" \"${ASK_BUDGET_TOKENS:-}\" "
                "\"${VERIFY_BUDGET_TOKENS:-}\" >> \"$CALL_LOG\"\n"
                "exit 0\n",
                encoding="utf-8",
            )
            bash_stub.chmod(0o755)
            env = dict(os.environ)
            env["BASH_BIN"] = str(bash_stub)
            env["CALL_LOG"] = str(call_log)
            env["MODEL_PATH"] = "/tmp/model"
            env["GPU_QUEUE"] = "1"
            env["SAMPLES_PER_QUESTION"] = "2"
            env["FULL_TASKS"] = "math"
            env["SKIP_LLM_RECHECK"] = "1"
            result = subprocess.run(
                ["bash", str(SCRIPT_PATH), "fixed-budget"],
                env=env,
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )

            log_text = call_log.read_text(encoding="utf-8")

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(log_text.count("eval/evaluate_forhelp.bash"), 3)
        self.assertIn("OUTPUT_TAG=fixed_budget_32_32 ASK=32 VERIFY=32", log_text)
        self.assertIn("OUTPUT_TAG=fixed_budget_64_96 ASK=64 VERIFY=96", log_text)
        self.assertIn("OUTPUT_TAG=fixed_budget_128_192 ASK=128 VERIFY=192", log_text)


if __name__ == "__main__":
    unittest.main()
