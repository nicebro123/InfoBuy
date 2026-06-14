import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_full_pipeline_2gpu.sh"


class FullPipeline2GPUTest(unittest.TestCase):
    def test_pipeline_materializes_teacher_and_worker_queues(self):
        with tempfile.TemporaryDirectory() as directory:
            bin_dir = Path(directory) / "bin"
            store = Path(directory) / "store"
            bin_dir.mkdir()
            python_stub = bin_dir / "python"
            python_stub.write_text(
                "#!/usr/bin/env bash\n"
                "case \"$*\" in\n"
                "  *launch_hsp_experiments.py*)\n"
                "    spec=''\n"
                "    while [[ $# -gt 0 ]]; do\n"
                "      if [[ \"$1\" == '--spec' ]]; then spec=\"$2\"; shift 2; else shift; fi\n"
                "    done\n"
                "    study=$(basename \"$spec\" .yaml)\n"
                "    mkdir -p \"$INFOBUY_STORE/experiments/$study\"\n"
                "    printf '#!/usr/bin/env bash\\necho %s\\n' \"$study\" > \"$INFOBUY_STORE/experiments/$study/run_gpu1.sh\"\n"
                "    chmod +x \"$INFOBUY_STORE/experiments/$study/run_gpu1.sh\"\n"
                "    ;;\n"
                "  *check_hsp_full_data.py*) exit 0 ;;\n"
                "  *) /usr/bin/python3 \"$@\" ;;\n"
                "esac\n",
                encoding="utf-8",
            )
            python_stub.chmod(0o755)
            env = dict(os.environ)
            env["PATH"] = f"{bin_dir}:{env['PATH']}"
            env["INFOBUY_STORE"] = str(store)
            env["INFOBUY_CKPT"] = str(store / "checkpoints")
            env["INFOBUY_GENERATED_DATA"] = str(store / "datasets" / "infobuy")
            env["INFOBUY_PYTHON"] = str(python_stub)
            env["TEACHER_GPU"] = "0"
            env["WORKER_GPU"] = "1"
            result = subprocess.run(
                ["bash", str(SCRIPT), "--skip-sft"],
                cwd=REPO_ROOT,
                env=env,
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )

            queue_root = store / "experiments" / "full_pipeline_2gpu"
            teacher_queue = queue_root / "run_teacher_gpu0.sh"
            worker_queue = queue_root / "run_worker_gpu1.sh"

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("[ready] teacher queue", result.stdout)
            self.assertIn("[ready] worker queue", result.stdout)
            self.assertTrue(teacher_queue.exists())
            self.assertTrue(worker_queue.exists())
            teacher_text = teacher_queue.read_text(encoding="utf-8")
            worker_text = worker_queue.read_text(encoding="utf-8")
            self.assertIn(f"export INFOBUY_STORE={store}", teacher_text)
            self.assertIn(f"export INFOBUY_STORE={store}", worker_text)
            self.assertIn("hsp_official/run_gpu1.sh", worker_text)
            self.assertIn("hsp_analysis/run_gpu1.sh", worker_text)

    def test_pipeline_rejects_overlapping_teacher_and_worker_gpu(self):
        env = dict(os.environ)
        env["TEACHER_GPU"] = "1"
        env["WORKER_GPU"] = "1"
        result = subprocess.run(
            ["bash", str(SCRIPT), "--skip-sft"],
            cwd=REPO_ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn("must be distinct", result.stderr)


if __name__ == "__main__":
    unittest.main()
