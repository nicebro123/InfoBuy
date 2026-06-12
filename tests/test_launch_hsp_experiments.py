import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "launch_hsp_experiments.py"


class HSPExperimentLauncherTest(unittest.TestCase):
    def run_launcher(self, store: Path, *args: str) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env["INFOBUY_STORE"] = str(store)
        env["INFOBUY_CKPT"] = str(store / "checkpoints")
        env["INFOBUY_GENERATED_DATA"] = str(store / "datasets" / "infobuy")
        return subprocess.run(
            [sys.executable, str(LAUNCHER), *args],
            cwd=REPO_ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

    def test_materializes_manifest_run_configs_and_gpu_queues(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Path(directory)
            result = self.run_launcher(
                store,
                "--spec",
                "configs/experiments/hsp_pilot.yaml",
                "--gpu-pairs",
                "0;1",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            study_dir = store / "experiments" / "hsp_pilot"
            manifest = yaml.safe_load((study_dir / "launch_manifest.yaml").read_text(encoding="utf-8"))
            run_names = [run["run_name"] for run in manifest["runs"]]
            self.assertEqual(run_names, ["qwen3_hsp_grpo_main", "qwen3_hsp_grpo_shaped"])
            self.assertTrue((study_dir / "run_gpu0.sh").exists())
            self.assertTrue((study_dir / "run_gpu1.sh").exists())
            self.assertTrue((study_dir / "launch_tmux.sh").exists())

            main_config = yaml.safe_load(
                (study_dir / "qwen3_hsp_grpo_main" / "run_config.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(main_config["trainer"]["experiment_name"], "qwen3_hsp_grpo_main")
            self.assertEqual(main_config["worker"]["reward"]["reward_function_kwargs"]["wrong_reject_weight"], 0.5)

    def test_gpu_queue_captures_external_store_environment(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Path(directory)
            result = self.run_launcher(
                store,
                "--spec",
                "configs/experiments/hsp_smoke.yaml",
                "--gpus",
                "2",
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            queue = store / "experiments" / "hsp_smoke" / "run_gpu2.sh"
            script = queue.read_text(encoding="utf-8")
            self.assertIn(f"export INFOBUY_STORE={store}", script)
            self.assertIn(f"export INFOBUY_CKPT={store / 'checkpoints'}", script)
            self.assertIn("source setup/env.sh >/dev/null", script)
            self.assertIn("export CUDA_VISIBLE_DEVICES=2", script)
            run_config = yaml.safe_load(
                (store / "experiments" / "hsp_smoke" / "qwen3_hsp_grpo_smoke" / "run_config.yaml").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(run_config["trainer"]["max_steps"], 2)
            self.assertIn("train_smoke", run_config["data"]["train_files"])

    def test_existing_completion_marker_marks_run_as_existing(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Path(directory)
            first = self.run_launcher(store, "--spec", "configs/experiments/hsp_smoke.yaml")
            self.assertEqual(first.returncode, 0, first.stderr)
            marker = store / "experiments" / "hsp_smoke" / "qwen3_hsp_grpo_smoke" / "markers" / "done"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("done\n", encoding="utf-8")

            second = self.run_launcher(store, "--spec", "configs/experiments/hsp_smoke.yaml")
            self.assertEqual(second.returncode, 0, second.stderr)
            manifest = yaml.safe_load(
                (store / "experiments" / "hsp_smoke" / "launch_manifest.yaml").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["runs"][0]["status"], "exists")


if __name__ == "__main__":
    unittest.main()
