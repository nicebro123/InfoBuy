import contextlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "launch_hsp_experiments.py"


class TeacherHealthHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        payload = {"results": [{"text": "ok", "finish_reason": "stop", "token_count": 1}]}
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


@contextlib.contextmanager
def teacher_health_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), TeacherHealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


class HSPExperimentLauncherTest(unittest.TestCase):
    def run_launcher(
        self, store: Path, *args: str, env_extra: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess[str]:
        env = dict(os.environ)
        env["INFOBUY_STORE"] = str(store)
        env["INFOBUY_CKPT"] = str(store / "checkpoints")
        env["INFOBUY_GENERATED_DATA"] = str(store / "datasets" / "infobuy")
        if env_extra:
            env.update(env_extra)
        return subprocess.run(
            [sys.executable, str(LAUNCHER), *args],
            cwd=REPO_ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )

    def make_fake_tmux(self, directory: Path) -> tuple[Path, Path]:
        log_path = directory / "tmux.log"
        tmux_path = directory / "fake_tmux"
        tmux_path.write_text(
            "#!/usr/bin/env bash\n"
            "if [[ \"$1\" == \"has-session\" ]]; then\n"
            "  exit 1\n"
            "fi\n"
            "printf '%q ' \"$@\" >> \"$TMP_FAKE_TMUX_LOG\"\n"
            "printf '\\n' >> \"$TMP_FAKE_TMUX_LOG\"\n",
            encoding="utf-8",
        )
        tmux_path.chmod(0o755)
        return tmux_path, log_path

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

    def test_launch_requires_explicit_training_gpu_assignment(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Path(directory)
            result = self.run_launcher(
                store,
                "--spec",
                "configs/experiments/hsp_smoke.yaml",
                "--launch-tmux",
                "--skip-teacher-check",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("requires explicit training GPUs", result.stderr)

    def test_launch_rejects_training_teacher_gpu_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Path(directory)
            result = self.run_launcher(
                store,
                "--spec",
                "configs/experiments/hsp_smoke.yaml",
                "--launch-tmux",
                "--skip-teacher-check",
                "--gpus",
                "1",
                "--teacher-gpus",
                "1",
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("overlap with teacher GPU", result.stderr)

    def test_launch_health_check_then_starts_tmux_with_distinct_gpus(self):
        with tempfile.TemporaryDirectory() as directory, teacher_health_server() as port:
            store = Path(directory) / "store"
            tmux_path, log_path = self.make_fake_tmux(Path(directory))
            spec_path = Path(directory) / "spec.yaml"
            spec_path.write_text(
                "\n".join(
                    [
                        "study_name: hsp_health_test",
                        "base_config: RL_stage/examples/config_hsp_smoke.yaml",
                        "output_root: ${INFOBUY_STORE}/experiments",
                        "checkpoint_root: ${INFOBUY_CKPT}/rl",
                        "launcher: RL_stage/examples/qwen3_hsp_grpo.sh",
                        "defaults:",
                        "  gpu: \"0\"",
                        "  model_path: ${INFOBUY_CKPT}/sft/qwen3-0.6b-hsp-sft",
                        f"  teacher_port: {port}",
                        "experiments:",
                        "  - name: rl_smoke",
                        "    run_name: qwen3_hsp_grpo_smoke",
                        "    config: RL_stage/examples/config_hsp_smoke.yaml",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            result = self.run_launcher(
                store,
                "--spec",
                str(spec_path),
                "--launch-tmux",
                "--gpus",
                "0",
                "--teacher-gpus",
                "1",
                "--teacher-check-timeout",
                "2",
                env_extra={
                    "TMUX_BIN": str(tmux_path),
                    "TMP_FAKE_TMUX_LOG": str(log_path),
                },
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("teacher health check passed", result.stdout)
            self.assertIn("new-session", log_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
