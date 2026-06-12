#!/usr/bin/env python3
"""Generate and optionally launch InfoBuy HSP experiment queues.

The workflow mirrors the public experiment pattern used in EvoCo-style repos:

1. write a compact study spec under ``configs/experiments``;
2. keep base RL configs stable under ``RL_stage/examples``;
3. materialize one immutable ``run_config.yaml`` per run outside the repo;
4. write a manifest plus per-GPU shell queues;
5. optionally start those queues in tmux.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping

import yaml


DEFAULT_SPEC = "configs/experiments/hsp_pilot.yaml"
DEFAULT_LAUNCHER = "RL_stage/examples/qwen3_hsp_grpo.sh"
STORAGE_ENV_KEYS = (
    "INFOBUY_STORE",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HF_DATASETS_CACHE",
    "STORAGE_PATH",
    "INFOBUY_MODELS",
    "INFOBUY_DATASETS",
    "INFOBUY_CKPT",
    "INFOBUY_LOGS",
    "INFOBUY_SERVICES",
    "INFOBUY_TMP",
    "INFOBUY_PRETRAINED_MODELS",
    "INFOBUY_TEACHER_MODELS",
    "INFOBUY_GENERATED_DATA",
)


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")
    return data


def write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False, allow_unicode=True)


def expand_path(value: str | os.PathLike[str], *, cwd: Path) -> Path:
    expanded = os.path.expandvars(str(value))
    path = Path(expanded).expanduser()
    if not path.is_absolute():
        path = (cwd / path).resolve()
    return path


def slug(value: str) -> str:
    chars: list[str] = []
    for char in value.strip().lower():
        if char.isalnum():
            chars.append(char)
        elif char in {"-", "_", ".", "+"}:
            chars.append(char)
        elif char.isspace() or char in {",", "/", ":"}:
            chars.append("_")
    return "".join(chars).strip("._-") or "run"


def quote_command(parts: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def set_by_path(payload: MutableMapping[str, Any], dotted_key: str, value: Any) -> None:
    if not dotted_key or dotted_key.startswith(".") or dotted_key.endswith("."):
        raise ValueError(f"Invalid override key: {dotted_key!r}")
    current: MutableMapping[str, Any] = payload
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        child = current.get(part)
        if child is None:
            child = {}
            current[part] = child
        if not isinstance(child, MutableMapping):
            raise ValueError(f"Cannot set {dotted_key!r}: {part!r} is not a mapping.")
        current = child
    current[parts[-1]] = value


def apply_overrides(base: Mapping[str, Any], overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    config = deepcopy(dict(base))
    for key, value in (overrides or {}).items():
        set_by_path(config, str(key), value)
    return config


def parse_gpu_pairs(raw: str | None) -> list[str]:
    if raw is None or not str(raw).strip():
        return []
    pairs = [item.strip() for item in str(raw).split(";") if item.strip()]
    if not pairs:
        raise ValueError("GPU pair list is empty.")
    for pair in pairs:
        ids = [item.strip() for item in pair.split(",") if item.strip()]
        if not ids:
            raise ValueError(f"Invalid GPU pair: {pair!r}")
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate GPU id in pair: {pair!r}")
        for gpu_id in ids:
            if not gpu_id.isdigit():
                raise ValueError(f"GPU ids must be integers: {pair!r}")
    return pairs


def spec_experiments(spec: Mapping[str, Any], gpu_pairs: list[str], gpu_override: str | None) -> list[dict[str, Any]]:
    experiments = spec.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("Spec must contain a non-empty 'experiments' list.")
    defaults = spec.get("defaults", {}) or {}
    if not isinstance(defaults, Mapping):
        raise ValueError("Spec 'defaults' must be a mapping.")

    output: list[dict[str, Any]] = []
    for index, item in enumerate(experiments):
        if not isinstance(item, Mapping):
            raise ValueError(f"experiments[{index}] must be a mapping.")
        merged = dict(defaults)
        merged.update(dict(item))
        if gpu_pairs:
            merged["gpu"] = gpu_pairs[index % len(gpu_pairs)]
        elif gpu_override:
            merged["gpu"] = gpu_override
        if "name" not in merged:
            raise ValueError(f"experiments[{index}] is missing required field 'name'.")
        output.append(merged)
    return output


def run_name(index: int, experiment: Mapping[str, Any]) -> str:
    if experiment.get("run_name"):
        return slug(str(experiment["run_name"]))
    return f"{index:02d}_{slug(str(experiment['name']))}"


def build_run(
    *,
    spec: Mapping[str, Any],
    spec_path: Path,
    repo_root: Path,
    experiment: Mapping[str, Any],
    index: int,
    overwrite: bool,
) -> dict[str, Any]:
    base_config_raw = experiment.get("config", spec.get("base_config"))
    if not base_config_raw:
        raise ValueError("Spec or experiment must define 'base_config' or 'config'.")
    base_config_path = expand_path(str(base_config_raw), cwd=repo_root)
    base_config = read_yaml(base_config_path)

    overrides: dict[str, Any] = {}
    spec_overrides = spec.get("overrides", {}) or {}
    if not isinstance(spec_overrides, Mapping):
        raise ValueError("Spec 'overrides' must be a mapping.")
    overrides.update(dict(spec_overrides))

    experiment_overrides = experiment.get("overrides", {}) or {}
    if not isinstance(experiment_overrides, Mapping):
        raise ValueError(f"Experiment {experiment['name']!r} overrides must be a mapping.")
    overrides.update(dict(experiment_overrides))

    name = run_name(index, experiment)
    study_name = slug(str(spec.get("study_name") or spec_path.stem))
    output_root = expand_path(str(spec.get("output_root", "${INFOBUY_STORE}/experiments")), cwd=repo_root)
    checkpoint_root = expand_path(str(spec.get("checkpoint_root", "${INFOBUY_CKPT}/rl")), cwd=repo_root)
    study_dir = output_root / study_name
    run_dir = study_dir / name
    marker_dir = run_dir / "markers"
    completion_marker = marker_dir / "done"
    train_log = run_dir / "train.log"
    run_config_path = run_dir / "run_config.yaml"
    save_path = expand_path(str(experiment.get("save_path", checkpoint_root / name)), cwd=repo_root)
    model_path = expand_path(str(experiment.get("model_path", "${INFOBUY_CKPT}/sft/qwen3-0.6b-hsp-sft")), cwd=repo_root)
    teacher_port = int(experiment.get("teacher_port", 7778))
    gpu = str(experiment.get("gpu", "0"))
    launcher = expand_path(str(experiment.get("launcher", spec.get("launcher", DEFAULT_LAUNCHER))), cwd=repo_root)

    overrides.setdefault("trainer.experiment_name", name)
    materialized = apply_overrides(base_config, overrides)
    write_yaml(run_config_path, materialized)

    env = dict(spec.get("env", {}) or {})
    env.update(dict(experiment.get("env", {}) or {}))
    status = "exists" if completion_marker.exists() and not overwrite else "ready"

    command = [
        "bash",
        str(launcher),
        f"worker.rollout.port={teacher_port}",
    ]
    for extra in experiment.get("extra_overrides", []) or []:
        command.append(str(extra))

    return {
        "index": index,
        "name": str(experiment["name"]),
        "run_name": name,
        "status": status,
        "gpu": gpu,
        "base_config": str(base_config_path),
        "run_config": str(run_config_path),
        "run_dir": str(run_dir),
        "save_path": str(save_path),
        "model_path": str(model_path),
        "teacher_port": teacher_port,
        "train_log": str(train_log),
        "completion_marker": str(completion_marker),
        "command": command,
        "env": env,
        "overwrite": overwrite,
    }


def shell_exports(env: Mapping[str, Any]) -> list[str]:
    lines = []
    for key, value in env.items():
        if not str(key).replace("_", "").isalnum():
            raise ValueError(f"Invalid environment variable name: {key!r}")
        lines.append(f"export {key}={shlex.quote(str(value))}")
    return lines


def captured_storage_exports() -> list[str]:
    exports: list[str] = []
    for key in STORAGE_ENV_KEYS:
        value = os.environ.get(key)
        if value:
            exports.append(f"export {key}={shlex.quote(value)}")
    return exports


def write_gpu_queue(path: Path, runs: list[Mapping[str, Any]], *, repo_root: Path) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {shlex.quote(str(repo_root))}",
    ]
    lines.extend(captured_storage_exports())
    lines.extend(
        [
            "source setup/env.sh >/dev/null",
            "",
        ]
    )
    for run in runs:
        lines.extend(
            [
                f"echo '==> {run['run_name']} on GPU {run['gpu']}'",
                f"mkdir -p {shlex.quote(str(Path(str(run['train_log'])).parent))} {shlex.quote(str(Path(str(run['completion_marker'])).parent))}",
                f"if [[ -f {shlex.quote(str(run['completion_marker']))} && {str(run['overwrite']).lower()} != true ]]; then",
                f"  echo 'skip existing {run['run_name']}'",
                "else",
                f"  export CUDA_VISIBLE_DEVICES={shlex.quote(str(run['gpu']))}",
                f"  export MODEL_PATH={shlex.quote(str(run['model_path']))}",
                f"  export SAVE_PATH={shlex.quote(str(run['save_path']))}",
                f"  export HSP_CONFIG={shlex.quote(str(run['run_config']))}",
                "  export PYTHONUNBUFFERED=1",
            ]
        )
        for export_line in shell_exports(run.get("env", {})):
            lines.append(f"  {export_line}")
        lines.extend(
            [
                f"  {quote_command(run['command'])} 2>&1 | tee {shlex.quote(str(run['train_log']))}",
                f"  touch {shlex.quote(str(run['completion_marker']))}",
                "fi",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)


def write_tmux_launcher(path: Path, queues: list[Path], *, study_name: str) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "TMUX_BIN=${TMUX_BIN:-tmux}",
        f"SESSION_PREFIX=${{INFOBUY_TMUX_SESSION_PREFIX:-infobuy_{slug(study_name)}}}",
        "",
    ]
    for queue in queues:
        suffix = slug(queue.stem.replace("run_", ""))
        lines.extend(
            [
                f"session=\"${{SESSION_PREFIX}}_{suffix}\"",
                f"cmd={shlex.quote('bash ' + shlex.quote(str(queue)))}",
                "if \"$TMUX_BIN\" has-session -t \"$session\" 2>/dev/null; then",
                "  echo \"tmux session already exists: $session\"",
                "else",
                "  \"$TMUX_BIN\" new-session -d -s \"$session\" \"$cmd\"",
                "  echo \"started tmux session: $session\"",
                "fi",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")
    path.chmod(0o755)


def write_launch_files(
    runs: list[dict[str, Any]], *, study_dir: Path, study_name: str, repo_root: Path
) -> dict[str, Any]:
    by_gpu: dict[str, list[Mapping[str, Any]]] = {}
    for run in runs:
        by_gpu.setdefault(slug(str(run["gpu"])), []).append(run)

    queues: list[Path] = []
    for gpu_slug, gpu_runs in by_gpu.items():
        queue_path = study_dir / f"run_gpu{gpu_slug}.sh"
        write_gpu_queue(queue_path, list(gpu_runs), repo_root=repo_root)
        queues.append(queue_path)

    tmux_path = study_dir / "launch_tmux.sh"
    write_tmux_launcher(tmux_path, queues, study_name=study_name)
    return {"queues": [str(path) for path in queues], "tmux_launcher": str(tmux_path)}


def launch_foreground(queues: Iterable[str]) -> None:
    for queue in queues:
        subprocess.run(["bash", queue], check=True)


def launch_tmux(tmux_launcher: str) -> None:
    subprocess.run(["bash", tmux_launcher], check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and launch InfoBuy HSP experiment queues.")
    parser.add_argument("--spec", default=DEFAULT_SPEC, help="Experiment spec YAML.")
    parser.add_argument("--launch", action="store_true", help="Run generated queues sequentially in this process.")
    parser.add_argument("--launch-tmux", action="store_true", help="Start generated queues in tmux.")
    parser.add_argument("--tmux", action="store_true", help="Alias for --launch-tmux.")
    parser.add_argument("--dry-run", action="store_true", help="Generate files only. This is the default.")
    parser.add_argument("--overwrite", action="store_true", help="Run even when completion markers exist.")
    parser.add_argument("--gpu-pairs", default=None, help="Semicolon-separated GPU workers, e.g. '0;1;2,3'.")
    parser.add_argument("--gpus", default=None, help="Override every run GPU assignment, e.g. '0' or '0,1'.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    spec_path = expand_path(args.spec, cwd=repo_root)
    spec = read_yaml(spec_path)

    gpu_pairs = parse_gpu_pairs(
        args.gpu_pairs or os.environ.get("INFOBUY_GPU_PAIRS") or os.environ.get("HSP_GPU_PAIRS")
    )
    experiments = spec_experiments(spec, gpu_pairs, args.gpus or os.environ.get("INFOBUY_GPUS"))
    study_name = slug(str(spec.get("study_name") or spec_path.stem))
    output_root = expand_path(str(spec.get("output_root", "${INFOBUY_STORE}/experiments")), cwd=repo_root)
    study_dir = output_root / study_name
    study_dir.mkdir(parents=True, exist_ok=True)

    runs = [
        build_run(
            spec=spec,
            spec_path=spec_path,
            repo_root=repo_root,
            experiment=experiment,
            index=index,
            overwrite=args.overwrite,
        )
        for index, experiment in enumerate(experiments)
    ]
    launch_files = write_launch_files(runs, study_dir=study_dir, study_name=study_name, repo_root=repo_root)
    manifest = {
        "study_name": study_name,
        "spec": str(spec_path),
        "study_dir": str(study_dir),
        "runs": runs,
        **launch_files,
    }
    manifest_path = study_dir / "launch_manifest.yaml"
    write_yaml(manifest_path, manifest)

    print(f"study: {study_name}")
    print(f"manifest: {manifest_path}")
    for run in runs:
        print(f"- {run['run_name']} [{run['status']}] gpu={run['gpu']} config={run['run_config']}")

    if args.launch_tmux or args.tmux:
        launch_tmux(str(launch_files["tmux_launcher"]))
    elif args.launch:
        launch_foreground(launch_files["queues"])


if __name__ == "__main__":
    main()
