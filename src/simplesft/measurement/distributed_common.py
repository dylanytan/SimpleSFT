"""Shared helpers for distributed measurement launchers."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..results.artifacts import _load_measurement_config, _load_model_spec
from .topology import maybe_apply_cross_numa_nccl_env
from ..types import ModelSpec, TrainingConfig


def repo_root() -> Path:
    """Return the repository root used for torchrun subprocesses."""

    return Path(__file__).resolve().parents[2]


def serialize_model(model: str | ModelSpec) -> str | dict[str, Any]:
    """Serialize a model reference for a distributed measurement request."""

    if isinstance(model, str):
        return model
    return asdict(model)


def load_request(*, path: str | Path) -> tuple[str | ModelSpec, TrainingConfig]:
    """Load a distributed measurement request from JSON."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    model_payload = raw["model"]
    model: str | ModelSpec = model_payload
    if isinstance(model_payload, dict):
        model = _load_model_spec(raw=model_payload)
    return model, _load_measurement_config(raw=raw["config"])


def torchrun_command(
    *,
    module_name: str,
    input_path: Path,
    output_path: Path,
    gpus_per_node: int,
) -> list[str]:
    """Build a `torchrun` command for a distributed measurement worker."""

    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={gpus_per_node}",
        "-m",
        module_name,
        "--input",
        str(input_path),
        "--output",
        str(output_path),
    ]


def build_torchrun_env() -> tuple[dict[str, str], bool, str]:
    """Build the subprocess environment for distributed measurement launches."""

    launch_env = dict(os.environ)
    triton_cache_dir = (
        Path("/tmp") / os.environ.get("USER", "simplesft") / "triton-cache"
    )
    triton_cache_dir.mkdir(parents=True, exist_ok=True)
    launch_env["TRITON_CACHE_DIR"] = str(triton_cache_dir)
    applied, reason = maybe_apply_cross_numa_nccl_env(env=launch_env)
    return launch_env, applied, reason
