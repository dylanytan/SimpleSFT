"""Topology-aware NCCL environment helpers."""

from __future__ import annotations

import re
import subprocess
from collections.abc import MutableMapping

ANSI_ESCAPE_PATTERN = re.compile(pattern=r"\x1B\[[0-?]*[ -/]*[@-~]")
GPU_NAME_PATTERN = re.compile(pattern=r"GPU\d+")


def _strip_ansi_sequences(*, text: str) -> str:
    """Remove ANSI escape bytes from terminal output.

    Args:
        text: Raw command output that may include terminal formatting.

    Returns:
        Plain-text output.
    """

    return ANSI_ESCAPE_PATTERN.sub("", text)


def read_topology_output() -> str | None:
    """Read `nvidia-smi topo -m` when available.

    Returns:
        Plain-text topology output or `None` if probing fails.

    Example:
        >>> isinstance(read_topology_output(), (str, type(None)))
        True
    """

    try:
        completed = subprocess.run(
            args=["nvidia-smi", "topo", "-m"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return _strip_ansi_sequences(text=completed.stdout)


def read_gpu_product_names() -> list[str] | None:
    """Read installed GPU product names from `nvidia-smi`.

    Returns:
        Ordered GPU product names or `None` if probing fails.
    """

    try:
        completed = subprocess.run(
            args=["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def has_required_cross_numa_gpu_shape(*, gpu_product_names: list[str]) -> bool:
    """Return whether the host matches the 2xA100 cross-NUMA policy shape.

    Args:
        gpu_product_names: GPU product names reported by `nvidia-smi`.

    Returns:
        `True` only when there are exactly two A100 GPUs.
    """

    if len(gpu_product_names) != 2:
        return False
    return all("A100" in gpu_name.upper() for gpu_name in gpu_product_names)


def is_cross_numa_topology(*, topology_text: str) -> bool:
    """Return whether the probed topology indicates cross-NUMA GPU links.

    Args:
        topology_text: Plain-text output from `nvidia-smi topo -m`.

    Returns:
        `True` when at least one GPU-to-GPU link is `SYS`.
    """

    cleaned_text = _strip_ansi_sequences(text=topology_text)
    gpu_names = []
    for raw_line in cleaned_text.splitlines():
        line = raw_line.strip()
        if "CPU Affinity" not in line:
            continue
        gpu_names = GPU_NAME_PATTERN.findall(line)
        break
    if len(gpu_names) < 2:
        return False
    for raw_line in cleaned_text.splitlines():
        line = raw_line.strip()
        if not line.startswith("GPU") or "CPU Affinity" in line:
            continue
        tokens = line.split()
        if tokens[0] not in gpu_names:
            continue
        link_tokens = tokens[1 : 1 + len(gpu_names)]
        if "SYS" in link_tokens:
            return True
    return False


def maybe_apply_cross_numa_nccl_env(*, env: MutableMapping[str, str]) -> tuple[bool, str]:
    """Apply NCCL env overrides when the host matches the known bad topology.

    Args:
        env: Mutable environment mapping for the launched subprocess.

    Returns:
        Tuple of `(applied, reason)` describing the policy outcome.
    """

    gpu_product_names = read_gpu_product_names()
    if gpu_product_names is None:
        return False, "gpu query unavailable"
    if not has_required_cross_numa_gpu_shape(gpu_product_names=gpu_product_names):
        return False, "requires exactly two A100 GPUs"
    topology_text = read_topology_output()
    if topology_text is None:
        return False, "topology probe unavailable"
    if not is_cross_numa_topology(topology_text=topology_text):
        return False, "topology is not cross-NUMA"
    env["NCCL_P2P_DISABLE"] = "1"
    return True, "cross-NUMA topology detected"
