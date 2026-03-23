"""Utility helpers shared across the SimpleSFT package."""

from __future__ import annotations

from typing import Any

import torch

from .constants import BYTES_PER_GB


DTYPE_BITS = {
    "fp32": 32,
    "float32": 32,
    "bf16": 16,
    "bfloat16": 16,
    "fp16": 16,
    "float16": 16,
    "fp8": 8,
    "int8": 8,
}


def bits_for_dtype(dtype_name: str) -> int:
    """Return the bit width for a dtype name.

    Args:
        dtype_name: Canonical or alias dtype string.

    Returns:
        Bit width for the dtype.
    """

    normalized_name = dtype_name.lower()
    assert normalized_name in DTYPE_BITS, f"Unsupported dtype: {dtype_name}"
    return DTYPE_BITS[normalized_name]


def bytes_for_dtype(dtype_name: str) -> int:
    """Return the byte width for a dtype name."""

    return bits_for_dtype(dtype_name) // 8


def bytes_to_gb(num_bytes: int) -> float:
    """Convert bytes to gibibytes for display."""

    return num_bytes / BYTES_PER_GB


def is_cuda_available() -> bool:
    """Return whether CUDA is available in the current runtime."""

    return torch.cuda.is_available()


def optimizer_state_in_baseline(*, warmup_steps: int, optimizer_name: str) -> bool:
    """Return whether optimizer state exists before the measured step.

    Args:
        warmup_steps: Number of warmup steps run before measurement.
        optimizer_name: Configured optimizer name.

    Returns:
        Whether optimizer state is expected to be materialized at baseline.

    Example:
        >>> optimizer_state_in_baseline(warmup_steps=1, optimizer_name="adamw")
        True
    """

    return warmup_steps > 0 and optimizer_name.lower() == "adamw"


def canonical_torch_dtype(dtype_name: str) -> torch.dtype:
    """Map a user dtype string to a torch dtype.

    Args:
        dtype_name: Canonical dtype string.

    Returns:
        Torch dtype.
    """

    mapping: dict[str, torch.dtype] = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
    }
    normalized_name = dtype_name.lower()
    assert normalized_name in mapping, f"Unsupported torch dtype: {dtype_name}"
    return mapping[normalized_name]


def maybe_get_peft() -> Any:
    """Import `peft` when available.

    Returns:
        Imported `peft` module.
    """

    try:
        import peft  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "LoRA support requires the optional dependency `peft`."
        ) from exc
    return peft


def maybe_get_deepspeed() -> Any:
    """Import `deepspeed` when available.

    Returns:
        Imported `deepspeed` module.
    """

    try:
        import deepspeed  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "ZeRO-2 measurement requires the optional dependency `deepspeed`."
        ) from exc
    return deepspeed
