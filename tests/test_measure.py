"""Tests for measurement-path behavior that do not require CUDA."""

import pytest

import simplesft.distributed_zero2_measure as zero2_measure_module
import simplesft.measure as measure_module
from simplesft.measure import measure_peak_memory
from simplesft.types import MemoryComponentBreakdown, MemoryResult, PhaseMemoryRecord
from simplesft.types import ModelSpec, TrainingConfig


def _toy_model_spec() -> ModelSpec:
    """Return a compact model spec for measurement tests."""

    return ModelSpec(
        model_name="toy",
        model_type="llama",
        num_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        intermediate_size=64,
        vocab_size=128,
        max_position_embeddings=128,
        total_params=10_000,
        trainable_linear_layers=(),
    )


def test_measure_peak_memory_requires_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Measurement should fail clearly when CUDA is unavailable."""

    monkeypatch.setattr(measure_module, "is_cuda_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is required"):
        measure_peak_memory(
            model=_toy_model_spec(),
            config=TrainingConfig(tuning_mode="full_ft"),
        )


def test_measure_peak_memory_zero2_requires_optional_backend() -> None:
    """ZeRO-2 measurement should fail clearly without its runtime stack."""

    with pytest.raises(RuntimeError):
        measure_peak_memory(
            model=_toy_model_spec(),
            config=TrainingConfig(tuning_mode="full_ft", distributed_mode="zero2"),
        )


def test_measure_peak_memory_delegates_zero2_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Top-level measurement should delegate multi-rank ZeRO-2 runs."""

    expected_result = MemoryResult(
        mode="measure",
        model_name="toy",
        config=TrainingConfig(
            tuning_mode="full_ft",
            distributed_mode="zero2",
            gpus_per_node=2,
            gpu_memory_gb=80.0,
        ),
        breakdown=MemoryComponentBreakdown(parameter_bytes=1),
        phase_records=(
            PhaseMemoryRecord(
                phase_name="optimizer_step",
                allocated_bytes=12,
                reserved_bytes=12,
                peak_allocated_bytes=12,
                peak_reserved_bytes=12,
                delta_allocated_bytes=12,
                delta_reserved_bytes=12,
            ),
        ),
        peak_phase="optimizer_step",
        global_peak_bytes=12,
        feasible=True,
    )
    monkeypatch.setattr(measure_module, "is_cuda_available", lambda: True)
    monkeypatch.setattr(measure_module, "maybe_get_deepspeed", lambda: object())
    monkeypatch.setattr(
        zero2_measure_module,
        "run_zero2_measurement",
        lambda model, config: expected_result,
    )
    result = measure_peak_memory(
        model="toy-model",
        config=TrainingConfig(
            tuning_mode="full_ft",
            distributed_mode="zero2",
            gpus_per_node=2,
        ),
    )
    assert result.global_peak_bytes == expected_result.global_peak_bytes
