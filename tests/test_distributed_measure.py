"""Tests for DDP measurement aggregation and delegation."""

import pytest

import simplesft.distributed_measure as distributed_measure_module
import simplesft.measure as measure_module
from simplesft.measure import aggregate_rank_results, measure_peak_memory
from simplesft.types import (
    MemoryComponentBreakdown,
    MemoryResult,
    PhaseMemoryRecord,
    TrainingConfig,
)


def _phase_record(*, phase_name: str, reserved_bytes: int) -> PhaseMemoryRecord:
    """Build a compact phase record for aggregation tests."""

    return PhaseMemoryRecord(
        phase_name=phase_name,
        allocated_bytes=reserved_bytes,
        reserved_bytes=reserved_bytes,
        peak_allocated_bytes=reserved_bytes,
        peak_reserved_bytes=reserved_bytes,
        delta_allocated_bytes=reserved_bytes,
        delta_reserved_bytes=reserved_bytes,
    )


def _memory_result(*, global_peak_bytes: int) -> MemoryResult:
    """Build a synthetic measurement result for DDP aggregation tests."""

    config = TrainingConfig(
        tuning_mode="full_ft",
        distributed_mode="ddp",
        gpus_per_node=2,
        gpu_memory_gb=80.0,
    )
    return MemoryResult(
        mode="measure",
        model_name="toy",
        config=config,
        breakdown=MemoryComponentBreakdown(
            parameter_bytes=1,
            gradient_bytes=2,
            optimizer_state_bytes=3,
            activation_bytes=4,
            transient_bytes=5,
            runtime_reserve_bytes=6,
        ),
        phase_records=(
            _phase_record(phase_name="forward", reserved_bytes=global_peak_bytes - 1),
            _phase_record(phase_name="optimizer_step", reserved_bytes=global_peak_bytes),
        ),
        peak_phase="optimizer_step",
        global_peak_bytes=global_peak_bytes,
        feasible=True,
    )


def test_aggregate_rank_results_uses_max_across_ranks() -> None:
    """DDP aggregation should take a conservative max across ranks."""

    aggregated = aggregate_rank_results(
        results=[_memory_result(global_peak_bytes=10), _memory_result(global_peak_bytes=12)]
    )
    assert aggregated.global_peak_bytes == 12
    assert aggregated.metadata["aggregated_across_ranks"] is True
    assert aggregated.metadata["per_rank_global_peak_bytes"] == [10, 12]


def test_measure_peak_memory_delegates_ddp_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Top-level measurement should delegate multi-rank DDP runs to torchrun."""

    expected_result = _memory_result(global_peak_bytes=12)
    monkeypatch.setattr(measure_module, "is_cuda_available", lambda: True)
    monkeypatch.setattr(
        distributed_measure_module,
        "run_ddp_measurement",
        lambda model, config: expected_result,
    )
    result = measure_peak_memory(
        model="toy-model",
        config=TrainingConfig(
            tuning_mode="full_ft",
            distributed_mode="ddp",
            gpus_per_node=2,
        ),
    )
    assert result.global_peak_bytes == expected_result.global_peak_bytes
