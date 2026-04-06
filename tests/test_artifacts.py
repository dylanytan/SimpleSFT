"""Tests for artifact serialization and loading."""

from pathlib import Path

from simplesft.results.artifacts import (
    load_comparison_result,
    load_memory_result,
    save_comparison_result,
    save_memory_result,
)
from simplesft.types import (
    ComparisonResult,
    MemoryComponentBreakdown,
    MemoryResult,
    PhaseMemoryRecord,
    TrainingConfig,
)


def test_memory_result_roundtrip(tmp_path: Path) -> None:
    """Memory results should round-trip through JSON artifacts."""

    result = MemoryResult(
        mode="estimate",
        model_name="toy",
        config=TrainingConfig(tuning_mode="full_ft"),
        breakdown=MemoryComponentBreakdown(parameter_bytes=123),
        phase_records=(
            PhaseMemoryRecord(
                phase_name="forward",
                allocated_bytes=1,
                reserved_bytes=2,
                peak_allocated_bytes=3,
                peak_reserved_bytes=4,
                delta_allocated_bytes=1,
                delta_reserved_bytes=2,
            ),
        ),
        peak_phase="forward",
        global_peak_bytes=4,
        feasible=True,
        metadata={"x": 1},
        assumptions=("a",),
    )
    artifact_path = tmp_path / "result.json"
    save_memory_result(result=result, path=artifact_path)
    loaded_result = load_memory_result(path=artifact_path)
    assert loaded_result.global_peak_bytes == result.global_peak_bytes
    assert loaded_result.breakdown.parameter_bytes == 123


def test_comparison_result_roundtrip_preserves_phase_aligned_fields(
    tmp_path: Path,
) -> None:
    """Comparison artifacts should preserve phase-aligned error metadata."""

    result = MemoryResult(
        mode="estimate",
        model_name="toy",
        config=TrainingConfig(tuning_mode="full_ft"),
        breakdown=MemoryComponentBreakdown(parameter_bytes=123),
        phase_records=(),
        peak_phase="forward",
        global_peak_bytes=4,
        feasible=True,
    )
    comparison = ComparisonResult(
        model_name="toy",
        measured=result,
        estimated=result,
        global_peak_error_bytes=0,
        global_peak_relative_error=0.0,
        phase_peak_error_bytes={"forward": 0},
        phase_peak_relative_error={"forward": 0.0},
        component_error_bytes={"parameter_bytes": 0},
        component_relative_error={"parameter_bytes": 0.0},
        phase_aligned_component_error_bytes={"transient_bytes": 2},
        phase_aligned_component_relative_error={"transient_bytes": 0.25},
        retained_forward_proxy_error_bytes=3,
        retained_forward_proxy_relative_error=0.5,
    )
    artifact_path = tmp_path / "comparison.json"
    save_comparison_result(result=comparison, path=artifact_path)
    loaded_result = load_comparison_result(path=artifact_path)
    assert loaded_result.phase_aligned_component_error_bytes["transient_bytes"] == 2
    assert loaded_result.retained_forward_proxy_error_bytes == 3
