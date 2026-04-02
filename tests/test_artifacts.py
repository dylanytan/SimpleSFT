"""Tests for artifact serialization and loading."""

from pathlib import Path

from simplesft.artifacts import load_memory_result, save_memory_result
from simplesft.types import (
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
