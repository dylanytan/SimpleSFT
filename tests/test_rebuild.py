"""Tests for rebuilding benchmark suites from saved measurements."""

from pathlib import Path

from simplesft.results.artifacts import (
    load_memory_result,
    save_benchmark_suite_result,
    save_memory_result,
)
from simplesft.results.rebuild import rebuild_benchmark_suite_from_measurements
from simplesft.types import (
    BenchmarkCase,
    BenchmarkCaseResult,
    BenchmarkSuiteResult,
    MemoryComponentBreakdown,
    MemoryResult,
    ModelSpec,
    PhaseMemoryRecord,
    TrainingConfig,
)


def _toy_model_spec() -> ModelSpec:
    """Return a compact model spec for rebuild tests."""

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


def test_rebuild_benchmark_suite_from_measurements(
    tmp_path: Path,
) -> None:
    """Rebuild should preserve measurements and regenerate estimates."""

    source_dir = tmp_path / "iter1"
    output_dir = tmp_path / "iter2"
    case = BenchmarkCase(
        name="toy-single-gpu",
        model=_toy_model_spec(),
        config=TrainingConfig(tuning_mode="full_ft"),
    )
    measurement_path = source_dir / case.artifact_slug() / "measurement.json"
    measurement = MemoryResult(
        mode="measure",
        model_name="toy",
        config=case.config,
        breakdown=MemoryComponentBreakdown(parameter_bytes=64),
        phase_records=(
            PhaseMemoryRecord(
                phase_name="forward",
                allocated_bytes=64,
                reserved_bytes=64,
                peak_allocated_bytes=64,
                peak_reserved_bytes=64,
                delta_allocated_bytes=64,
                delta_reserved_bytes=64,
            ),
        ),
        peak_phase="forward",
        global_peak_bytes=64,
        feasible=True,
    )
    save_memory_result(result=measurement, path=measurement_path)
    save_benchmark_suite_result(
        result=BenchmarkSuiteResult(
            output_dir=str(source_dir),
            case_results=(
                BenchmarkCaseResult(
                    case=case,
                    estimate_path=str(
                        source_dir / case.artifact_slug() / "estimate.json"
                    ),
                    measurement_path=str(measurement_path),
                ),
            ),
        ),
        path=source_dir / "suite_index.json",
    )

    rebuilt_suite, comparisons = rebuild_benchmark_suite_from_measurements(
        source_dir=source_dir,
        output_dir=output_dir,
    )

    assert len(comparisons) == 1
    rebuilt_case = rebuilt_suite.case_results[0]
    assert rebuilt_case.measurement_path is not None
    assert rebuilt_case.comparison_path is not None
    assert Path(rebuilt_case.estimate_path).exists()
    assert Path(rebuilt_case.measurement_path).exists()
    assert Path(rebuilt_case.comparison_path).exists()
    rebuilt_measurement = load_memory_result(path=rebuilt_case.measurement_path)
    assert rebuilt_measurement.global_peak_bytes == measurement.global_peak_bytes
