"""Rebuild benchmark artifacts from saved measurement corpora."""

from __future__ import annotations

from pathlib import Path

from .artifacts import (
    load_benchmark_suite_result,
    load_memory_result,
    save_benchmark_suite_result,
    save_comparison_result,
    save_memory_result,
)
from .compare import compare_measurement_to_estimate
from .estimate import estimate_peak_memory
from .types import (
    BenchmarkCase,
    BenchmarkCaseResult,
    BenchmarkSuiteResult,
    ComparisonResult,
    EstimatorConfig,
    MeasurementConfig,
)


def _measurement_to_estimator_config(
    *, config: EstimatorConfig | MeasurementConfig
) -> EstimatorConfig:
    """Return the structural estimator config for a saved measurement config."""

    if isinstance(config, EstimatorConfig):
        return config
    return config.to_estimator_config()


def _case_output_dir(*, output_dir: str | Path, case: BenchmarkCase) -> Path:
    """Return the output directory for one rebuilt benchmark case."""

    return Path(output_dir) / case.artifact_slug()


def _measurement_path_from_case_result(case_result: BenchmarkCaseResult) -> Path:
    """Return the saved measurement path for one benchmark case result."""

    assert (
        case_result.measurement_path is not None
    ), f"Measurement artifact missing for benchmark case {case_result.case.name}."
    return Path(case_result.measurement_path)


def _rebuild_case_from_measurement(
    *,
    case_result: BenchmarkCaseResult,
    output_dir: str | Path,
) -> tuple[BenchmarkCaseResult, ComparisonResult]:
    """Rebuild estimate and comparison artifacts from a saved measurement."""

    measurement = load_memory_result(
        path=_measurement_path_from_case_result(case_result)
    )
    estimate = estimate_peak_memory(
        model=case_result.case.model,
        config=_measurement_to_estimator_config(config=measurement.config),
    )
    comparison = compare_measurement_to_estimate(
        measured=measurement,
        estimated=estimate,
    )
    case_output_dir = _case_output_dir(output_dir=output_dir, case=case_result.case)
    measurement_path = case_output_dir / "measurement.json"
    estimate_path = case_output_dir / "estimate.json"
    comparison_path = case_output_dir / "comparison.json"
    save_memory_result(result=measurement, path=measurement_path)
    save_memory_result(result=estimate, path=estimate_path)
    save_comparison_result(result=comparison, path=comparison_path)
    return (
        BenchmarkCaseResult(
            case=case_result.case,
            estimate_path=str(estimate_path),
            measurement_path=str(measurement_path),
            comparison_path=str(comparison_path),
        ),
        comparison,
    )


def rebuild_benchmark_suite_from_measurements(
    *,
    source_dir: str | Path,
    output_dir: str | Path,
) -> tuple[BenchmarkSuiteResult, list[ComparisonResult]]:
    """Rebuild a benchmark suite from saved measurements.

    Args:
        source_dir: Existing benchmark suite with saved `measurement.json` artifacts.
        output_dir: Destination directory for rebuilt estimates and comparisons.

    Returns:
        A rebuilt suite index and the newly generated comparisons.

    Example:
        >>> from pathlib import Path
        >>> rebuild_benchmark_suite_from_measurements(
        ...     source_dir=Path('benchmark_artifacts/iter1'),
        ...     output_dir=Path('benchmark_artifacts/iter2'),
        ... )
        Traceback (most recent call last):
        ...
        FileNotFoundError: ...
    """

    source_suite = load_benchmark_suite_result(
        path=Path(source_dir) / "suite_index.json",
    )
    rebuilt_case_results = []
    comparisons = []
    for case_result in source_suite.case_results:
        rebuilt_case_result, comparison = _rebuild_case_from_measurement(
            case_result=case_result,
            output_dir=output_dir,
        )
        rebuilt_case_results.append(rebuilt_case_result)
        comparisons.append(comparison)
    rebuilt_suite = BenchmarkSuiteResult(
        output_dir=str(output_dir),
        case_results=tuple(rebuilt_case_results),
        notes=("Rebuilt from saved measurements using the current estimator.",),
    )
    save_benchmark_suite_result(
        result=rebuilt_suite,
        path=Path(output_dir) / "suite_index.json",
    )
    return rebuilt_suite, comparisons
