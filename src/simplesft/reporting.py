"""Developer-facing reporting for comparison and calibration iterations."""

from __future__ import annotations

from statistics import mean, median

from .types import BenchmarkSuiteResult, ComparisonResult
from .utils import bytes_to_gb


def _format_percent(value: float) -> str:
    """Return a percent string for a relative error value."""

    return f"{value * 100:.2f}%"


def _aggregate_max_relative_errors(
    *,
    comparisons: list[ComparisonResult],
    field_name: str,
) -> list[tuple[str, float]]:
    """Aggregate maximum relative errors across comparisons."""

    aggregate: dict[str, float] = {}
    for comparison in comparisons:
        raw_mapping = getattr(comparison, field_name)
        for key, value in raw_mapping.items():
            aggregate[key] = max(aggregate.get(key, 0.0), value)
    return sorted(aggregate.items(), key=lambda item: item[1], reverse=True)


def render_comparison_report(
    *,
    iteration_name: str,
    comparisons: list[ComparisonResult],
) -> str:
    """Render a developer-readable iteration report.

    Args:
        iteration_name: Label for the iteration report.
        comparisons: Comparison results for the benchmark corpus.

    Returns:
        Markdown text describing benchmark errors and next debugging targets.
    """

    assert comparisons, "At least one comparison is required to render a report."
    mean_peak_error_gb = mean(
        bytes_to_gb(comparison.global_peak_error_bytes) for comparison in comparisons
    )
    median_peak_relative_error = median(
        comparison.global_peak_relative_error for comparison in comparisons
    )
    worst_component_errors = _aggregate_max_relative_errors(
        comparisons=comparisons,
        field_name="component_relative_error",
    )
    worst_phase_errors = _aggregate_max_relative_errors(
        comparisons=comparisons,
        field_name="phase_peak_relative_error",
    )
    worst_workspace_errors = _aggregate_max_relative_errors(
        comparisons=comparisons,
        field_name="workspace_proxy_relative_error",
    )
    lines = [f"# {iteration_name}", "", "## Benchmarks"]
    for comparison in comparisons:
        lines.append(
            f"- {comparison.model_name} / {comparison.measured.config.tuning_mode} / "
            f"{comparison.measured.config.distributed_mode} / "
            f"seq={comparison.measured.config.max_seq_len} / "
            f"mb={comparison.measured.config.micro_batch_size_per_gpu}"
        )
    lines.extend(
        [
            "",
            "## Summary",
            f"- Mean global peak error (GiB): {mean_peak_error_gb:.3f}",
            f"- Median global relative error: {_format_percent(median_peak_relative_error)}",
            "",
            "## Largest mismatches",
        ]
    )
    sorted_comparisons = sorted(
        comparisons,
        key=lambda comparison: comparison.global_peak_error_bytes,
        reverse=True,
    )
    for comparison in sorted_comparisons[:5]:
        lines.append(
            f"- {comparison.model_name}: peak error "
            f"{bytes_to_gb(comparison.global_peak_error_bytes):.3f} GiB; "
            f"relative={_format_percent(comparison.global_peak_relative_error)}; "
            f"notes={'; '.join(comparison.notes)}"
        )
    lines.extend(["", "## Worst component relative errors"])
    for component_name, relative_error in worst_component_errors[:5]:
        lines.append(f"- {component_name}: {_format_percent(relative_error)}")
    lines.extend(["", "## Worst phase relative errors"])
    for phase_name, relative_error in worst_phase_errors[:5]:
        lines.append(f"- {phase_name}: {_format_percent(relative_error)}")
    if worst_workspace_errors:
        lines.extend(["", "## Worst workspace-proxy relative errors"])
        for proxy_name, relative_error in worst_workspace_errors[:5]:
            lines.append(f"- {proxy_name}: {_format_percent(relative_error)}")
    lines.extend(
        [
            "",
            "## What changed",
            "- Update this section manually after each developer iteration.",
            "",
            "## What remains unexplained",
            "- Review residual_bytes and transient_bytes errors for the worst cases.",
        ]
    )
    return "\n".join(lines)


def render_suite_report(
    *,
    iteration_name: str,
    suite_result: BenchmarkSuiteResult,
    comparisons: list[ComparisonResult],
) -> str:
    """Render a suite-level benchmark report with artifact outcomes."""

    failed_cases = [
        case_result
        for case_result in suite_result.case_results
        if case_result.error_message is not None
    ]
    lines = [
        f"# {iteration_name}",
        "",
        "## Suite",
        f"- Output directory: {suite_result.output_dir}",
        f"- Total cases: {len(suite_result.case_results)}",
        f"- Cases with comparison artifacts: {len(comparisons)}",
        f"- Cases with failures: {len(failed_cases)}",
    ]
    if failed_cases:
        lines.extend(["", "## Failures"])
        for case_result in failed_cases:
            lines.append(f"- {case_result.case.name}: {case_result.error_message}")
    if comparisons:
        lines.extend(
            [
                "",
                render_comparison_report(
                    iteration_name="Comparison Summary",
                    comparisons=comparisons,
                ),
            ]
        )
    return "\n".join(lines)
