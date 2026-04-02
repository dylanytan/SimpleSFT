"""Tests for corpus-cleaning utilities."""

from __future__ import annotations

from pathlib import Path

from simplesft.artifacts import (
    save_benchmark_suite_result,
    save_comparison_result,
    save_memory_result,
)
from simplesft.compare import compare_measurement_to_estimate
from simplesft.corpus_cleaning import clean_measurement_corpus
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


def _toy_model() -> ModelSpec:
    """Return a compact model spec for corpus-cleaning tests."""

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


def _memory_result(
    *,
    peak_gib: float,
    config: TrainingConfig,
    mode: str,
) -> MemoryResult:
    """Return a compact memory result with one peak phase record."""

    peak_bytes = int(peak_gib * (1024**3))
    record = PhaseMemoryRecord(
        phase_name="forward",
        allocated_bytes=peak_bytes,
        reserved_bytes=peak_bytes,
        peak_allocated_bytes=peak_bytes,
        peak_reserved_bytes=peak_bytes,
        delta_allocated_bytes=peak_bytes,
        delta_reserved_bytes=peak_bytes,
    )
    return MemoryResult(
        mode=mode,
        model_name="toy",
        config=config,
        breakdown=MemoryComponentBreakdown(parameter_bytes=peak_bytes),
        phase_records=(record,),
        peak_phase="forward",
        global_peak_bytes=peak_bytes,
        feasible=True,
    )


def _write_suite_case(
    *,
    root_dir: Path,
    artifact_dir: str,
    case_name: str,
    peak_gib: float,
) -> None:
    """Write one measured suite case into a temporary artifact tree."""

    config = TrainingConfig(
        tuning_mode="full_ft",
        distributed_mode="single_gpu",
        max_seq_len=128,
        gpu_memory_gb=40.0,
    )
    estimate = _memory_result(peak_gib=peak_gib + 1.0, config=config, mode="estimate")
    measure = _memory_result(peak_gib=peak_gib, config=config, mode="measure")
    comparison = compare_measurement_to_estimate(
        measured=measure,
        estimated=estimate,
    )
    case_dir = root_dir / artifact_dir / case_name
    estimate_path = case_dir / "estimate.json"
    measure_path = case_dir / "measurement.json"
    comparison_path = case_dir / "comparison.json"
    save_memory_result(result=estimate, path=estimate_path)
    save_memory_result(result=measure, path=measure_path)
    save_comparison_result(result=comparison, path=comparison_path)
    suite_result = BenchmarkSuiteResult(
        output_dir=str(root_dir / artifact_dir),
        case_results=(
            BenchmarkCaseResult(
                case=BenchmarkCase(
                    name=case_name,
                    model=_toy_model(),
                    config=config,
                ),
                estimate_path=str(estimate_path),
                measurement_path=str(measure_path),
                comparison_path=str(comparison_path),
            ),
        ),
    )
    save_benchmark_suite_result(
        result=suite_result,
        path=root_dir / artifact_dir / "suite_index.json",
    )


def test_clean_measurement_corpus_dedupes_and_keeps_latest(tmp_path: Path) -> None:
    """Cleaner should keep one canonical row and preserve duplicate history."""

    _write_suite_case(
        root_dir=tmp_path,
        artifact_dir="toy_suite_iter1",
        case_name="case-a",
        peak_gib=10.0,
    )
    _write_suite_case(
        root_dir=tmp_path,
        artifact_dir="toy_suite_iter2",
        case_name="case-a",
        peak_gib=10.5,
    )
    result = clean_measurement_corpus(
        root_dir=tmp_path,
        output_dir=tmp_path / "_cleaned",
    )
    canonical_csv = (tmp_path / "_cleaned" / "canonical_measurements.csv").read_text()
    duplicate_csv = (
        tmp_path / "_cleaned" / "duplicate_historical_measurements.csv"
    ).read_text()
    assert result.canonical_rows == 1
    assert result.duplicate_rows == 1
    assert "toy_suite_iter2" in canonical_csv
    assert "toy_suite_iter1" in duplicate_csv
    assert "0.5" in canonical_csv


def test_clean_measurement_corpus_normalizes_report_only_oom_rows(
    tmp_path: Path,
) -> None:
    """Cleaner should normalize report-only markdown tables into CSV rows."""

    report_dir = tmp_path / "report_only_grid"
    report_dir.mkdir(parents=True)
    (report_dir / "report.md").write_text(
        "\n".join(
            [
                "# Report",
                "",
                "| Model | Tuning | Mode | Seq | Est GiB | Est Phase | Est Feasible | Status | Meas GiB | Meas Phase | Error GiB |",
                "| --- | --- | --- | ---: | ---: | --- | --- | --- | ---: | --- | ---: |",
                "| toy | full_ft | zero3 | 2048 | 12.0 | backward | True | measured | 11.0 | backward | 1.0 |",
                "| toy | full_ft | zero3 | 4096 | 24.0 | backward | False | oom |  |  |  |",
            ]
        ),
        encoding="utf-8",
    )
    result = clean_measurement_corpus(
        root_dir=tmp_path,
        output_dir=tmp_path / "_cleaned",
    )
    report_csv = (tmp_path / "_cleaned" / "normalized_report_rows.csv").read_text()
    oom_csv = (tmp_path / "_cleaned" / "normalized_oom_rows.csv").read_text()
    assert result.report_only_dirs == 1
    assert result.report_only_rows == 2
    assert result.report_only_oom_rows == 1
    assert "report_only_grid" in report_csv
    assert "oom" in oom_csv
