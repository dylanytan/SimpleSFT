"""Tests for corpus-cleaning utilities."""

from __future__ import annotations

from pathlib import Path

from simplesft.results.artifacts import (
    save_benchmark_suite_result,
    save_comparison_result,
    save_memory_result,
)
from simplesft.results.compare import compare_measurement_to_estimate
from simplesft.results.corpus_cleaning import clean_measurement_corpus
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
    model_name: str = "toy",
    metadata: dict[str, object] | None = None,
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
        model_name=model_name,
        config=config,
        breakdown=MemoryComponentBreakdown(parameter_bytes=peak_bytes),
        phase_records=(record,),
        peak_phase="forward",
        global_peak_bytes=peak_bytes,
        feasible=True,
        metadata={} if metadata is None else metadata,
    )


def _write_suite_case(
    *,
    root_dir: Path,
    artifact_dir: str,
    case_name: str,
    peak_gib: float,
    gradient_checkpointing: bool = False,
    attention_backend: str = "sdpa",
    runtime_attention_implementation: str | None = None,
) -> None:
    """Write one measured suite case into a temporary artifact tree."""

    config = TrainingConfig(
        tuning_mode="full_ft",
        distributed_mode="single_gpu",
        attention_backend=attention_backend,
        max_seq_len=128,
        gpu_memory_gb=40.0,
        gradient_checkpointing=gradient_checkpointing,
    )
    estimate = _memory_result(peak_gib=peak_gib + 1.0, config=config, mode="estimate")
    measurement_metadata: dict[str, object] = {}
    if runtime_attention_implementation is not None:
        measurement_metadata["runtime_attention_implementation"] = (
            runtime_attention_implementation
        )
    measure = _memory_result(
        peak_gib=peak_gib,
        config=config,
        mode="measure",
        metadata=measurement_metadata,
    )
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


def _write_standalone_measurement_case(
    *,
    root_dir: Path,
    artifact_dir: str,
    peak_gib: float,
    distributed_mode: str,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    attention_backend: str = "sdpa",
    runtime_attention_implementation: str = "sdpa",
) -> None:
    """Write one standalone measurement artifact without suite metadata.

    Args:
        root_dir: Temporary artifact root.
        artifact_dir: Directory relative to the root where measurement lives.
        peak_gib: Measured global peak in GiB.
        distributed_mode: Distributed backend recorded in the measurement.
        model_name: Resolvable model id used for synthetic estimate replay.
    """

    config = TrainingConfig(
        tuning_mode="full_ft",
        distributed_mode=distributed_mode,
        attention_backend=attention_backend,
        max_seq_len=128,
        gpu_memory_gb=40.0,
        gradient_checkpointing=True,
        gpus_per_node=2,
    )
    measurement = _memory_result(
        peak_gib=peak_gib,
        config=config,
        mode="measure",
        model_name=model_name,
        metadata={
            "runtime_attention_implementation": runtime_attention_implementation,
        },
    )
    measurement_path = root_dir / artifact_dir / "measurement.json"
    save_memory_result(result=measurement, path=measurement_path)


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


def test_clean_measurement_corpus_excludes_checkpointed_rows(tmp_path: Path) -> None:
    """Cleaner should drop checkpointed measurements from the corpus."""

    _write_suite_case(
        root_dir=tmp_path,
        artifact_dir="toy_suite_ckpt",
        case_name="case-ckpt",
        peak_gib=10.0,
        gradient_checkpointing=True,
    )
    _write_suite_case(
        root_dir=tmp_path,
        artifact_dir="toy_suite_nonckpt",
        case_name="case-nonckpt",
        peak_gib=10.0,
        gradient_checkpointing=False,
    )
    result = clean_measurement_corpus(
        root_dir=tmp_path,
        output_dir=tmp_path / "_cleaned",
    )
    canonical_csv = (tmp_path / "_cleaned" / "canonical_measurements.csv").read_text()
    assert result.canonical_rows == 1
    assert result.checkpointed_rows_dropped == 1
    assert result.checkpointed_rows_included == 0
    assert "toy_suite_nonckpt" in canonical_csv
    assert "toy_suite_ckpt" not in canonical_csv


def test_clean_measurement_corpus_includes_curated_checkpointed_rows(
    tmp_path: Path,
) -> None:
    """Cleaner should keep checkpointed rows from the curated checkpoint batch."""

    _write_suite_case(
        root_dir=tmp_path,
        artifact_dir="single_gpu_80gb_ckpt_20260406/toy_suite_ckpt",
        case_name="case-ckpt",
        peak_gib=10.0,
        gradient_checkpointing=True,
    )
    _write_suite_case(
        root_dir=tmp_path,
        artifact_dir="toy_suite_nonckpt",
        case_name="case-nonckpt",
        peak_gib=10.0,
        gradient_checkpointing=False,
    )
    result = clean_measurement_corpus(
        root_dir=tmp_path,
        output_dir=tmp_path / "_cleaned",
    )
    canonical_csv = (tmp_path / "_cleaned" / "canonical_measurements.csv").read_text()
    assert result.canonical_rows == 2
    assert result.checkpointed_rows_dropped == 0
    assert result.checkpointed_rows_included == 1
    assert "toy_suite_nonckpt" in canonical_csv
    assert "toy_suite_ckpt" in canonical_csv


def test_clean_measurement_corpus_includes_curated_distributed_checkpoint_rows(
    tmp_path: Path,
) -> None:
    """Cleaner should synthesize rows for curated standalone checkpoint batches."""

    _write_standalone_measurement_case(
        root_dir=tmp_path,
        artifact_dir="zero2_h100_ckpt_20260406/toy_suite_zero2_ckpt",
        peak_gib=10.0,
        distributed_mode="zero2",
    )
    _write_standalone_measurement_case(
        root_dir=tmp_path,
        artifact_dir="zero3_h100_ckpt_20260406/toy_suite_zero3_ckpt",
        peak_gib=11.0,
        distributed_mode="zero3",
    )
    result = clean_measurement_corpus(
        root_dir=tmp_path,
        output_dir=tmp_path / "_cleaned",
    )
    canonical_csv = (tmp_path / "_cleaned" / "canonical_measurements.csv").read_text()
    assert result.canonical_rows == 2
    assert result.checkpointed_rows_dropped == 0
    assert result.checkpointed_rows_included == 2
    assert "toy_suite_zero2_ckpt" in canonical_csv
    assert "toy_suite_zero3_ckpt" in canonical_csv
    assert (
        tmp_path / "zero2_h100_ckpt_20260406" / "toy_suite_zero2_ckpt" / "estimate.json"
    ).exists()
    assert (
        tmp_path
        / "zero2_h100_ckpt_20260406"
        / "toy_suite_zero2_ckpt"
        / "comparison.json"
    ).exists()


def test_clean_measurement_corpus_includes_curated_olmo_a100_checkpoint_rows(
    tmp_path: Path,
) -> None:
    """Cleaner should synthesize rows for the curated OLMo A100 checkpoint batch."""

    _write_standalone_measurement_case(
        root_dir=tmp_path,
        artifact_dir="olmo3_7b_a10080gb_ckpt_flash2_clean_20260406/lora_zero3_ckpt_flash2_seq32768_mb1",
        peak_gib=12.0,
        distributed_mode="zero3",
        model_name="allenai/Olmo-3-1025-7B",
        attention_backend="flash2",
        runtime_attention_implementation="flash_attention_2",
    )
    result = clean_measurement_corpus(
        root_dir=tmp_path,
        output_dir=tmp_path / "_cleaned",
    )
    canonical_csv = (tmp_path / "_cleaned" / "canonical_measurements.csv").read_text()
    assert result.canonical_rows == 1
    assert result.checkpointed_rows_dropped == 0
    assert result.checkpointed_rows_included == 1
    assert "lora_zero3_ckpt_flash2_seq32768_mb1" in canonical_csv
    assert ",flash2," in canonical_csv
    assert (
        tmp_path
        / "olmo3_7b_a10080gb_ckpt_flash2_clean_20260406"
        / "lora_zero3_ckpt_flash2_seq32768_mb1"
        / "estimate.json"
    ).exists()
    assert (
        tmp_path
        / "olmo3_7b_a10080gb_ckpt_flash2_clean_20260406"
        / "lora_zero3_ckpt_flash2_seq32768_mb1"
        / "comparison.json"
    ).exists()


def test_clean_measurement_corpus_normalizes_standard_backend_rows(
    tmp_path: Path,
) -> None:
    """Cleaner should replace legacy `standard` rows with resolved backends."""

    _write_suite_case(
        root_dir=tmp_path,
        artifact_dir="toy_suite_runtime_sdpa",
        case_name="case-a",
        peak_gib=10.0,
        attention_backend="standard",
        runtime_attention_implementation="sdpa",
    )
    result = clean_measurement_corpus(
        root_dir=tmp_path,
        output_dir=tmp_path / "_cleaned",
    )
    canonical_csv = (tmp_path / "_cleaned" / "canonical_measurements.csv").read_text()
    assert result.canonical_rows == 1
    assert result.ambiguous_backend_rows_dropped == 0
    assert ",sdpa," in canonical_csv
    assert ",standard," not in canonical_csv


def test_clean_measurement_corpus_drops_ambiguous_standard_rows(
    tmp_path: Path,
) -> None:
    """Cleaner should drop legacy `standard` rows without runtime metadata."""

    _write_suite_case(
        root_dir=tmp_path,
        artifact_dir="toy_suite_ambiguous_standard",
        case_name="case-a",
        peak_gib=10.0,
        attention_backend="standard",
    )
    result = clean_measurement_corpus(
        root_dir=tmp_path,
        output_dir=tmp_path / "_cleaned",
    )
    canonical_csv = (tmp_path / "_cleaned" / "canonical_measurements.csv").read_text()
    assert result.canonical_rows == 0
    assert result.ambiguous_backend_rows_dropped == 1
    assert canonical_csv == ""


def test_clean_measurement_corpus_removes_stale_legacy_outputs(
    tmp_path: Path,
) -> None:
    """Cleaner should delete old report-only outputs from prior builds."""

    cleaned_dir = tmp_path / "_cleaned"
    cleaned_dir.mkdir()
    for filename in (
        "report_only_artifacts.csv",
        "normalized_report_rows.csv",
        "normalized_oom_rows.csv",
    ):
        (cleaned_dir / filename).write_text("stale\n", encoding="utf-8")
    _write_suite_case(
        root_dir=tmp_path,
        artifact_dir="toy_suite_iter1",
        case_name="case-a",
        peak_gib=10.0,
    )
    clean_measurement_corpus(
        root_dir=tmp_path,
        output_dir=cleaned_dir,
    )
    for filename in (
        "report_only_artifacts.csv",
        "normalized_report_rows.csv",
        "normalized_oom_rows.csv",
    ):
        assert not (cleaned_dir / filename).exists()
