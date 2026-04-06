"""Canonicalize saved benchmark-measurement artifacts.

This module turns the historical `benchmark_artifacts` tree into a replayable
evaluation corpus with one canonical row per unique measured configuration plus
duplicate-history audit rows.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from .artifacts import (
    load_benchmark_suite_result,
    load_comparison_result,
    load_memory_result,
    save_comparison_result,
    save_memory_result,
    write_json_artifact,
    write_text_artifact,
)
from .compare import compare_measurement_to_estimate
from ..estimator.estimate import estimate_peak_memory
from ..types import EstimatorConfig

RUNTIME_BACKEND_MAP = {
    "eager": "standard",
    "flash_attention_2": "flash2",
    "flash2": "flash2",
    "sdpa": "sdpa",
}


@dataclass(frozen=True)
class CorpusMeasurementRow:
    """One measured benchmark row derived from a suite comparison artifact.

    Args:
        artifact_dir: Suite directory name.
        case_name: Stable benchmark case name.
        model_name: Model identifier.
        tuning_mode: Training mode such as `full_ft` or `lora`.
        distributed_mode: Runtime backend such as `single_gpu`, `ddp`, or `zero2`.
        optimizer_name: Optimizer identifier.
        attention_backend: Requested attention backend.
        gradient_checkpointing: Whether checkpointing was enabled.
        max_seq_len: Sequence length in tokens.
        micro_batch_size_per_gpu: Micro-batch size per rank.
        gpus_per_node: GPU count per node.
        num_nodes: Number of nodes.
        gpu_memory_gb: VRAM per GPU.
        measured_global_peak_gib: Measured peak memory per rank in GiB.
        estimated_global_peak_gib: Predicted peak memory per rank in GiB.
        absolute_error_gib: Absolute peak error in GiB.
        absolute_relative_error_pct: Absolute relative peak error in percent.
        measured_peak_phase: Measured peak phase name.
        estimated_peak_phase: Predicted peak phase name.
        comparison_path: Comparison artifact path.
        measurement_path: Measurement artifact path, if present.
        estimate_path: Estimate artifact path.
        config_json: Canonical JSON payload for the full training config.
        config_fingerprint: Stable hash of model plus full config.

    Returns:
        Frozen row suitable for CSV export.
    """

    artifact_dir: str
    case_name: str
    model_name: str
    tuning_mode: str
    distributed_mode: str
    optimizer_name: str
    attention_backend: str
    gradient_checkpointing: bool
    max_seq_len: int
    micro_batch_size_per_gpu: int
    gpus_per_node: int
    num_nodes: int
    gpu_memory_gb: float
    measured_global_peak_gib: float
    estimated_global_peak_gib: float
    absolute_error_gib: float
    absolute_relative_error_pct: float
    measured_peak_phase: str
    estimated_peak_phase: str
    comparison_path: str
    measurement_path: str
    estimate_path: str
    config_json: str
    config_fingerprint: str

    def to_csv_row(self) -> dict[str, Any]:
        """Return a flat CSV-ready row."""

        return asdict(self)


@dataclass(frozen=True)
class DuplicateStats:
    """Variance summary for repeated measurements of one config."""

    duplicate_count: int
    measured_mean_gib: float
    measured_variance_gib2: float
    measured_std_gib: float
    measured_range_gib: float
    measured_values_json: str

    def to_csv_row(self) -> dict[str, Any]:
        """Return a flat CSV-ready row."""

        return asdict(self)


@dataclass(frozen=True)
class CorpusCleaningResult:
    """Summary of one corpus-cleaning run."""

    root_dir: str
    output_dir: str
    raw_suite_dirs: int
    canonical_rows: int
    duplicate_rows: int
    exact_duplicate_groups: int
    nonzero_duplicate_groups: int
    ambiguous_backend_rows_dropped: int
    checkpointed_rows_dropped: int
    checkpointed_rows_included: int

    def to_markdown(self) -> str:
        """Return a concise markdown summary for the cleaned corpus."""

        lines = [
            "# Cleaned Measurement Corpus",
            "",
            f"- Root dir: `{self.root_dir}`",
            f"- Output dir: `{self.output_dir}`",
            f"- Raw suite dirs included: {self.raw_suite_dirs}",
            f"- Canonical rows: {self.canonical_rows}",
            f"- Duplicate historical rows: {self.duplicate_rows}",
            f"- Exact duplicate groups: {self.exact_duplicate_groups}",
            f"- Nonzero-variance duplicate groups: {self.nonzero_duplicate_groups}",
            f"- Ambiguous backend rows dropped: {self.ambiguous_backend_rows_dropped}",
            f"- Checkpointed rows dropped: {self.checkpointed_rows_dropped}",
            f"- Checkpointed rows included: {self.checkpointed_rows_included}",
        ]
        return "\n".join(lines) + "\n"


def _write_csv(*, path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write a sequence of dictionaries to CSV."""

    materialized_rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not materialized_rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized_rows[0].keys()))
        writer.writeheader()
        writer.writerows(materialized_rows)


def _remove_stale_cleaned_outputs(*, output_dir: Path) -> None:
    """Delete legacy cleaner artifacts that are no longer produced.

    Args:
        output_dir: Destination directory for cleaned corpus outputs.

    Returns:
        None. Obsolete files are removed in place when present.

    Example:
        >>> _remove_stale_cleaned_outputs(output_dir=Path("benchmark_artifacts/_cleaned"))
    """

    stale_filenames = (
        "report_only_artifacts.csv",
        "normalized_report_rows.csv",
        "normalized_oom_rows.csv",
    )
    for filename in stale_filenames:
        stale_path = output_dir / filename
        if stale_path.exists():
            stale_path.unlink()


def _suite_sort_key(*, artifact_dir: str) -> tuple[int, int, str]:
    """Return a deterministic sort key preferring later iteration suffixes."""

    if match := re.search(r"_iter(\d+)$", artifact_dir):
        return (2, int(match.group(1)), artifact_dir)
    if match := re.search(r"_v(\d+)$", artifact_dir):
        return (1, int(match.group(1)), artifact_dir)
    return (0, 0, artifact_dir)


def _is_excluded_suite_dir(*, artifact_dir: str) -> bool:
    """Return whether a suite directory should be excluded from canonical rows."""

    return artifact_dir.startswith("_") or "_analytic_iter" in artifact_dir


def _is_curated_checkpoint_suite_dir(*, suite_dir: Path) -> bool:
    """Return whether a suite dir belongs to the curated checkpoint batch.

    We only admit checkpointed rows from explicit checkpoint batches so that
    older buggy checkpoint measurements stay excluded from the canonical set.
    """

    curated_prefixes = (
        "single_gpu_80gb_ckpt_",
        "zero2_h100_ckpt_",
        "zero3_h100_ckpt_",
    )
    return any(
        part.startswith(curated_prefix)
        for part in suite_dir.parts
        for curated_prefix in curated_prefixes
    )


def _config_json_and_fingerprint(
    *,
    model_name: str,
    config: Any,
) -> tuple[str, str]:
    """Return canonical config JSON and a stable hash fingerprint."""

    config_json = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    fingerprint_payload = json.dumps(
        {"model_name": model_name, "config": json.loads(config_json)},
        sort_keys=True,
        separators=(",", ":"),
    )
    fingerprint = hashlib.sha1(fingerprint_payload.encode("utf-8")).hexdigest()
    return config_json, fingerprint


def _canonical_attention_backend(
    *, requested_backend: str, runtime_backend: object
) -> str | None:
    """Return the canonical backend label for one measured row.

    Args:
        requested_backend: Backend recorded in the saved training config.
        runtime_backend: Runtime attention implementation from measurement metadata.

    Returns:
        Canonical backend label used by the current estimator, or `None` when a
        legacy `standard` row lacks runtime metadata and cannot be replayed
        unambiguously.
    """

    normalized_requested = requested_backend.strip().lower()
    if normalized_requested in {"flash_attention_2", "flash2"}:
        return "flash2"
    if normalized_requested == "eager":
        return "standard"
    if normalized_requested == "sdpa":
        return "sdpa"
    if normalized_requested != "standard":
        return normalized_requested
    if not isinstance(runtime_backend, str):
        return None
    return RUNTIME_BACKEND_MAP.get(runtime_backend.strip().lower())


def _measurement_row_from_comparison(
    *,
    artifact_dir: str,
    case_name: str,
    comparison_path: Path,
    measurement_path: str,
    estimate_path: str,
    attention_backend: str,
) -> CorpusMeasurementRow:
    """Load one suite comparison into a normalized measurement row."""

    comparison = load_comparison_result(path=comparison_path)
    config = comparison.measured.config
    config_json, fingerprint = _config_json_and_fingerprint(
        model_name=comparison.model_name,
        config=config,
    )
    return CorpusMeasurementRow(
        artifact_dir=artifact_dir,
        case_name=case_name,
        model_name=comparison.model_name,
        tuning_mode=config.tuning_mode,
        distributed_mode=config.distributed_mode,
        optimizer_name=config.optimizer_name,
        attention_backend=attention_backend,
        gradient_checkpointing=config.gradient_checkpointing,
        max_seq_len=config.max_seq_len,
        micro_batch_size_per_gpu=config.micro_batch_size_per_gpu,
        gpus_per_node=config.gpus_per_node,
        num_nodes=config.num_nodes,
        gpu_memory_gb=config.gpu_memory_gb,
        measured_global_peak_gib=comparison.measured.global_peak_gb(),
        estimated_global_peak_gib=comparison.estimated.global_peak_gb(),
        absolute_error_gib=abs(comparison.global_peak_error_bytes) / (1024**3),
        absolute_relative_error_pct=abs(comparison.global_peak_relative_error) * 100.0,
        measured_peak_phase=comparison.measured.peak_phase,
        estimated_peak_phase=comparison.estimated.peak_phase,
        comparison_path=str(comparison_path),
        measurement_path=measurement_path,
        estimate_path=estimate_path,
        config_json=config_json,
        config_fingerprint=fingerprint,
    )


def _synthetic_estimate_and_comparison(
    *,
    measurement_path: Path,
) -> tuple[Path, Path]:
    """Return sibling estimate/comparison paths for one measurement artifact.

    Args:
        measurement_path: Standalone measurement artifact path.

    Returns:
        Tuple of `(estimate_path, comparison_path)` next to the measurement.
    """

    artifact_dir = measurement_path.parent
    return artifact_dir / "estimate.json", artifact_dir / "comparison.json"


def _measurement_row_from_standalone_measurement(
    *,
    measurement_path: Path,
    attention_backend: str,
) -> CorpusMeasurementRow:
    """Promote one standalone measurement into a synthetic corpus row.

    Args:
        measurement_path: Measured artifact without suite/comparison metadata.

    Returns:
        Normalized corpus row using a synthesized estimate/comparison pair.
    """

    measured = load_memory_result(path=measurement_path)
    estimator_config = measured.config
    if not isinstance(estimator_config, EstimatorConfig):
        estimator_config = estimator_config.to_estimator_config()
    estimate_path, comparison_path = _synthetic_estimate_and_comparison(
        measurement_path=measurement_path
    )
    estimated = estimate_peak_memory(
        model=measured.model_name,
        config=estimator_config,
    )
    comparison = compare_measurement_to_estimate(
        measured=measured,
        estimated=estimated,
    )
    save_memory_result(result=estimated, path=estimate_path)
    save_comparison_result(result=comparison, path=comparison_path)
    return _measurement_row_from_comparison(
        artifact_dir=measurement_path.parent.name,
        case_name=measurement_path.parent.name,
        comparison_path=comparison_path,
        measurement_path=str(measurement_path),
        estimate_path=str(estimate_path),
        attention_backend=attention_backend,
    )


def _iter_suite_rows(
    *, root_dir: Path
) -> tuple[list[CorpusMeasurementRow], int, int, int, set[Path]]:
    """Return normalized rows from raw measured benchmark suites.

    Checkpointed rows are excluded from the returned corpus and counted
    separately so the caller can report how many rows were dropped. Ambiguous
    legacy `standard` rows are also dropped here so the canonical corpus
    remains directly replayable.
    """

    rows: list[CorpusMeasurementRow] = []
    ambiguous_backend_rows_dropped = 0
    checkpointed_rows_dropped = 0
    checkpointed_rows_included = 0
    seen_measurement_paths: set[Path] = set()
    suite_index_paths = sorted(
        root_dir.rglob("suite_index.json"), key=lambda path: path.parent.as_posix()
    )
    for suite_index_path in suite_index_paths:
        suite_dir = suite_index_path.parent
        if _is_excluded_suite_dir(artifact_dir=suite_dir.name):
            continue
        suite_result = load_benchmark_suite_result(path=suite_index_path)
        for case_result in suite_result.case_results:
            if case_result.error_message is not None:
                continue
            if (
                case_result.measurement_path is None
                or case_result.comparison_path is None
            ):
                continue
            comparison_path = Path(case_result.comparison_path)
            measurement_path = Path(case_result.measurement_path)
            if not comparison_path.exists() or not measurement_path.exists():
                continue
            seen_measurement_paths.add(measurement_path.resolve())
            measured = load_memory_result(path=measurement_path)
            attention_backend = _canonical_attention_backend(
                requested_backend=measured.config.attention_backend,
                runtime_backend=measured.metadata.get("runtime_attention_implementation"),
            )
            if attention_backend is None:
                ambiguous_backend_rows_dropped += 1
                continue
            is_checkpointed = measured.config.gradient_checkpointing
            if is_checkpointed and not _is_curated_checkpoint_suite_dir(
                suite_dir=suite_dir
            ):
                checkpointed_rows_dropped += 1
                continue
            if is_checkpointed:
                checkpointed_rows_included += 1
            rows.append(
                _measurement_row_from_comparison(
                    artifact_dir=suite_dir.name,
                    case_name=case_result.case.name,
                    comparison_path=comparison_path,
                    measurement_path=case_result.measurement_path,
                    estimate_path=case_result.estimate_path,
                    attention_backend=attention_backend,
                )
            )
    return (
        rows,
        ambiguous_backend_rows_dropped,
        checkpointed_rows_dropped,
        checkpointed_rows_included,
        seen_measurement_paths,
    )


def _iter_standalone_curated_rows(
    *,
    root_dir: Path,
    seen_measurement_paths: set[Path],
) -> tuple[list[CorpusMeasurementRow], int, int]:
    """Return curated standalone checkpoint rows missing suite metadata.

    Args:
        root_dir: Root benchmark-artifact directory.
        seen_measurement_paths: Measurement artifacts already covered by suites.

    Returns:
        Tuple of synthetic standalone rows, included row count, and dropped
        ambiguous-backend row count.
    """

    rows: list[CorpusMeasurementRow] = []
    ambiguous_backend_rows_dropped = 0
    for measurement_path in sorted(root_dir.rglob("measurement.json")):
        resolved_path = measurement_path.resolve()
        if resolved_path in seen_measurement_paths:
            continue
        if not _is_curated_checkpoint_suite_dir(suite_dir=measurement_path.parent):
            continue
        measured = load_memory_result(path=measurement_path)
        if not measured.config.gradient_checkpointing:
            continue
        attention_backend = _canonical_attention_backend(
            requested_backend=measured.config.attention_backend,
            runtime_backend=measured.metadata.get("runtime_attention_implementation"),
        )
        if attention_backend is None:
            ambiguous_backend_rows_dropped += 1
            continue
        rows.append(
            _measurement_row_from_standalone_measurement(
                measurement_path=measurement_path,
                attention_backend=attention_backend,
            )
        )
    return rows, len(rows), ambiguous_backend_rows_dropped


def _duplicate_stats(*, rows: list[CorpusMeasurementRow]) -> DuplicateStats:
    """Return measured-peak variance statistics for one duplicate group."""

    measured_values = [row.measured_global_peak_gib for row in rows]
    variance = statistics.pvariance(measured_values) if len(rows) > 1 else 0.0
    measured_range = max(measured_values) - min(measured_values)
    return DuplicateStats(
        duplicate_count=len(rows),
        measured_mean_gib=sum(measured_values) / len(measured_values),
        measured_variance_gib2=variance,
        measured_std_gib=variance**0.5,
        measured_range_gib=measured_range,
        measured_values_json=json.dumps(sorted(measured_values)),
    )


def _canonical_and_duplicates(
    *,
    rows: list[CorpusMeasurementRow],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int]:
    """Split raw rows into canonical rows and duplicate historical rows."""

    grouped: dict[str, list[CorpusMeasurementRow]] = {}
    for row in rows:
        grouped.setdefault(row.config_fingerprint, []).append(row)
    canonical_rows: list[dict[str, Any]] = []
    duplicate_rows: list[dict[str, Any]] = []
    exact_groups = 0
    nonzero_groups = 0
    for group_rows in grouped.values():
        sorted_rows = sorted(
            group_rows,
            key=lambda row: _suite_sort_key(artifact_dir=row.artifact_dir),
        )
        canonical_row = sorted_rows[-1]
        stats = _duplicate_stats(rows=sorted_rows)
        if len(sorted_rows) > 1 and stats.measured_variance_gib2 == 0.0:
            exact_groups += 1
        if len(sorted_rows) > 1 and stats.measured_variance_gib2 > 0.0:
            nonzero_groups += 1
        canonical_rows.append(
            {
                **canonical_row.to_csv_row(),
                **stats.to_csv_row(),
                "canonical_artifact_dir": canonical_row.artifact_dir,
                "is_canonical": True,
            }
        )
        for duplicate_row in sorted_rows[:-1]:
            duplicate_rows.append(
                {
                    **duplicate_row.to_csv_row(),
                    **stats.to_csv_row(),
                    "canonical_artifact_dir": canonical_row.artifact_dir,
                    "is_canonical": False,
                }
            )
    return canonical_rows, duplicate_rows, exact_groups, nonzero_groups


def clean_measurement_corpus(
    *,
    root_dir: str | Path,
    output_dir: str | Path,
) -> CorpusCleaningResult:
    """Build a canonical measurement corpus from historical artifacts.

    Args:
        root_dir: Root `benchmark_artifacts` directory to audit.
        output_dir: Directory where cleaned CSV and summary artifacts are written.

    Returns:
        Summary dataclass describing the cleaned corpus outputs.

    Example:
        >>> result = clean_measurement_corpus(
        ...     root_dir='benchmark_artifacts',
        ...     output_dir='benchmark_artifacts/_cleaned_corpus',
        ... )
        >>> result.canonical_rows >= 0
        True
    """

    root_path = Path(root_dir)
    output_path = Path(output_dir)
    (
        measurement_rows,
        ambiguous_backend_rows_dropped,
        checkpointed_rows_dropped,
        checkpointed_rows_included,
        seen_measurement_paths,
    ) = (
        _iter_suite_rows(root_dir=root_path)
    )
    (
        standalone_rows,
        standalone_checkpoint_rows,
        standalone_ambiguous_rows_dropped,
    ) = _iter_standalone_curated_rows(
        root_dir=root_path,
        seen_measurement_paths=seen_measurement_paths,
    )
    measurement_rows.extend(standalone_rows)
    ambiguous_backend_rows_dropped += standalone_ambiguous_rows_dropped
    checkpointed_rows_included += standalone_checkpoint_rows
    canonical_rows, duplicate_rows, exact_groups, nonzero_groups = (
        _canonical_and_duplicates(rows=measurement_rows)
    )
    _remove_stale_cleaned_outputs(output_dir=output_path)
    _write_csv(path=output_path / "canonical_measurements.csv", rows=canonical_rows)
    _write_csv(
        path=output_path / "duplicate_historical_measurements.csv",
        rows=duplicate_rows,
    )
    result = CorpusCleaningResult(
        root_dir=str(root_path),
        output_dir=str(output_path),
        raw_suite_dirs=len({row.artifact_dir for row in measurement_rows}),
        canonical_rows=len(canonical_rows),
        duplicate_rows=len(duplicate_rows),
        exact_duplicate_groups=exact_groups,
        nonzero_duplicate_groups=nonzero_groups,
        ambiguous_backend_rows_dropped=ambiguous_backend_rows_dropped,
        checkpointed_rows_dropped=checkpointed_rows_dropped,
        checkpointed_rows_included=checkpointed_rows_included,
    )
    write_json_artifact(
        path=output_path / "cleaning_summary.json",
        payload=asdict(result),
    )
    write_text_artifact(
        path=output_path / "cleaning_summary.md",
        content=result.to_markdown(),
    )
    return result
