"""Canonicalize and audit saved benchmark-measurement artifacts.

This module turns the historical `benchmark_artifacts` tree into a cleaner
evaluation corpus with:

- one canonical row per unique measured configuration,
- explicit duplicate historical rows,
- audit rows for report-only artifacts,
- normalized OOM rows extracted from report-only markdown tables.
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
    write_json_artifact,
    write_text_artifact,
)


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
class ReportArtifactAuditRow:
    """Audit information for a report-only artifact directory."""

    artifact_dir: str
    report_path: str
    parsed_row_count: int
    oom_row_count: int
    notes: str

    def to_csv_row(self) -> dict[str, Any]:
        """Return a flat CSV-ready row."""

        return asdict(self)


@dataclass(frozen=True)
class NormalizedReportRow:
    """Normalized row extracted from a report-only markdown table."""

    artifact_dir: str
    report_path: str
    model: str
    tuning: str
    mode: str
    seq: str
    est_gib: str
    est_phase: str
    est_feasible: str
    status: str
    meas_gib: str
    meas_phase: str
    error_gib: str
    needs_promotion: bool

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
    report_only_dirs: int
    report_only_rows: int
    report_only_oom_rows: int

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
            f"- Report-only dirs needing promotion: {self.report_only_dirs}",
            f"- Normalized report-only rows: {self.report_only_rows}",
            f"- Normalized report-only OOM rows: {self.report_only_oom_rows}",
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


def _measurement_row_from_comparison(
    *,
    artifact_dir: str,
    case_name: str,
    comparison_path: Path,
    measurement_path: str,
    estimate_path: str,
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
        attention_backend=config.attention_backend,
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


def _iter_suite_rows(*, root_dir: Path) -> list[CorpusMeasurementRow]:
    """Return normalized rows from raw measured benchmark suites."""

    rows: list[CorpusMeasurementRow] = []
    for suite_dir in sorted(root_dir.iterdir()):
        if not suite_dir.is_dir() or _is_excluded_suite_dir(
            artifact_dir=suite_dir.name
        ):
            continue
        suite_index_path = suite_dir / "suite_index.json"
        if not suite_index_path.exists():
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
            rows.append(
                _measurement_row_from_comparison(
                    artifact_dir=suite_dir.name,
                    case_name=case_result.case.name,
                    comparison_path=comparison_path,
                    measurement_path=case_result.measurement_path,
                    estimate_path=case_result.estimate_path,
                )
            )
    return rows


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


def _parse_markdown_table(*, text: str) -> list[dict[str, str]]:
    """Parse the first markdown table found in a report."""

    lines = text.splitlines()
    for index in range(len(lines) - 1):
        if not lines[index].startswith("|"):
            continue
        if not re.match(r"^\|\s*[-:| ]+\|?\s*$", lines[index + 1]):
            continue
        headers = [cell.strip() for cell in lines[index].strip("|").split("|")]
        rows: list[dict[str, str]] = []
        for row_line in lines[index + 2 :]:
            if not row_line.startswith("|"):
                break
            cells = [cell.strip() for cell in row_line.strip("|").split("|")]
            if len(cells) != len(headers):
                continue
            rows.append(dict(zip(headers, cells)))
        if rows:
            return rows
    return []


def _normalized_report_rows(
    *, artifact_dir: str, report_path: Path
) -> list[NormalizedReportRow]:
    """Return normalized table rows from one report-only artifact."""

    table_rows = _parse_markdown_table(text=report_path.read_text(encoding="utf-8"))
    normalized_rows: list[NormalizedReportRow] = []
    for raw_row in table_rows:
        normalized_rows.append(
            NormalizedReportRow(
                artifact_dir=artifact_dir,
                report_path=str(report_path),
                model=raw_row.get("Model", ""),
                tuning=raw_row.get("Tuning", ""),
                mode=raw_row.get("Mode", ""),
                seq=raw_row.get("Seq", ""),
                est_gib=raw_row.get("Est GiB", ""),
                est_phase=raw_row.get("Est Phase", ""),
                est_feasible=raw_row.get("Est Feasible", ""),
                status=raw_row.get("Status", ""),
                meas_gib=raw_row.get("Meas GiB", ""),
                meas_phase=raw_row.get("Meas Phase", ""),
                error_gib=raw_row.get("Error GiB", ""),
                needs_promotion=True,
            )
        )
    return normalized_rows


def _report_only_audit_rows(
    *, root_dir: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return audit rows, normalized report rows, and normalized OOM rows."""

    audit_rows: list[dict[str, Any]] = []
    normalized_rows: list[dict[str, Any]] = []
    oom_rows: list[dict[str, Any]] = []
    for artifact_dir in sorted(root_dir.iterdir()):
        if not artifact_dir.is_dir():
            continue
        suite_index_path = artifact_dir / "suite_index.json"
        report_path = artifact_dir / "report.md"
        if suite_index_path.exists() or not report_path.exists():
            continue
        parsed_rows = _normalized_report_rows(
            artifact_dir=artifact_dir.name,
            report_path=report_path,
        )
        oom_count = sum(
            1 for row in parsed_rows if row.status.lower() in {"oom", "runtime oom"}
        )
        audit_rows.append(
            ReportArtifactAuditRow(
                artifact_dir=artifact_dir.name,
                report_path=str(report_path),
                parsed_row_count=len(parsed_rows),
                oom_row_count=oom_count,
                notes="report-only artifact needs suite promotion",
            ).to_csv_row()
        )
        normalized_rows.extend(row.to_csv_row() for row in parsed_rows)
        oom_rows.extend(
            row.to_csv_row()
            for row in parsed_rows
            if row.status.lower() in {"oom", "runtime oom"}
        )
    return audit_rows, normalized_rows, oom_rows


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
    measurement_rows = _iter_suite_rows(root_dir=root_path)
    canonical_rows, duplicate_rows, exact_groups, nonzero_groups = (
        _canonical_and_duplicates(rows=measurement_rows)
    )
    audit_rows, normalized_report_rows, oom_rows = _report_only_audit_rows(
        root_dir=root_path
    )
    _write_csv(path=output_path / "canonical_measurements.csv", rows=canonical_rows)
    _write_csv(
        path=output_path / "duplicate_historical_measurements.csv",
        rows=duplicate_rows,
    )
    _write_csv(path=output_path / "report_only_artifacts.csv", rows=audit_rows)
    _write_csv(
        path=output_path / "normalized_report_rows.csv",
        rows=normalized_report_rows,
    )
    _write_csv(path=output_path / "normalized_oom_rows.csv", rows=oom_rows)
    result = CorpusCleaningResult(
        root_dir=str(root_path),
        output_dir=str(output_path),
        raw_suite_dirs=len({row.artifact_dir for row in measurement_rows}),
        canonical_rows=len(canonical_rows),
        duplicate_rows=len(duplicate_rows),
        exact_duplicate_groups=exact_groups,
        nonzero_duplicate_groups=nonzero_groups,
        report_only_dirs=len(audit_rows),
        report_only_rows=len(normalized_report_rows),
        report_only_oom_rows=len(oom_rows),
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
