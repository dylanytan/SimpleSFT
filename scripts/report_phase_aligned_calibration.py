"""Replay the canonical corpus and report phase-aligned calibration residuals."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean, median
from typing import Callable

from simplesft.results.artifacts import load_memory_result
from simplesft.results.compare import compare_measurement_to_estimate
from simplesft.estimator.estimate import estimate_peak_memory
from simplesft.types import ComparisonResult, EstimatorConfig


@dataclass(frozen=True)
class ReplayRecord:
    """Compact replay record for one canonical benchmark row.

    Args:
        model_name: Model identifier.
        attention_backend: Canonical backend stored in the cleaned corpus.
        sequence_length: Sequence length used by the case.
        distributed_mode: Distributed strategy name.
        tuning_mode: Tuning mode name.
        measured_phase: Measured peak phase.
        estimated_phase: Estimated peak phase.
        comparison: Full comparison result for the row.

    Returns:
        One replayable comparison record.
    """

    model_name: str
    attention_backend: str
    sequence_length: int
    distributed_mode: str
    tuning_mode: str
    measured_phase: str
    estimated_phase: str
    comparison: ComparisonResult

    def signed_global_error_gib(self) -> float:
        """Return signed global peak error in GiB."""

        delta_bytes = (
            self.comparison.estimated.global_peak_bytes
            - self.comparison.measured.global_peak_bytes
        )
        return delta_bytes / 2**30

    def retained_proxy_error_gib(self) -> float:
        """Return retained-forward proxy error in GiB."""

        return self.comparison.retained_forward_proxy_error_bytes / 2**30


def _comparison_for_row(*, row: dict[str, str]) -> ReplayRecord:
    """Replay one canonical row through the current estimator."""

    measured = load_memory_result(path=Path(row["measurement_path"]))
    replay_config = measured.config
    if not isinstance(replay_config, EstimatorConfig):
        replay_config = replay_config.to_estimator_config()
    replay_config = replace(
        replay_config,
        attention_backend=row["attention_backend"],
    )
    estimated = estimate_peak_memory(model=row["model_name"], config=replay_config)
    comparison = compare_measurement_to_estimate(
        measured=measured,
        estimated=estimated,
    )
    return ReplayRecord(
        model_name=row["model_name"],
        attention_backend=row["attention_backend"],
        sequence_length=int(row["max_seq_len"]),
        distributed_mode=row["distributed_mode"],
        tuning_mode=row["tuning_mode"],
        measured_phase=measured.peak_phase,
        estimated_phase=estimated.peak_phase,
        comparison=comparison,
    )


def _format_gib(value: float) -> str:
    """Return one float formatted as GiB with three decimals."""

    return f"{value:.3f} GiB"


def _format_pct(value: float) -> str:
    """Return one float formatted as a percentage with one decimal."""

    return f"{value * 100:.1f}%"


def _format_pct_tex(value: float) -> str:
    """Return one percentage formatted for LaTeX text mode."""

    return _escape_tex(_format_pct(value))


def _mean_abs_gib(values: list[int]) -> float:
    """Return mean absolute bytes converted to GiB."""

    return mean(abs(value) for value in values) / 2**30 if values else 0.0


def _mean_abs_relative(values: list[float]) -> float:
    """Return mean absolute relative error."""

    return mean(values) if values else 0.0


def _phase_match_rate(*, records: list[ReplayRecord]) -> float:
    """Return the phase match rate across replay records."""

    if not records:
        return 0.0
    matches = sum(record.measured_phase == record.estimated_phase for record in records)
    return matches / len(records)


def _component_mae_gib(
    *, records: list[ReplayRecord], key: str, phase_aligned: bool
) -> float:
    """Return component MAE in GiB for raw or phase-aligned comparisons."""

    if phase_aligned:
        values = [
            record.comparison.phase_aligned_component_error_bytes.get(key, 0)
            for record in records
        ]
    else:
        values = [
            record.comparison.component_error_bytes.get(key, 0) for record in records
        ]
    return _mean_abs_gib(values=values)


def _grouped_summary(
    *,
    records: list[ReplayRecord],
    label: str,
    key_fn: Callable[[ReplayRecord], object],
) -> list[str]:
    """Return one markdown table for a grouped residual summary."""

    groups: dict[str, list[ReplayRecord]] = {}
    for record in records:
        groups.setdefault(str(key_fn(record)), []).append(record)
    lines = [
        f"### By {label}",
        "",
        "| Group | Count | Global MAE | Proxy MAE | Phase Match |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key in sorted(groups):
        group_records = groups[key]
        global_errors = [
            record.comparison.global_peak_error_bytes for record in group_records
        ]
        proxy_errors = [record.retained_proxy_error_gib() for record in group_records]
        lines.append(
            "| "
            + " | ".join(
                [
                    key,
                    str(len(group_records)),
                    _format_gib(_mean_abs_gib(values=global_errors)),
                    _format_gib(mean(proxy_errors) if proxy_errors else 0.0),
                    _format_pct(_phase_match_rate(records=group_records)),
                ]
            )
            + " |"
        )
    lines.extend(["", ""])
    return lines


def _targeted_slice_records(
    *,
    records: list[ReplayRecord],
    label: str,
) -> list[ReplayRecord]:
    """Return one targeted replay slice used by acceptance checks.

    Args:
        records: All replay records.
        label: Stable targeted-slice label.

    Returns:
        Records belonging to the requested acceptance slice.
    """

    if label == "non_eager_long_seq_full_ft":
        return [
            record
            for record in records
            if record.tuning_mode == "full_ft"
            and record.sequence_length >= 4096
            and record.attention_backend != "standard"
        ]
    if label == "non_eager_long_seq_lora":
        return [
            record
            for record in records
            if record.tuning_mode == "lora"
            and record.sequence_length >= 4096
            and record.attention_backend != "standard"
        ]
    if label == "zero2":
        return [
            record for record in records if record.distributed_mode == "zero2"
        ]
    assert label == "zero3", f"Unknown slice: {label}"
    return [
        record for record in records if record.distributed_mode == "zero3"
    ]


def _targeted_slice_lines(*, records: list[ReplayRecord]) -> list[str]:
    """Return targeted acceptance-slice summary lines.

    Args:
        records: Replay records from the canonical cleaned corpus.

    Returns:
        Markdown lines summarizing the default acceptance slices.
    """

    labels = (
        ("non_eager_long_seq_full_ft", "Non-Eager Long-Seq Full FT"),
        ("non_eager_long_seq_lora", "Non-Eager Long-Seq LoRA"),
        ("zero2", "ZeRO-2"),
        ("zero3", "ZeRO-3"),
    )
    lines = [
        "## Targeted Slices",
        "",
        "| Slice | Count | Global MAE | Proxy MAE | Phase Match |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label, title in labels:
        slice_records = _targeted_slice_records(records=records, label=label)
        global_errors = [
            record.comparison.global_peak_error_bytes for record in slice_records
        ]
        proxy_errors = [record.retained_proxy_error_gib() for record in slice_records]
        lines.append(
            "| "
            + " | ".join(
                [
                    title,
                    str(len(slice_records)),
                    _format_gib(_mean_abs_gib(values=global_errors)),
                    _format_gib(mean(proxy_errors) if proxy_errors else 0.0),
                    _format_pct(_phase_match_rate(records=slice_records)),
                ]
            )
            + " |"
        )
    lines.extend(["", ""])
    return lines


def _component_table_lines(*, records: list[ReplayRecord]) -> list[str]:
    """Return one markdown table for raw and phase-aligned component MAE."""

    lines = [
        "## Component MAE",
        "",
        "| Metric | Raw | Phase-Aligned |",
        "| --- | ---: | ---: |",
    ]
    for key in (
        "parameter_bytes",
        "gradient_bytes",
        "optimizer_state_bytes",
        "activation_bytes",
        "transient_bytes",
        "runtime_reserve_bytes",
    ):
        phase_aligned_value = (
            "`n/a`"
            if key == "activation_bytes"
            else _format_gib(
                _component_mae_gib(
                    records=records,
                    key=key,
                    phase_aligned=True,
                )
            )
        )
        raw_value = _format_gib(
            _component_mae_gib(records=records, key=key, phase_aligned=False)
        )
        lines.append(f"| {key} | {raw_value} | {phase_aligned_value} |")
    lines.extend(["", ""])
    return lines


def _overview_lines(
    *,
    records: list[ReplayRecord],
    failures: list[str],
) -> list[str]:
    """Return overview markdown lines for the replay report."""

    global_errors = [record.comparison.global_peak_error_bytes for record in records]
    relative_errors = [
        record.comparison.global_peak_relative_error for record in records
    ]
    proxy_errors = [record.retained_proxy_error_gib() for record in records]
    return [
        "# Phase-Aligned Calibration Replay",
        "",
        f"- Replayed rows: `{len(records)}`",
        f"- Failed rows: `{len(failures)}`",
        f"- Global MAE: `{_format_gib(_mean_abs_gib(values=global_errors))}`",
        f"- Mean absolute relative error: `{_format_pct(_mean_abs_relative(relative_errors))}`",
        f"- Median absolute relative error: `{_format_pct(median(relative_errors) if relative_errors else 0.0)}`",
        f"- Phase match rate: `{_format_pct(_phase_match_rate(records=records))}`",
        f"- Retained-forward proxy MAE: `{_format_gib(mean(proxy_errors) if proxy_errors else 0.0)}`",
        "",
    ]


def _phase_confusion_lines(*, records: list[ReplayRecord]) -> list[str]:
    """Return one markdown table for the phase confusion matrix."""

    confusion_rows = _phase_confusion_rows(records=records)
    lines = [
        "## Phase Confusion",
        "",
        "| Measured | Estimated | Count |",
        "| --- | --- | ---: |",
    ]
    for measured_phase, estimated_phase, count in confusion_rows:
        lines.append(f"| {measured_phase} | {estimated_phase} | {count} |")
    lines.extend(["", ""])
    return lines


def _phase_confusion_rows(*, records: list[ReplayRecord]) -> list[tuple[str, str, int]]:
    """Return sorted phase-confusion rows for markdown and TeX output."""

    confusion_counts: dict[tuple[str, str], int] = {}
    for record in records:
        key = (record.measured_phase, record.estimated_phase)
        confusion_counts[key] = confusion_counts.get(key, 0) + 1
    return [
        (measured_phase, estimated_phase, confusion_counts[(measured_phase, estimated_phase)])
        for measured_phase, estimated_phase in sorted(confusion_counts)
    ]


def _escape_tex(text: str) -> str:
    """Return one string escaped for LaTeX text mode."""

    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    escaped_text = text
    for old, new in replacements.items():
        escaped_text = escaped_text.replace(old, new)
    return escaped_text


def build_accuracy_tex(*, records: list[ReplayRecord], failures: list[str]) -> str:
    """Return a TeX include with replay-derived accuracy metrics."""

    global_errors = [record.comparison.global_peak_error_bytes for record in records]
    relative_errors = [
        record.comparison.global_peak_relative_error for record in records
    ]
    proxy_errors = [record.retained_proxy_error_gib() for record in records]
    lines = [
        r"\section{Quantitative Accuracy}",
        (
            "The current canonical replay covers "
            f"{len(records)} rows and skips {len(failures)} rows that still fail "
            "for artifact or support reasons."
        ),
        "",
        r"\begin{align*}",
        r"\text{Global MAE} &= " + _format_gib(_mean_abs_gib(values=global_errors)).replace(" GiB", r"\ \mathrm{GiB}") + r" \\",
        r"\text{Mean Absolute Relative Error} &= " + _format_pct_tex(_mean_abs_relative(relative_errors)) + r" \\",
        r"\text{Median Absolute Relative Error} &= " + _format_pct_tex(median(relative_errors) if relative_errors else 0.0) + r" \\",
        r"\text{Phase Match Rate} &= " + _format_pct_tex(_phase_match_rate(records=records)) + r" \\",
        r"\text{Retained-Forward Proxy MAE} &= " + _format_gib(mean(proxy_errors) if proxy_errors else 0.0).replace(" GiB", r"\ \mathrm{GiB}"),
        r"\end{align*}",
        "",
        r"\subsection{Phase Confusion}",
        r"\begin{center}",
        r"\begin{tabular}{lll}",
        r"\toprule",
        r"Measured & Estimated & Count \\",
        r"\midrule",
    ]
    for measured_phase, estimated_phase, count in _phase_confusion_rows(records=records):
        lines.append(
            f"{_escape_tex(measured_phase)} & {_escape_tex(estimated_phase)} & {count} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
            "",
            r"\subsection{Targeted Slices}",
            r"\begin{center}",
            r"\begin{tabular}{p{2.9in}rrr}",
            r"\toprule",
            r"Slice & Count & Global MAE & Phase Match \\",
            r"\midrule",
        ]
    )
    for label, title in (
        ("non_eager_long_seq_full_ft", "Non-Eager Long-Seq Full FT"),
        ("non_eager_long_seq_lora", "Non-Eager Long-Seq LoRA"),
        ("zero2", "ZeRO-2"),
        ("zero3", "ZeRO-3"),
    ):
        slice_records = _targeted_slice_records(records=records, label=label)
        slice_errors = [
            record.comparison.global_peak_error_bytes for record in slice_records
        ]
        lines.append(
            " & ".join(
                [
                    _escape_tex(title),
                    str(len(slice_records)),
                    _escape_tex(_format_gib(_mean_abs_gib(values=slice_errors))),
                    _format_pct_tex(_phase_match_rate(records=slice_records)),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{center}",
            "",
        ]
    )
    return "\n".join(lines)


def _top_rows(*, records: list[ReplayRecord], reverse: bool) -> list[str]:
    """Return one markdown table for over- or under-estimates."""

    ordered = sorted(records, key=lambda record: record.signed_global_error_gib())
    if reverse:
        ordered = list(reversed(ordered))
    title = "Top 20 Overestimates" if reverse else "Top 20 Underestimates"
    lines = [
        f"## {title}",
        "",
        "| Model | Backend | Seq | Dist | Tune | Measured | Estimated | Signed Error |",
        "| --- | --- | ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for record in ordered[:20]:
        lines.append(
            "| "
            + " | ".join(
                [
                    record.model_name,
                    record.attention_backend,
                    str(record.sequence_length),
                    record.distributed_mode,
                    record.tuning_mode,
                    _format_gib(record.comparison.measured.global_peak_bytes / 2**30),
                    _format_gib(record.comparison.estimated.global_peak_bytes / 2**30),
                    _format_gib(record.signed_global_error_gib()),
                ]
            )
            + " |"
        )
    lines.extend(["", ""])
    return lines


def _phase_mismatch_rows(*, records: list[ReplayRecord]) -> list[str]:
    """Return one markdown table for the largest phase mismatches."""

    mismatches = [
        record for record in records if record.measured_phase != record.estimated_phase
    ]
    mismatches.sort(
        key=lambda record: record.comparison.global_peak_error_bytes,
        reverse=True,
    )
    lines = [
        "## Top 20 Phase Mismatches",
        "",
        "| Model | Backend | Seq | Dist | Tune | Measured Phase | Estimated Phase | Global Error | Proxy Error |",
        "| --- | --- | ---: | --- | --- | --- | --- | ---: | ---: |",
    ]
    for record in mismatches[:20]:
        lines.append(
            "| "
            + " | ".join(
                [
                    record.model_name,
                    record.attention_backend,
                    str(record.sequence_length),
                    record.distributed_mode,
                    record.tuning_mode,
                    record.measured_phase,
                    record.estimated_phase,
                    _format_gib(record.comparison.global_peak_error_bytes / 2**30),
                    _format_gib(record.retained_proxy_error_gib()),
                ]
            )
            + " |"
        )
    lines.extend(["", ""])
    return lines


def build_report(
    *,
    records: list[ReplayRecord],
    failures: list[str],
) -> str:
    """Return the phase-aligned calibration report as markdown."""

    lines = _overview_lines(
        records=records,
        failures=failures,
    )
    lines.extend(_component_table_lines(records=records))
    lines.extend(_phase_confusion_lines(records=records))
    lines.extend(_targeted_slice_lines(records=records))
    lines.extend(
        _grouped_summary(
            records=records,
            label="measured phase",
            key_fn=lambda record: record.measured_phase,
        )
    )
    lines.extend(
        _grouped_summary(
            records=records,
            label="attention backend",
            key_fn=lambda record: record.attention_backend,
        )
    )
    lines.extend(
        _grouped_summary(
            records=records,
            label="sequence length",
            key_fn=lambda record: record.sequence_length,
        )
    )
    lines.extend(
        _grouped_summary(
            records=records,
            label="distributed mode",
            key_fn=lambda record: record.distributed_mode,
        )
    )
    lines.extend(
        _grouped_summary(
            records=records,
            label="tuning mode",
            key_fn=lambda record: record.tuning_mode,
        )
    )
    lines.extend(_top_rows(records=records, reverse=True))
    lines.extend(_top_rows(records=records, reverse=False))
    lines.extend(_phase_mismatch_rows(records=records))
    if failures:
        lines.extend(["## Failures", ""])
        lines.extend(f"- `{failure}`" for failure in failures)
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """Return parsed CLI args for the phase-aligned report."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canonical-csv",
        type=Path,
        default=Path(
            "/users/PAA0201/ollieproudman/work/SimpleSFT/benchmark_artifacts/_cleaned_corpus/canonical_measurements.csv"
        ),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tex-output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """Replay the canonical corpus and emit a markdown report."""

    args = parse_args()
    rows = list(csv.DictReader(args.canonical_csv.open()))
    records: list[ReplayRecord] = []
    failures: list[str] = []
    for row in rows:
        try:
            record = _comparison_for_row(row=row)
            records.append(record)
        except Exception as error:
            failures.append(f"{row['model_name']}::{row['case_name']}::{error}")
    report = build_report(
        records=records,
        failures=failures,
    )
    if args.tex_output is not None:
        args.tex_output.parent.mkdir(parents=True, exist_ok=True)
        args.tex_output.write_text(
            build_accuracy_tex(records=records, failures=failures),
            encoding="utf-8",
        )
    if args.output is None:
        print(report)
        return
    args.output.write_text(report, encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
