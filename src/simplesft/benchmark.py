"""Benchmark-suite helpers for developer-led calibration iterations."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .artifacts import (
    save_benchmark_suite_result,
    save_comparison_result,
    save_memory_result,
)
from .compare import compare_measurement_to_estimate
from .estimate import estimate_peak_memory
from .measure import measure_peak_memory
from .types import (
    BenchmarkCase,
    BenchmarkCaseResult,
    BenchmarkSuiteResult,
    ComparisonResult,
    LoRAConfig,
    ModelSpec,
    TrainingConfig,
)


def build_default_benchmark_cases(
    *,
    model: str | ModelSpec,
    seq_lens: Iterable[int],
    micro_batches: Iterable[int],
    tuning_modes: Iterable[str],
    distributed_modes: Iterable[str],
    attention_backends: Iterable[str] = ("standard",),
    gpu_memory_gb: float = 24.0,
    gpus_per_node: int = 1,
    lora_rank: int = 16,
    gradient_checkpointing: bool = False,
) -> tuple[BenchmarkCase, ...]:
    """Build a default benchmark corpus from cartesian-product settings."""

    cases: list[BenchmarkCase] = []
    for tuning_mode in tuning_modes:
        for distributed_mode in distributed_modes:
            for attention_backend in attention_backends:
                for seq_len in seq_lens:
                    for micro_batch_size in micro_batches:
                        lora = None
                        if tuning_mode == "lora":
                            lora = LoRAConfig(rank=lora_rank)
                        config = TrainingConfig(
                            tuning_mode=tuning_mode,
                            micro_batch_size_per_gpu=micro_batch_size,
                            max_seq_len=seq_len,
                            gradient_checkpointing=gradient_checkpointing,
                            distributed_mode=distributed_mode,
                            attention_backend=attention_backend,
                            gpu_memory_gb=gpu_memory_gb,
                            gpus_per_node=gpus_per_node,
                            lora=lora,
                        )
                        checkpoint_tag = "-ckpt" if gradient_checkpointing else ""
                        case_name = (
                            f"{tuning_mode}-{distributed_mode}-{attention_backend}"
                            f"{checkpoint_tag}-seq{seq_len}-mb{micro_batch_size}"
                        )
                        cases.append(
                            BenchmarkCase(
                                name=case_name,
                                model=model,
                                config=config,
                                tags=(tuning_mode, distributed_mode, attention_backend),
                            )
                        )
    return tuple(cases)


def _case_dir(*, output_dir: str | Path, case: BenchmarkCase) -> Path:
    """Return the artifact directory for one benchmark case."""

    return Path(output_dir) / case.artifact_slug()


def _run_case_estimate(*, case: BenchmarkCase):
    """Run the estimate path for one benchmark case."""

    return estimate_peak_memory(model=case.model, config=case.config)


def _run_case_measure(*, case: BenchmarkCase):
    """Run the measurement path for one benchmark case."""

    return measure_peak_memory(model=case.model, config=case.config)


def _save_case_artifacts(
    *,
    case: BenchmarkCase,
    output_dir: str | Path,
    include_measurement: bool,
    allow_measurement_failures: bool,
) -> tuple[BenchmarkCaseResult, ComparisonResult | None]:
    """Run one case and persist its artifacts."""

    case_output_dir = _case_dir(output_dir=output_dir, case=case)
    estimate = _run_case_estimate(case=case)
    estimate_path = case_output_dir / "estimate.json"
    save_memory_result(result=estimate, path=estimate_path)
    try:
        if not include_measurement:
            return BenchmarkCaseResult(case=case, estimate_path=str(estimate_path)), None
        measurement = _run_case_measure(case=case)
        measurement_path = case_output_dir / "measurement.json"
        save_memory_result(result=measurement, path=measurement_path)
        comparison = compare_measurement_to_estimate(
            measured=measurement,
            estimated=estimate,
        )
        comparison_path = case_output_dir / "comparison.json"
        save_comparison_result(result=comparison, path=comparison_path)
        return (
            BenchmarkCaseResult(
                case=case,
                estimate_path=str(estimate_path),
                measurement_path=str(measurement_path),
                comparison_path=str(comparison_path),
            ),
            comparison,
        )
    except Exception as exc:
        if not allow_measurement_failures:
            raise
        return (
            BenchmarkCaseResult(
                case=case,
                estimate_path=str(estimate_path),
                error_message=str(exc),
            ),
            None,
        )


def run_benchmark_suite(
    *,
    cases: Iterable[BenchmarkCase],
    output_dir: str | Path,
    include_measurement: bool = False,
    allow_measurement_failures: bool = True,
) -> tuple[BenchmarkSuiteResult, list[ComparisonResult]]:
    """Run a benchmark suite and persist suite-level artifacts."""

    case_results: list[BenchmarkCaseResult] = []
    comparisons: list[ComparisonResult] = []
    for case in cases:
        case_result, comparison = _save_case_artifacts(
            case=case,
            output_dir=output_dir,
            include_measurement=include_measurement,
            allow_measurement_failures=allow_measurement_failures,
        )
        case_results.append(case_result)
        if comparison is not None:
            comparisons.append(comparison)
    suite_result = BenchmarkSuiteResult(
        output_dir=str(output_dir),
        case_results=tuple(case_results),
        notes=("Measurement failures are allowed by default during estimate-first iteration.",),
    )
    save_benchmark_suite_result(
        result=suite_result,
        path=Path(output_dir) / "suite_index.json",
    )
    return suite_result, comparisons
