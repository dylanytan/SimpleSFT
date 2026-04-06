"""Tests for developer-facing reporting."""

from simplesft.results.benchmark import build_default_benchmark_cases, run_benchmark_suite
from simplesft.results.compare import compare_measurement_to_estimate
from simplesft.results.reporting import render_comparison_report, render_suite_report
from simplesft.types import (
    MemoryComponentBreakdown,
    MemoryResult,
    ModelSpec,
    PhaseMemoryRecord,
    TrainingConfig,
)


def test_render_comparison_report_mentions_iteration_name() -> None:
    """Reports should include the requested iteration title."""

    config = TrainingConfig(tuning_mode="full_ft")
    record = PhaseMemoryRecord(
        phase_name="forward",
        allocated_bytes=1,
        reserved_bytes=1,
        peak_allocated_bytes=1,
        peak_reserved_bytes=1,
        delta_allocated_bytes=1,
        delta_reserved_bytes=1,
    )
    result = MemoryResult(
        mode="estimate",
        model_name="toy",
        config=config,
        breakdown=MemoryComponentBreakdown(parameter_bytes=1),
        phase_records=(record,),
        peak_phase="forward",
        global_peak_bytes=1,
        feasible=True,
    )
    comparison = compare_measurement_to_estimate(measured=result, estimated=result)
    report = render_comparison_report(
        iteration_name="Iteration 1",
        comparisons=[comparison],
    )
    assert "# Iteration 1" in report


def test_render_suite_report_mentions_suite() -> None:
    """Suite reports should mention suite-level totals."""

    model_spec = ModelSpec(
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
    case = build_default_benchmark_cases(
        model=model_spec,
        seq_lens=(16,),
        micro_batches=(1,),
        tuning_modes=("full_ft",),
        distributed_modes=("single_gpu",),
    )[0]
    suite_result, _comparisons = run_benchmark_suite(
        cases=(case,),
        output_dir="tmp-report-suite",
        include_measurement=False,
    )
    report = render_suite_report(
        iteration_name="Iteration 2",
        suite_result=suite_result,
        comparisons=[],
    )
    assert "## Suite" in report


def test_build_default_benchmark_cases_include_checkpoint_tag() -> None:
    """Checkpointed benchmark cases should get distinct artifact names."""

    model_spec = ModelSpec(
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
    case = build_default_benchmark_cases(
        model=model_spec,
        seq_lens=(16,),
        micro_batches=(1,),
        tuning_modes=("full_ft",),
        distributed_modes=("single_gpu",),
        gradient_checkpointing=True,
    )[0]
    assert "-ckpt-" in case.name
