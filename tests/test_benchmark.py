"""Tests for benchmark-suite utilities."""

from pathlib import Path

from simplesft.results.benchmark import build_default_benchmark_cases, run_benchmark_suite
from simplesft.types import ModelSpec


def _toy_model_spec() -> ModelSpec:
    """Return a compact model spec for benchmark tests."""

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


def test_build_default_benchmark_cases_creates_cartesian_product() -> None:
    """Default benchmark corpus should enumerate the requested case grid."""

    cases = build_default_benchmark_cases(
        model=_toy_model_spec(),
        seq_lens=(16, 32),
        micro_batches=(1, 2),
        tuning_modes=("full_ft",),
        distributed_modes=("single_gpu", "ddp"),
    )
    assert len(cases) == 8


def test_run_benchmark_suite_saves_suite_index(tmp_path: Path) -> None:
    """Benchmark suites should persist a suite index artifact."""

    case = build_default_benchmark_cases(
        model=_toy_model_spec(),
        seq_lens=(16,),
        micro_batches=(1,),
        tuning_modes=("full_ft",),
        distributed_modes=("single_gpu",),
    )[0]
    suite_result, comparisons = run_benchmark_suite(
        cases=(case,),
        output_dir=tmp_path,
        include_measurement=False,
    )
    assert len(suite_result.case_results) == 1
    assert comparisons == []
    assert (tmp_path / "suite_index.json").exists()


def test_build_default_benchmark_cases_applies_optimizer_specific_overrides() -> None:
    """Benchmark cases should accept per-optimizer config overrides."""

    cases = build_default_benchmark_cases(
        model=_toy_model_spec(),
        seq_lens=(16,),
        micro_batches=(1,),
        tuning_modes=("full_ft",),
        distributed_modes=("single_gpu",),
        optimizer_names=("sgd", "adamw"),
        optimizer_overrides_by_name={
            "sgd": {"optimizer_momentum": 0.9},
            "adamw": {"optimizer_beta1": 0.8},
        },
    )
    case_by_optimizer = {case.config.optimizer_name: case for case in cases}
    assert case_by_optimizer["sgd"].config.optimizer_momentum == 0.9
    assert case_by_optimizer["adamw"].config.optimizer_beta1 == 0.8
