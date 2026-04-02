"""Tests for comparison and search helpers."""

from simplesft.compare import compare_measurement_to_estimate
from simplesft.search import search_configurations
from simplesft.types import (
    EstimatorConfig,
    MemoryComponentBreakdown,
    MemoryResult,
    ModelLinearLayerSpec,
    ModelSpec,
    PhaseMemoryRecord,
)


def _result(
    global_peak_bytes: int, distributed_mode: str, feasible: bool
) -> MemoryResult:
    """Build a compact memory result for tests."""

    config = EstimatorConfig(
        tuning_mode="full_ft",
        distributed_mode=distributed_mode,
        gpu_memory_gb=100.0,
    )
    record = PhaseMemoryRecord(
        phase_name="forward",
        allocated_bytes=global_peak_bytes,
        reserved_bytes=global_peak_bytes,
        peak_allocated_bytes=global_peak_bytes,
        peak_reserved_bytes=global_peak_bytes,
        delta_allocated_bytes=global_peak_bytes,
        delta_reserved_bytes=global_peak_bytes,
        notes=(),
    )
    return MemoryResult(
        mode="estimate",
        model_name="toy",
        config=config,
        breakdown=MemoryComponentBreakdown(parameter_bytes=global_peak_bytes),
        phase_records=(record,),
        peak_phase="forward",
        global_peak_bytes=global_peak_bytes,
        feasible=feasible,
    )


def _toy_model_spec() -> ModelSpec:
    """Return a compact model spec for search tests."""

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
        trainable_linear_layers=(
            ModelLinearLayerSpec("layers.0.self_attn.q_proj", 32, 32, "attention"),
        ),
    )


def test_compare_measurement_to_estimate_reports_zero_error_for_equal_results() -> None:
    """Comparison should report zero error for identical inputs."""

    measured = _result(
        global_peak_bytes=10, distributed_mode="single_gpu", feasible=True
    )
    estimated = _result(
        global_peak_bytes=10, distributed_mode="single_gpu", feasible=True
    )
    comparison = compare_measurement_to_estimate(measured=measured, estimated=estimated)
    assert comparison.global_peak_error_bytes == 0
    assert comparison.global_peak_relative_error == 0.0
    assert comparison.component_error_bytes["parameter_bytes"] == 0


def test_compare_measurement_to_estimate_tracks_workspace_proxy_errors() -> None:
    """Comparison should expose workspace-proxy deltas from metadata."""

    measured = _result(
        global_peak_bytes=10, distributed_mode="single_gpu", feasible=True
    )
    estimated = _result(
        global_peak_bytes=10, distributed_mode="single_gpu", feasible=True
    )
    measured.metadata["forward_workspace_proxy_bytes"] = 7
    estimated.metadata["forward_workspace_proxy_bytes"] = 5
    comparison = compare_measurement_to_estimate(measured=measured, estimated=estimated)
    assert comparison.workspace_proxy_error_bytes["forward_workspace_proxy_bytes"] == 2
    assert comparison.workspace_proxy_relative_error[
        "forward_workspace_proxy_bytes"
    ] == (2 / 7)


def test_compare_measurement_to_estimate_tracks_intermediate_term_errors() -> None:
    """Comparison should expose non-additive term deltas from metadata."""

    measured = _result(
        global_peak_bytes=10, distributed_mode="single_gpu", feasible=True
    )
    estimated = _result(
        global_peak_bytes=10, distributed_mode="single_gpu", feasible=True
    )
    measured.metadata["hook_visible_activation_bytes"] = 9
    estimated.metadata["hook_visible_activation_bytes"] = 6
    comparison = compare_measurement_to_estimate(measured=measured, estimated=estimated)
    assert (
        comparison.intermediate_term_error_bytes["hook_visible_activation_bytes"] == 3
    )
    assert comparison.intermediate_term_relative_error[
        "hook_visible_activation_bytes"
    ] == (3 / 9)


def test_search_configurations_prefers_simpler_feasible_modes() -> None:
    """Search should rank feasible simpler configs ahead of complex ones."""

    class FakeEstimate:
        """Minimal fake estimate result for search ranking."""

    result = search_configurations(
        model=_toy_model_spec(),
        configs=[
            EstimatorConfig(tuning_mode="full_ft", distributed_mode="zero2"),
            EstimatorConfig(tuning_mode="full_ft", distributed_mode="single_gpu"),
        ],
    )
    assert result.candidates[0].config.distributed_mode == "single_gpu"


def test_search_configurations_ranks_zero3_after_zero2() -> None:
    """Search simplicity ordering should keep ZeRO-3 behind ZeRO-2."""

    result = search_configurations(
        model=_toy_model_spec(),
        configs=[
            EstimatorConfig(
                tuning_mode="full_ft",
                distributed_mode="zero3",
                gpus_per_node=2,
            ),
            EstimatorConfig(
                tuning_mode="full_ft",
                distributed_mode="zero2",
                gpus_per_node=2,
            ),
        ],
    )
    assert result.candidates[0].config.distributed_mode == "zero2"
