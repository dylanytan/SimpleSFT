"""Tests for comparison and search helpers."""

from simplesft.results.compare import (
    compare_measurement_to_estimate,
    project_estimated_breakdown_for_phase,
)
from simplesft.results.search import search_configurations
from simplesft.types import (
    ActivationDebug,
    EstimatorConfig,
    EstimatorDebugInfo,
    MemoryComponentBreakdown,
    MemoryResult,
    ModelLinearLayerSpec,
    ModelSpec,
    PhasePeakDebug,
    PhaseMemoryRecord,
    ResidentStateDebug,
    TrainingConfig,
    WorkspaceDebug,
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


def _debugged_estimate_result() -> MemoryResult:
    """Build an estimate result with explicit debug terms for compare tests."""

    config = EstimatorConfig(
        tuning_mode="full_ft",
        distributed_mode="single_gpu",
        gpu_memory_gb=100.0,
    )
    return MemoryResult(
        mode="estimate",
        model_name="toy",
        config=config,
        breakdown=MemoryComponentBreakdown(
            parameter_bytes=10,
            gradient_bytes=4,
            optimizer_state_bytes=6,
            activation_bytes=30,
            transient_bytes=1,
            runtime_reserve_bytes=3,
        ),
        phase_records=(
            PhaseMemoryRecord(
                phase_name="forward",
                allocated_bytes=30,
                reserved_bytes=30,
                peak_allocated_bytes=30,
                peak_reserved_bytes=30,
                delta_allocated_bytes=0,
                delta_reserved_bytes=0,
            ),
            PhaseMemoryRecord(
                phase_name="backward",
                allocated_bytes=25,
                reserved_bytes=25,
                peak_allocated_bytes=25,
                peak_reserved_bytes=25,
                delta_allocated_bytes=0,
                delta_reserved_bytes=0,
            ),
        ),
        peak_phase="forward",
        global_peak_bytes=30,
        feasible=True,
        metadata={"retained_forward_proxy_bytes": 30},
        debug=EstimatorDebugInfo(
            resident_state=ResidentStateDebug(
                parameter_bytes=10,
                gradient_bytes=4,
                optimizer_state_bytes=6,
                master_weight_bytes=2,
                runtime_support_bytes=1,
                persistent_backend_buffer_bytes=5,
                trainable_parameter_bytes=10,
            ),
            activations=ActivationDebug(
                base_hook_visible_activation_bytes=12,
                visible_propagation_bytes=12,
                checkpoint_resident_block_input_bytes=0,
                saved_linear_input_bytes=8,
                saved_input_overlap_bytes=0,
                mlp_intermediate_bytes=0,
                parameter_gradient_context_bytes=8,
                residual_norm_bytes=2,
                checkpoint_boundary_bytes=0,
                attention_saved_bytes=0,
                loss_state_bytes=20,
                lora_low_rank_bytes=0,
                lora_backward_logits_context_bytes=0,
                expanded_query_saved_bytes=0,
                query_output_context_bytes=0,
                key_output_context_bytes=0,
                value_output_context_bytes=0,
                output_proj_input_context_bytes=0,
                output_proj_output_context_bytes=0,
                retained_forward_proxy_bytes=12,
                forward_phase_activation_bytes=30,
                backward_phase_activation_bytes=12,
                hook_visible_activation_bytes=12,
            ),
            workspace=WorkspaceDebug(
                attention_forward_workspace_bytes=1,
                backward_kernel_workspace_bytes=5,
                recompute_workspace_bytes=0,
                loss_workspace_bytes=0,
                optimizer_update_workspace_bytes=3,
                ddp_reducer_bucket_bytes=0,
                ddp_comm_overlap_bytes=0,
                zero_allgather_bucket_bytes=0,
                zero_reduce_bucket_bytes=0,
                zero_prefetch_bucket_bytes=0,
                zero_fetch_window_bytes=0,
                zero_update_window_bytes=0,
                zero_comm_window_bytes=0,
                tensor_parallel_comm_window_bytes=0,
                sequence_parallel_comm_window_bytes=0,
            ),
            phase_peaks=PhasePeakDebug(
                forward_peak_bytes=30,
                backward_peak_bytes=25,
                optimizer_peak_bytes=20,
                global_peak_bytes=30,
                global_peak_phase="forward",
                soft_reserved_global_peak_bytes=30,
                stressed_reserved_global_peak_bytes=30,
                backward_end_state_bytes=8,
            ),
        ),
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


def test_compare_measurement_to_estimate_uses_allocated_phase_peaks_when_requested() -> (
    None
):
    """Phase errors should respect allocated-mode measurement peaks."""

    measured_record = PhaseMemoryRecord(
        phase_name="forward",
        allocated_bytes=80,
        reserved_bytes=120,
        peak_allocated_bytes=90,
        peak_reserved_bytes=120,
        delta_allocated_bytes=80,
        delta_reserved_bytes=120,
        notes=(),
    )
    estimated_record = PhaseMemoryRecord(
        phase_name="forward",
        allocated_bytes=95,
        reserved_bytes=95,
        peak_allocated_bytes=95,
        peak_reserved_bytes=95,
        delta_allocated_bytes=95,
        delta_reserved_bytes=95,
        notes=(),
    )
    measured = MemoryResult(
        mode="measure",
        model_name="toy",
        config=TrainingConfig(
            tuning_mode="full_ft",
            allocator_peak_mode="allocated",
        ),
        breakdown=MemoryComponentBreakdown(parameter_bytes=10),
        phase_records=(measured_record,),
        peak_phase="forward",
        global_peak_bytes=90,
        feasible=True,
    )
    estimated = MemoryResult(
        mode="estimate",
        model_name="toy",
        config=EstimatorConfig(tuning_mode="full_ft"),
        breakdown=MemoryComponentBreakdown(parameter_bytes=10),
        phase_records=(estimated_record,),
        peak_phase="forward",
        global_peak_bytes=95,
        feasible=True,
    )
    comparison = compare_measurement_to_estimate(
        measured=measured,
        estimated=estimated,
    )
    assert comparison.phase_peak_error_bytes["forward"] == 5


def test_compare_measurement_to_estimate_projects_breakdown_to_measured_phase() -> None:
    """Phase-aligned component errors should use the measured peak phase."""

    measured = MemoryResult(
        mode="measure",
        model_name="toy",
        config=TrainingConfig(tuning_mode="full_ft"),
        breakdown=MemoryComponentBreakdown(
            parameter_bytes=10,
            gradient_bytes=4,
            optimizer_state_bytes=6,
            activation_bytes=30,
            transient_bytes=7,
            runtime_reserve_bytes=3,
        ),
        phase_records=(),
        peak_phase="backward",
        global_peak_bytes=25,
        feasible=True,
        metadata={"retained_activation_bytes": 30},
    )
    estimated = _debugged_estimate_result()
    comparison = compare_measurement_to_estimate(
        measured=measured,
        estimated=estimated,
    )
    assert comparison.component_error_bytes["transient_bytes"] == 6
    assert comparison.phase_aligned_component_error_bytes["transient_bytes"] == 2
    assert "activation_bytes" not in comparison.phase_aligned_component_error_bytes


def test_projected_breakdown_keeps_persistent_backend_buffers() -> None:
    """Phase projection should keep persistent backend buffers in runtime reserve."""

    projected = project_estimated_breakdown_for_phase(
        result=_debugged_estimate_result(),
        phase_name="backward",
    )
    assert projected.runtime_reserve_bytes == 8


def test_compare_measurement_to_estimate_splits_retained_forward_proxy_error() -> None:
    """Retained-forward proxy error should be reported independently."""

    measured = MemoryResult(
        mode="measure",
        model_name="toy",
        config=TrainingConfig(tuning_mode="full_ft"),
        breakdown=MemoryComponentBreakdown(
            parameter_bytes=10,
            gradient_bytes=4,
            optimizer_state_bytes=6,
            activation_bytes=50,
            transient_bytes=7,
            runtime_reserve_bytes=3,
        ),
        phase_records=(),
        peak_phase="backward",
        global_peak_bytes=25,
        feasible=True,
        metadata={"retained_activation_bytes": 50},
    )
    estimated = _debugged_estimate_result()
    comparison = compare_measurement_to_estimate(
        measured=measured,
        estimated=estimated,
    )
    assert comparison.component_error_bytes["activation_bytes"] == 20
    assert comparison.retained_forward_proxy_error_bytes == 20
    assert comparison.retained_forward_proxy_relative_error == (20 / 50)


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
