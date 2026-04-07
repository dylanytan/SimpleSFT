"""Tests for artifact serialization and loading."""

import json
from pathlib import Path

from simplesft.results.artifacts import (
    load_comparison_result,
    load_memory_result,
    save_comparison_result,
    save_memory_result,
)
from simplesft.types import (
    ComparisonResult,
    MemoryComponentBreakdown,
    MemoryResult,
    PhaseMemoryRecord,
    TrainingConfig,
)


def test_memory_result_roundtrip(tmp_path: Path) -> None:
    """Memory results should round-trip through JSON artifacts."""

    result = MemoryResult(
        mode="estimate",
        model_name="toy",
        config=TrainingConfig(tuning_mode="full_ft"),
        breakdown=MemoryComponentBreakdown(parameter_bytes=123),
        phase_records=(
            PhaseMemoryRecord(
                phase_name="forward",
                allocated_bytes=1,
                reserved_bytes=2,
                peak_allocated_bytes=3,
                peak_reserved_bytes=4,
                delta_allocated_bytes=1,
                delta_reserved_bytes=2,
            ),
        ),
        peak_phase="forward",
        global_peak_bytes=4,
        feasible=True,
        metadata={"x": 1},
        assumptions=("a",),
    )
    artifact_path = tmp_path / "result.json"
    save_memory_result(result=result, path=artifact_path)
    loaded_result = load_memory_result(path=artifact_path)
    assert loaded_result.global_peak_bytes == result.global_peak_bytes
    assert loaded_result.breakdown.parameter_bytes == 123


def test_comparison_result_roundtrip_preserves_phase_aligned_fields(
    tmp_path: Path,
) -> None:
    """Comparison artifacts should preserve phase-aligned error metadata."""

    result = MemoryResult(
        mode="estimate",
        model_name="toy",
        config=TrainingConfig(tuning_mode="full_ft"),
        breakdown=MemoryComponentBreakdown(parameter_bytes=123),
        phase_records=(),
        peak_phase="forward",
        global_peak_bytes=4,
        feasible=True,
    )
    comparison = ComparisonResult(
        model_name="toy",
        measured=result,
        estimated=result,
        global_peak_error_bytes=0,
        global_peak_relative_error=0.0,
        phase_peak_error_bytes={"forward": 0},
        phase_peak_relative_error={"forward": 0.0},
        component_error_bytes={"parameter_bytes": 0},
        component_relative_error={"parameter_bytes": 0.0},
        phase_aligned_component_error_bytes={"transient_bytes": 2},
        phase_aligned_component_relative_error={"transient_bytes": 0.25},
        retained_forward_proxy_error_bytes=3,
        retained_forward_proxy_relative_error=0.5,
    )
    artifact_path = tmp_path / "comparison.json"
    save_comparison_result(result=comparison, path=artifact_path)
    loaded_result = load_comparison_result(path=artifact_path)
    assert loaded_result.phase_aligned_component_error_bytes["transient_bytes"] == 2
    assert loaded_result.retained_forward_proxy_error_bytes == 3


def test_load_comparison_result_ignores_legacy_estimator_debug_keys(
    tmp_path: Path,
) -> None:
    """Artifact loading should tolerate legacy estimator-debug fields."""

    comparison_path = tmp_path / "comparison.json"
    comparison_path.write_text(
        json.dumps(
            {
                "model_name": "toy",
                "measured": {
                    "mode": "measure",
                    "model_name": "toy",
                    "config": {"tuning_mode": "full_ft"},
                    "breakdown": {"parameter_bytes": 1},
                    "phase_records": [],
                    "peak_phase": "forward",
                    "global_peak_bytes": 1,
                    "feasible": True,
                    "metadata": {},
                    "debug": None,
                    "assumptions": [],
                },
                "estimated": {
                    "mode": "estimate",
                    "model_name": "toy",
                    "config": {"tuning_mode": "full_ft"},
                    "breakdown": {"parameter_bytes": 1},
                    "phase_records": [],
                    "peak_phase": "forward",
                    "global_peak_bytes": 1,
                    "feasible": True,
                    "metadata": {},
                    "debug": {
                        "resident_state": {
                            "parameter_bytes": 1,
                            "gradient_bytes": 0,
                            "optimizer_state_bytes": 0,
                            "master_weight_bytes": 0,
                            "runtime_support_bytes": 0,
                            "persistent_backend_buffer_bytes": 0,
                            "trainable_parameter_bytes": 1,
                            "runtime_floor_bytes": 99,
                        },
                        "activations": {
                            "base_hook_visible_activation_bytes": 0,
                            "visible_propagation_bytes": 0,
                            "checkpoint_resident_block_input_bytes": 0,
                            "saved_linear_input_bytes": 0,
                            "saved_input_overlap_bytes": 0,
                            "mlp_intermediate_bytes": 0,
                            "parameter_gradient_context_bytes": 0,
                            "residual_norm_bytes": 0,
                            "checkpoint_boundary_bytes": 0,
                            "attention_saved_bytes": 0,
                            "loss_state_bytes": 0,
                            "lora_low_rank_bytes": 0,
                            "lora_backward_logits_context_bytes": 0,
                            "expanded_query_saved_bytes": 0,
                            "query_output_context_bytes": 0,
                            "key_output_context_bytes": 0,
                            "value_output_context_bytes": 0,
                            "output_proj_input_context_bytes": 0,
                            "output_proj_output_context_bytes": 0,
                            "retained_forward_proxy_bytes": 0,
                            "forward_phase_activation_bytes": 0,
                            "backward_phase_activation_bytes": 0,
                            "hook_visible_activation_bytes": 0,
                        },
                        "workspace": {
                            "attention_forward_workspace_bytes": 0,
                            "backward_kernel_workspace_bytes": 0,
                            "recompute_workspace_bytes": 0,
                            "loss_workspace_bytes": 0,
                            "optimizer_update_workspace_bytes": 0,
                            "ddp_reducer_bucket_bytes": 0,
                            "ddp_comm_overlap_bytes": 0,
                            "zero_allgather_bucket_bytes": 0,
                            "zero_reduce_bucket_bytes": 0,
                            "zero_prefetch_bucket_bytes": 0,
                            "zero_fetch_window_bytes": 0,
                            "zero_update_window_bytes": 0,
                            "zero_comm_window_bytes": 0,
                            "tensor_parallel_comm_window_bytes": 0,
                            "sequence_parallel_comm_window_bytes": 0,
                        },
                        "phase_peaks": {
                            "forward_peak_bytes": 1,
                            "backward_peak_bytes": 1,
                            "optimizer_peak_bytes": 1,
                            "global_peak_bytes": 1,
                            "global_peak_phase": "forward",
                            "soft_reserved_global_peak_bytes": 1,
                            "stressed_reserved_global_peak_bytes": 1,
                            "backward_end_state_bytes": 0,
                        },
                    },
                    "assumptions": [],
                },
                "global_peak_error_bytes": 0,
                "global_peak_relative_error": 0.0,
                "phase_peak_error_bytes": {},
                "phase_peak_relative_error": {},
                "component_error_bytes": {},
                "component_relative_error": {},
            }
        ),
        encoding="utf-8",
    )
    loaded_result = load_comparison_result(path=comparison_path)
    assert loaded_result.estimated.debug is not None
    assert loaded_result.estimated.debug.resident_state.parameter_bytes == 1
