"""Comparison helpers for measured and estimated memory results."""

from __future__ import annotations

from .types import ComparisonResult, MemoryComponentBreakdown, MemoryResult


def _component_dict(breakdown: MemoryComponentBreakdown) -> dict[str, int]:
    """Return a flat component mapping from a breakdown object."""

    return {
        "parameter_bytes": breakdown.parameter_bytes,
        "gradient_bytes": breakdown.gradient_bytes,
        "optimizer_state_bytes": breakdown.optimizer_state_bytes,
        "activation_bytes": breakdown.activation_bytes,
        "transient_bytes": breakdown.transient_bytes,
        "residual_bytes": breakdown.residual_bytes,
        "runtime_reserve_bytes": breakdown.runtime_reserve_bytes,
    }


def _workspace_proxy_dict(result: MemoryResult) -> dict[str, int]:
    """Return workspace-like metadata exposed by measurement and estimation."""

    metadata = result.comparable_metadata()
    workspace_keys = (
        "attention_forward_workspace_bytes",
        "backward_kernel_workspace_bytes",
        "recompute_workspace_bytes",
        "optimizer_update_workspace_bytes",
        "ddp_reducer_bucket_bytes",
        "ddp_comm_overlap_bytes",
        "zero_fetch_window_bytes",
        "zero_update_window_bytes",
        "zero_comm_window_bytes",
        "tensor_parallel_comm_window_bytes",
        "sequence_parallel_comm_window_bytes",
        "forward_workspace_proxy_bytes",
        "backward_workspace_proxy_bytes",
        "optimizer_workspace_proxy_bytes",
    )
    return {key: int(metadata.get(key, 0)) for key in workspace_keys}


def _intermediate_term_dict(result: MemoryResult) -> dict[str, int]:
    """Return comparable non-additive metadata terms from a result."""

    metadata = result.comparable_metadata()
    metadata_keys = (
        "hook_visible_activation_bytes",
        "saved_linear_input_bytes",
        "residual_norm_bytes",
        "checkpoint_boundary_bytes",
        "attention_saved_bytes",
        "loss_state_bytes",
        "lora_low_rank_bytes",
        "expanded_query_saved_bytes",
        "forward_phase_activation_bytes",
        "backward_phase_activation_bytes",
        "backward_end_state_bytes",
    )
    return {key: int(metadata[key]) for key in metadata_keys if key in metadata}


def _relative_error(measured_value: int, estimated_value: int) -> float:
    """Return a safe relative error against the measured value."""

    if measured_value == 0:
        return 0.0 if estimated_value == 0 else 1.0
    return abs(measured_value - estimated_value) / measured_value


def compare_measurement_to_estimate(
    *,
    measured: MemoryResult,
    estimated: MemoryResult,
) -> ComparisonResult:
    """Compare a measured result against an estimate.

    Args:
        measured: Ground-truth measurement result.
        estimated: Analytical estimate result.

    Returns:
        ComparisonResult describing errors at peak, phase, and component level.

    Example:
        >>> from simplesft.types import MemoryComponentBreakdown, MemoryResult, TrainingConfig
        >>> config = TrainingConfig(tuning_mode="full_ft")
        >>> result = MemoryResult(
        ...     mode="estimate",
        ...     model_name="toy",
        ...     config=config,
        ...     breakdown=MemoryComponentBreakdown(),
        ...     phase_records=(),
        ...     peak_phase="forward",
        ...     global_peak_bytes=0,
        ...     feasible=True,
        ... )
        >>> compare_measurement_to_estimate(measured=result, estimated=result).global_peak_error_bytes
        0
    """

    measured_by_phase = {
        record.phase_name: record.peak_reserved_bytes
        for record in measured.phase_records
    }
    estimated_by_phase = {
        record.phase_name: record.peak_reserved_bytes
        for record in estimated.phase_records
    }
    phase_names = sorted(set(measured_by_phase) | set(estimated_by_phase))
    phase_peak_error_bytes = {
        phase_name: abs(
            measured_by_phase.get(phase_name, 0) - estimated_by_phase.get(phase_name, 0)
        )
        for phase_name in phase_names
    }
    phase_peak_relative_error = {
        phase_name: _relative_error(
            measured_value=measured_by_phase.get(phase_name, 0),
            estimated_value=estimated_by_phase.get(phase_name, 0),
        )
        for phase_name in phase_names
    }
    measured_components = _component_dict(breakdown=measured.breakdown)
    estimated_components = _component_dict(breakdown=estimated.breakdown)
    measured_workspace = _workspace_proxy_dict(result=measured)
    estimated_workspace = _workspace_proxy_dict(result=estimated)
    measured_terms = _intermediate_term_dict(result=measured)
    estimated_terms = _intermediate_term_dict(result=estimated)
    intermediate_term_names = sorted(set(measured_terms) & set(estimated_terms))
    component_error_bytes = {
        key: abs(measured_components.get(key, 0) - estimated_components.get(key, 0))
        for key in measured_components
    }
    component_relative_error = {
        key: _relative_error(
            measured_value=measured_components.get(key, 0),
            estimated_value=estimated_components.get(key, 0),
        )
        for key in measured_components
    }
    workspace_proxy_error_bytes = {
        key: abs(measured_workspace.get(key, 0) - estimated_workspace.get(key, 0))
        for key in measured_workspace
    }
    workspace_proxy_relative_error = {
        key: _relative_error(
            measured_value=measured_workspace.get(key, 0),
            estimated_value=estimated_workspace.get(key, 0),
        )
        for key in measured_workspace
    }
    intermediate_term_error_bytes = {
        key: abs(measured_terms.get(key, 0) - estimated_terms.get(key, 0))
        for key in intermediate_term_names
    }
    intermediate_term_relative_error = {
        key: _relative_error(
            measured_value=measured_terms.get(key, 0),
            estimated_value=estimated_terms.get(key, 0),
        )
        for key in intermediate_term_names
    }
    notes = []
    if estimated.global_peak_bytes < measured.global_peak_bytes:
        notes.append("Estimator under-predicts measured global peak.")
    elif estimated.global_peak_bytes > measured.global_peak_bytes:
        notes.append("Estimator over-predicts measured global peak.")
    else:
        notes.append("Estimator matches measured global peak.")
    return ComparisonResult(
        model_name=measured.model_name,
        measured=measured,
        estimated=estimated,
        global_peak_error_bytes=abs(
            measured.global_peak_bytes - estimated.global_peak_bytes
        ),
        global_peak_relative_error=_relative_error(
            measured_value=measured.global_peak_bytes,
            estimated_value=estimated.global_peak_bytes,
        ),
        phase_peak_error_bytes=phase_peak_error_bytes,
        phase_peak_relative_error=phase_peak_relative_error,
        component_error_bytes=component_error_bytes,
        component_relative_error=component_relative_error,
        workspace_proxy_error_bytes=workspace_proxy_error_bytes,
        workspace_proxy_relative_error=workspace_proxy_relative_error,
        intermediate_term_error_bytes=intermediate_term_error_bytes,
        intermediate_term_relative_error=intermediate_term_relative_error,
        benchmark_metadata={
            "tuning_mode": measured.config.tuning_mode,
            "optimizer_name": measured.config.optimizer_name,
            "distributed_mode": measured.config.distributed_mode,
            "sequence_length": measured.config.max_seq_len,
            "micro_batch_size_per_gpu": measured.config.micro_batch_size_per_gpu,
            "attention_backend": measured.config.attention_backend,
            "gradient_checkpointing": measured.config.gradient_checkpointing,
        },
        notes=tuple(notes),
    )
