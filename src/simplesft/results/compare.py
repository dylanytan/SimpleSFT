"""Comparison helpers for measured and estimated memory results."""

from __future__ import annotations

from ..types import (
    ComparisonResult,
    MeasurementConfig,
    MemoryComponentBreakdown,
    MemoryResult,
)

PHASE_ALIGNED_COMPONENT_KEYS = (
    "parameter_bytes",
    "gradient_bytes",
    "optimizer_state_bytes",
    "transient_bytes",
    "residual_bytes",
    "runtime_reserve_bytes",
)


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


def _phase_aligned_component_dict(
    *, breakdown: MemoryComponentBreakdown
) -> dict[str, int]:
    """Return only phase-local component buckets that can be compared fairly."""

    components = _component_dict(breakdown=breakdown)
    return {key: components[key] for key in PHASE_ALIGNED_COMPONENT_KEYS}


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
        "zero_allgather_bucket_bytes",
        "zero_reduce_bucket_bytes",
        "zero_prefetch_bucket_bytes",
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
        "base_hook_visible_activation_bytes",
        "visible_propagation_bytes",
        "checkpoint_resident_block_input_bytes",
        "saved_linear_input_bytes",
        "mlp_intermediate_bytes",
        "residual_norm_bytes",
        "checkpoint_boundary_bytes",
        "attention_saved_bytes",
        "loss_state_bytes",
        "lora_low_rank_bytes",
        "expanded_query_saved_bytes",
        "query_output_context_bytes",
        "key_output_context_bytes",
        "value_output_context_bytes",
        "output_proj_input_context_bytes",
        "output_proj_output_context_bytes",
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


def _phase_peak_bytes_for_comparison(
    *,
    result: MemoryResult,
    phase_name: str,
) -> int:
    """Return the phase peak bytes used for comparison."""

    phase_record = next(
        (record for record in result.phase_records if record.phase_name == phase_name),
        None,
    )
    if phase_record is None:
        return 0
    if result.mode == "measure" and isinstance(result.config, MeasurementConfig):
        if result.config.normalized_allocator_peak_mode() == "allocated":
            return phase_record.peak_allocated_bytes
    return phase_record.peak_reserved_bytes


def _forward_workspace_bytes(result: MemoryResult) -> int:
    """Return the projected forward workspace bytes for one estimate result."""

    assert result.debug is not None, "Estimate debug info is required for projection."
    workspace = result.debug.workspace
    return (
        workspace.attention_forward_workspace_bytes
        + workspace.tensor_parallel_comm_window_bytes
        + workspace.sequence_parallel_comm_window_bytes
    )


def _backward_workspace_bytes(result: MemoryResult) -> int:
    """Return the projected backward workspace bytes for one estimate result."""

    assert result.debug is not None, "Estimate debug info is required for projection."
    workspace = result.debug.workspace
    return (
        workspace.backward_kernel_workspace_bytes
        + workspace.recompute_workspace_bytes
        + workspace.tensor_parallel_comm_window_bytes
        + workspace.sequence_parallel_comm_window_bytes
    )


def _optimizer_workspace_bytes(result: MemoryResult) -> int:
    """Return the projected optimizer-step workspace bytes for one estimate."""

    assert result.debug is not None, "Estimate debug info is required for projection."
    workspace = result.debug.workspace
    if result.config.distributed_mode == "ddp":
        return max(
            workspace.ddp_comm_overlap_bytes,
            workspace.optimizer_update_workspace_bytes,
        )
    if result.config.distributed_mode in {"zero2", "zero3"}:
        return max(
            workspace.zero_fetch_window_bytes,
            workspace.zero_update_window_bytes,
            workspace.zero_comm_window_bytes,
        )
    return workspace.optimizer_update_workspace_bytes


def project_estimated_breakdown_for_phase(
    *,
    result: MemoryResult,
    phase_name: str,
) -> MemoryComponentBreakdown:
    """Project an estimate result into the requested phase breakdown.

    Args:
        result: Analytical estimate result to project.
        phase_name: One of `forward`, `backward`, or `optimizer_step`.

    Returns:
        Estimated breakdown aligned to the requested phase.

    Example:
        >>> from simplesft.types import EstimatorConfig, MemoryResult
        >>> # See tests for a concrete construction example.
    """

    if result.mode != "estimate" or result.debug is None:
        return result.breakdown
    resident = result.debug.resident_state
    runtime_reserve_bytes = (
        resident.runtime_support_bytes
        + resident.persistent_backend_buffer_bytes
        + resident.master_weight_bytes
    )
    if phase_name == "forward":
        return MemoryComponentBreakdown(
            parameter_bytes=resident.parameter_bytes,
            gradient_bytes=resident.gradient_bytes,
            optimizer_state_bytes=resident.optimizer_state_bytes,
            activation_bytes=result.debug.activations.forward_phase_activation_bytes,
            transient_bytes=_forward_workspace_bytes(result=result),
            residual_bytes=0,
            runtime_reserve_bytes=runtime_reserve_bytes,
        )
    if phase_name == "backward":
        return MemoryComponentBreakdown(
            parameter_bytes=resident.parameter_bytes,
            gradient_bytes=resident.gradient_bytes,
            optimizer_state_bytes=resident.optimizer_state_bytes,
            activation_bytes=result.debug.activations.backward_phase_activation_bytes,
            transient_bytes=_backward_workspace_bytes(result=result),
            residual_bytes=0,
            runtime_reserve_bytes=runtime_reserve_bytes,
        )
    assert phase_name == "optimizer_step", f"Unsupported phase: {phase_name}"
    optimizer_record = next(
        record for record in result.phase_records if record.phase_name == "optimizer_step"
    )
    resident_floor_bytes = (
        resident.parameter_bytes
        + resident.gradient_bytes
        + resident.optimizer_state_bytes
        + runtime_reserve_bytes
    )
    return MemoryComponentBreakdown(
        parameter_bytes=resident.parameter_bytes,
        gradient_bytes=resident.gradient_bytes,
        optimizer_state_bytes=resident.optimizer_state_bytes,
        activation_bytes=0,
        transient_bytes=max(
            0,
            optimizer_record.peak_reserved_bytes - resident_floor_bytes,
        ),
        residual_bytes=0,
        runtime_reserve_bytes=runtime_reserve_bytes,
    )


def _retained_forward_proxy_bytes(result: MemoryResult) -> int:
    """Return the retained-forward activation proxy for one result."""

    metadata = result.comparable_metadata()
    if result.mode == "measure":
        return int(
            metadata.get("retained_activation_bytes", result.breakdown.activation_bytes)
        )
    return int(
        metadata.get(
            "retained_forward_proxy_bytes",
            result.breakdown.activation_bytes,
        )
    )


def _error_dict(
    *,
    measured_values: dict[str, int],
    estimated_values: dict[str, int],
) -> tuple[dict[str, int], dict[str, float]]:
    """Return absolute and relative error mappings for aligned dictionaries."""

    error_bytes = {
        key: abs(measured_values.get(key, 0) - estimated_values.get(key, 0))
        for key in measured_values
    }
    relative_error = {
        key: _relative_error(
            measured_value=measured_values.get(key, 0),
            estimated_value=estimated_values.get(key, 0),
        )
        for key in measured_values
    }
    return error_bytes, relative_error


def _phase_peak_comparison(
    *,
    measured: MemoryResult,
    estimated: MemoryResult,
) -> tuple[dict[str, int], dict[str, float]]:
    """Return phase-level peak errors for measured and estimated results."""

    phase_names = sorted(
        {
            record.phase_name
            for record in measured.phase_records + estimated.phase_records
        }
    )
    measured_by_phase = {
        phase_name: _phase_peak_bytes_for_comparison(
            result=measured,
            phase_name=phase_name,
        )
        for phase_name in phase_names
    }
    estimated_by_phase = {
        phase_name: _phase_peak_bytes_for_comparison(
            result=estimated,
            phase_name=phase_name,
        )
        for phase_name in phase_names
    }
    return _error_dict(
        measured_values=measured_by_phase,
        estimated_values=estimated_by_phase,
    )


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
        ComparisonResult describing global, phase, component, and proxy errors.
    """

    phase_peak_error_bytes, phase_peak_relative_error = _phase_peak_comparison(
        measured=measured,
        estimated=estimated,
    )
    measured_components = _component_dict(breakdown=measured.breakdown)
    estimated_components = _component_dict(breakdown=estimated.breakdown)
    component_error_bytes, component_relative_error = _error_dict(
        measured_values=measured_components,
        estimated_values=estimated_components,
    )
    projected_breakdown = project_estimated_breakdown_for_phase(
        result=estimated,
        phase_name=measured.peak_phase,
    )
    phase_aligned_error_bytes, phase_aligned_relative_error = _error_dict(
        measured_values=_phase_aligned_component_dict(breakdown=measured.breakdown),
        estimated_values=_phase_aligned_component_dict(breakdown=projected_breakdown),
    )
    measured_proxy_bytes = _retained_forward_proxy_bytes(result=measured)
    estimated_proxy_bytes = _retained_forward_proxy_bytes(result=estimated)
    measured_workspace = _workspace_proxy_dict(result=measured)
    estimated_workspace = _workspace_proxy_dict(result=estimated)
    measured_terms = _intermediate_term_dict(result=measured)
    estimated_terms = _intermediate_term_dict(result=estimated)
    workspace_proxy_error_bytes, workspace_proxy_relative_error = _error_dict(
        measured_values=measured_workspace,
        estimated_values=estimated_workspace,
    )
    intermediate_term_names = sorted(set(measured_terms) & set(estimated_terms))
    intermediate_term_error_bytes, intermediate_term_relative_error = _error_dict(
        measured_values={key: measured_terms[key] for key in intermediate_term_names},
        estimated_values={key: estimated_terms[key] for key in intermediate_term_names},
    )
    notes = []
    if estimated.global_peak_bytes < measured.global_peak_bytes:
        notes.append("Estimator under-predicts measured global peak.")
    elif estimated.global_peak_bytes > measured.global_peak_bytes:
        notes.append("Estimator over-predicts measured global peak.")
    else:
        notes.append("Estimator matches measured global peak.")
    if measured.peak_phase != estimated.peak_phase:
        notes.append(
            "Peak phases differ; phase-aligned component errors use the measured phase."
        )
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
        phase_aligned_component_error_bytes=phase_aligned_error_bytes,
        phase_aligned_component_relative_error=phase_aligned_relative_error,
        retained_forward_proxy_error_bytes=abs(
            measured_proxy_bytes - estimated_proxy_bytes
        ),
        retained_forward_proxy_relative_error=_relative_error(
            measured_value=measured_proxy_bytes,
            estimated_value=estimated_proxy_bytes,
        ),
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
            "measured_peak_phase": measured.peak_phase,
            "estimated_peak_phase": estimated.peak_phase,
        },
        notes=tuple(notes),
    )
