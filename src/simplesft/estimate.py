"""Analytical peak-memory estimation for structural estimator configs."""

from __future__ import annotations

from dataclasses import replace

from .activation_model import build_activation_terms
from .inspect import inspect_model
from .phase_model import PhaseInputs, build_phase_peak_debug, build_phase_records
from .resident_state_model import (
    build_resident_state_terms,
    estimate_lora_parameter_count,
    resolved_optimizer_state_dtype,
    trainable_parameter_count,
)
from .types import (
    EstimatorConfig,
    EstimatorDebugInfo,
    MemoryComponentBreakdown,
    MemoryResult,
    ModelSpec,
)
from .workspace_model import build_workspace_terms


def _coerce_model_spec(*, model: str | ModelSpec) -> ModelSpec:
    """Return a `ModelSpec` for estimation.

    Args:
        model: Model id/path or a precomputed `ModelSpec`.

    Returns:
        Inspected `ModelSpec`.
    """

    if isinstance(model, ModelSpec):
        return model
    return inspect_model(model_ref=model)


def _build_breakdown(
    *,
    resident_terms,
    activation_terms,
    workspace_terms,
    peak_phase: str,
) -> MemoryComponentBreakdown:
    """Return the public component breakdown for the selected peak phase.

    Args:
        resident_terms: Resident-state terms from the structural model.
        activation_terms: Retained activation terms from the structural model.
        workspace_terms: Workspace terms from the structural model.
        peak_phase: Selected peak phase.

    Returns:
        Public `MemoryComponentBreakdown` aligned to the peak phase.
    """

    activation_bytes = activation_terms.debug.forward_phase_activation_bytes
    transient_bytes = workspace_terms.forward_workspace_bytes
    if peak_phase == "backward":
        activation_bytes = activation_terms.debug.backward_phase_activation_bytes
        transient_bytes = workspace_terms.backward_workspace_bytes
    if peak_phase == "optimizer_step":
        activation_bytes = 0
        transient_bytes = (
            workspace_terms.backward_end_state_bytes
            + workspace_terms.optimizer_workspace_bytes
        )
    return MemoryComponentBreakdown(
        parameter_bytes=resident_terms.debug.parameter_bytes,
        gradient_bytes=resident_terms.debug.gradient_bytes,
        optimizer_state_bytes=resident_terms.debug.optimizer_state_bytes,
        activation_bytes=activation_bytes,
        transient_bytes=transient_bytes,
        residual_bytes=0,
        runtime_reserve_bytes=resident_terms.debug.runtime_floor_bytes
        + resident_terms.debug.master_weight_bytes,
    )


def _build_assumptions(*, config: EstimatorConfig) -> tuple[str, ...]:
    """Return human-readable assumptions for one estimate.

    Args:
        config: Structural estimator config.

    Returns:
        Short tuple of assumptions used by the simplified estimator.
    """

    assumptions = [
        "Peak memory uses allocated peak plus a fixed backend runtime floor.",
        "Hook-visible activations are diagnostic only and do not drive the retained activation term.",
        "Checkpointing moves rematerializable tensors into backward recompute workspace.",
    ]
    if config.distributed_mode in {"zero2", "zero3"}:
        assumptions.append(
            "ZeRO optimizer peak uses bucket-local fetch, update, and communication subphases."
        )
        assumptions.append("ZeRO optimizer update scratch is sharded by world size.")
    if config.tensor_parallel_degree > 1:
        assumptions.append(
            "Tensor-parallel resident state shards attention/MLP matrices across TP ranks."
        )
    if config.sequence_parallel:
        assumptions.append(
            "Sequence parallel shards sequence-local activation state across TP ranks."
        )
    return tuple(assumptions)


def estimate_peak_memory(
    model: str | ModelSpec,
    config: EstimatorConfig,
) -> MemoryResult:
    """Estimate per-rank peak memory for one structural training configuration.

    Args:
        model: Hugging Face model id or precomputed `ModelSpec`.
        config: Structural estimator configuration.

    Returns:
        Analytical estimate with typed debug internals and phase timeline.

    Example:
        >>> from simplesft.types import EstimatorConfig
        >>> result = estimate_peak_memory(
        ...     model="sshleifer/tiny-gpt2",
        ...     config=EstimatorConfig(tuning_mode="full_ft", max_seq_len=16),
        ... )
        >>> result.mode
        'estimate'
    """

    assert isinstance(config, EstimatorConfig), (
        "estimate_peak_memory() now requires EstimatorConfig. "
        "Project measurement configs with to_estimator_config()."
    )
    model_spec = _coerce_model_spec(model=model)
    resident_terms = build_resident_state_terms(model_spec=model_spec, config=config)
    activation_terms = build_activation_terms(model_spec=model_spec, config=config)
    workspace_terms = build_workspace_terms(
        model_spec=model_spec,
        config=config,
        parameter_bytes=resident_terms.debug.parameter_bytes,
        gradient_bytes=resident_terms.debug.gradient_bytes,
        optimizer_state_bytes=resident_terms.debug.optimizer_state_bytes,
    )
    phase_records = build_phase_records(
        inputs=PhaseInputs(
            parameter_bytes=resident_terms.debug.parameter_bytes,
            gradient_bytes=resident_terms.debug.gradient_bytes,
            optimizer_state_bytes=resident_terms.debug.optimizer_state_bytes,
            master_weight_bytes=resident_terms.debug.master_weight_bytes,
            runtime_floor_bytes=resident_terms.debug.runtime_floor_bytes,
            forward_activation_bytes=activation_terms.debug.forward_phase_activation_bytes,
            backward_activation_bytes=activation_terms.debug.backward_phase_activation_bytes,
            forward_workspace_bytes=workspace_terms.forward_workspace_bytes,
            backward_workspace_bytes=workspace_terms.backward_workspace_bytes,
            optimizer_workspace_bytes=workspace_terms.optimizer_workspace_bytes,
            backward_end_state_bytes=workspace_terms.backward_end_state_bytes,
        )
    )
    phase_debug = build_phase_peak_debug(
        phase_records=phase_records,
        backward_end_state_bytes=workspace_terms.backward_end_state_bytes,
    )
    debug_info = EstimatorDebugInfo(
        resident_state=resident_terms.debug,
        activations=activation_terms.debug,
        workspace=workspace_terms.debug,
        phase_peaks=phase_debug,
    )
    breakdown = _build_breakdown(
        resident_terms=resident_terms,
        activation_terms=activation_terms,
        workspace_terms=workspace_terms,
        peak_phase=phase_debug.global_peak_phase,
    )
    metadata = {
        "model_type": model_spec.model_type,
        "world_size": config.world_size(),
        "data_parallel_degree": config.data_parallel_degree(),
        "tensor_parallel_degree": config.tensor_parallel_degree_resolved(),
        "sequence_parallel": config.sequence_parallel,
        "optimizer_state_dtype_resolved": resolved_optimizer_state_dtype(config=config),
        "trainable_params": trainable_parameter_count(
            model_spec=model_spec,
            config=config,
        ),
        "estimated_lora_params": (
            estimate_lora_parameter_count(
                model_spec=model_spec,
                lora_config=config.lora,
            )
            if config.lora is not None
            else 0
        ),
    }
    return MemoryResult(
        mode="estimate",
        model_name=model_spec.model_name,
        config=replace(config),
        breakdown=breakdown,
        phase_records=phase_records,
        peak_phase=phase_debug.global_peak_phase,
        global_peak_bytes=phase_debug.global_peak_bytes,
        feasible=phase_debug.global_peak_bytes <= int(config.gpu_memory_gb * (1024**3)),
        metadata=metadata,
        debug=debug_info,
        assumptions=_build_assumptions(config=config),
    )
