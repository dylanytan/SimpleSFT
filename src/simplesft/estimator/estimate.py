"""Analytical peak-memory estimation for structural estimator configs."""

from __future__ import annotations

from dataclasses import replace

from .activation import build_activation_terms
from ..models.inspect import inspect_model
from .optimizer import optimizer_is_zero_tested
from .phase import PhaseInputs, build_phase_peak_debug, build_phase_records
from ..models.precomputed_model_specs import resolve_model_spec
from .resident_state import (
    build_resident_state_terms,
    estimate_lora_parameter_count,
    resolved_optimizer_state_dtype,
    trainable_parameter_count,
)
from ..types import (
    EstimatorConfig,
    EstimatorDebugInfo,
    MemoryComponentBreakdown,
    MemoryResult,
    ModelSpec,
)
from .workspace import build_workspace_terms


def _coerce_model_spec(*, model: str | ModelSpec) -> ModelSpec:
    """Return a `ModelSpec` for estimation.

    Args:
        model: Model id/path or a precomputed `ModelSpec`.

    Returns:
        Inspected `ModelSpec`.
    """

    if isinstance(model, ModelSpec):
        return model
    return resolve_model_spec(model_ref=model, inspect_fn=inspect_model)


def _build_breakdown(
    *,
    resident_terms,
    activation_terms,
    workspace_terms,
    phase_records,
    peak_phase: str,
) -> MemoryComponentBreakdown:
    """Return the public component breakdown for the selected peak phase.

    Args:
        resident_terms: Resident-state terms from the structural model.
        activation_terms: Retained activation terms from the structural model.
        workspace_terms: Workspace terms from the structural model.
        phase_records: Estimated phase timeline used to recover phase-local
            transient peaks.
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
        optimizer_record = next(
            record
            for record in phase_records
            if record.phase_name == "optimizer_step"
        )
        resident_floor_bytes = (
            resident_terms.debug.parameter_bytes
            + resident_terms.debug.gradient_bytes
            + resident_terms.debug.optimizer_state_bytes
            + resident_terms.debug.runtime_support_bytes
            + resident_terms.debug.persistent_backend_buffer_bytes
            + resident_terms.debug.master_weight_bytes
        )
        transient_bytes = max(
            0,
            optimizer_record.peak_reserved_bytes - resident_floor_bytes,
        )
    return MemoryComponentBreakdown(
        parameter_bytes=resident_terms.debug.parameter_bytes,
        gradient_bytes=resident_terms.debug.gradient_bytes,
        optimizer_state_bytes=resident_terms.debug.optimizer_state_bytes,
        activation_bytes=activation_bytes,
        transient_bytes=transient_bytes,
        residual_bytes=0,
        runtime_reserve_bytes=resident_terms.debug.runtime_support_bytes
        + resident_terms.debug.persistent_backend_buffer_bytes
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
        "Runtime support is modeled as named CUDA, allocator, NCCL, and DeepSpeed support objects.",
        "Hook-visible activations are diagnostic only and do not drive the retained activation term.",
        "Checkpointing moves rematerializable tensors into backward recompute workspace.",
        "Architecture support is driven by explicit family manifests and typed attention / tensor-layout metadata.",
    ]
    if config.distributed_mode in {"zero2", "zero3"}:
        assumptions.append(
            "ZeRO optimizer peak uses bucket-local fetch, update, and communication subphases."
        )
        assumptions.append("ZeRO optimizer update scratch is sharded by world size.")
    if config.tuning_mode == "full_ft" and config.is_zero_mode():
        assumptions.append(
            "ZeRO full fine-tuning includes explicit persistent backend support buffers."
        )
    if config.tensor_parallel_degree > 1:
        assumptions.append(
            "Tensor-parallel resident state shards explicit row/column/vocab tensor roles across TP ranks."
        )
    if config.sequence_parallel:
        assumptions.append(
            "Sequence parallel shards sequence-local activation state across TP ranks."
        )
    if model_ref := getattr(config, "attention_backend", None):
        assumptions.append(
            f"Attention workspace uses the `{model_ref}` backend together with the inspected head layout."
        )
    return tuple(assumptions)


def _optimizer_reserved_carryover_bytes(
    *,
    config: EstimatorConfig,
    activation_terms,
    workspace_terms,
) -> int:
    """Return optimizer-step reserved carryover bytes.

    Args:
        config: Structural estimator config.
        activation_terms: Retained activation terms from the structural model.
        workspace_terms: Workspace terms from the structural model.

    Returns:
        Additional optimizer-step bytes representing checkpointed ZeRO reserved
        carryover from backward activation state.
    """

    if not config.gradient_checkpointing:
        if (
            config.tuning_mode == "full_ft"
            and config.distributed_mode == "zero2"
            and optimizer_is_zero_tested(config=config)
        ):
            retained_forward_without_loss_bytes = max(
                0,
                activation_terms.debug.retained_forward_proxy_bytes
                - activation_terms.debug.loss_state_bytes,
            )
            return max(
                workspace_terms.backward_workspace_bytes,
                retained_forward_without_loss_bytes,
            )
        if (
            config.tuning_mode == "full_ft"
            and config.distributed_mode == "zero2"
            and config.normalized_attention_backend() == "sdpa"
        ):
            return activation_terms.debug.backward_phase_activation_bytes
        return 0
    if (
        config.tuning_mode == "lora"
        and config.distributed_mode == "single_gpu"
    ):
        return (
            activation_terms.debug.forward_phase_activation_bytes
            + workspace_terms.backward_workspace_bytes
        )
    if config.tuning_mode != "full_ft":
        return 0
    if not config.is_zero_mode():
        return 0
    return activation_terms.debug.backward_phase_activation_bytes


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
        trainable_parameter_bytes=resident_terms.debug.trainable_parameter_bytes,
        gradient_bytes=resident_terms.debug.gradient_bytes,
        optimizer_state_bytes=resident_terms.debug.optimizer_state_bytes,
    )
    phase_records = build_phase_records(
        inputs=PhaseInputs(
            parameter_bytes=resident_terms.debug.parameter_bytes,
            gradient_bytes=resident_terms.debug.gradient_bytes,
            optimizer_state_bytes=resident_terms.debug.optimizer_state_bytes,
            master_weight_bytes=resident_terms.debug.master_weight_bytes,
            runtime_support_bytes=resident_terms.debug.runtime_support_bytes,
            persistent_backend_buffer_bytes=(
                resident_terms.debug.persistent_backend_buffer_bytes
            ),
            forward_activation_bytes=activation_terms.debug.forward_phase_activation_bytes,
            backward_activation_bytes=activation_terms.debug.backward_phase_activation_bytes,
            forward_workspace_bytes=workspace_terms.forward_workspace_bytes,
            backward_workspace_bytes=workspace_terms.backward_workspace_bytes,
            optimizer_workspace_bytes=workspace_terms.optimizer_workspace_bytes,
            backward_end_state_bytes=workspace_terms.backward_end_state_bytes,
            optimizer_reserved_carryover_bytes=_optimizer_reserved_carryover_bytes(
                config=config,
                activation_terms=activation_terms,
                workspace_terms=workspace_terms,
            ),
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
        phase_records=phase_records,
        peak_phase=phase_debug.global_peak_phase,
    )
    metadata = {
        "model_type": model_spec.model_type,
        "architecture_family": model_spec.architecture_family.family_label,
        "world_size": config.world_size(),
        "data_parallel_degree": config.data_parallel_degree(),
        "tensor_parallel_degree": config.tensor_parallel_degree_resolved(),
        "sequence_parallel": config.sequence_parallel,
        "num_query_heads": model_spec.attention.num_query_heads,
        "num_key_value_heads": model_spec.attention.num_key_value_heads,
        "sliding_window_size": model_spec.attention.sliding_window_size or 0,
        "optimizer_state_dtype_resolved": resolved_optimizer_state_dtype(config=config),
        "retained_forward_proxy_bytes": (
            activation_terms.debug.retained_forward_proxy_bytes
        ),
        "retained_forward_proxy_source": "retained_forward_proxy_bytes",
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
