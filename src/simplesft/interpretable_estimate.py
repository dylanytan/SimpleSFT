"""Interpretable memory-estimation terms with explicit state accounting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .constants import TRAINING_PHASES, model_type_uses_sdpa
from .optimizer_model import (
    normalized_optimizer_name,
    optimizer_state_numel,
    optimizer_update_numel,
    resolved_optimizer_update_dtype,
)
from .types import (
    LoRAConfig,
    MemoryComponentBreakdown,
    ModelSpec,
    PhaseMemoryRecord,
    TrainingConfig,
)
from .utils import bytes_for_dtype, optimizer_state_in_baseline


@dataclass(frozen=True)
class InterpretableEstimateTerms:
    """Named state and workspace terms for one analytical estimate.

    Args:
        breakdown: Persistent memory components for one rank.
        forward_activation_bytes: Retained activation bytes visible at forward peak.
        backward_activation_bytes: Retained activation bytes visible at backward peak.
        forward_workspace_bytes: Temporary bytes added in forward.
        backward_workspace_bytes: Temporary bytes added in backward.
        optimizer_workspace_bytes: Temporary bytes added in optimizer step.
        runtime_allocated_support_bytes: Runtime-support bytes that stay allocated.
        runtime_reserved_only_bytes: Runtime-support bytes that affect reserved memory only.
        metadata: Named intermediate terms for reports and debugging.
        assumptions: Human-readable assumptions behind the estimate.

    Returns:
        Frozen dataclass describing the full interpretable estimate.
    """

    breakdown: MemoryComponentBreakdown
    forward_activation_bytes: int
    backward_activation_bytes: int
    forward_workspace_bytes: int
    forward_reserved_workspace_bytes: int
    backward_workspace_bytes: int
    backward_reserved_workspace_bytes: int
    optimizer_workspace_bytes: int
    optimizer_reserved_workspace_bytes: int
    runtime_allocated_support_bytes: int
    runtime_reserved_only_bytes: int
    metadata: dict[str, Any]
    assumptions: tuple[str, ...]


@dataclass(frozen=True)
class WorkspaceTerms:
    """Allocated and reserved workspace terms for one analytical estimate."""

    forward_allocated_bytes: int
    forward_reserved_bytes: int
    backward_allocated_bytes: int
    backward_reserved_bytes: int
    optimizer_allocated_bytes: int
    optimizer_reserved_bytes: int
    metadata: dict[str, int]


@dataclass(frozen=True)
class ActivationTermBreakdown:
    """Explicit retained-activation terms for one microstep.

    Args:
        hook_visible_bytes: Module-hook-visible outputs across all layers.
        base_hook_visible_bytes: Base-model visible outputs without LoRA wrappers.
        visible_propagation_bytes: Visible outputs retained for gradient propagation.
        parameter_gradient_context_bytes: Saved tensors needed for trainable weight
            gradient computation.
        resident_autograd_context_bytes: Hidden autograd context that remains
            resident into backward.
        linear_input_bytes: Inputs saved for trainable linear backward paths.
        norm_residual_bytes: Residual-stream and norm inputs retained for backward.
        mlp_intermediate_bytes: MLP nonlinearity intermediates retained for backward.
        lora_low_rank_bytes: Low-rank adapter intermediates retained for backward.
        checkpoint_boundary_bytes: Hidden-state boundaries kept under checkpointing.
        attention_score_bytes: Exact attention-score storage retained into backward.
        checkpoint_overlap_bytes: Extra checkpointed overlap between visible and
            hidden forward context in sharded expanded-query full fine-tuning.
        retained_formula_bytes: Direct analytical retained-activation total.
        retained_total_bytes: Final retained-activation term used by the estimator.

    Returns:
        Frozen dataclass describing retained-activation components.
    """

    hook_visible_bytes: int
    base_hook_visible_bytes: int
    visible_propagation_bytes: int
    parameter_gradient_context_bytes: int
    resident_autograd_context_bytes: int
    linear_input_bytes: int
    norm_residual_bytes: int
    mlp_intermediate_bytes: int
    lora_low_rank_bytes: int
    checkpoint_boundary_bytes: int
    attention_score_bytes: int
    checkpoint_overlap_bytes: int
    retained_formula_bytes: int
    retained_total_bytes: int


def estimate_lora_parameter_count(
    *,
    model_spec: ModelSpec,
    lora_config: LoRAConfig,
) -> int:
    """Estimate total LoRA parameters from targeted linear layers.

    Args:
        model_spec: Inspected model summary.
        lora_config: LoRA adapter configuration.

    Returns:
        Trainable LoRA parameter count.
    """

    total_params = 0
    for layer in model_spec.trainable_linear_layers:
        if layer.module_name.endswith(lora_config.target_modules):
            total_params += lora_config.rank * (layer.input_dim + layer.output_dim)
    return total_params


def resolved_optimizer_state_dtype(*, config: TrainingConfig) -> str:
    """Return the explicit optimizer-state dtype for the estimate.

    Args:
        config: Training configuration being estimated.

    Returns:
        Canonical dtype string used for optimizer-state tensors.
    """

    if config.optimizer_state_dtype != "auto":
        return config.optimizer_state_dtype
    if config.tuning_mode == "lora":
        return config.adapter_state_dtype()
    optimizer_name = normalized_optimizer_name(config=config)
    if config.is_zero_mode() and optimizer_name in {"adam", "adamw", "adafactor"}:
        return "fp32"
    return config.weight_dtype


def resolved_gradient_dtype(*, config: TrainingConfig) -> str:
    """Return the gradient dtype used for trainable parameters."""

    if config.tuning_mode == "lora":
        return config.adapter_gradient_dtype()
    return config.grad_dtype


def _scaled_bytes(*, value: int, factor: float) -> int:
    """Return byte counts scaled by a configurable factor."""

    return int(round(value * factor))


def _image_tokens_per_sample(*, model_spec: ModelSpec, config: TrainingConfig) -> int:
    """Return extra image tokens injected into each sample, if any."""

    if not model_spec.supports_vision_inputs() or config.vision_images_per_sample <= 0:
        return 0
    assert model_spec.vision is not None
    return config.vision_images_per_sample * model_spec.vision.image_token_count(
        image_size=config.vision_image_size,
    )


def _token_count(*, model_spec: ModelSpec, config: TrainingConfig) -> int:
    """Return text-plus-image tokens processed per transformer layer."""

    return model_spec.effective_tokens_per_layer(
        batch_size=config.micro_batch_size_per_gpu,
        sequence_length=config.max_seq_len,
        image_tokens_per_sample=_image_tokens_per_sample(
            model_spec=model_spec,
            config=config,
        ),
    )


def _raw_optimizer_state_bytes(*, model_spec: ModelSpec, config: TrainingConfig) -> int:
    """Return unsharded optimizer-state bytes before backend residency rules."""

    state_dtype = resolved_optimizer_state_dtype(config=config)
    state_numel = optimizer_state_numel(model_spec=model_spec, config=config)
    return state_numel * bytes_for_dtype(state_dtype)


def _zero2_state_is_sharded(*, config: TrainingConfig) -> bool:
    """Return whether ZeRO-2 should shard optimizer state for this optimizer."""

    optimizer_name = normalized_optimizer_name(config=config)
    if optimizer_name in config.normalized_zero_tested_optimizer_names():
        return True
    return config.zero_untested_optimizer_state_is_sharded


def _zero2_replica_floor_bytes(
    *,
    model_spec: ModelSpec,
    config: TrainingConfig,
) -> int:
    """Return fallback replicated optimizer-like bytes for untested ZeRO runs."""

    trainable_params = trainable_parameter_count(model_spec=model_spec, config=config)
    replica_elements = int(
        round(trainable_params * config.zero_untested_optimizer_replica_tensor_count)
    )
    replica_dtype = config.zero_untested_replica_dtype()
    return replica_elements * bytes_for_dtype(replica_dtype)


def _zero2_optimizer_state_residency_bytes(
    *,
    model_spec: ModelSpec,
    config: TrainingConfig,
) -> int:
    """Return persistent optimizer-state bytes resident on one ZeRO-2 rank."""

    state_bytes = _raw_optimizer_state_bytes(model_spec=model_spec, config=config)
    if state_bytes == 0:
        return 0
    if _zero2_state_is_sharded(config=config):
        return state_bytes // max(config.world_size(), 1)
    return max(
        state_bytes,
        _zero2_replica_floor_bytes(model_spec=model_spec, config=config),
    )


def _zero2_prefetch_dtype(*, model_spec: ModelSpec, config: TrainingConfig) -> str:
    """Return the dtype governing ZeRO-2 optimizer prefetch bucket sizing."""

    state_dtype = resolved_optimizer_state_dtype(config=config)
    if _zero2_state_is_sharded(config=config):
        return state_dtype
    state_bytes = _raw_optimizer_state_bytes(model_spec=model_spec, config=config)
    if state_bytes == 0:
        return state_dtype
    replica_floor = _zero2_replica_floor_bytes(model_spec=model_spec, config=config)
    if replica_floor > state_bytes:
        return config.zero_untested_replica_dtype()
    return state_dtype


def _zero2_update_is_sharded(*, config: TrainingConfig) -> bool:
    """Return whether ZeRO-2 should shard optimizer update scratch buffers."""

    optimizer_name = normalized_optimizer_name(config=config)
    if optimizer_name in config.normalized_zero_tested_optimizer_names():
        return True
    return config.zero_untested_optimizer_update_is_sharded


def _zero2_update_replica_floor_bytes(
    *,
    model_spec: ModelSpec,
    config: TrainingConfig,
) -> int:
    """Return fallback replicated optimizer-update bytes for ZeRO-2."""

    trainable_params = trainable_parameter_count(model_spec=model_spec, config=config)
    replica_elements = int(
        round(
            trainable_params
            * config.zero_untested_optimizer_update_replica_tensor_count
        )
    )
    return replica_elements * bytes_for_dtype(config.zero_untested_update_dtype())


def _zero2_optimizer_update_residency_bytes(
    *,
    model_spec: ModelSpec,
    config: TrainingConfig,
    optimizer_update_bytes: int,
) -> int:
    """Return ZeRO-2 optimizer-update bytes resident on one rank."""

    if optimizer_update_bytes == 0:
        return 0
    if _zero2_update_is_sharded(config=config):
        return optimizer_update_bytes // max(config.world_size(), 1)
    return max(
        optimizer_update_bytes,
        _zero2_update_replica_floor_bytes(model_spec=model_spec, config=config),
    )


def trainable_parameter_count(*, model_spec: ModelSpec, config: TrainingConfig) -> int:
    """Return the trainable parameter count for the selected tuning mode.

    Args:
        model_spec: Inspected model summary.
        config: Training configuration.

    Returns:
        Trainable parameter count.
    """

    if config.tuning_mode == "full_ft":
        return model_spec.total_params
    assert (
        config.tuning_mode == "lora"
    ), f"Unsupported tuning mode: {config.tuning_mode}"
    assert config.lora is not None, "LoRA config is required for tuning_mode='lora'."
    return estimate_lora_parameter_count(
        model_spec=model_spec,
        lora_config=config.lora,
    )


def _first_block_layers(*, model_spec: ModelSpec) -> tuple:
    """Return linear layers belonging to the first transformer block."""

    block_layers = tuple(
        layer
        for layer in model_spec.trainable_linear_layers
        if "layers.0." in layer.module_name or ".h.0." in layer.module_name
    )
    if block_layers:
        return block_layers
    return model_spec.trainable_linear_layers


def _block_layer_by_suffix(
    *,
    model_spec: ModelSpec,
    suffix: str,
):
    """Return the first-block linear layer matching the given suffix."""

    for layer in _first_block_layers(model_spec=model_spec):
        if layer.module_name.endswith(suffix):
            return layer
    return None


def _linear_output_elements_per_block(*, model_spec: ModelSpec) -> int:
    """Return per-token hook-visible linear output elements for one block."""

    return sum(layer.output_dim for layer in _first_block_layers(model_spec=model_spec))


def _aux_output_elements_per_block(*, model_spec: ModelSpec) -> int:
    """Return extra hook-visible non-linear output elements for one block."""

    q_layer = _block_layer_by_suffix(model_spec=model_spec, suffix="q_proj")
    k_layer = _block_layer_by_suffix(model_spec=model_spec, suffix="k_proj")
    if q_layer is None or k_layer is None:
        return (5 * model_spec.hidden_size) + model_spec.intermediate_size
    return (
        q_layer.output_dim
        + k_layer.output_dim
        + (5 * model_spec.hidden_size)
        + model_spec.intermediate_size
    )


def _lora_hook_extra_elements_per_block(
    *,
    model_spec: ModelSpec,
    lora_config: LoRAConfig,
) -> int:
    """Return PEFT hook-visible extra outputs introduced by LoRA wrappers."""

    extra_elements = 0
    for layer in _first_block_layers(model_spec=model_spec):
        if layer.module_name.endswith(lora_config.target_modules):
            extra_elements += (
                (3 * layer.output_dim) + (2 * layer.input_dim) + lora_config.rank
            )
    return extra_elements


def _hook_visible_activation_bytes(
    *,
    model_spec: ModelSpec,
    config: TrainingConfig,
) -> int:
    """Return module-hook-visible activation bytes for one microstep."""

    token_count = _token_count(model_spec=model_spec, config=config)
    saved_elements = _linear_output_elements_per_block(model_spec=model_spec)
    saved_elements += _aux_output_elements_per_block(model_spec=model_spec)
    if config.tuning_mode == "lora":
        assert (
            config.lora is not None
        ), "LoRA config is required for tuning_mode='lora'."
        saved_elements += _lora_hook_extra_elements_per_block(
            model_spec=model_spec,
            lora_config=config.lora,
        )
    return (
        token_count
        * model_spec.num_layers
        * saved_elements
        * bytes_for_dtype(config.weight_dtype)
    )


def _base_hook_visible_activation_bytes(
    *,
    model_spec: ModelSpec,
    config: TrainingConfig,
) -> int:
    """Return hook-visible bytes from the base model without LoRA wrapper extras."""

    token_count = _token_count(model_spec=model_spec, config=config)
    saved_elements = _linear_output_elements_per_block(model_spec=model_spec)
    saved_elements += _aux_output_elements_per_block(model_spec=model_spec)
    return (
        token_count
        * model_spec.num_layers
        * saved_elements
        * bytes_for_dtype(config.weight_dtype)
    )


def _retained_lora_visible_extra_bytes(
    *,
    model_spec: ModelSpec,
    config: TrainingConfig,
) -> int:
    """Return retained LoRA-only hook-visible bytes beyond the base model floor.

    Args:
        model_spec: Inspected model summary.
        config: Training configuration being estimated.

    Returns:
        Retained LoRA-only visible bytes scaled by the backend-specific
        retention fraction.
    """

    if config.tuning_mode != "lora":
        return 0
    hook_visible_bytes = _hook_visible_activation_bytes(
        model_spec=model_spec,
        config=config,
    )
    base_visible_bytes = _base_hook_visible_activation_bytes(
        model_spec=model_spec,
        config=config,
    )
    extra_visible_bytes = max(0, hook_visible_bytes - base_visible_bytes)
    return _scaled_bytes(
        value=extra_visible_bytes,
        factor=config.lora_visible_activation_extra_fraction(),
    )


def _tokenwise_retained_activation_bytes(
    *,
    model_spec: ModelSpec,
    config: TrainingConfig,
) -> int:
    """Return retained tokenwise activations excluding attention-score storage."""

    token_count = _token_count(model_spec=model_spec, config=config)
    retained_elements = (2 * model_spec.hidden_size) + model_spec.intermediate_size
    if config.tuning_mode == "lora" and config.lora is not None:
        retained_elements += config.lora.rank * len(config.lora.target_modules)
    return (
        token_count
        * model_spec.num_layers
        * retained_elements
        * bytes_for_dtype(config.weight_dtype)
    )


def _linear_input_elements_per_block(*, model_spec: ModelSpec) -> int:
    """Return per-token trainable linear input elements for one block."""

    return sum(layer.input_dim for layer in _first_block_layers(model_spec=model_spec))


def _retained_linear_input_elements_per_block(
    *,
    model_spec: ModelSpec,
    config: TrainingConfig,
) -> int:
    """Return per-token linear inputs that must survive for backward.

    Args:
        model_spec: Inspected model summary.
        config: Training configuration being estimated.

    Returns:
        Linear input elements retained for trainable weight backward paths.
    """

    if config.tuning_mode == "full_ft":
        return _linear_input_elements_per_block(model_spec=model_spec)
    assert config.lora is not None, "LoRA config is required for tuning_mode='lora'."
    return sum(
        layer.input_dim
        for layer in _first_block_layers(model_spec=model_spec)
        if layer.module_name.endswith(config.lora.target_modules)
    )


def _visible_propagation_bytes(
    *,
    model_spec: ModelSpec,
    hook_visible_bytes: int,
    base_hook_visible_bytes: int,
    config: TrainingConfig,
) -> int:
    """Return visible outputs that must remain for backward propagation.

    Args:
        hook_visible_bytes: Hook-visible outputs including LoRA wrapper extras.
        base_hook_visible_bytes: Base-model hook-visible outputs.
        config: Training configuration being estimated.

    Returns:
        Visible activation bytes that remain relevant for backward propagation.
    """

    if config.tuning_mode == "lora":
        if config.is_zero_mode():
            if config.gradient_checkpointing:
                return hook_visible_bytes
            if (
                _expanded_forward_autograd_context_bytes(
                    model_spec=model_spec,
                    config=config,
                )
                > 0
            ):
                return hook_visible_bytes
            extra_visible_bytes = max(
                0,
                hook_visible_bytes - base_hook_visible_bytes,
            )
            return base_hook_visible_bytes + _scaled_bytes(
                value=extra_visible_bytes,
                factor=config.lora_visible_activation_extra_fraction(),
            )
        return hook_visible_bytes
    return base_hook_visible_bytes


def _norm_residual_elements_per_block(*, model_spec: ModelSpec) -> int:
    """Return residual-stream elements retained outside trainable matmul inputs."""

    return 2 * model_spec.hidden_size


def _lora_low_rank_elements_per_block(
    *,
    model_spec: ModelSpec,
    lora_config: LoRAConfig | None,
) -> int:
    """Return retained low-rank adapter intermediates for one block."""

    if lora_config is None:
        return 0
    target_count = sum(
        1
        for layer in _first_block_layers(model_spec=model_spec)
        if layer.module_name.endswith(lora_config.target_modules)
    )
    return target_count * lora_config.rank


def _checkpoint_boundary_bytes(
    *,
    model_spec: ModelSpec,
    config: TrainingConfig,
) -> int:
    """Return hidden-state boundaries retained under checkpointing."""

    token_count = _token_count(model_spec=model_spec, config=config)
    boundary_elements = (
        token_count * (model_spec.num_layers + 1) * model_spec.hidden_size
    )
    return boundary_elements * bytes_for_dtype(config.weight_dtype)


def _expanded_attention_dimensions(
    *,
    model_spec: ModelSpec,
) -> tuple[int, int, int, int, int]:
    """Return expanded attention-path dimensions beyond the hidden-width baseline."""

    q_layer = _block_layer_by_suffix(model_spec=model_spec, suffix="q_proj")
    k_layer = _block_layer_by_suffix(model_spec=model_spec, suffix="k_proj")
    v_layer = _block_layer_by_suffix(model_spec=model_spec, suffix="v_proj")
    o_layer = _block_layer_by_suffix(model_spec=model_spec, suffix="o_proj")
    hidden_size = model_spec.hidden_size
    query_elements = q_layer.output_dim if q_layer is not None else hidden_size
    if query_elements <= hidden_size:
        return 0, 0, 0, 0, 0
    key_elements = k_layer.output_dim if k_layer is not None else hidden_size
    value_elements = v_layer.output_dim if v_layer is not None else hidden_size
    output_input_elements = o_layer.input_dim if o_layer is not None else hidden_size
    output_output_elements = o_layer.output_dim if o_layer is not None else hidden_size
    return (
        query_elements - hidden_size,
        key_elements,
        value_elements,
        output_input_elements,
        output_output_elements,
    )


def _expanded_autograd_context_elements_per_block(
    *,
    model_spec: ModelSpec,
    config: TrainingConfig,
) -> int:
    """Return extra autograd-context elements caused by expanded query paths."""

    (
        expanded_query_elements,
        key_elements,
        value_elements,
        output_input_elements,
        output_output_elements,
    ) = _expanded_attention_dimensions(model_spec=model_spec)
    if expanded_query_elements == 0:
        return 0
    return (
        _scaled_bytes(
            value=expanded_query_elements,
            factor=config.forward_autograd_expanded_query_context_copies,
        )
        + _scaled_bytes(
            value=key_elements,
            factor=config.forward_autograd_expanded_key_context_copies,
        )
        + _scaled_bytes(
            value=value_elements,
            factor=config.forward_autograd_expanded_value_context_copies,
        )
        + _scaled_bytes(
            value=output_input_elements,
            factor=config.forward_autograd_expanded_output_input_context_copies,
        )
        + _scaled_bytes(
            value=output_output_elements,
            factor=config.forward_autograd_expanded_output_output_context_copies,
        )
        + _scaled_bytes(
            value=model_spec.hidden_size,
            factor=config.forward_autograd_expanded_hidden_context_copies,
        )
    )


def _autograd_context_elements_per_block(
    *,
    model_spec: ModelSpec,
    config: TrainingConfig,
) -> int:
    """Return non-hook autograd context elements that survive to backward."""

    q_layer = _block_layer_by_suffix(model_spec=model_spec, suffix="q_proj")
    k_layer = _block_layer_by_suffix(model_spec=model_spec, suffix="k_proj")
    v_layer = _block_layer_by_suffix(model_spec=model_spec, suffix="v_proj")
    o_layer = _block_layer_by_suffix(model_spec=model_spec, suffix="o_proj")
    hidden_size = model_spec.hidden_size
    query_elements = q_layer.output_dim if q_layer is not None else hidden_size
    key_elements = k_layer.output_dim if k_layer is not None else hidden_size
    value_elements = v_layer.output_dim if v_layer is not None else hidden_size
    output_input_elements = o_layer.input_dim if o_layer is not None else hidden_size
    output_output_elements = o_layer.output_dim if o_layer is not None else hidden_size
    base_elements = _linear_input_elements_per_block(model_spec=model_spec)
    base_elements += _scaled_bytes(
        value=query_elements,
        factor=config.forward_autograd_query_context_copies,
    )
    base_elements += _scaled_bytes(
        value=key_elements,
        factor=config.forward_autograd_key_context_copies,
    )
    base_elements += _scaled_bytes(
        value=hidden_size,
        factor=config.forward_autograd_hidden_context_copies,
    )
    base_elements += _expanded_autograd_context_elements_per_block(
        model_spec=model_spec,
        config=config,
    )
    return base_elements


def _forward_autograd_context_bytes(
    *,
    model_spec: ModelSpec,
    config: TrainingConfig,
) -> int:
    """Return end-of-forward autograd context bytes not visible to hooks."""

    token_count = _token_count(model_spec=model_spec, config=config)
    return (
        token_count
        * model_spec.num_layers
        * _autograd_context_elements_per_block(
            model_spec=model_spec,
            config=config,
        )
        * bytes_for_dtype(config.weight_dtype)
    )


def _expanded_forward_autograd_context_bytes(
    *,
    model_spec: ModelSpec,
    config: TrainingConfig,
) -> int:
    """Return forward-end autograd bytes unique to expanded-query attention."""

    token_count = _token_count(model_spec=model_spec, config=config)
    return (
        token_count
        * model_spec.num_layers
        * _expanded_autograd_context_elements_per_block(
            model_spec=model_spec,
            config=config,
        )
        * bytes_for_dtype(config.weight_dtype)
    )


def _resident_forward_autograd_context_bytes(
    *,
    model_spec: ModelSpec,
    config: TrainingConfig,
) -> int:
    """Return hidden forward autograd context that remains resident at peak.

    Args:
        model_spec: Inspected model summary.
        config: Training configuration being estimated.

    Returns:
        Resident hidden autograd-context bytes after applying the mode-specific
        retention fraction.
    """

    forward_autograd_context_bytes = _forward_autograd_context_bytes(
        model_spec=model_spec,
        config=config,
    )
    if _uses_expanded_query_full_ft_residency(
        model_spec=model_spec,
        config=config,
    ):
        return forward_autograd_context_bytes
    return _scaled_bytes(
        value=forward_autograd_context_bytes,
        factor=config.retained_forward_autograd_context_fraction(),
    )


def _uses_expanded_query_full_ft_residency(
    *,
    model_spec: ModelSpec,
    config: TrainingConfig,
) -> bool:
    """Return whether full FT should retain explicit forward autograd context.

    Args:
        model_spec: Inspected model summary.
        config: Training configuration being estimated.

    Returns:
        True when expanded-query attention introduces a hidden autograd object
        that should remain distinct from visible propagation at peak.
    """

    if config.tuning_mode != "full_ft" or config.is_zero_mode():
        return False
    return (
        _expanded_forward_autograd_context_bytes(
            model_spec=model_spec,
            config=config,
        )
        > 0
    )


def _uses_checkpointed_nonexpanded_sharded_full_ft(
    *,
    model_spec: ModelSpec,
    config: TrainingConfig,
) -> bool:
    """Return whether checkpointed sharded full FT should use phase activations.

    Args:
        model_spec: Inspected model summary.
        config: Training configuration being estimated.

    Returns:
        True when checkpointed sharded full fine-tuning on a non-expanded model
        should split forward and backward retained activations.
    """

    if (
        not config.gradient_checkpointing
        or config.tuning_mode != "full_ft"
        or not config.is_zero_mode()
    ):
        return False
    return (
        _expanded_forward_autograd_context_bytes(
            model_spec=model_spec,
            config=config,
        )
        == 0
    )


def _expanded_query_retained_activation_bytes(
    *,
    visible_propagation_bytes: int,
    resident_autograd_context_bytes: int,
    checkpoint_boundary_bytes: int,
    attention_score_bytes: int,
    config: TrainingConfig,
) -> int:
    """Return retained activations for expanded-query full fine-tuning.

    Args:
        visible_propagation_bytes: Hook-visible outputs retained for backward.
        resident_autograd_context_bytes: Hidden forward autograd context that
            remains resident at peak.
        checkpoint_boundary_bytes: Hidden-state boundaries kept under
            checkpointing.
        attention_score_bytes: Retained attention-score storage.
        config: Training configuration being estimated.

    Returns:
        Retained activation bytes when visible propagation and hidden autograd
        context must coexist instead of collapsing to a max.
    """

    visible_floor_bytes = visible_propagation_bytes
    if config.gradient_checkpointing:
        visible_floor_bytes = max(visible_floor_bytes, checkpoint_boundary_bytes)
    return visible_floor_bytes + resident_autograd_context_bytes + attention_score_bytes


def _checkpoint_expanded_query_overlap_bytes(
    *,
    model_spec: ModelSpec,
    config: TrainingConfig,
    visible_propagation_bytes: int,
    resident_autograd_context_bytes: int,
) -> int:
    """Return checkpointed overlap between visible and hidden forward context.

    Args:
        model_spec: Inspected model summary.
        config: Training configuration being estimated.
        visible_propagation_bytes: Visible forward outputs retained for backward.
        resident_autograd_context_bytes: Hidden forward context still resident.

    Returns:
        Extra overlap bytes kept when checkpointed sharded full fine-tuning on
        expanded-query models retains both visible and hidden forward state.
    """

    if (
        config.tuning_mode != "full_ft"
        or not config.is_zero_mode()
        or _expanded_forward_autograd_context_bytes(
            model_spec=model_spec,
            config=config,
        )
        == 0
    ):
        return 0
    return _scaled_bytes(
        value=min(
            visible_propagation_bytes,
            resident_autograd_context_bytes,
        ),
        factor=config.checkpoint_expanded_query_overlap_fraction(),
    )


def _checkpointed_retained_activation_bytes(
    *,
    model_spec: ModelSpec,
    config: TrainingConfig,
    visible_propagation_bytes: int,
    resident_autograd_context_bytes: int,
    checkpoint_boundary_bytes: int,
    attention_score_bytes: int,
    lora_low_rank_bytes: int,
) -> tuple[int, int, int]:
    """Return checkpointed retained-activation totals and overlap term.

    Args:
        model_spec: Inspected model summary.
        config: Training configuration being estimated.
        visible_propagation_bytes: Visible outputs retained for backward.
        resident_autograd_context_bytes: Hidden forward context still resident.
        checkpoint_boundary_bytes: Boundary hidden states retained by checkpointing.
        attention_score_bytes: Retained attention-score storage.
        lora_low_rank_bytes: LoRA low-rank retained intermediates.

    Returns:
        Tuple of `(retained_formula_bytes, retained_total_bytes,
        checkpoint_overlap_bytes)`.
    """

    checkpoint_overlap_bytes = _checkpoint_expanded_query_overlap_bytes(
        model_spec=model_spec,
        config=config,
        visible_propagation_bytes=visible_propagation_bytes,
        resident_autograd_context_bytes=resident_autograd_context_bytes,
    )
    retained_formula_bytes = (
        checkpoint_boundary_bytes
        + resident_autograd_context_bytes
        + attention_score_bytes
        + lora_low_rank_bytes
        + checkpoint_overlap_bytes
    )
    retained_total_bytes = (
        max(
            visible_propagation_bytes,
            checkpoint_boundary_bytes + resident_autograd_context_bytes,
        )
        + lora_low_rank_bytes
        + attention_score_bytes
        + checkpoint_overlap_bytes
    )
    return retained_formula_bytes, retained_total_bytes, checkpoint_overlap_bytes


def _checkpointed_nonexpanded_phase_activation_bytes(
    *,
    model_spec: ModelSpec,
    config: TrainingConfig,
    linear_input_bytes: int,
    norm_residual_bytes: int,
    checkpoint_boundary_bytes: int,
    attention_score_bytes: int,
) -> tuple[int, int]:
    """Return forward/backward retained activations for checkpointed non-expanded FT.

    Args:
        model_spec: Inspected model summary.
        config: Training configuration being estimated.
        linear_input_bytes: Saved trainable linear input bytes.
        norm_residual_bytes: Residual-stream bytes retained for backward.
        checkpoint_boundary_bytes: Boundary hidden states kept by checkpointing.
        attention_score_bytes: Retained attention-score storage.

    Returns:
        Tuple of `(forward_activation_bytes, backward_activation_bytes)`.
    """

    if not _uses_checkpointed_nonexpanded_sharded_full_ft(
        model_spec=model_spec,
        config=config,
    ):
        fallback_activation_bytes = checkpoint_boundary_bytes + attention_score_bytes
        return fallback_activation_bytes, fallback_activation_bytes
    forward_activation_bytes = checkpoint_boundary_bytes + attention_score_bytes
    backward_activation_bytes = (
        linear_input_bytes
        + norm_residual_bytes
        + checkpoint_boundary_bytes
        + attention_score_bytes
    )
    return forward_activation_bytes, backward_activation_bytes


def _uses_checkpointed_sharded_lora_phase_split(
    *,
    model_spec: ModelSpec,
    config: TrainingConfig,
) -> bool:
    """Return whether checkpointed sharded LoRA should use phase-specific activations.

    Args:
        model_spec: Inspected model summary.
        config: Training configuration being estimated.

    Returns:
        ``True`` when checkpointed sharded LoRA runs on a non-expanded model
        where forward-visible activations should be treated as rematerializable
        boundary state rather than fully retained propagation state.
    """

    return (
        config.tuning_mode == "lora"
        and config.gradient_checkpointing
        and config.is_zero_mode()
        and _expanded_forward_autograd_context_bytes(
            model_spec=model_spec,
            config=config,
        )
        == 0
    )


def _checkpointed_lora_phase_activation_bytes(
    *,
    parameter_gradient_context_bytes: int,
    checkpoint_boundary_bytes: int,
    attention_score_bytes: int,
    lora_low_rank_bytes: int,
) -> tuple[int, int]:
    """Return forward/backward retained activations for checkpointed sharded LoRA.

    Args:
        parameter_gradient_context_bytes: Saved tensors required for adapter
            gradient formation during backward recompute.
        checkpoint_boundary_bytes: Hidden-state boundaries kept by checkpointing.
        attention_score_bytes: Retained attention-score storage.
        lora_low_rank_bytes: Low-rank adapter intermediates retained for backward.

    Returns:
        Tuple of `(forward_activation_bytes, backward_activation_bytes)`.

    Example:
        Checkpointed ZeRO LoRA keeps boundary state through forward and only
        adapter-relevant saved tensors at the backward peak.
    """

    forward_activation_bytes = checkpoint_boundary_bytes + attention_score_bytes
    backward_activation_bytes = (
        parameter_gradient_context_bytes
        + checkpoint_boundary_bytes
        + attention_score_bytes
        + lora_low_rank_bytes
    )
    return forward_activation_bytes, backward_activation_bytes


def _activation_term_breakdown(
    *,
    model_spec: ModelSpec,
    config: TrainingConfig,
) -> ActivationTermBreakdown:
    """Return an explicit retained-activation decomposition.

    Args:
        model_spec: Inspected model summary.
        config: Training configuration being estimated.

    Returns:
        Named retained-activation terms built from tensor classes rather than
        only from aggregate scaling rules.
    """

    token_count = _token_count(model_spec=model_spec, config=config)
    element_bytes = bytes_for_dtype(config.weight_dtype)
    hook_visible_bytes = _hook_visible_activation_bytes(
        model_spec=model_spec,
        config=config,
    )
    base_hook_visible_bytes = _base_hook_visible_activation_bytes(
        model_spec=model_spec,
        config=config,
    )
    linear_input_bytes = (
        token_count
        * model_spec.num_layers
        * _retained_linear_input_elements_per_block(
            model_spec=model_spec,
            config=config,
        )
        * element_bytes
    )
    norm_residual_bytes = (
        token_count
        * model_spec.num_layers
        * _norm_residual_elements_per_block(model_spec=model_spec)
        * element_bytes
    )
    mlp_intermediate_bytes = (
        token_count
        * model_spec.num_layers
        * model_spec.intermediate_size
        * element_bytes
    )
    lora_low_rank_bytes = (
        token_count
        * model_spec.num_layers
        * _lora_low_rank_elements_per_block(
            model_spec=model_spec,
            lora_config=config.lora,
        )
        * element_bytes
    )
    checkpoint_boundary_bytes = _checkpoint_boundary_bytes(
        model_spec=model_spec,
        config=config,
    )
    attention_score_bytes = _attention_score_bytes(
        model_spec=model_spec,
        config=config,
    )
    resident_autograd_context_bytes = _resident_forward_autograd_context_bytes(
        model_spec=model_spec,
        config=config,
    )
    visible_propagation_bytes = _visible_propagation_bytes(
        model_spec=model_spec,
        hook_visible_bytes=hook_visible_bytes,
        base_hook_visible_bytes=base_hook_visible_bytes,
        config=config,
    )
    parameter_gradient_context_bytes = linear_input_bytes + mlp_intermediate_bytes
    if _uses_expanded_query_full_ft_residency(
        model_spec=model_spec,
        config=config,
    ):
        retained_total_bytes = _expanded_query_retained_activation_bytes(
            visible_propagation_bytes=visible_propagation_bytes,
            resident_autograd_context_bytes=resident_autograd_context_bytes,
            checkpoint_boundary_bytes=checkpoint_boundary_bytes,
            attention_score_bytes=attention_score_bytes,
            config=config,
        )
        retained_formula_bytes = retained_total_bytes
        return ActivationTermBreakdown(
            hook_visible_bytes=hook_visible_bytes,
            base_hook_visible_bytes=base_hook_visible_bytes,
            visible_propagation_bytes=visible_propagation_bytes,
            parameter_gradient_context_bytes=parameter_gradient_context_bytes,
            resident_autograd_context_bytes=resident_autograd_context_bytes,
            linear_input_bytes=linear_input_bytes,
            norm_residual_bytes=norm_residual_bytes,
            mlp_intermediate_bytes=mlp_intermediate_bytes,
            lora_low_rank_bytes=lora_low_rank_bytes,
            checkpoint_boundary_bytes=checkpoint_boundary_bytes,
            attention_score_bytes=attention_score_bytes,
            checkpoint_overlap_bytes=0,
            retained_formula_bytes=retained_formula_bytes,
            retained_total_bytes=retained_total_bytes,
        )
    checkpoint_overlap_bytes = 0
    if config.gradient_checkpointing:
        (
            retained_formula_bytes,
            retained_total_bytes,
            checkpoint_overlap_bytes,
        ) = _checkpointed_retained_activation_bytes(
            model_spec=model_spec,
            config=config,
            visible_propagation_bytes=visible_propagation_bytes,
            resident_autograd_context_bytes=resident_autograd_context_bytes,
            checkpoint_boundary_bytes=checkpoint_boundary_bytes,
            attention_score_bytes=attention_score_bytes,
            lora_low_rank_bytes=lora_low_rank_bytes,
        )
    else:
        retained_formula_bytes = (
            parameter_gradient_context_bytes
            + resident_autograd_context_bytes
            + norm_residual_bytes
            + lora_low_rank_bytes
            + attention_score_bytes
        )
        retained_total_bytes = max(
            visible_propagation_bytes,
            parameter_gradient_context_bytes + resident_autograd_context_bytes,
        )
        retained_total_bytes += norm_residual_bytes + lora_low_rank_bytes
        retained_total_bytes += attention_score_bytes
    return ActivationTermBreakdown(
        hook_visible_bytes=hook_visible_bytes,
        base_hook_visible_bytes=base_hook_visible_bytes,
        visible_propagation_bytes=visible_propagation_bytes,
        parameter_gradient_context_bytes=parameter_gradient_context_bytes,
        resident_autograd_context_bytes=resident_autograd_context_bytes,
        linear_input_bytes=linear_input_bytes,
        norm_residual_bytes=norm_residual_bytes,
        mlp_intermediate_bytes=mlp_intermediate_bytes,
        lora_low_rank_bytes=lora_low_rank_bytes,
        checkpoint_boundary_bytes=checkpoint_boundary_bytes,
        attention_score_bytes=attention_score_bytes,
        checkpoint_overlap_bytes=checkpoint_overlap_bytes,
        retained_formula_bytes=retained_formula_bytes,
        retained_total_bytes=retained_total_bytes,
    )


def _checkpoint_retained_activation_bytes(
    *,
    model_spec: ModelSpec,
    config: TrainingConfig,
) -> int:
    """Return the tokenwise retained-activation term under checkpointing."""

    token_count = _token_count(model_spec=model_spec, config=config)
    checkpoint_elements = (
        token_count * (model_spec.num_layers + 1) * model_spec.hidden_size
    )
    checkpoint_bytes = checkpoint_elements * bytes_for_dtype(config.weight_dtype)
    return _scaled_bytes(
        value=checkpoint_bytes,
        factor=config.gradient_checkpointing_activation_factor,
    )


def _logits_elements(*, model_spec: ModelSpec, config: TrainingConfig) -> int:
    """Return the number of logits elements materialized in one microstep."""

    token_count = _token_count(model_spec=model_spec, config=config)
    return token_count * model_spec.vocab_size


def _logits_workspace_bytes(*, model_spec: ModelSpec, config: TrainingConfig) -> int:
    """Return explicit logits/loss materialization bytes for one microstep."""

    return _logits_elements(
        model_spec=model_spec,
        config=config,
    ) * bytes_for_dtype(config.loss_output_resolved_dtype())


def _loss_output_bytes(
    *,
    model_spec: ModelSpec,
    config: TrainingConfig,
    activation_bytes: int,
) -> int:
    """Return model-output bytes that can persist beyond backward."""

    output_bytes = _scaled_bytes(
        value=_logits_elements(
            model_spec=model_spec,
            config=config,
        )
        * bytes_for_dtype(config.loss_output_resolved_dtype()),
        factor=config.loss_output_logits_fraction,
    )
    return min(output_bytes, activation_bytes)


def _head_dim(*, model_spec: ModelSpec) -> int:
    """Return the per-head hidden dimension."""

    return model_spec.hidden_size // model_spec.num_attention_heads


def _attention_path_elements_per_block(*, model_spec: ModelSpec) -> int:
    """Return per-token attention-path elements used for tiled workspace."""

    q_layer = _block_layer_by_suffix(model_spec=model_spec, suffix="q_proj")
    k_layer = _block_layer_by_suffix(model_spec=model_spec, suffix="k_proj")
    v_layer = _block_layer_by_suffix(model_spec=model_spec, suffix="v_proj")
    o_layer = _block_layer_by_suffix(model_spec=model_spec, suffix="o_proj")
    if any(layer is None for layer in (q_layer, k_layer, v_layer, o_layer)):
        return 3 * model_spec.hidden_size
    assert q_layer is not None
    assert k_layer is not None
    assert v_layer is not None
    assert o_layer is not None
    return (
        q_layer.output_dim
        + k_layer.output_dim
        + v_layer.output_dim
        + o_layer.output_dim
    )


def _attention_projection_workspace_bytes(
    *,
    model_spec: ModelSpec,
    config: TrainingConfig,
) -> int:
    """Return attention projection-path workspace common across backends."""

    token_count = _token_count(model_spec=model_spec, config=config)
    return (
        token_count
        * _attention_path_elements_per_block(model_spec=model_spec)
        * bytes_for_dtype(config.weight_dtype)
    )


def _effective_attention_backend(
    *,
    model_spec: ModelSpec,
    config: TrainingConfig,
) -> str:
    """Return the effective runtime attention backend for estimation."""

    backend = config.normalized_attention_backend()
    if backend != "standard":
        return backend
    if model_type_uses_sdpa(model_type=model_spec.model_type):
        return "sdpa"
    return "eager"


def _attention_score_bytes(
    *,
    model_spec: ModelSpec,
    config: TrainingConfig,
) -> int:
    """Return retained attention-score bytes that survive into backward."""

    batch_size = config.micro_batch_size_per_gpu
    seq_len = config.max_seq_len
    layers = model_spec.num_layers
    heads = model_spec.num_attention_heads
    score_bytes = bytes_for_dtype(config.attention_score_dtype)
    backend = _effective_attention_backend(model_spec=model_spec, config=config)
    if backend in {"standard", "eager"}:
        score_elements = batch_size * layers * heads * seq_len * seq_len
        return score_elements * score_bytes
    return 0


def _attention_temporary_score_bytes(
    *,
    model_spec: ModelSpec,
    config: TrainingConfig,
) -> int:
    """Return temporary tiled score workspace for non-eager attention kernels."""

    batch_size = config.micro_batch_size_per_gpu
    seq_len = config.max_seq_len
    heads = model_spec.num_attention_heads
    score_bytes = bytes_for_dtype(config.attention_score_dtype)
    backend = _effective_attention_backend(model_spec=model_spec, config=config)
    if backend == "sdpa":
        tile_size = min(seq_len, config.sdpa_attention_tile_size)
        return batch_size * heads * seq_len * tile_size * score_bytes
    if backend in {"flash", "flash2", "flashattention2", "flash_attention_2"}:
        tile_size = min(seq_len, config.flash_attention_tile_size)
        return batch_size * heads * seq_len * tile_size * score_bytes
    return 0


def _attention_temporary_workspace_bytes(
    *,
    model_spec: ModelSpec,
    config: TrainingConfig,
) -> int:
    """Return temporary attention workspace bytes for the selected backend."""

    projection_bytes = _attention_projection_workspace_bytes(
        model_spec=model_spec,
        config=config,
    )
    return projection_bytes + _attention_temporary_score_bytes(
        model_spec=model_spec,
        config=config,
    )


def _attention_backward_workspace_bytes(
    *,
    model_spec: ModelSpec,
    config: TrainingConfig,
) -> int:
    """Return one-layer attention scratch visible during backward.

    Args:
        model_spec: Inspected model summary.
        config: Training configuration being estimated.

    Returns:
        Backward attention scratch sized as one layer of score-state pressure.
    """

    batch_size = config.micro_batch_size_per_gpu
    seq_len = config.max_seq_len
    heads = model_spec.num_attention_heads
    score_bytes = bytes_for_dtype(config.attention_score_dtype)
    backend = _effective_attention_backend(model_spec=model_spec, config=config)
    if backend in {"standard", "eager"}:
        return batch_size * heads * seq_len * seq_len * score_bytes
    return _attention_temporary_score_bytes(
        model_spec=model_spec,
        config=config,
    )


def _retained_activation_bytes(
    *,
    model_spec: ModelSpec,
    config: TrainingConfig,
) -> int:
    """Return forward-end resident activations from explicit tensor classes."""

    return _activation_term_breakdown(
        model_spec=model_spec,
        config=config,
    ).retained_total_bytes


def _checkpoint_recompute_bytes(
    *,
    model_spec: ModelSpec,
    config: TrainingConfig,
) -> int:
    """Return one-layer recompute bytes introduced by gradient checkpointing."""

    if not config.gradient_checkpointing:
        return 0
    token_count = _token_count(model_spec=model_spec, config=config)
    layer_output_elements = _linear_output_elements_per_block(model_spec=model_spec)
    layer_output_elements += _aux_output_elements_per_block(model_spec=model_spec)
    if config.tuning_mode == "lora":
        assert (
            config.lora is not None
        ), "LoRA config is required for tuning_mode='lora'."
        layer_output_elements += _lora_hook_extra_elements_per_block(
            model_spec=model_spec,
            lora_config=config.lora,
        )
    layer_recompute_bytes = (
        token_count * layer_output_elements * bytes_for_dtype(config.weight_dtype)
    )
    attention_layer_bytes = _attention_temporary_workspace_bytes(
        model_spec=model_spec,
        config=config,
    )
    recompute_bytes = layer_recompute_bytes + attention_layer_bytes
    return _scaled_bytes(
        value=recompute_bytes,
        factor=config.gradient_checkpointing_backward_workspace_factor,
    )


def _backward_workspace_total_bytes(
    *, backward_bytes: int, checkpoint_recompute_bytes: int
) -> int:
    """Return the visible backward workspace after checkpointing overlap.

    Args:
        backward_bytes: Base backward workspace already present without recompute.
        checkpoint_recompute_bytes: Extra recompute workspace introduced by
            checkpointing.

    Returns:
        Backward workspace visible at peak, assuming recompute reuses the same
        allocator pool instead of summing with the base backward term.
    """

    return max(backward_bytes, checkpoint_recompute_bytes)


def _combined_reserved_transient_bytes(
    *, carried_reserved_bytes: int, transient_reserved_bytes: int
) -> int:
    """Return reserved transients visible after allocator-pool reuse.

    Args:
        carried_reserved_bytes: Reserved bytes carried from the previous phase.
        transient_reserved_bytes: Local reserved bytes created in the current
            phase.

    Returns:
        The larger of the carried pool and the new transient demand.
    """

    return max(carried_reserved_bytes, transient_reserved_bytes)


def _stacked_reserved_transient_bytes(
    *,
    carried_reserved_bytes: int,
    transient_reserved_bytes: int,
    stack_fraction: float,
) -> int:
    """Return reserved bytes when carryover and local pools partially stack.

    Args:
        carried_reserved_bytes: Reserved bytes carried from the previous phase.
        transient_reserved_bytes: Local reserved bytes created in the current
            phase.
        stack_fraction: Fraction of the smaller pool that stacks instead of
            reusing the larger allocator pool.

    Returns:
        Combined reserved bytes after accounting for pool reuse and stacking.
    """

    overlap_free_bytes = max(carried_reserved_bytes, transient_reserved_bytes)
    stacked_bytes = _scaled_bytes(
        value=min(carried_reserved_bytes, transient_reserved_bytes),
        factor=stack_fraction,
    )
    return overlap_free_bytes + stacked_bytes


def _runtime_component_split(
    *, total_bytes: int, allocated_fraction: float
) -> tuple[int, int]:
    """Split a runtime component into allocated and reserved-only bytes."""

    allocated_bytes = _scaled_bytes(value=total_bytes, factor=allocated_fraction)
    bounded_allocated = max(0, min(total_bytes, allocated_bytes))
    return bounded_allocated, total_bytes - bounded_allocated


def _runtime_support_bytes(*, config: TrainingConfig) -> tuple[int, int]:
    """Return runtime bytes split into allocated support and reserved-only pool."""

    if config.reserved_vram_gb_per_gpu is not None:
        return 0, int(config.reserved_vram_gb_per_gpu * (1024**3))
    component_specs = [
        (
            int(config.runtime_cuda_context_gb * (1024**3)),
            config.runtime_cuda_context_allocated_fraction,
        ),
        (
            int(config.runtime_allocator_pool_gb * (1024**3)),
            config.runtime_allocator_pool_allocated_fraction,
        ),
    ]
    if config.distributed_mode != "single_gpu":
        component_specs.append(
            (
                int(config.runtime_nccl_gb * (1024**3)),
                config.runtime_nccl_allocated_fraction,
            )
        )
    if config.is_zero_mode():
        component_specs.append(
            (
                int(config.runtime_deepspeed_gb * (1024**3)),
                config.runtime_deepspeed_allocated_fraction,
            )
        )
    allocated_bytes = 0
    reserved_only_bytes = 0
    for total_bytes, allocated_fraction in component_specs:
        component_allocated, component_reserved = _runtime_component_split(
            total_bytes=total_bytes,
            allocated_fraction=allocated_fraction,
        )
        allocated_bytes += component_allocated
        reserved_only_bytes += component_reserved
    return allocated_bytes, reserved_only_bytes


def _persistent_backend_buffer_bytes(
    *,
    model_spec: ModelSpec,
    config: TrainingConfig,
) -> int:
    """Return persistent backend support bytes sized in trainable-param copies.

    Args:
        model_spec: Inspected model summary.
        config: Training configuration being estimated.

    Returns:
        Bytes held persistently by the backend outside named model state.
    """

    buffer_count = config.persistent_backend_buffer_count()
    if buffer_count <= 0:
        return 0
    zero2_pool_bytes = _zero2_full_persistent_backend_buffer_bytes(
        model_spec=model_spec,
        config=config,
    )
    if zero2_pool_bytes is not None:
        return _scaled_bytes(value=zero2_pool_bytes, factor=buffer_count)
    trainable_params = trainable_parameter_count(model_spec=model_spec, config=config)
    return _scaled_bytes(
        value=(
            trainable_params
            * bytes_for_dtype(config.persistent_backend_buffer_resolved_dtype())
        ),
        factor=buffer_count,
    )


def _zero2_full_persistent_backend_buffer_bytes(
    *,
    model_spec: ModelSpec,
    config: TrainingConfig,
) -> int | None:
    """Return the default ZeRO-2 full-FT pool size when sharded pools are explicit.

    Args:
        model_spec: Inspected model summary.
        config: Training configuration being estimated.

    Returns:
        Sharded persistent pool bytes for tested ZeRO-2 full fine-tuning, or
        ``None`` when the generic trainable-copy sizing should be used.

    Example:
        For AdamW ZeRO-2 full fine-tuning on three ranks, this returns one
        parameter-partition window plus one gradient-shard window.
    """

    if config.persistent_backend_buffer_tensor_count is not None:
        return None
    if config.distributed_mode != "zero2" or config.tuning_mode != "full_ft":
        return None
    optimizer_name = normalized_optimizer_name(config=config)
    if optimizer_name not in config.normalized_zero_tested_optimizer_names():
        return None
    world_size = max(config.world_size(), 1)
    trainable_params = trainable_parameter_count(model_spec=model_spec, config=config)
    grad_shard_bytes = (
        trainable_params * bytes_for_dtype(resolved_gradient_dtype(config=config))
    ) // world_size
    parameter_partition_bytes = (
        model_spec.total_params * bytes_for_dtype(config.weight_dtype)
    ) // world_size
    return grad_shard_bytes + parameter_partition_bytes


def _parameter_bytes(*, model_spec: ModelSpec, config: TrainingConfig) -> int:
    """Return persistent parameter bytes resident on one rank."""

    base_bytes = model_spec.total_params * bytes_for_dtype(config.weight_dtype)
    if config.distributed_mode == "zero3":
        base_bytes = base_bytes // max(config.world_size(), 1)
    if config.tuning_mode != "lora":
        if not config.use_master_weights or config.is_zero_mode():
            return base_bytes
        return base_bytes + (
            trainable_parameter_count(model_spec=model_spec, config=config)
            * bytes_for_dtype(config.master_weight_dtype)
        )
    adapter_params = trainable_parameter_count(model_spec=model_spec, config=config)
    adapter_bytes = adapter_params * bytes_for_dtype(config.adapter_parameter_dtype())
    if not config.use_master_weights or config.is_zero_mode():
        return base_bytes + adapter_bytes
    master_bytes = adapter_params * bytes_for_dtype(config.master_weight_dtype)
    return base_bytes + adapter_bytes + master_bytes


def _gradient_bytes(*, model_spec: ModelSpec, config: TrainingConfig) -> int:
    """Return persistent gradient bytes resident on one rank."""

    if config.is_zero_mode():
        return 0
    trainable_params = trainable_parameter_count(model_spec=model_spec, config=config)
    grad_dtype = resolved_gradient_dtype(config=config)
    return trainable_params * bytes_for_dtype(grad_dtype)


def _optimizer_state_bytes(*, model_spec: ModelSpec, config: TrainingConfig) -> int:
    """Return persistent optimizer-state bytes resident on one rank."""

    state_bytes = _raw_optimizer_state_bytes(model_spec=model_spec, config=config)
    if not config.is_zero_mode():
        return state_bytes
    optimizer_shard_bytes = _zero2_optimizer_state_residency_bytes(
        model_spec=model_spec,
        config=config,
    )
    if not config.use_master_weights:
        return optimizer_shard_bytes
    trainable_params = trainable_parameter_count(model_spec=model_spec, config=config)
    world_size = max(config.world_size(), 1)
    master_shard_bytes = (
        trainable_params * bytes_for_dtype(config.master_weight_dtype)
    ) // world_size
    return optimizer_shard_bytes + master_shard_bytes


def _bucket_bytes(*, num_elements: int, dtype: str) -> int:
    """Return byte size for a communication or prefetch bucket."""

    return num_elements * bytes_for_dtype(dtype)


def _workspace_terms(
    *,
    model_spec: ModelSpec,
    config: TrainingConfig,
    activation_terms: ActivationTermBreakdown,
    parameter_bytes: int,
    persistent_backend_buffer_bytes: int,
) -> WorkspaceTerms:
    """Return explicit workspace terms for forward, backward, and optimizer step.

    Args:
        model_spec: Inspected model summary.
        config: Training configuration being estimated.
        activation_terms: Explicit retained-activation breakdown.
        parameter_bytes: Persistent parameter bytes.

    Returns:
        Allocated and reserved workspace terms plus named metadata.
    """

    trainable_params = trainable_parameter_count(model_spec=model_spec, config=config)
    optimizer_update_bytes = optimizer_update_numel(
        model_spec=model_spec,
        config=config,
    ) * bytes_for_dtype(resolved_optimizer_update_dtype(config=config))
    zero_optimizer_update_bytes = _zero2_optimizer_update_residency_bytes(
        model_spec=model_spec,
        config=config,
        optimizer_update_bytes=optimizer_update_bytes,
    )
    logits_workspace_bytes = _logits_workspace_bytes(
        model_spec=model_spec,
        config=config,
    )
    attention_projection_workspace_bytes = _attention_projection_workspace_bytes(
        model_spec=model_spec,
        config=config,
    )
    attention_temporary_workspace_bytes = _attention_temporary_workspace_bytes(
        model_spec=model_spec,
        config=config,
    )
    attention_backward_workspace_bytes = _attention_backward_workspace_bytes(
        model_spec=model_spec,
        config=config,
    )
    checkpoint_recompute_bytes = _checkpoint_recompute_bytes(
        model_spec=model_spec,
        config=config,
    )
    world_size = max(config.world_size(), 1)
    grad_dtype = resolved_gradient_dtype(config=config)
    grad_bytes = trainable_params * bytes_for_dtype(grad_dtype)
    grad_shard_bytes = grad_bytes // world_size
    reduce_bucket_bytes = min(
        max(grad_bytes if not config.is_zero_mode() else grad_shard_bytes, 0),
        _bucket_bytes(num_elements=config.zero_bucket_elements, dtype=grad_dtype),
    )
    optimizer_state_dtype = resolved_optimizer_state_dtype(config=config)
    state_bytes = _raw_optimizer_state_bytes(model_spec=model_spec, config=config)
    state_shard_bytes = _zero2_optimizer_state_residency_bytes(
        model_spec=model_spec,
        config=config,
    )
    prefetch_dtype = optimizer_state_dtype
    if config.is_zero_mode():
        prefetch_dtype = _zero2_prefetch_dtype(
            model_spec=model_spec,
            config=config,
        )
    prefetch_bytes = min(
        max(state_bytes if not config.is_zero_mode() else state_shard_bytes, 0),
        _bucket_bytes(
            num_elements=config.zero_prefetch_elements,
            dtype=prefetch_dtype,
        ),
    )
    full_parameter_bytes = model_spec.total_params * bytes_for_dtype(
        config.weight_dtype
    )
    parameter_partition_bytes = full_parameter_bytes // world_size
    allgather_bucket_bytes = min(
        full_parameter_bytes,
        _bucket_bytes(
            num_elements=config.zero_bucket_elements, dtype=config.weight_dtype
        ),
    )
    backward_buffer_overlap_bytes = _scaled_bytes(
        value=persistent_backend_buffer_bytes,
        factor=config.persistent_backend_buffer_backward_overlap_fraction,
    )
    optimizer_buffer_overlap_bytes = _scaled_bytes(
        value=persistent_backend_buffer_bytes,
        factor=config.persistent_backend_buffer_optimizer_overlap_fraction,
    )
    checkpoint_forward_factor = (
        config.gradient_checkpointing_forward_workspace_factor
        if config.gradient_checkpointing
        else 1.0
    )
    if config.distributed_mode == "single_gpu" and config.tuning_mode == "full_ft":
        forward_bytes = _scaled_bytes(
            value=attention_temporary_workspace_bytes,
            factor=checkpoint_forward_factor,
        ) + _scaled_bytes(
            value=logits_workspace_bytes,
            factor=config.single_full_forward_logits_copies * checkpoint_forward_factor,
        )
        return WorkspaceTerms(
            forward_allocated_bytes=forward_bytes,
            forward_reserved_bytes=forward_bytes,
            backward_allocated_bytes=checkpoint_recompute_bytes,
            backward_reserved_bytes=checkpoint_recompute_bytes,
            optimizer_allocated_bytes=optimizer_update_bytes,
            optimizer_reserved_bytes=optimizer_update_bytes,
            metadata={
                "logits_workspace_bytes": logits_workspace_bytes,
                "attention_projection_workspace_bytes": attention_projection_workspace_bytes,
                "attention_temporary_workspace_bytes": attention_temporary_workspace_bytes,
                "checkpoint_recompute_bytes": checkpoint_recompute_bytes,
                "optimizer_update_bytes": optimizer_update_bytes,
            },
        )
    if config.distributed_mode == "single_gpu":
        forward_bytes = _scaled_bytes(
            value=attention_temporary_workspace_bytes,
            factor=checkpoint_forward_factor,
        ) + _scaled_bytes(
            value=logits_workspace_bytes,
            factor=checkpoint_forward_factor,
        )
        return WorkspaceTerms(
            forward_allocated_bytes=forward_bytes,
            forward_reserved_bytes=forward_bytes,
            backward_allocated_bytes=checkpoint_recompute_bytes,
            backward_reserved_bytes=checkpoint_recompute_bytes,
            optimizer_allocated_bytes=optimizer_update_bytes,
            optimizer_reserved_bytes=optimizer_update_bytes,
            metadata={
                "logits_workspace_bytes": logits_workspace_bytes,
                "attention_projection_workspace_bytes": attention_projection_workspace_bytes,
                "attention_temporary_workspace_bytes": attention_temporary_workspace_bytes,
                "checkpoint_recompute_bytes": checkpoint_recompute_bytes,
                "optimizer_update_bytes": optimizer_update_bytes,
            },
        )
    if config.distributed_mode == "ddp" and config.tuning_mode == "full_ft":
        forward_bytes = _scaled_bytes(
            value=attention_temporary_workspace_bytes,
            factor=checkpoint_forward_factor,
        ) + _scaled_bytes(
            value=logits_workspace_bytes,
            factor=config.ddp_full_forward_logits_copies * checkpoint_forward_factor,
        )
        raw_backward_bytes = _scaled_bytes(
            value=reduce_bucket_bytes,
            factor=config.ddp_backward_reduce_bucket_copies,
        )
        backward_bytes = max(0, raw_backward_bytes - backward_buffer_overlap_bytes)
        return WorkspaceTerms(
            forward_allocated_bytes=forward_bytes,
            forward_reserved_bytes=forward_bytes,
            backward_allocated_bytes=backward_bytes + checkpoint_recompute_bytes,
            backward_reserved_bytes=raw_backward_bytes + checkpoint_recompute_bytes,
            optimizer_allocated_bytes=optimizer_update_bytes,
            optimizer_reserved_bytes=optimizer_update_bytes,
            metadata={
                "logits_workspace_bytes": logits_workspace_bytes,
                "attention_projection_workspace_bytes": attention_projection_workspace_bytes,
                "attention_temporary_workspace_bytes": attention_temporary_workspace_bytes,
                "ddp_reduce_bucket_bytes": reduce_bucket_bytes,
                "persistent_backend_buffer_bytes": persistent_backend_buffer_bytes,
                "checkpoint_recompute_bytes": checkpoint_recompute_bytes,
                "optimizer_update_bytes": optimizer_update_bytes,
            },
        )
    if config.distributed_mode == "ddp":
        forward_bytes = _scaled_bytes(
            value=attention_temporary_workspace_bytes,
            factor=checkpoint_forward_factor,
        ) + _scaled_bytes(
            value=logits_workspace_bytes,
            factor=checkpoint_forward_factor,
        )
        backward_bytes = _scaled_bytes(
            value=reduce_bucket_bytes,
            factor=config.ddp_backward_reduce_bucket_copies,
        )
        backward_total_bytes = _backward_workspace_total_bytes(
            backward_bytes=backward_bytes,
            checkpoint_recompute_bytes=checkpoint_recompute_bytes,
        )
        return WorkspaceTerms(
            forward_allocated_bytes=forward_bytes,
            forward_reserved_bytes=forward_bytes,
            backward_allocated_bytes=backward_total_bytes,
            backward_reserved_bytes=backward_total_bytes,
            optimizer_allocated_bytes=optimizer_update_bytes,
            optimizer_reserved_bytes=optimizer_update_bytes,
            metadata={
                "logits_workspace_bytes": logits_workspace_bytes,
                "attention_projection_workspace_bytes": attention_projection_workspace_bytes,
                "attention_temporary_workspace_bytes": attention_temporary_workspace_bytes,
                "ddp_reduce_bucket_bytes": reduce_bucket_bytes,
                "checkpoint_recompute_bytes": checkpoint_recompute_bytes,
                "optimizer_update_bytes": optimizer_update_bytes,
            },
        )
    if config.distributed_mode == "zero2" and config.tuning_mode == "full_ft":
        master_shard_bytes = 0
        if config.use_master_weights:
            master_shard_bytes = (
                trainable_params * bytes_for_dtype(config.master_weight_dtype)
            ) // world_size
        forward_bytes = _scaled_bytes(
            value=attention_temporary_workspace_bytes,
            factor=checkpoint_forward_factor,
        ) + _scaled_bytes(
            value=logits_workspace_bytes,
            factor=config.zero2_full_forward_logits_copies * checkpoint_forward_factor,
        )
        backward_bytes = _scaled_bytes(
            value=grad_shard_bytes,
            factor=config.zero2_full_backward_grad_shard_copies,
        ) + _scaled_bytes(
            value=reduce_bucket_bytes,
            factor=config.zero2_full_backward_reduce_bucket_copies,
        )
        raw_optimizer_bytes = (
            _scaled_bytes(
                value=master_shard_bytes,
                factor=config.zero2_full_optimizer_master_shard_copies,
            )
            + _scaled_bytes(
                value=grad_shard_bytes,
                factor=config.zero2_full_optimizer_grad_shard_copies,
            )
            + _scaled_bytes(
                value=reduce_bucket_bytes,
                factor=config.zero2_full_optimizer_reduce_bucket_copies,
            )
            + zero_optimizer_update_bytes
            + _scaled_bytes(
                value=parameter_partition_bytes,
                factor=config.zero2_full_optimizer_parameter_partition_copies,
            )
            + _scaled_bytes(
                value=allgather_bucket_bytes,
                factor=config.zero2_full_optimizer_allgather_bucket_copies,
            )
            + _scaled_bytes(
                value=prefetch_bytes,
                factor=config.zero2_full_optimizer_prefetch_copies,
            )
        )
        optimizer_bytes = max(0, raw_optimizer_bytes - optimizer_buffer_overlap_bytes)
        backward_total_bytes = _backward_workspace_total_bytes(
            backward_bytes=backward_bytes,
            checkpoint_recompute_bytes=checkpoint_recompute_bytes,
        )
        return WorkspaceTerms(
            forward_allocated_bytes=forward_bytes,
            forward_reserved_bytes=forward_bytes,
            backward_allocated_bytes=backward_total_bytes,
            backward_reserved_bytes=backward_total_bytes,
            optimizer_allocated_bytes=optimizer_bytes,
            optimizer_reserved_bytes=raw_optimizer_bytes,
            metadata={
                "logits_workspace_bytes": logits_workspace_bytes,
                "attention_projection_workspace_bytes": attention_projection_workspace_bytes,
                "attention_temporary_workspace_bytes": attention_temporary_workspace_bytes,
                "zero2_master_shard_bytes": master_shard_bytes,
                "zero2_grad_shard_bytes": grad_shard_bytes,
                "zero2_reduce_bucket_bytes": reduce_bucket_bytes,
                "zero2_parameter_partition_bytes": parameter_partition_bytes,
                "zero2_allgather_bucket_bytes": allgather_bucket_bytes,
                "zero2_optimizer_prefetch_bytes": prefetch_bytes,
                "zero2_optimizer_update_bytes": zero_optimizer_update_bytes,
                "persistent_backend_buffer_bytes": persistent_backend_buffer_bytes,
                "checkpoint_recompute_bytes": checkpoint_recompute_bytes,
                "optimizer_update_bytes": optimizer_update_bytes,
            },
        )
    if config.distributed_mode == "zero2":
        model_bucket_bytes = _scaled_bytes(
            value=min(
                full_parameter_bytes,
                _bucket_bytes(
                    num_elements=config.zero_bucket_elements,
                    dtype=config.weight_dtype,
                ),
            ),
            factor=config.zero2_lora_backward_model_bucket_copies,
        )
        forward_bytes = _scaled_bytes(
            value=attention_temporary_workspace_bytes,
            factor=checkpoint_forward_factor,
        ) + _scaled_bytes(
            value=logits_workspace_bytes,
            factor=checkpoint_forward_factor,
        )
        backward_allocated_bytes = (
            _scaled_bytes(
                value=grad_shard_bytes,
                factor=config.zero2_lora_backward_grad_shard_copies,
            )
            + attention_backward_workspace_bytes
            + activation_terms.lora_low_rank_bytes
        )
        optimizer_bytes = (
            _scaled_bytes(
                value=grad_shard_bytes,
                factor=config.zero2_lora_optimizer_grad_shard_copies,
            )
            + prefetch_bytes
            + zero_optimizer_update_bytes
        )
        backward_allocated_total_bytes = _backward_workspace_total_bytes(
            backward_bytes=backward_allocated_bytes,
            checkpoint_recompute_bytes=checkpoint_recompute_bytes,
        )
        backward_reserved_total_bytes = (
            backward_allocated_total_bytes + model_bucket_bytes
        )
        return WorkspaceTerms(
            forward_allocated_bytes=forward_bytes,
            forward_reserved_bytes=forward_bytes,
            backward_allocated_bytes=backward_allocated_total_bytes,
            backward_reserved_bytes=backward_reserved_total_bytes,
            optimizer_allocated_bytes=optimizer_bytes,
            optimizer_reserved_bytes=optimizer_bytes,
            metadata={
                "logits_workspace_bytes": logits_workspace_bytes,
                "attention_projection_workspace_bytes": attention_projection_workspace_bytes,
                "attention_temporary_workspace_bytes": attention_temporary_workspace_bytes,
                "attention_backward_workspace_bytes": (
                    attention_backward_workspace_bytes
                ),
                "zero2_lora_backward_model_bucket_bytes": model_bucket_bytes,
                "zero2_grad_shard_bytes": grad_shard_bytes,
                "zero2_backward_lora_low_rank_bytes": (
                    activation_terms.lora_low_rank_bytes
                ),
                "zero2_optimizer_prefetch_bytes": prefetch_bytes,
                "zero2_optimizer_update_bytes": zero_optimizer_update_bytes,
                "checkpoint_recompute_bytes": checkpoint_recompute_bytes,
                "optimizer_update_bytes": optimizer_update_bytes,
            },
        )
    if config.tuning_mode == "full_ft":
        master_shard_bytes = 0
        if config.use_master_weights:
            master_shard_bytes = (
                trainable_params * bytes_for_dtype(config.master_weight_dtype)
            ) // world_size
        forward_bytes = (
            _scaled_bytes(
                value=attention_temporary_workspace_bytes,
                factor=checkpoint_forward_factor,
            )
            + _scaled_bytes(
                value=logits_workspace_bytes,
                factor=config.zero3_full_forward_logits_copies
                * checkpoint_forward_factor,
            )
            + _scaled_bytes(
                value=allgather_bucket_bytes,
                factor=config.zero3_full_forward_allgather_bucket_copies,
            )
            + _scaled_bytes(
                value=parameter_partition_bytes,
                factor=config.zero3_full_forward_parameter_partition_copies,
            )
        )
        backward_bytes = (
            _scaled_bytes(
                value=grad_shard_bytes,
                factor=config.zero3_full_backward_grad_shard_copies,
            )
            + _scaled_bytes(
                value=reduce_bucket_bytes,
                factor=config.zero3_full_backward_reduce_bucket_copies,
            )
            + _scaled_bytes(
                value=allgather_bucket_bytes,
                factor=config.zero3_full_backward_allgather_bucket_copies,
            )
            + _scaled_bytes(
                value=parameter_partition_bytes,
                factor=config.zero3_full_backward_parameter_partition_copies,
            )
        )
        raw_optimizer_bytes = (
            _scaled_bytes(
                value=master_shard_bytes,
                factor=config.zero3_full_optimizer_master_shard_copies,
            )
            + _scaled_bytes(
                value=grad_shard_bytes,
                factor=config.zero3_full_optimizer_grad_shard_copies,
            )
            + _scaled_bytes(
                value=parameter_partition_bytes,
                factor=config.zero3_full_optimizer_parameter_partition_copies,
            )
            + _scaled_bytes(
                value=allgather_bucket_bytes,
                factor=config.zero3_full_optimizer_allgather_bucket_copies,
            )
            + _scaled_bytes(
                value=prefetch_bytes,
                factor=config.zero3_full_optimizer_prefetch_copies,
            )
            + _scaled_bytes(
                value=reduce_bucket_bytes,
                factor=config.zero3_full_optimizer_reduce_bucket_copies,
            )
            + zero_optimizer_update_bytes
        )
        optimizer_bytes = max(0, raw_optimizer_bytes - optimizer_buffer_overlap_bytes)
        backward_total_bytes = _backward_workspace_total_bytes(
            backward_bytes=backward_bytes,
            checkpoint_recompute_bytes=checkpoint_recompute_bytes,
        )
        return WorkspaceTerms(
            forward_allocated_bytes=forward_bytes,
            forward_reserved_bytes=forward_bytes,
            backward_allocated_bytes=backward_total_bytes,
            backward_reserved_bytes=backward_total_bytes,
            optimizer_allocated_bytes=optimizer_bytes,
            optimizer_reserved_bytes=raw_optimizer_bytes,
            metadata={
                "logits_workspace_bytes": logits_workspace_bytes,
                "attention_projection_workspace_bytes": attention_projection_workspace_bytes,
                "attention_temporary_workspace_bytes": attention_temporary_workspace_bytes,
                "zero3_master_shard_bytes": master_shard_bytes,
                "zero3_grad_shard_bytes": grad_shard_bytes,
                "zero3_reduce_bucket_bytes": reduce_bucket_bytes,
                "zero3_parameter_partition_bytes": parameter_partition_bytes,
                "zero3_allgather_bucket_bytes": allgather_bucket_bytes,
                "zero3_optimizer_prefetch_bytes": prefetch_bytes,
                "zero3_optimizer_update_bytes": zero_optimizer_update_bytes,
                "persistent_backend_buffer_bytes": persistent_backend_buffer_bytes,
                "checkpoint_recompute_bytes": checkpoint_recompute_bytes,
                "optimizer_update_bytes": optimizer_update_bytes,
            },
        )
    forward_bytes = (
        _scaled_bytes(
            value=attention_temporary_workspace_bytes,
            factor=checkpoint_forward_factor,
        )
        + _scaled_bytes(
            value=logits_workspace_bytes,
            factor=checkpoint_forward_factor,
        )
        + _scaled_bytes(
            value=allgather_bucket_bytes,
            factor=config.zero3_lora_forward_allgather_bucket_copies,
        )
    )
    backward_bytes = _scaled_bytes(
        value=allgather_bucket_bytes,
        factor=config.zero3_lora_backward_allgather_bucket_copies,
    ) + (
        _scaled_bytes(
            value=grad_shard_bytes,
            factor=config.zero3_lora_backward_grad_shard_copies,
        )
        + attention_backward_workspace_bytes
        + activation_terms.lora_low_rank_bytes
    )
    optimizer_bytes = (
        _scaled_bytes(
            value=grad_shard_bytes,
            factor=config.zero3_lora_optimizer_grad_shard_copies,
        )
        + prefetch_bytes
        + zero_optimizer_update_bytes
    )
    backward_total_bytes = _backward_workspace_total_bytes(
        backward_bytes=backward_bytes,
        checkpoint_recompute_bytes=checkpoint_recompute_bytes,
    )
    return WorkspaceTerms(
        forward_allocated_bytes=forward_bytes,
        forward_reserved_bytes=forward_bytes,
        backward_allocated_bytes=backward_total_bytes,
        backward_reserved_bytes=backward_total_bytes,
        optimizer_allocated_bytes=optimizer_bytes,
        optimizer_reserved_bytes=optimizer_bytes,
        metadata={
            "logits_workspace_bytes": logits_workspace_bytes,
            "attention_projection_workspace_bytes": attention_projection_workspace_bytes,
            "attention_temporary_workspace_bytes": attention_temporary_workspace_bytes,
            "attention_backward_workspace_bytes": attention_backward_workspace_bytes,
            "zero3_grad_shard_bytes": grad_shard_bytes,
            "zero3_backward_lora_low_rank_bytes": activation_terms.lora_low_rank_bytes,
            "zero3_allgather_bucket_bytes": allgather_bucket_bytes,
            "zero3_optimizer_prefetch_bytes": prefetch_bytes,
            "zero3_optimizer_update_bytes": zero_optimizer_update_bytes,
            "checkpoint_recompute_bytes": checkpoint_recompute_bytes,
            "optimizer_update_bytes": optimizer_update_bytes,
        },
    )


def _assumptions(*, model_spec: ModelSpec, config: TrainingConfig) -> tuple[str, ...]:
    """Return mode-specific assumptions for the interpretable estimator."""

    common = (
        "Estimator uses explicit parameter, gradient, optimizer-state, and workspace terms.",
        "Estimator separates forward-end resident activations from hook-visible outputs and temporary workspace.",
        "Attention backend changes explicit attention-score storage and temporary workspace terms.",
        "The `standard` backend models the default Hugging Face runtime path for the model family.",
        "Gradient checkpointing changes the tokenwise retained-activation term and may replace, rather than add to, backward workspace.",
        "Optimizer-state tensors are derived from explicit optimizer semantics or a state-count override.",
    )
    if model_spec.supports_vision_inputs() and config.vision_images_per_sample > 0:
        common += (
            "Vision-language models are approximated by folding image tokens into the effective sequence length.",
        )
    if config.distributed_mode == "single_gpu":
        return common + (
            "Single-GPU mode models no communication buckets or sharding.",
        )
    if config.distributed_mode == "ddp":
        return common + (
            "DDP models replicated parameters and optimizer states with backward reduce buckets.",
        )
    if config.distributed_mode == "zero2":
        return common + (
            "ZeRO-2 models replicated parameters with optimizer-state residency controlled by explicit optimizer support assumptions.",
        )
    return common + (
        "ZeRO-3 models shard parameters, gradients, and optimizer states with stage-3 all-gather buckets.",
    )


def build_interpretable_terms(
    *,
    model_spec: ModelSpec,
    config: TrainingConfig,
) -> InterpretableEstimateTerms:
    """Build explicit persistent-state and workspace terms for one config.

    Args:
        model_spec: Inspected model summary.
        config: Training configuration being estimated.

    Returns:
        Interpretable persistent and transient state terms.

    Example:
        >>> from simplesft.types import TrainingConfig
        >>> terms = build_interpretable_terms(
        ...     model_spec=ModelSpec(
        ...         model_name="toy",
        ...         model_type="qwen3",
        ...         num_layers=2,
        ...         hidden_size=64,
        ...         num_attention_heads=4,
        ...         intermediate_size=256,
        ...         vocab_size=32000,
        ...         max_position_embeddings=4096,
        ...         total_params=1000,
        ...         trainable_linear_layers=(),
        ...     ),
        ...     config=TrainingConfig(tuning_mode="full_ft", distributed_mode="ddp", gpus_per_node=2),
        ... )
        >>> terms.breakdown.parameter_bytes > 0
        True
    """

    parameter_bytes = _parameter_bytes(model_spec=model_spec, config=config)
    activation_terms = _activation_term_breakdown(
        model_spec=model_spec,
        config=config,
    )
    activation_bytes = activation_terms.retained_total_bytes
    forward_activation_bytes = activation_bytes
    backward_activation_bytes = activation_bytes
    if _uses_checkpointed_sharded_lora_phase_split(
        model_spec=model_spec,
        config=config,
    ):
        (
            forward_activation_bytes,
            backward_activation_bytes,
        ) = _checkpointed_lora_phase_activation_bytes(
            parameter_gradient_context_bytes=(
                activation_terms.parameter_gradient_context_bytes
            ),
            checkpoint_boundary_bytes=activation_terms.checkpoint_boundary_bytes,
            attention_score_bytes=activation_terms.attention_score_bytes,
            lora_low_rank_bytes=activation_terms.lora_low_rank_bytes,
        )
        activation_bytes = backward_activation_bytes
    elif _uses_checkpointed_nonexpanded_sharded_full_ft(
        model_spec=model_spec,
        config=config,
    ):
        (
            forward_activation_bytes,
            backward_activation_bytes,
        ) = _checkpointed_nonexpanded_phase_activation_bytes(
            model_spec=model_spec,
            config=config,
            linear_input_bytes=activation_terms.linear_input_bytes,
            norm_residual_bytes=activation_terms.norm_residual_bytes,
            checkpoint_boundary_bytes=activation_terms.checkpoint_boundary_bytes,
            attention_score_bytes=activation_terms.attention_score_bytes,
        )
        activation_bytes = backward_activation_bytes
    loss_output_bytes = _loss_output_bytes(
        model_spec=model_spec,
        config=config,
        activation_bytes=backward_activation_bytes,
    )
    base_hook_visible_activation_bytes = activation_terms.base_hook_visible_bytes
    hook_visible_activation_bytes = activation_terms.hook_visible_bytes
    autograd_context_bytes = _forward_autograd_context_bytes(
        model_spec=model_spec,
        config=config,
    )
    expanded_autograd_context_bytes = _expanded_forward_autograd_context_bytes(
        model_spec=model_spec,
        config=config,
    )
    resident_autograd_context_bytes = activation_terms.resident_autograd_context_bytes
    tokenwise_retained_activation_bytes = (
        activation_terms.linear_input_bytes
        + activation_terms.norm_residual_bytes
        + activation_terms.mlp_intermediate_bytes
        + activation_terms.lora_low_rank_bytes
    )
    attention_score_bytes = activation_terms.attention_score_bytes
    gradient_bytes = _gradient_bytes(model_spec=model_spec, config=config)
    optimizer_state_bytes = _optimizer_state_bytes(model_spec=model_spec, config=config)
    runtime_allocated_support_bytes, runtime_reserved_only_bytes = (
        _runtime_support_bytes(
            config=config,
        )
    )
    persistent_backend_buffer_bytes = _persistent_backend_buffer_bytes(
        model_spec=model_spec,
        config=config,
    )
    workspace_terms = _workspace_terms(
        model_spec=model_spec,
        config=config,
        activation_terms=activation_terms,
        parameter_bytes=parameter_bytes,
        persistent_backend_buffer_bytes=persistent_backend_buffer_bytes,
    )
    metadata_dict: dict[str, Any] = dict(workspace_terms.metadata)
    metadata_dict.update(
        {
            "retained_activation_bytes": activation_bytes,
            "forward_phase_activation_bytes": forward_activation_bytes,
            "backward_phase_activation_bytes": backward_activation_bytes,
            "tokenwise_retained_activation_bytes": tokenwise_retained_activation_bytes,
            "base_hook_visible_activation_bytes": base_hook_visible_activation_bytes,
            "hook_visible_activation_bytes": hook_visible_activation_bytes,
            "visible_propagation_bytes": activation_terms.visible_propagation_bytes,
            "parameter_gradient_context_bytes": (
                activation_terms.parameter_gradient_context_bytes
            ),
            "activation_linear_input_bytes": activation_terms.linear_input_bytes,
            "activation_norm_residual_bytes": activation_terms.norm_residual_bytes,
            "activation_mlp_intermediate_bytes": (
                activation_terms.mlp_intermediate_bytes
            ),
            "activation_lora_low_rank_bytes": activation_terms.lora_low_rank_bytes,
            "activation_checkpoint_boundary_bytes": (
                activation_terms.checkpoint_boundary_bytes
            ),
            "checkpoint_expanded_query_overlap_bytes": (
                activation_terms.checkpoint_overlap_bytes
            ),
            "activation_formula_bytes": activation_terms.retained_formula_bytes,
            "forward_autograd_context_bytes": autograd_context_bytes,
            "resident_forward_autograd_context_bytes": (
                resident_autograd_context_bytes
            ),
            "expanded_forward_autograd_context_bytes": expanded_autograd_context_bytes,
            "attention_score_bytes": attention_score_bytes,
            "loss_output_bytes": loss_output_bytes,
            "effective_attention_backend": _effective_attention_backend(
                model_spec=model_spec,
                config=config,
            ),
            "persistent_backend_buffer_bytes": persistent_backend_buffer_bytes,
            "runtime_allocated_support_bytes": runtime_allocated_support_bytes,
            "runtime_reserved_only_bytes": runtime_reserved_only_bytes,
            "forward_workspace_bytes": workspace_terms.forward_allocated_bytes,
            "forward_reserved_workspace_bytes": workspace_terms.forward_reserved_bytes,
            "backward_workspace_bytes": workspace_terms.backward_allocated_bytes,
            "backward_reserved_workspace_bytes": workspace_terms.backward_reserved_bytes,
            "optimizer_workspace_bytes": workspace_terms.optimizer_allocated_bytes,
            "optimizer_reserved_workspace_bytes": workspace_terms.optimizer_reserved_bytes,
        }
    )
    runtime_allocated_support_bytes += persistent_backend_buffer_bytes
    breakdown = MemoryComponentBreakdown(
        parameter_bytes=parameter_bytes,
        gradient_bytes=gradient_bytes,
        optimizer_state_bytes=optimizer_state_bytes,
        activation_bytes=activation_bytes,
        runtime_reserve_bytes=runtime_reserved_only_bytes,
    )
    return InterpretableEstimateTerms(
        breakdown=breakdown,
        forward_activation_bytes=forward_activation_bytes,
        backward_activation_bytes=backward_activation_bytes,
        forward_workspace_bytes=workspace_terms.forward_allocated_bytes,
        forward_reserved_workspace_bytes=workspace_terms.forward_reserved_bytes,
        backward_workspace_bytes=workspace_terms.backward_allocated_bytes,
        backward_reserved_workspace_bytes=workspace_terms.backward_reserved_bytes,
        optimizer_workspace_bytes=workspace_terms.optimizer_allocated_bytes,
        optimizer_reserved_workspace_bytes=workspace_terms.optimizer_reserved_bytes,
        runtime_allocated_support_bytes=runtime_allocated_support_bytes,
        runtime_reserved_only_bytes=runtime_reserved_only_bytes,
        metadata=metadata_dict,
        assumptions=_assumptions(model_spec=model_spec, config=config),
    )


def _phase_end_allocated_floor_bytes(
    *,
    model_load_allocated_bytes: int,
    baseline_allocated_bytes: int,
    post_step_baseline_allocated_bytes: int,
    breakdown: MemoryComponentBreakdown,
    forward_activation_bytes: int,
    loss_output_bytes: int,
) -> dict[str, int]:
    """Return non-transient allocated bytes visible at phase end."""

    return {
        "model_load": model_load_allocated_bytes,
        "optimizer_create": baseline_allocated_bytes,
        "post_init_baseline": baseline_allocated_bytes,
        "batch_materialization": baseline_allocated_bytes,
        "forward": baseline_allocated_bytes + forward_activation_bytes,
        "loss_materialization": baseline_allocated_bytes + forward_activation_bytes,
        "backward": baseline_allocated_bytes
        + breakdown.gradient_bytes
        + loss_output_bytes,
        "optimizer_step": (
            post_step_baseline_allocated_bytes
            + breakdown.gradient_bytes
            + loss_output_bytes
        ),
        "zero_grad": post_step_baseline_allocated_bytes + loss_output_bytes,
        "step_end": post_step_baseline_allocated_bytes + loss_output_bytes,
    }


def _phase_peak_base_bytes(
    *,
    model_load_allocated_bytes: int,
    baseline_allocated_bytes: int,
    post_step_baseline_allocated_bytes: int,
    breakdown: MemoryComponentBreakdown,
    forward_activation_bytes: int,
    backward_activation_bytes: int,
    loss_output_bytes: int,
) -> dict[str, int]:
    """Return allocated bytes guaranteed live at each phase peak."""

    return {
        "model_load": model_load_allocated_bytes,
        "optimizer_create": baseline_allocated_bytes,
        "post_init_baseline": baseline_allocated_bytes,
        "batch_materialization": baseline_allocated_bytes,
        "forward": baseline_allocated_bytes + forward_activation_bytes,
        "loss_materialization": baseline_allocated_bytes + forward_activation_bytes,
        "backward": (
            baseline_allocated_bytes
            + backward_activation_bytes
            + breakdown.gradient_bytes
        ),
        "optimizer_step": (
            post_step_baseline_allocated_bytes
            + breakdown.gradient_bytes
            + loss_output_bytes
        ),
        "zero_grad": post_step_baseline_allocated_bytes + loss_output_bytes,
        "step_end": post_step_baseline_allocated_bytes + loss_output_bytes,
    }


def _phase_local_peak_allocated_transient_bytes(
    *,
    forward_workspace_bytes: int,
    backward_workspace_bytes: int,
    optimizer_workspace_bytes: int,
) -> dict[str, int]:
    """Return local allocated transient spikes created within each phase."""

    return {
        "model_load": 0,
        "optimizer_create": 0,
        "post_init_baseline": 0,
        "batch_materialization": 0,
        "forward": forward_workspace_bytes,
        "loss_materialization": 0,
        "backward": backward_workspace_bytes,
        "optimizer_step": optimizer_workspace_bytes,
        "zero_grad": 0,
        "step_end": 0,
    }


def _phase_local_peak_reserved_transient_bytes(
    *,
    forward_reserved_workspace_bytes: int,
    backward_reserved_workspace_bytes: int,
    optimizer_reserved_workspace_bytes: int,
) -> dict[str, int]:
    """Return local reserved transient spikes created within each phase."""

    return {
        "model_load": 0,
        "optimizer_create": 0,
        "post_init_baseline": 0,
        "batch_materialization": 0,
        "forward": forward_reserved_workspace_bytes,
        "loss_materialization": 0,
        "backward": backward_reserved_workspace_bytes,
        "optimizer_step": optimizer_reserved_workspace_bytes,
        "zero_grad": 0,
        "step_end": 0,
    }


def build_interpretable_phase_records(
    *,
    config: TrainingConfig,
    breakdown: MemoryComponentBreakdown,
    forward_activation_bytes: int,
    backward_activation_bytes: int,
    forward_workspace_bytes: int,
    forward_reserved_workspace_bytes: int,
    backward_workspace_bytes: int,
    backward_reserved_workspace_bytes: int,
    optimizer_workspace_bytes: int,
    optimizer_reserved_workspace_bytes: int,
    runtime_allocated_support_bytes: int,
    loss_output_bytes: int,
    optimizer_activation_release_bytes: int = 0,
) -> tuple[PhaseMemoryRecord, ...]:
    """Build an explicit phase timeline from persistent and workspace terms.

    Args:
        config: Training configuration being estimated.
        breakdown: Persistent memory components for one rank.
        forward_activation_bytes: Retained activation bytes visible at forward peak.
        forward_workspace_bytes: Forward-only workspace bytes.
        backward_workspace_bytes: Backward-only workspace bytes.
        optimizer_workspace_bytes: Optimizer-step-only workspace bytes.
        runtime_allocated_support_bytes: Runtime/backend bytes that stay allocated.
        loss_output_bytes: Persistent model-output bytes surviving into step end.
        optimizer_activation_release_bytes: Activation-derived reserved bytes
            released before optimizer step instead of carrying from backward.

    Returns:
        Phase timeline with deterministic reserved and allocated bytes.
    """

    optimizer_ready = optimizer_state_in_baseline(
        warmup_steps=config.warmup_steps,
        optimizer_state_in_baseline_after_warmup=(
            config.optimizer_state_in_baseline_after_warmup
        ),
    )
    model_load_allocated_bytes = (
        breakdown.parameter_bytes + runtime_allocated_support_bytes
    )
    baseline_allocated_bytes = model_load_allocated_bytes
    if optimizer_ready:
        baseline_allocated_bytes += breakdown.optimizer_state_bytes
    post_step_baseline_allocated_bytes = (
        model_load_allocated_bytes + breakdown.optimizer_state_bytes
    )
    phase_end_floors = _phase_end_allocated_floor_bytes(
        model_load_allocated_bytes=model_load_allocated_bytes,
        baseline_allocated_bytes=baseline_allocated_bytes,
        post_step_baseline_allocated_bytes=post_step_baseline_allocated_bytes,
        breakdown=breakdown,
        forward_activation_bytes=forward_activation_bytes,
        loss_output_bytes=loss_output_bytes,
    )
    phase_peak_bases = _phase_peak_base_bytes(
        model_load_allocated_bytes=model_load_allocated_bytes,
        baseline_allocated_bytes=baseline_allocated_bytes,
        post_step_baseline_allocated_bytes=post_step_baseline_allocated_bytes,
        breakdown=breakdown,
        forward_activation_bytes=forward_activation_bytes,
        backward_activation_bytes=backward_activation_bytes,
        loss_output_bytes=loss_output_bytes,
    )
    phase_peak_allocated_transients = _phase_local_peak_allocated_transient_bytes(
        forward_workspace_bytes=forward_workspace_bytes,
        backward_workspace_bytes=backward_workspace_bytes,
        optimizer_workspace_bytes=optimizer_workspace_bytes,
    )
    phase_peak_reserved_transients = _phase_local_peak_reserved_transient_bytes(
        forward_reserved_workspace_bytes=forward_reserved_workspace_bytes,
        backward_reserved_workspace_bytes=backward_reserved_workspace_bytes,
        optimizer_reserved_workspace_bytes=optimizer_reserved_workspace_bytes,
    )
    previous_allocated = 0
    previous_reserved = 0
    previous_phase = ""
    records: list[PhaseMemoryRecord] = []
    for phase_name in TRAINING_PHASES:
        local_peak_allocated_transient_bytes = phase_peak_allocated_transients[
            phase_name
        ]
        local_peak_reserved_transient_bytes = phase_peak_reserved_transients[phase_name]
        end_allocated_transient_bytes = _scaled_bytes(
            value=local_peak_allocated_transient_bytes,
            factor=config.phase_end_allocated_transient_fraction(
                phase_name=phase_name,
            ),
        )
        end_reserved_transient_bytes = _scaled_bytes(
            value=local_peak_reserved_transient_bytes,
            factor=config.phase_end_reserved_transient_fraction(
                phase_name=phase_name,
            ),
        )
        allocated_bytes = phase_end_floors[phase_name] + end_allocated_transient_bytes
        reserved_floor_bytes = allocated_bytes + breakdown.runtime_reserve_bytes
        carried_reserved_bytes = 0
        if previous_phase:
            release_bytes = 0
            if previous_phase == "backward" and phase_name == "optimizer_step":
                release_bytes = optimizer_activation_release_bytes
            excess_reserved_bytes = max(
                0, previous_reserved - reserved_floor_bytes - release_bytes
            )
            carried_reserved_bytes = _scaled_bytes(
                value=excess_reserved_bytes,
                factor=config.reserved_carryover_fraction(
                    previous_phase=previous_phase,
                    next_phase=phase_name,
                ),
            )
        optimizer_stack_fraction = 0.0
        if phase_name == "optimizer_step":
            optimizer_stack_fraction = config.optimizer_reserved_stack_fraction()
        end_phase_reserved_bytes = _stacked_reserved_transient_bytes(
            carried_reserved_bytes=carried_reserved_bytes,
            transient_reserved_bytes=end_reserved_transient_bytes,
            stack_fraction=optimizer_stack_fraction,
        )
        reserved_bytes = max(
            reserved_floor_bytes,
            reserved_floor_bytes + end_phase_reserved_bytes,
        )
        peak_floor_reserved_bytes = (
            phase_peak_bases[phase_name] + breakdown.runtime_reserve_bytes
        )
        peak_carried_reserved_bytes = 0
        if previous_phase:
            release_bytes = 0
            if previous_phase == "backward" and phase_name == "optimizer_step":
                release_bytes = optimizer_activation_release_bytes
            peak_excess_reserved_bytes = max(
                0, previous_reserved - peak_floor_reserved_bytes - release_bytes
            )
            peak_carried_reserved_bytes = _scaled_bytes(
                value=peak_excess_reserved_bytes,
                factor=config.reserved_carryover_fraction(
                    previous_phase=previous_phase,
                    next_phase=phase_name,
                ),
            )
        peak_phase_reserved_bytes = _stacked_reserved_transient_bytes(
            carried_reserved_bytes=peak_carried_reserved_bytes,
            transient_reserved_bytes=local_peak_reserved_transient_bytes,
            stack_fraction=optimizer_stack_fraction,
        )
        peak_allocated_bytes = max(
            allocated_bytes,
            phase_peak_bases[phase_name] + local_peak_allocated_transient_bytes,
        )
        peak_reserved_bytes = max(
            reserved_bytes,
            peak_allocated_bytes + breakdown.runtime_reserve_bytes,
            peak_floor_reserved_bytes + peak_phase_reserved_bytes,
        )
        records.append(
            PhaseMemoryRecord(
                phase_name=phase_name,
                allocated_bytes=allocated_bytes,
                reserved_bytes=reserved_bytes,
                peak_allocated_bytes=peak_allocated_bytes,
                peak_reserved_bytes=peak_reserved_bytes,
                delta_allocated_bytes=allocated_bytes - previous_allocated,
                delta_reserved_bytes=reserved_bytes - previous_reserved,
                notes=(
                    "estimated",
                    "interpretable",
                    f"carryover={carried_reserved_bytes}",
                ),
            )
        )
        previous_allocated = allocated_bytes
        previous_reserved = reserved_bytes
        previous_phase = phase_name
    return tuple(records)


def transient_bytes_for_peak(
    *,
    peak_phase: str,
    config: TrainingConfig,
    breakdown: MemoryComponentBreakdown,
    phase_records: tuple[PhaseMemoryRecord, ...],
    peak_bytes: int | None = None,
) -> int:
    """Return transient bytes implied by the dominant peak phase.

    Args:
        peak_phase: Peak phase name.
        config: Training configuration being estimated.
        breakdown: Persistent memory components for one rank.
        phase_records: Deterministic phase timeline.
        peak_bytes: Selected peak bytes for the phase. When omitted, uses the
            modeled soft reserved peak.

    Returns:
        Peak-phase bytes above the persistent floor.
    """

    peak_record = next(
        record for record in phase_records if record.phase_name == peak_phase
    )
    persistent_reserved_bytes = (
        peak_record.allocated_bytes + breakdown.runtime_reserve_bytes
    )
    selected_peak_bytes = (
        peak_record.peak_reserved_bytes if peak_bytes is None else peak_bytes
    )
    return max(0, selected_peak_bytes - persistent_reserved_bytes)
