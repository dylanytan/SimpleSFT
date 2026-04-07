"""Retained activation accounting for the simplified analytical estimator."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Union

from ..models.architecture_types import (
    LINEAR_ROLE_ATTENTION_KEY,
    LINEAR_ROLE_ATTENTION_OUTPUT,
    LINEAR_ROLE_ATTENTION_QKV,
    LINEAR_ROLE_ATTENTION_QUERY,
    LINEAR_ROLE_ATTENTION_VALUE,
    LINEAR_ROLE_LM_HEAD,
    LINEAR_ROLE_MLP_DOWN,
    LINEAR_ROLE_MLP_GATE,
    LINEAR_ROLE_MLP_GATE_UP,
    LINEAR_ROLE_MLP_UP,
    LINEAR_ROLE_ROUTER,
)
from .parallelism import (
    local_linear_input_dim,
    local_linear_output_dim,
    sequence_parallel_divisor,
    tensor_parallel_degree,
)
from ..types import (
    ActivationDebug,
    EstimatorConfig,
    MeasurementConfig,
    ModelLinearLayerSpec,
    ModelSpec,
)
from ..utils import bytes_for_dtype


ConfigLike = Union[EstimatorConfig, MeasurementConfig]
CHECKPOINT_SEGMENT_LAYERS = 1
ATTENTION_SCORE_DTYPE = "fp32"


@dataclass(frozen=True)
class ActivationTerms:
    """Retained activation terms for one analytical estimate."""

    debug: ActivationDebug


@dataclass(frozen=True)
class _SavedInputGroup:
    """Saved-input source group used for retained activation accounting.

    Args:
        name: Stable source-tensor label.
        repeats_per_block: Whether the source appears once per transformer block.

    Returns:
        One normalized saved-input source group.
    """

    name: str
    repeats_per_block: bool


def _scaled_int(*, value: int, factor: float) -> int:
    """Return one integer scaled by a floating-point factor.

    Args:
        value: Integer magnitude to scale.
        factor: Scalar multiplier.

    Returns:
        Rounded scaled integer.
    """

    return int(round(value * factor))


def _first_block_layers(*, model_spec: ModelSpec) -> tuple[ModelLinearLayerSpec, ...]:
    """Return the linear layers belonging to the first transformer block.

    Args:
        model_spec: Inspected model summary.

    Returns:
        First-block linear layers when recognizable, else all trainable layers.
    """

    block_layers = tuple(
        layer
        for layer in model_spec.trainable_linear_layers
        if "layers.0." in layer.module_name or ".h.0." in layer.module_name
    )
    if block_layers:
        return block_layers
    return tuple(model_spec.trainable_linear_layers)


def _first_block_layer_by_role(
    *,
    model_spec: ModelSpec,
    roles: tuple[str, ...],
) -> ModelLinearLayerSpec | None:
    """Return the first block layer matching one of the requested roles.

    Args:
        model_spec: Inspected model summary.
        roles: Candidate linear-role labels.

    Returns:
        Matching first-block linear layer, or `None`.
    """

    for layer in _first_block_layers(model_spec=model_spec):
        if layer.role in roles:
            return layer
    return None


def _token_count(*, model_spec: ModelSpec, config: ConfigLike) -> int:
    """Return the total token count processed by one layer."""

    return model_spec.tokens_per_layer(
        batch_size=config.micro_batch_size_per_gpu,
        sequence_length=config.max_seq_len,
    )


def _sequence_local_token_count(*, model_spec: ModelSpec, config: ConfigLike) -> int:
    """Return the local token count for sequence-sharded activation tensors."""

    return math.ceil(
        _token_count(model_spec=model_spec, config=config)
        / sequence_parallel_divisor(config=config)
    )


def _target_layers(
    *, model_spec: ModelSpec, config: ConfigLike
) -> tuple[ModelLinearLayerSpec, ...]:
    """Return the trainable linear layers relevant for saved activation state."""

    if config.tuning_mode == "full_ft" or config.lora is None:
        return tuple(model_spec.trainable_linear_layers)
    return tuple(
        layer
        for layer in model_spec.trainable_linear_layers
        if layer.module_name.endswith(config.lora.target_modules)
    )


def _local_lora_rank(*, config: ConfigLike) -> int:
    """Return the TP-local LoRA rank.

    Args:
        config: Estimator or measurement config.

    Returns:
        Local adapter rank used by one TP shard.
    """

    assert config.lora is not None, "LoRA config is required for local rank."
    if not config.uses_tensor_parallel():
        return config.lora.rank
    return math.ceil(config.lora.rank / tensor_parallel_degree(config=config))


def _lora_visible_activation_extra_fraction(*, config: ConfigLike) -> float:
    """Return the retained fraction of LoRA-only visible activation extras.

    Args:
        config: Estimator or measurement config.

    Returns:
        Fraction of LoRA-only visible outputs retained under the current
        distributed mode.
    """

    if config.tuning_mode != "lora":
        return 0.0
    if config.distributed_mode == "zero2":
        return 1.0 / 6.0
    if config.distributed_mode == "zero3":
        return 1.0
    return 0.0


def _uses_checkpointed_distributed_lora_proxy(
    *,
    config: ConfigLike,
    checkpointed: bool,
) -> bool:
    """Return whether checkpointed distributed LoRA uses the slim proxy path.

    Args:
        config: Estimator or measurement config.
        checkpointed: Whether gradient checkpointing is enabled.

    Returns:
        Whether the run is a non-single-GPU checkpointed LoRA configuration.
    """

    return (
        checkpointed
        and config.tuning_mode == "lora"
        and config.distributed_mode != "single_gpu"
    )


def _single_gpu_lora_visible_extra_fraction(*, model_spec: ModelSpec) -> float:
    """Return retained LoRA-visible extra fraction for single-GPU non-widened paths.

    Args:
        model_spec: Inspected model summary.

    Returns:
        Fraction of LoRA-only visible extras retained for non-widened
        single-GPU paths.
    """

    if model_spec.attention.num_key_value_heads < model_spec.attention.num_query_heads:
        return 0.5
    return 1.0 / 3.0


def _linear_output_elements_per_block(
    *, model_spec: ModelSpec, config: ConfigLike
) -> int:
    """Return first-block linear outputs visible to hooks.

    Args:
        model_spec: Inspected model summary.
        config: Estimator or measurement config.

    Returns:
        TP-local linear-output elements produced by one block.
    """

    return sum(
        local_linear_output_dim(layer_spec=layer, config=config)
        for layer in _first_block_layers(model_spec=model_spec)
    )


def _aux_output_elements_per_block(*, model_spec: ModelSpec, config: ConfigLike) -> int:
    """Return first-block non-linear outputs visible to hooks.

    Args:
        model_spec: Inspected model summary.
        config: Estimator or measurement config.

    Returns:
        Additional hook-visible outputs beyond linear modules.
    """

    query_layer = _first_block_layer_by_role(
        model_spec=model_spec,
        roles=(LINEAR_ROLE_ATTENTION_QUERY, LINEAR_ROLE_ATTENTION_QKV),
    )
    key_layer = _first_block_layer_by_role(
        model_spec=model_spec,
        roles=(LINEAR_ROLE_ATTENTION_KEY,),
    )
    query_elements = (
        local_linear_output_dim(layer_spec=query_layer, config=config)
        if query_layer is not None
        else 0
    )
    key_elements = (
        local_linear_output_dim(layer_spec=key_layer, config=config)
        if key_layer is not None
        else 0
    )
    return query_elements + key_elements + (5 * model_spec.hidden_size) + (
        model_spec.intermediate_size
    )


def _lora_hook_extra_elements_per_block(
    *, model_spec: ModelSpec, config: ConfigLike
) -> int:
    """Return LoRA wrapper outputs visible to hooks in one block.

    Args:
        model_spec: Inspected model summary.
        config: Estimator or measurement config.

    Returns:
        Extra hook-visible elements produced by LoRA wrappers.
    """

    if config.lora is None:
        return 0
    extra_elements = 0
    local_rank = _local_lora_rank(config=config)
    for layer in _first_block_layers(model_spec=model_spec):
        if layer.module_name.endswith(config.lora.target_modules):
            extra_elements += (
                (3 * local_linear_output_dim(layer_spec=layer, config=config))
                + (2 * local_linear_input_dim(layer_spec=layer, config=config))
                + local_rank
            )
    return extra_elements


def _base_hook_visible_activation_bytes(
    *, model_spec: ModelSpec, config: ConfigLike
) -> int:
    """Return base-model hook-visible activation bytes.

    Args:
        model_spec: Inspected model summary.
        config: Estimator or measurement config.

    Returns:
        First-block base-model hook-visible activations scaled over depth.
    """

    visible_elements = _linear_output_elements_per_block(
        model_spec=model_spec,
        config=config,
    ) + _aux_output_elements_per_block(model_spec=model_spec, config=config)
    return (
        _sequence_local_token_count(model_spec=model_spec, config=config)
        * model_spec.num_layers
        * visible_elements
        * bytes_for_dtype(config.weight_dtype)
    )


def _hook_visible_activation_bytes(*, model_spec: ModelSpec, config: ConfigLike) -> int:
    """Return hook-visible activation bytes including LoRA wrapper extras.

    Args:
        model_spec: Inspected model summary.
        config: Estimator or measurement config.

    Returns:
        Hook-visible activations observed after one forward microstep.
    """

    hook_visible_bytes = _base_hook_visible_activation_bytes(
        model_spec=model_spec,
        config=config,
    )
    if config.tuning_mode != "lora":
        return hook_visible_bytes
    return hook_visible_bytes + (
        _sequence_local_token_count(model_spec=model_spec, config=config)
        * model_spec.num_layers
        * _lora_hook_extra_elements_per_block(
            model_spec=model_spec,
            config=config,
        )
        * bytes_for_dtype(config.weight_dtype)
    )


def _visible_propagation_bytes(
    *,
    model_spec: ModelSpec,
    config: ConfigLike,
    base_hook_visible_bytes: int,
    hook_visible_bytes: int,
) -> int:
    """Return visible outputs that must remain for backward propagation.

    Args:
        model_spec: Inspected model summary.
        config: Estimator or measurement config.
        base_hook_visible_bytes: Base-model hook-visible outputs.
        hook_visible_bytes: Hook-visible outputs including LoRA extras.

    Returns:
        Visible activation bytes retained for backward propagation.
    """

    expanded_query_width = model_spec.attention.expanded_query_width(
        hidden_size=model_spec.hidden_size
    )
    if config.tuning_mode != "lora":
        return base_hook_visible_bytes
    if not config.is_zero_mode():
        if config.gradient_checkpointing or expanded_query_width > 0:
            return hook_visible_bytes
        extra_visible_bytes = max(0, hook_visible_bytes - base_hook_visible_bytes)
        return base_hook_visible_bytes + _scaled_int(
            value=extra_visible_bytes,
            factor=_single_gpu_lora_visible_extra_fraction(model_spec=model_spec),
        )
    if config.gradient_checkpointing or expanded_query_width > 0:
        return hook_visible_bytes
    extra_visible_bytes = max(0, hook_visible_bytes - base_hook_visible_bytes)
    return base_hook_visible_bytes + _scaled_int(
        value=extra_visible_bytes,
        factor=_lora_visible_activation_extra_fraction(config=config),
    )


def _saved_input_group_for_layer(
    *, layer_spec: ModelLinearLayerSpec
) -> _SavedInputGroup:
    """Return the unique retained-input source represented by one linear layer.

    Args:
        layer_spec: One inspected trainable linear layer.

    Returns:
        Group label indicating whether this layer shares its saved input with
        other consumers in the same block or is a one-off global tensor.
    """

    role = layer_spec.role
    if role in {
        LINEAR_ROLE_ATTENTION_QUERY,
        LINEAR_ROLE_ATTENTION_KEY,
        LINEAR_ROLE_ATTENTION_VALUE,
        LINEAR_ROLE_ATTENTION_QKV,
    }:
        return _SavedInputGroup(name="attention_input_hidden", repeats_per_block=True)
    if role == LINEAR_ROLE_ATTENTION_OUTPUT:
        return _SavedInputGroup(name="attention_output_context", repeats_per_block=True)
    if role in {
        LINEAR_ROLE_MLP_GATE,
        LINEAR_ROLE_MLP_UP,
        LINEAR_ROLE_MLP_GATE_UP,
    }:
        return _SavedInputGroup(name="mlp_input_hidden", repeats_per_block=True)
    if role == LINEAR_ROLE_MLP_DOWN:
        return _SavedInputGroup(name="mlp_down_context", repeats_per_block=True)
    if role == LINEAR_ROLE_ROUTER:
        return _SavedInputGroup(name="router_input_hidden", repeats_per_block=True)
    if role == LINEAR_ROLE_LM_HEAD:
        return _SavedInputGroup(name="lm_head_input_hidden", repeats_per_block=False)
    return _SavedInputGroup(
        name=f"fallback:{layer_spec.category}:{role}",
        repeats_per_block=True,
    )


def _saved_input_elements(
    *,
    model_spec: ModelSpec,
    config: ConfigLike,
    checkpointed: bool,
) -> int:
    """Return deduplicated saved-input elements before dtype scaling.

    Args:
        model_spec: Inspected model summary.
        config: Estimator or measurement config.
        checkpointed: Whether checkpointing is enabled.

    Returns:
        Local saved-input element count after deduplicating shared source tensors.

    Example:
        Separate Q/K/V projections fed by one hidden state contribute that hidden
        state once, not three times.
    """

    block_group_inputs: dict[str, int] = {}
    global_group_inputs: dict[str, int] = {}
    for layer in _target_layers(model_spec=model_spec, config=config):
        group = _saved_input_group_for_layer(layer_spec=layer)
        group_inputs = (
            block_group_inputs if group.repeats_per_block else global_group_inputs
        )
        group_inputs[group.name] = max(
            group_inputs.get(group.name, 0),
            local_linear_input_dim(layer_spec=layer, config=config),
        )
    segment_layers = (
        CHECKPOINT_SEGMENT_LAYERS if checkpointed else model_spec.num_layers
    )
    return (
        segment_layers * sum(block_group_inputs.values())
        + sum(global_group_inputs.values())
    )


def _saved_linear_input_bytes(
    *,
    model_spec: ModelSpec,
    config: ConfigLike,
    checkpointed: bool,
) -> int:
    """Return saved trainable linear-input bytes."""

    return (
        _sequence_local_token_count(model_spec=model_spec, config=config)
        * _saved_input_elements(
            model_spec=model_spec,
            config=config,
            checkpointed=checkpointed,
        )
        * bytes_for_dtype(config.weight_dtype)
    )


def _checkpoint_resident_block_input_bytes(
    *,
    model_spec: ModelSpec,
    config: ConfigLike,
    checkpointed: bool,
) -> int:
    """Return one block-input snapshot per layer for checkpointed single-GPU LoRA.

    Args:
        model_spec: Inspected model summary.
        config: Estimator or measurement config.
        checkpointed: Whether checkpointing is enabled.

    Returns:
        Bytes for the largest repeating block-input source kept resident across
        the checkpointed forward pass.
    """

    if not checkpointed or config.tuning_mode != "lora":
        return 0
    if config.distributed_mode != "single_gpu":
        return 0
    block_group_inputs: dict[str, int] = {}
    for layer in _target_layers(model_spec=model_spec, config=config):
        group = _saved_input_group_for_layer(layer_spec=layer)
        if not group.repeats_per_block:
            continue
        block_group_inputs[group.name] = max(
            block_group_inputs.get(group.name, 0),
            local_linear_input_dim(layer_spec=layer, config=config),
        )
    resident_elements = max(block_group_inputs.values(), default=0)
    return (
        _sequence_local_token_count(model_spec=model_spec, config=config)
        * model_spec.num_layers
        * resident_elements
        * bytes_for_dtype(config.weight_dtype)
    )


def _residual_norm_bytes(
    *, model_spec: ModelSpec, config: ConfigLike, checkpointed: bool
) -> int:
    """Return retained residual and normalization bytes."""

    segment_layers = (
        CHECKPOINT_SEGMENT_LAYERS if checkpointed else model_spec.num_layers
    )
    return (
        _sequence_local_token_count(model_spec=model_spec, config=config)
        * segment_layers
        * 2
        * model_spec.hidden_size
        * bytes_for_dtype(config.weight_dtype)
    )


def _checkpoint_boundary_bytes(*, model_spec: ModelSpec, config: ConfigLike) -> int:
    """Return checkpoint-boundary hidden-state bytes."""

    return (
        _sequence_local_token_count(model_spec=model_spec, config=config)
        * (model_spec.num_layers + 1)
        * model_spec.hidden_size
        * bytes_for_dtype(config.weight_dtype)
    )


def _mlp_intermediate_elements(*, model_spec: ModelSpec, config: ConfigLike) -> int:
    """Return TP-local MLP intermediate elements retained by one block.

    Args:
        model_spec: Inspected model summary.
        config: Estimator or measurement config.

    Returns:
        Local MLP intermediate width for one transformer block.
    """

    candidate_elements = []
    for layer in _first_block_layers(model_spec=model_spec):
        if layer.role in {
            LINEAR_ROLE_MLP_GATE,
            LINEAR_ROLE_MLP_UP,
            LINEAR_ROLE_MLP_GATE_UP,
        }:
            candidate_elements.append(
                local_linear_output_dim(layer_spec=layer, config=config)
            )
        if layer.role == LINEAR_ROLE_MLP_DOWN:
            candidate_elements.append(
                local_linear_input_dim(layer_spec=layer, config=config)
            )
    if candidate_elements:
        return max(candidate_elements)
    return model_spec.intermediate_size


def _mlp_intermediate_bytes(
    *, model_spec: ModelSpec, config: ConfigLike, checkpointed: bool
) -> int:
    """Return retained MLP intermediate bytes for full fine-tuning.

    Args:
        model_spec: Inspected model summary.
        config: Estimator or measurement config.
        checkpointed: Whether checkpointing is enabled.

    Returns:
        Retained MLP-intermediate bytes, or zero for LoRA.
    """

    if config.tuning_mode != "full_ft":
        return 0
    segment_layers = (
        CHECKPOINT_SEGMENT_LAYERS if checkpointed else model_spec.num_layers
    )
    return (
        _sequence_local_token_count(model_spec=model_spec, config=config)
        * segment_layers
        * _mlp_intermediate_elements(model_spec=model_spec, config=config)
        * bytes_for_dtype(config.weight_dtype)
    )


def _widened_attention_context_elements(
    *, model_spec: ModelSpec, config: ConfigLike
) -> tuple[int, int, int, int, int]:
    """Return TP-local widened attention-path element counts for one block.

    Args:
        model_spec: Inspected model summary.
        config: Estimator or measurement config.

    Returns:
        Tuple of query-extra, key, value, output-input, and output-output
        context elements. All values are zero when no widened path exists.
    """

    extra_query_width = max(
        model_spec.attention.query_width - model_spec.hidden_size,
        0,
    )
    if extra_query_width == 0 and (
        model_spec.attention.output_proj_input_width <= model_spec.hidden_size
    ):
        return 0, 0, 0, 0, 0
    tp_degree = tensor_parallel_degree(config=config)
    return (
        math.ceil(extra_query_width / tp_degree),
        math.ceil(model_spec.attention.key_width / tp_degree),
        math.ceil(model_spec.attention.value_width / tp_degree),
        math.ceil(model_spec.attention.output_proj_input_width / tp_degree),
        model_spec.hidden_size,
    )


def _widened_attention_context_bytes(
    *, model_spec: ModelSpec, config: ConfigLike, checkpointed: bool
) -> tuple[int, int, int, int, int]:
    """Return widened attention-path bytes split into named tensor classes.

    Args:
        model_spec: Inspected model summary.
        config: Estimator or measurement config.
        checkpointed: Whether checkpointing is enabled.

    Returns:
        Bytes for query-extra, key, value, output-input, and output-output
        retained contexts.
    """

    if config.tuning_mode != "full_ft":
        return 0, 0, 0, 0, 0
    segment_layers = (
        CHECKPOINT_SEGMENT_LAYERS if checkpointed else model_spec.num_layers
    )
    token_count = _sequence_local_token_count(model_spec=model_spec, config=config)
    element_bytes = bytes_for_dtype(config.weight_dtype)
    (
        query_output_elements,
        key_output_elements,
        value_output_elements,
        output_proj_input_elements,
        output_proj_output_elements,
    ) = _widened_attention_context_elements(
        model_spec=model_spec,
        config=config,
    )
    return (
        token_count * segment_layers * query_output_elements * element_bytes,
        token_count * segment_layers * key_output_elements * element_bytes,
        token_count * segment_layers * value_output_elements * element_bytes,
        token_count * segment_layers * output_proj_input_elements * element_bytes,
        token_count * segment_layers * output_proj_output_elements * element_bytes,
    )


def _expanded_query_saved_bytes(
    *,
    model_spec: ModelSpec,
    config: ConfigLike,
    checkpointed: bool,
) -> int:
    """Return explicit saved context created by expanded-query attention."""

    return sum(
        _widened_attention_context_bytes(
            model_spec=model_spec,
            config=config,
            checkpointed=checkpointed,
        )
    )


def _attention_saved_bytes(*, model_spec: ModelSpec, config: ConfigLike) -> int:
    """Return retained attention-score bytes for eager attention backends."""

    if config.normalized_attention_backend() != "standard":
        return 0
    score_bytes = bytes_for_dtype(ATTENTION_SCORE_DTYPE)
    local_query_heads = model_spec.attention.local_query_heads(
        tensor_parallel_degree=tensor_parallel_degree(config=config)
    )
    local_key_value_heads = model_spec.attention.local_key_value_heads(
        tensor_parallel_degree=tensor_parallel_degree(config=config)
    )
    effective_key_length = model_spec.attention.effective_window(
        sequence_length=config.max_seq_len
    )
    effective_head_count = max(local_query_heads, local_key_value_heads)
    return (
        config.micro_batch_size_per_gpu
        * effective_head_count
        * config.max_seq_len
        * effective_key_length
        * model_spec.num_layers
        * score_bytes
    )


def _loss_state_bytes(*, model_spec: ModelSpec, config: ConfigLike) -> int:
    """Return retained loss/logit state bytes."""

    return _scaled_int(
        value=(
            _sequence_local_token_count(model_spec=model_spec, config=config)
            * math.ceil(
                model_spec.vocab_size
                / (
                    tensor_parallel_degree(config=config)
                    if config.vocab_parallel_logits
                    else 1
                )
            )
            * bytes_for_dtype(config.loss_output_resolved_dtype())
        ),
        factor=config.loss_output_logits_fraction,
    )


def _lora_low_rank_bytes(
    *, model_spec: ModelSpec, config: ConfigLike, checkpointed: bool
) -> int:
    """Return retained LoRA low-rank intermediate bytes."""

    if config.lora is None:
        return 0
    segment_layers = (
        CHECKPOINT_SEGMENT_LAYERS if checkpointed else model_spec.num_layers
    )
    target_rank_sum = sum(
        (
            math.ceil(config.lora.rank / tensor_parallel_degree(config=config))
            if config.uses_tensor_parallel()
            else config.lora.rank
        )
        for layer in _first_block_layers(model_spec=model_spec)
        if layer.module_name.endswith(config.lora.target_modules)
    )
    return (
        _sequence_local_token_count(model_spec=model_spec, config=config)
        * segment_layers
        * target_rank_sum
        * bytes_for_dtype(config.adapter_parameter_dtype())
    )


def _saved_input_overlap_bytes(
    *,
    model_spec: ModelSpec,
    config: ConfigLike,
    checkpointed: bool,
    saved_linear_input_bytes: int,
) -> int:
    """Return explicit overlap between saved inputs and visible propagation.

    Args:
        model_spec: Inspected model summary.
        config: Estimator or measurement config.
        checkpointed: Whether checkpointing is enabled.
        saved_linear_input_bytes: Saved linear-input contexts before overlap.

    Returns:
        Saved-input bytes that should be deduplicated against visible outputs.

    Example:
        For large single-GPU LoRA SDPA configurations, this term removes saved
        input tensors that alias visible propagation tensors.
    """

    if checkpointed or config.tuning_mode != "lora":
        return 0
    if config.distributed_mode != "single_gpu":
        return 0
    if config.normalized_attention_backend() != "sdpa":
        return 0
    if (
        model_spec.hidden_size
        < config.single_lora_sdpa_saved_input_overlap_min_hidden_size
    ):
        return 0
    if model_spec.num_layers < config.single_lora_sdpa_saved_input_overlap_min_layers:
        return 0
    overlap_bytes = _scaled_int(
        value=saved_linear_input_bytes,
        factor=config.single_lora_sdpa_saved_input_overlap_fraction,
    )
    return min(saved_linear_input_bytes, overlap_bytes)


def _lora_backward_logits_context_bytes(
    *,
    config: ConfigLike,
    checkpointed: bool,
    loss_state_bytes: int,
) -> int:
    """Return backward-local logits context retained for flash2 LoRA.

    Args:
        config: Estimator or measurement config.
        checkpointed: Whether checkpointing is enabled.
        loss_state_bytes: Forward loss/logit state bytes.

    Returns:
        Backward-local logits context bytes.
    """

    if checkpointed or config.tuning_mode != "lora":
        return 0
    if config.distributed_mode != "single_gpu":
        return 0
    if config.normalized_attention_backend() != "flash2":
        return 0
    return _scaled_int(
        value=loss_state_bytes,
        factor=config.single_lora_flash2_backward_logits_fraction,
    )


def _retained_forward_proxy_bytes(
    *,
    model_spec: ModelSpec,
    config: ConfigLike,
    checkpointed: bool,
    visible_propagation_bytes: int,
    checkpoint_boundary_bytes: int,
    attention_saved_bytes: int,
    loss_state_bytes: int,
    lora_low_rank_bytes: int,
    saved_linear_input_bytes: int,
    saved_input_overlap_bytes: int,
    mlp_intermediate_bytes: int,
    expanded_query_saved_bytes: int,
    checkpoint_resident_block_input_bytes: int,
) -> int:
    """Return the forward-retained activation proxy used for calibration.

    Args:
        model_spec: Inspected model summary.
        config: Estimator or measurement config.
        checkpointed: Whether checkpointing is enabled.
        visible_propagation_bytes: Hook-visible outputs retained for backward.
        checkpoint_boundary_bytes: Checkpoint boundary hidden states.
        attention_saved_bytes: Eager attention score retention.
        loss_state_bytes: Forward loss/logit state bytes.
        lora_low_rank_bytes: LoRA low-rank retained intermediates.
        saved_linear_input_bytes: Saved linear-input contexts.
        saved_input_overlap_bytes: Explicit overlap with visible propagation.
        mlp_intermediate_bytes: Retained MLP intermediates.
        expanded_query_saved_bytes: Explicit widened-path retained contexts.
        checkpoint_resident_block_input_bytes: One resident block-input snapshot
            per layer for checkpointed single-GPU LoRA.

    Returns:
        Analytical proxy for retained end-of-forward activations.

    Example:
        >>> # See tests for concrete retained-proxy constructions.
    """

    if checkpointed:
        checkpoint_context_bytes = checkpoint_boundary_bytes
        if config.tuning_mode == "lora" and config.distributed_mode == "single_gpu":
            return (
                checkpoint_context_bytes
                + attention_saved_bytes
                + loss_state_bytes
                + lora_low_rank_bytes
                + checkpoint_resident_block_input_bytes
            )
        if _uses_checkpointed_distributed_lora_proxy(
            config=config,
            checkpointed=checkpointed,
        ):
            return (
                checkpoint_context_bytes
                + attention_saved_bytes
                + loss_state_bytes
                + lora_low_rank_bytes
            )
        return checkpoint_context_bytes + attention_saved_bytes
    distinct_saved_linear_input_bytes = max(
        0,
        saved_linear_input_bytes - saved_input_overlap_bytes,
    )
    proxy_bytes = visible_propagation_bytes + distinct_saved_linear_input_bytes
    if config.tuning_mode == "full_ft":
        proxy_bytes += mlp_intermediate_bytes + expanded_query_saved_bytes
    return proxy_bytes + attention_saved_bytes


def _parameter_gradient_context_bytes(
    *,
    saved_linear_input_bytes: int,
    mlp_intermediate_bytes: int,
) -> int:
    """Return backward-local contexts used to form trainable parameter gradients.

    Args:
        saved_linear_input_bytes: Saved trainable linear-input contexts.
        mlp_intermediate_bytes: Retained MLP intermediates for full fine-tuning.

    Returns:
        Explicit backward-only parameter-gradient context bytes.
    """

    return saved_linear_input_bytes + mlp_intermediate_bytes


def _backward_phase_activation_bytes(
    *,
    model_spec: ModelSpec,
    config: ConfigLike,
    checkpointed: bool,
    retained_forward_proxy_bytes: int,
    checkpoint_boundary_bytes: int,
    checkpoint_resident_block_input_bytes: int,
    visible_propagation_bytes: int,
    parameter_gradient_context_bytes: int,
    saved_linear_input_bytes: int,
    mlp_intermediate_bytes: int,
    residual_norm_bytes: int,
    attention_saved_bytes: int,
    lora_low_rank_bytes: int,
    lora_backward_logits_context_bytes: int,
    expanded_query_saved_bytes: int,
) -> int:
    """Return the backward-phase retained activation estimate.

    Args:
        model_spec: Inspected model summary.
        config: Estimator or measurement config.
        checkpointed: Whether checkpointing is enabled.
        retained_forward_proxy_bytes: End-of-forward retained proxy.
        checkpoint_boundary_bytes: Checkpoint boundary hidden states.
        checkpoint_resident_block_input_bytes: One resident block-input
            snapshot per layer for checkpointed single-GPU LoRA.
        visible_propagation_bytes: Hook-visible outputs retained for backward.
        saved_linear_input_bytes: Saved linear-input contexts.
        mlp_intermediate_bytes: Retained MLP intermediates.
        residual_norm_bytes: Residual and normalization states.
        attention_saved_bytes: Eager attention score retention.
        lora_low_rank_bytes: LoRA low-rank backward contexts.
        lora_backward_logits_context_bytes: Backward-local logits context.
        expanded_query_saved_bytes: Explicit widened-path retained contexts.

    Returns:
        Backward-phase retained activation bytes.
    """

    if checkpointed:
        checkpoint_context_bytes = checkpoint_boundary_bytes
        if _uses_checkpointed_distributed_lora_proxy(
            config=config,
            checkpointed=checkpointed,
        ):
            return (
                retained_forward_proxy_bytes
                + saved_linear_input_bytes
                + residual_norm_bytes
            )
        return (
            checkpoint_context_bytes
            + checkpoint_resident_block_input_bytes
            + saved_linear_input_bytes
            + mlp_intermediate_bytes
            + residual_norm_bytes
            + attention_saved_bytes
            + lora_low_rank_bytes
            + lora_backward_logits_context_bytes
            + expanded_query_saved_bytes
        )
    backward_local_bytes = (
        parameter_gradient_context_bytes
        + residual_norm_bytes
        + lora_low_rank_bytes
        + lora_backward_logits_context_bytes
    )
    return retained_forward_proxy_bytes + backward_local_bytes + attention_saved_bytes


def _checkpointed_sharded_lora_backward_visible_bytes(
    *,
    config: ConfigLike,
    visible_propagation_bytes: int,
    checkpoint_boundary_bytes: int,
) -> int:
    """Return backward-visible bytes kept under checkpointed ZeRO LoRA.

    Args:
        config: Estimator or measurement config.
        visible_propagation_bytes: Visible outputs retained without checkpointing.
        checkpoint_boundary_bytes: Checkpoint boundary hidden states already
            counted in the forward proxy.

    Returns:
        Extra backward-time visible bytes not already represented by checkpoint
        boundaries.
    """

    if config.tuning_mode != "lora" or not config.gradient_checkpointing:
        return 0
    return 0


def build_activation_terms(
    *,
    model_spec: ModelSpec,
    config: ConfigLike,
) -> ActivationTerms:
    """Build retained activation terms for one analytical estimate.

    Args:
        model_spec: Inspected model summary.
        config: Estimator or measurement config.

    Returns:
        Forward and backward retained activation terms plus diagnostics.
    """

    checkpoint_boundary_bytes = _checkpoint_boundary_bytes(
        model_spec=model_spec,
        config=config,
    )
    attention_saved_bytes = _attention_saved_bytes(model_spec=model_spec, config=config)
    loss_state_bytes = _loss_state_bytes(model_spec=model_spec, config=config)
    checkpointed = config.gradient_checkpointing
    saved_linear_input_bytes = _saved_linear_input_bytes(
        model_spec=model_spec,
        config=config,
        checkpointed=checkpointed,
    )
    checkpoint_resident_block_input_bytes = _checkpoint_resident_block_input_bytes(
        model_spec=model_spec,
        config=config,
        checkpointed=checkpointed,
    )
    saved_input_overlap_bytes = _saved_input_overlap_bytes(
        model_spec=model_spec,
        config=config,
        checkpointed=checkpointed,
        saved_linear_input_bytes=saved_linear_input_bytes,
    )
    mlp_intermediate_bytes = _mlp_intermediate_bytes(
        model_spec=model_spec,
        config=config,
        checkpointed=checkpointed,
    )
    residual_norm_bytes = _residual_norm_bytes(
        model_spec=model_spec,
        config=config,
        checkpointed=checkpointed,
    )
    lora_low_rank_bytes = _lora_low_rank_bytes(
        model_spec=model_spec,
        config=config,
        checkpointed=checkpointed,
    )
    lora_backward_logits_context_bytes = _lora_backward_logits_context_bytes(
        config=config,
        checkpointed=checkpointed,
        loss_state_bytes=loss_state_bytes,
    )
    (
        query_output_context_bytes,
        key_output_context_bytes,
        value_output_context_bytes,
        output_proj_input_context_bytes,
        output_proj_output_context_bytes,
    ) = _widened_attention_context_bytes(
        model_spec=model_spec,
        config=config,
        checkpointed=checkpointed,
    )
    expanded_query_saved_bytes = (
        query_output_context_bytes
        + key_output_context_bytes
        + value_output_context_bytes
        + output_proj_input_context_bytes
        + output_proj_output_context_bytes
    )
    base_hook_visible_activation_bytes = _base_hook_visible_activation_bytes(
        model_spec=model_spec,
        config=config,
    )
    hook_visible_activation_bytes = _hook_visible_activation_bytes(
        model_spec=model_spec,
        config=config,
    )
    visible_propagation_bytes = _visible_propagation_bytes(
        model_spec=model_spec,
        config=config,
        base_hook_visible_bytes=base_hook_visible_activation_bytes,
        hook_visible_bytes=hook_visible_activation_bytes,
    )
    checkpointed_sharded_lora_backward_visible_bytes = (
        _checkpointed_sharded_lora_backward_visible_bytes(
            config=config,
            visible_propagation_bytes=visible_propagation_bytes,
            checkpoint_boundary_bytes=checkpoint_boundary_bytes,
        )
    )
    parameter_gradient_context_bytes = _parameter_gradient_context_bytes(
        saved_linear_input_bytes=saved_linear_input_bytes,
        mlp_intermediate_bytes=mlp_intermediate_bytes,
    )
    if config.normalized_attention_backend() == "standard":
        if checkpointed:
            retained_forward_proxy_bytes = (
                checkpoint_boundary_bytes + attention_saved_bytes
            )
            forward_phase_activation_bytes = (
                retained_forward_proxy_bytes + loss_state_bytes
            )
            backward_phase_activation_bytes = (
                checkpoint_boundary_bytes
                + saved_linear_input_bytes
                + residual_norm_bytes
                + attention_saved_bytes
                + lora_low_rank_bytes
                + lora_backward_logits_context_bytes
                + expanded_query_saved_bytes
            )
        else:
            backward_phase_activation_bytes = (
                saved_linear_input_bytes
                + residual_norm_bytes
                + attention_saved_bytes
                + lora_low_rank_bytes
                + lora_backward_logits_context_bytes
                + expanded_query_saved_bytes
            )
            retained_forward_proxy_bytes = backward_phase_activation_bytes
            forward_phase_activation_bytes = (
                backward_phase_activation_bytes + loss_state_bytes
            )
    else:
        retained_forward_proxy_bytes = _retained_forward_proxy_bytes(
            model_spec=model_spec,
            config=config,
            checkpointed=checkpointed,
            visible_propagation_bytes=visible_propagation_bytes,
            checkpoint_boundary_bytes=checkpoint_boundary_bytes,
            attention_saved_bytes=attention_saved_bytes,
            loss_state_bytes=loss_state_bytes,
            lora_low_rank_bytes=lora_low_rank_bytes,
            saved_linear_input_bytes=saved_linear_input_bytes,
            saved_input_overlap_bytes=saved_input_overlap_bytes,
            mlp_intermediate_bytes=mlp_intermediate_bytes,
            expanded_query_saved_bytes=expanded_query_saved_bytes,
            checkpoint_resident_block_input_bytes=checkpoint_resident_block_input_bytes,
        )
        if checkpointed and config.tuning_mode == "lora":
            forward_phase_activation_bytes = retained_forward_proxy_bytes
        else:
            forward_phase_activation_bytes = (
                retained_forward_proxy_bytes + loss_state_bytes
            )
        backward_phase_activation_bytes = _backward_phase_activation_bytes(
            model_spec=model_spec,
            config=config,
            checkpointed=checkpointed,
            retained_forward_proxy_bytes=retained_forward_proxy_bytes,
            checkpoint_boundary_bytes=checkpoint_boundary_bytes,
            checkpoint_resident_block_input_bytes=checkpoint_resident_block_input_bytes,
            visible_propagation_bytes=visible_propagation_bytes,
            parameter_gradient_context_bytes=parameter_gradient_context_bytes,
            saved_linear_input_bytes=saved_linear_input_bytes,
            mlp_intermediate_bytes=mlp_intermediate_bytes,
            residual_norm_bytes=residual_norm_bytes,
            attention_saved_bytes=attention_saved_bytes,
            lora_low_rank_bytes=lora_low_rank_bytes,
            lora_backward_logits_context_bytes=lora_backward_logits_context_bytes,
            expanded_query_saved_bytes=expanded_query_saved_bytes,
        )
        backward_phase_activation_bytes += (
            checkpointed_sharded_lora_backward_visible_bytes
        )
    return ActivationTerms(
        debug=ActivationDebug(
            base_hook_visible_activation_bytes=base_hook_visible_activation_bytes,
            visible_propagation_bytes=visible_propagation_bytes,
            checkpoint_resident_block_input_bytes=checkpoint_resident_block_input_bytes,
            saved_linear_input_bytes=saved_linear_input_bytes,
            saved_input_overlap_bytes=saved_input_overlap_bytes,
            mlp_intermediate_bytes=mlp_intermediate_bytes,
            parameter_gradient_context_bytes=parameter_gradient_context_bytes,
            residual_norm_bytes=residual_norm_bytes,
            checkpoint_boundary_bytes=checkpoint_boundary_bytes,
            attention_saved_bytes=attention_saved_bytes,
            loss_state_bytes=loss_state_bytes,
            lora_low_rank_bytes=lora_low_rank_bytes,
            lora_backward_logits_context_bytes=lora_backward_logits_context_bytes,
            expanded_query_saved_bytes=expanded_query_saved_bytes,
            query_output_context_bytes=query_output_context_bytes,
            key_output_context_bytes=key_output_context_bytes,
            value_output_context_bytes=value_output_context_bytes,
            output_proj_input_context_bytes=output_proj_input_context_bytes,
            output_proj_output_context_bytes=output_proj_output_context_bytes,
            checkpointed_sharded_lora_backward_visible_bytes=(
                checkpointed_sharded_lora_backward_visible_bytes
            ),
            retained_forward_proxy_bytes=retained_forward_proxy_bytes,
            forward_phase_activation_bytes=forward_phase_activation_bytes,
            backward_phase_activation_bytes=backward_phase_activation_bytes,
            hook_visible_activation_bytes=hook_visible_activation_bytes,
        )
    )
