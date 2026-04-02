"""Retained activation accounting for the simplified analytical estimator."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Union

from .parallelism_model import (
    local_linear_input_dim,
    local_linear_output_dim,
    sequence_parallel_divisor,
    tensor_parallel_degree,
)
from .types import (
    ActivationDebug,
    EstimatorConfig,
    MeasurementConfig,
    ModelLinearLayerSpec,
    ModelSpec,
)
from .utils import bytes_for_dtype


ConfigLike = Union[EstimatorConfig, MeasurementConfig]
CHECKPOINT_SEGMENT_LAYERS = 1
ATTENTION_SCORE_DTYPE = "fp32"


@dataclass(frozen=True)
class ActivationTerms:
    """Retained activation terms for one analytical estimate."""

    debug: ActivationDebug


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


def _average_block_layers(*, model_spec: ModelSpec) -> tuple[ModelLinearLayerSpec, ...]:
    """Return one block's worth of linear layers by averaging over full-layer stats."""

    return tuple(model_spec.trainable_linear_layers)


def _target_layers(
    *, model_spec: ModelSpec, config: ConfigLike
) -> tuple[ModelLinearLayerSpec, ...]:
    """Return the trainable linear layers relevant for saved activation state."""

    if config.tuning_mode == "full_ft" or config.lora is None:
        return _average_block_layers(model_spec=model_spec)
    return tuple(
        layer
        for layer in _average_block_layers(model_spec=model_spec)
        if layer.module_name.endswith(config.lora.target_modules)
    )


def _saved_linear_input_bytes(
    *,
    model_spec: ModelSpec,
    config: ConfigLike,
    checkpointed: bool,
) -> int:
    """Return saved trainable linear-input bytes."""

    segment_layers = (
        CHECKPOINT_SEGMENT_LAYERS if checkpointed else model_spec.num_layers
    )
    input_elements = sum(
        local_linear_input_dim(layer_spec=layer, config=config)
        for layer in _target_layers(model_spec=model_spec, config=config)
    )
    return (
        _sequence_local_token_count(model_spec=model_spec, config=config)
        * segment_layers
        * input_elements
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


def _expanded_query_saved_bytes(
    *,
    model_spec: ModelSpec,
    config: ConfigLike,
    checkpointed: bool,
) -> int:
    """Return explicit saved context created by expanded-query attention."""

    q_proj = next(
        (
            layer
            for layer in model_spec.trainable_linear_layers
            if layer.module_name.endswith("q_proj")
        ),
        None,
    )
    o_proj = next(
        (
            layer
            for layer in model_spec.trainable_linear_layers
            if layer.module_name.endswith("o_proj")
        ),
        None,
    )
    if q_proj is None or o_proj is None or q_proj.output_dim <= model_spec.hidden_size:
        return 0
    segment_layers = (
        CHECKPOINT_SEGMENT_LAYERS if checkpointed else model_spec.num_layers
    )
    extra_elements = (q_proj.output_dim - model_spec.hidden_size) + max(
        o_proj.input_dim - model_spec.hidden_size, 0
    )
    return (
        _sequence_local_token_count(model_spec=model_spec, config=config)
        * segment_layers
        * math.ceil(extra_elements / tensor_parallel_degree(config=config))
        * bytes_for_dtype(config.weight_dtype)
    )


def _attention_saved_bytes(*, model_spec: ModelSpec, config: ConfigLike) -> int:
    """Return retained attention-score bytes for eager attention backends."""

    if config.normalized_attention_backend() != "standard":
        return 0
    token_count = _token_count(model_spec=model_spec, config=config)
    score_bytes = bytes_for_dtype(ATTENTION_SCORE_DTYPE)
    return (
        config.micro_batch_size_per_gpu
        * math.ceil(
            model_spec.num_attention_heads / tensor_parallel_degree(config=config)
        )
        * config.max_seq_len
        * config.max_seq_len
        * model_spec.num_layers
        * score_bytes
    )


def _loss_state_bytes(*, model_spec: ModelSpec, config: ConfigLike) -> int:
    """Return retained loss/logit state bytes."""

    return (
        _sequence_local_token_count(model_spec=model_spec, config=config)
        * math.ceil(
            model_spec.vocab_size
            / (
                tensor_parallel_degree(config=config)
                if config.vocab_parallel_logits
                else 1
            )
        )
        * bytes_for_dtype(config.weight_dtype)
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
        for _layer in _target_layers(model_spec=model_spec, config=config)
    )
    return (
        _sequence_local_token_count(model_spec=model_spec, config=config)
        * segment_layers
        * target_rank_sum
        * bytes_for_dtype(config.adapter_parameter_dtype())
    )


def _hook_visible_activation_bytes(*, model_spec: ModelSpec, config: ConfigLike) -> int:
    """Return hook-visible layer outputs as a diagnostic-only quantity."""

    visible_elements = sum(
        local_linear_output_dim(layer_spec=layer, config=config)
        for layer in model_spec.trainable_linear_layers
    )
    return (
        _sequence_local_token_count(model_spec=model_spec, config=config)
        * visible_elements
        * bytes_for_dtype(config.weight_dtype)
    )


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
    if checkpointed:
        forward_phase_activation_bytes = (
            checkpoint_boundary_bytes + attention_saved_bytes + loss_state_bytes
        )
        backward_phase_activation_bytes = (
            checkpoint_boundary_bytes
            + _saved_linear_input_bytes(
                model_spec=model_spec, config=config, checkpointed=True
            )
            + _residual_norm_bytes(
                model_spec=model_spec, config=config, checkpointed=True
            )
            + attention_saved_bytes
            + _lora_low_rank_bytes(
                model_spec=model_spec, config=config, checkpointed=True
            )
            + _expanded_query_saved_bytes(
                model_spec=model_spec, config=config, checkpointed=True
            )
        )
    else:
        backward_phase_activation_bytes = (
            _saved_linear_input_bytes(
                model_spec=model_spec, config=config, checkpointed=False
            )
            + _residual_norm_bytes(
                model_spec=model_spec, config=config, checkpointed=False
            )
            + attention_saved_bytes
            + _lora_low_rank_bytes(
                model_spec=model_spec, config=config, checkpointed=False
            )
            + _expanded_query_saved_bytes(
                model_spec=model_spec, config=config, checkpointed=False
            )
        )
        forward_phase_activation_bytes = (
            backward_phase_activation_bytes + loss_state_bytes
        )
    return ActivationTerms(
        debug=ActivationDebug(
            saved_linear_input_bytes=_saved_linear_input_bytes(
                model_spec=model_spec,
                config=config,
                checkpointed=checkpointed,
            ),
            residual_norm_bytes=_residual_norm_bytes(
                model_spec=model_spec,
                config=config,
                checkpointed=checkpointed,
            ),
            checkpoint_boundary_bytes=checkpoint_boundary_bytes,
            attention_saved_bytes=attention_saved_bytes,
            loss_state_bytes=loss_state_bytes,
            lora_low_rank_bytes=_lora_low_rank_bytes(
                model_spec=model_spec,
                config=config,
                checkpointed=checkpointed,
            ),
            expanded_query_saved_bytes=_expanded_query_saved_bytes(
                model_spec=model_spec,
                config=config,
                checkpointed=checkpointed,
            ),
            forward_phase_activation_bytes=forward_phase_activation_bytes,
            backward_phase_activation_bytes=backward_phase_activation_bytes,
            hook_visible_activation_bytes=_hook_visible_activation_bytes(
                model_spec=model_spec,
                config=config,
            ),
        )
    )
