"""Tensor- and sequence-parallel helpers for the structural estimator."""

from __future__ import annotations

import math
from typing import Union

from .types import (
    EstimatorConfig,
    MeasurementConfig,
    ModelLinearLayerSpec,
    ModelParameterSpec,
)


ConfigLike = Union[EstimatorConfig, MeasurementConfig]
TP_LINEAR_CATEGORIES = {"attention", "mlp"}
ROW_PARALLEL_TOKENS = ("o_proj", "down_proj", "c_proj", "wo")


def tensor_parallel_degree(*, config: ConfigLike) -> int:
    """Return the validated tensor-parallel degree."""

    return config.tensor_parallel_degree_resolved()


def sequence_parallel_divisor(*, config: ConfigLike) -> int:
    """Return the divisor applied to sequence-local activation tensors."""

    if not config.sequence_parallel or not config.uses_tensor_parallel():
        return 1
    return tensor_parallel_degree(config=config)


def is_tp_sharded_linear(
    *, layer_spec: ModelLinearLayerSpec, config: ConfigLike
) -> bool:
    """Return whether one linear layer is sharded by tensor parallelism."""

    return config.uses_tensor_parallel() and layer_spec.category in TP_LINEAR_CATEGORIES


def is_row_parallel_linear(*, layer_spec: ModelLinearLayerSpec) -> bool:
    """Return whether one linear layer behaves like a row-parallel Megatron op."""

    lower_name = layer_spec.module_name.lower()
    return any(token in lower_name for token in ROW_PARALLEL_TOKENS)


def local_linear_input_dim(
    *, layer_spec: ModelLinearLayerSpec, config: ConfigLike
) -> int:
    """Return the local saved-input width for one trainable linear layer."""

    if not is_tp_sharded_linear(layer_spec=layer_spec, config=config):
        return layer_spec.input_dim
    if not is_row_parallel_linear(layer_spec=layer_spec):
        return layer_spec.input_dim
    return math.ceil(layer_spec.input_dim / tensor_parallel_degree(config=config))


def local_linear_output_dim(
    *, layer_spec: ModelLinearLayerSpec, config: ConfigLike
) -> int:
    """Return the local output width for one trainable linear layer."""

    if not is_tp_sharded_linear(layer_spec=layer_spec, config=config):
        return layer_spec.output_dim
    if is_row_parallel_linear(layer_spec=layer_spec):
        return layer_spec.output_dim
    return math.ceil(layer_spec.output_dim / tensor_parallel_degree(config=config))


def tp_shard_divisor_for_parameter(
    *, parameter_spec: ModelParameterSpec, config: ConfigLike
) -> int:
    """Return the tensor-parallel divisor applied to one parameter tensor."""

    if not config.uses_tensor_parallel() or not parameter_spec.is_matrix():
        return 1
    if parameter_spec.category in TP_LINEAR_CATEGORIES:
        return tensor_parallel_degree(config=config)
    if parameter_spec.category == "embedding" and config.vocab_parallel_logits:
        return tensor_parallel_degree(config=config)
    return 1


def tp_shard_numel(*, numel: int, divisor: int) -> int:
    """Return the local element count for one sharded tensor."""

    return math.ceil(numel / max(divisor, 1))
