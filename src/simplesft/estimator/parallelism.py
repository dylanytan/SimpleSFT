"""Tensor- and sequence-parallel helpers for the structural estimator."""

from __future__ import annotations

import math
from typing import Union

from ..models.architecture_types import (
    TENSOR_PARALLEL_COLUMN,
    TENSOR_PARALLEL_ROW,
    TENSOR_PARALLEL_VOCAB,
)
from ..types import (
    EstimatorConfig,
    MeasurementConfig,
    ModelLinearLayerSpec,
    ModelParameterSpec,
)


ConfigLike = Union[EstimatorConfig, MeasurementConfig]


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
    """Return whether one linear layer is explicitly TP sharded."""

    return config.uses_tensor_parallel() and layer_spec.tensor_parallel_role in {
        TENSOR_PARALLEL_COLUMN,
        TENSOR_PARALLEL_ROW,
    }


def is_row_parallel_linear(*, layer_spec: ModelLinearLayerSpec) -> bool:
    """Return whether one linear layer is row parallel."""

    return layer_spec.tensor_parallel_role == TENSOR_PARALLEL_ROW


def local_linear_input_dim(
    *, layer_spec: ModelLinearLayerSpec, config: ConfigLike
) -> int:
    """Return the local saved-input width for one trainable linear layer."""

    if not is_tp_sharded_linear(layer_spec=layer_spec, config=config):
        return layer_spec.input_dim
    if is_row_parallel_linear(layer_spec=layer_spec):
        return math.ceil(layer_spec.input_dim / tensor_parallel_degree(config=config))
    return layer_spec.input_dim


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

    if not config.uses_tensor_parallel():
        return 1
    if parameter_spec.tensor_parallel_role in {
        TENSOR_PARALLEL_COLUMN,
        TENSOR_PARALLEL_ROW,
    }:
        return tensor_parallel_degree(config=config)
    if (
        parameter_spec.tensor_parallel_role == TENSOR_PARALLEL_VOCAB
        and config.vocab_parallel_logits
    ):
        return tensor_parallel_degree(config=config)
    return 1


def tp_shard_numel(*, numel: int, divisor: int) -> int:
    """Return the local element count for one sharded tensor."""

    return math.ceil(numel / max(divisor, 1))
