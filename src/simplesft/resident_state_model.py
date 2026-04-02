"""Resident-state accounting for the simplified analytical estimator."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union

from .optimizer_model import (
    optimizer_state_numel_for_parameter,
    trainable_parameter_specs,
)
from .parallelism_model import tp_shard_divisor_for_parameter, tp_shard_numel
from .types import (
    EstimatorConfig,
    LoRAConfig,
    MeasurementConfig,
    ModelParameterSpec,
    ModelSpec,
    ResidentStateDebug,
)
from .utils import bytes_for_dtype


ConfigLike = Union[EstimatorConfig, MeasurementConfig]
RUNTIME_FLOOR_GB_BY_MODE = {
    "single_gpu": 0.30,
    "ddp": 0.60,
    "zero2": 0.80,
    "zero3": 1.00,
}


@dataclass(frozen=True)
class ResidentStateTerms:
    """Resident memory terms for one analytical estimate."""

    debug: ResidentStateDebug
    trainable_parameter_count: int


def estimate_lora_parameter_count(
    *, model_spec: ModelSpec, lora_config: LoRAConfig
) -> int:
    """Return the LoRA adapter parameter count.

    Args:
        model_spec: Model summary containing trainable linear layers.
        lora_config: LoRA adapter settings.

    Returns:
        Total trainable LoRA adapter parameter count.
    """

    return sum(
        parameter_spec.numel()
        for parameter_spec in trainable_parameter_specs(
            model_spec=model_spec,
            config=EstimatorConfig(tuning_mode="lora", lora=lora_config),
        )
    )


def trainable_parameter_count(*, model_spec: ModelSpec, config: ConfigLike) -> int:
    """Return the trainable parameter count for one config."""

    return sum(
        parameter_spec.numel()
        for parameter_spec in trainable_parameter_specs(
            model_spec=model_spec,
            config=config,
        )
    )


def resolved_optimizer_state_dtype(*, config: ConfigLike) -> str:
    """Return the dtype used for optimizer-state residency."""

    if config.optimizer_state_dtype != "auto":
        return config.optimizer_state_dtype
    if config.is_zero_mode():
        return "fp32"
    if config.tuning_mode == "lora":
        return config.adapter_state_dtype()
    return config.weight_dtype


def _ceil_shard(num_bytes: int, *, world_size: int) -> int:
    """Return per-rank bytes for a uniformly sharded resident tensor."""

    return math.ceil(num_bytes / max(world_size, 1))


def _local_parameter_numel(
    *, parameter_spec: ModelParameterSpec, config: ConfigLike
) -> int:
    """Return the local parameter element count after TP sharding."""

    return tp_shard_numel(
        numel=parameter_spec.numel(),
        divisor=tp_shard_divisor_for_parameter(
            parameter_spec=parameter_spec,
            config=config,
        ),
    )


def _local_parameter_bytes_from_specs(
    *,
    parameter_specs: tuple[ModelParameterSpec, ...],
    config: ConfigLike,
    dtype_name: str,
    shard_for_zero: bool,
) -> int:
    """Return local parameter bytes after TP and optional ZeRO sharding."""

    total_bytes = 0
    dtype_bytes = bytes_for_dtype(dtype_name)
    for parameter_spec in parameter_specs:
        local_numel = _local_parameter_numel(
            parameter_spec=parameter_spec,
            config=config,
        )
        if shard_for_zero and config.is_zero_mode():
            local_numel = math.ceil(local_numel / config.data_parallel_degree())
        total_bytes += local_numel * dtype_bytes
    return total_bytes


def _parameter_bytes(*, model_spec: ModelSpec, config: ConfigLike) -> int:
    """Return resident parameter bytes for one backend."""

    dense_bytes = (
        _local_parameter_bytes_from_specs(
            parameter_specs=model_spec.parameter_specs,
            config=config,
            dtype_name=config.weight_dtype,
            shard_for_zero=config.distributed_mode == "zero3",
        )
        if model_spec.parameter_specs
        else model_spec.total_params * bytes_for_dtype(config.weight_dtype)
    )
    if config.tuning_mode == "full_ft":
        return dense_bytes
    adapter_bytes = _local_parameter_bytes_from_specs(
        parameter_specs=trainable_parameter_specs(model_spec=model_spec, config=config),
        config=config,
        dtype_name=config.adapter_parameter_dtype(),
        shard_for_zero=config.distributed_mode == "zero3",
    )
    return dense_bytes + adapter_bytes


def _gradient_bytes(*, model_spec: ModelSpec, config: ConfigLike) -> int:
    """Return resident gradient bytes for one backend."""

    return _local_parameter_bytes_from_specs(
        parameter_specs=trainable_parameter_specs(model_spec=model_spec, config=config),
        config=config,
        dtype_name=(
            config.adapter_gradient_dtype()
            if config.tuning_mode == "lora"
            else config.grad_dtype
        ),
        shard_for_zero=config.is_zero_mode(),
    )


def _optimizer_state_bytes(*, model_spec: ModelSpec, config: ConfigLike) -> int:
    """Return resident optimizer-state bytes for one backend."""

    total_bytes = 0
    for parameter_spec in trainable_parameter_specs(
        model_spec=model_spec, config=config
    ):
        local_numel = _local_parameter_numel(
            parameter_spec=parameter_spec,
            config=config,
        )
        state_ratio = optimizer_state_numel_for_parameter(
            parameter_spec=parameter_spec,
            config=config,
        ) / max(parameter_spec.numel(), 1)
        total_bytes += math.ceil(local_numel * state_ratio)
    total_bytes *= bytes_for_dtype(resolved_optimizer_state_dtype(config=config))
    if config.is_zero_mode():
        return _ceil_shard(total_bytes, world_size=config.data_parallel_degree())
    return total_bytes


def _master_weight_bytes(*, model_spec: ModelSpec, config: ConfigLike) -> int:
    """Return resident master-weight bytes for one backend."""

    if not config.use_master_weights:
        return 0
    return _local_parameter_bytes_from_specs(
        parameter_specs=trainable_parameter_specs(model_spec=model_spec, config=config),
        config=config,
        dtype_name=config.master_weight_dtype,
        shard_for_zero=config.is_zero_mode(),
    )


def runtime_floor_bytes(*, config: ConfigLike) -> int:
    """Return the fixed runtime floor in bytes."""

    runtime_floor_gb = getattr(config, "runtime_floor_gb_override", None)
    if runtime_floor_gb is None:
        runtime_floor_gb = RUNTIME_FLOOR_GB_BY_MODE[config.distributed_mode]
    return int(runtime_floor_gb * (1024**3))


def build_resident_state_terms(
    *,
    model_spec: ModelSpec,
    config: ConfigLike,
) -> ResidentStateTerms:
    """Build the resident-state terms for one estimate.

    Args:
        model_spec: Inspected model summary.
        config: Estimator or measurement config.

    Returns:
        Resident-state terms and the trainable parameter count.

    Example:
        >>> from simplesft.types import EstimatorConfig, ModelSpec
        >>> terms = build_resident_state_terms(
        ...     model_spec=ModelSpec(
        ...         model_name="toy",
        ...         model_type="llama",
        ...         num_layers=1,
        ...         hidden_size=16,
        ...         num_attention_heads=2,
        ...         intermediate_size=32,
        ...         vocab_size=128,
        ...         max_position_embeddings=128,
        ...         total_params=64,
        ...         trainable_linear_layers=(),
        ...     ),
        ...     config=EstimatorConfig(tuning_mode="full_ft"),
        ... )
        >>> terms.debug.parameter_bytes > 0
        True
    """

    trainable_params = trainable_parameter_count(model_spec=model_spec, config=config)
    local_trainable_bytes = _local_parameter_bytes_from_specs(
        parameter_specs=trainable_parameter_specs(model_spec=model_spec, config=config),
        config=config,
        dtype_name=(
            config.adapter_parameter_dtype()
            if config.tuning_mode == "lora"
            else config.weight_dtype
        ),
        shard_for_zero=False,
    )
    return ResidentStateTerms(
        debug=ResidentStateDebug(
            parameter_bytes=_parameter_bytes(model_spec=model_spec, config=config),
            gradient_bytes=_gradient_bytes(model_spec=model_spec, config=config),
            optimizer_state_bytes=_optimizer_state_bytes(
                model_spec=model_spec,
                config=config,
            ),
            master_weight_bytes=_master_weight_bytes(
                model_spec=model_spec,
                config=config,
            ),
            runtime_floor_bytes=runtime_floor_bytes(config=config),
            trainable_parameter_bytes=local_trainable_bytes,
        ),
        trainable_parameter_count=trainable_params,
    )
