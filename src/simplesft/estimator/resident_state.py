"""Resident-state accounting for the simplified analytical estimator."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union

from .optimizer import (
    optimizer_state_numel_for_parameter,
    normalized_optimizer_name,
    trainable_parameter_specs,
    zero_optimizer_state_is_sharded,
    zero_untested_optimizer_state_replica_floor_bytes,
)
from .parallelism import tp_shard_divisor_for_parameter, tp_shard_numel
from ..types import (
    EstimatorConfig,
    LoRAConfig,
    MeasurementConfig,
    ModelParameterSpec,
    ModelSpec,
    ResidentStateDebug,
)
from ..utils import bytes_for_dtype


ConfigLike = Union[EstimatorConfig, MeasurementConfig]


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
    if zero_optimizer_state_is_sharded(config=config):
        return _ceil_shard(total_bytes, world_size=config.data_parallel_degree())
    if config.is_zero_mode():
        return max(
            total_bytes,
            zero_untested_optimizer_state_replica_floor_bytes(
                model_spec=model_spec,
                config=config,
            ),
        )
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


def runtime_support_bytes(*, config: ConfigLike) -> int:
    """Return named runtime support in bytes.

    Args:
        config: Estimator or measurement config.

    Returns:
        Total runtime support derived from the configured CUDA, allocator,
        NCCL, and DeepSpeed support components, or from the total runtime
        support override when one is present.
    """

    return int(config.runtime_support_gb() * (1024**3))


def _scaled_bytes(*, value: int, factor: float) -> int:
    """Return bytes scaled by one fractional copy count.

    Args:
        value: Unscaled byte count.
        factor: Multiplicative copy count or overlap factor.

    Returns:
        Rounded byte count after applying the factor.
    """

    return int(round(value * factor))


def _persistent_backend_buffer_bytes(
    *,
    config: ConfigLike,
    parameter_bytes: int,
    gradient_bytes: int,
    trainable_parameter_count: int,
) -> int:
    """Return persistent backend support buffers for ZeRO full fine-tuning.

    Args:
        config: Estimator or measurement config.
        parameter_bytes: Resident parameter bytes visible on the rank.
        gradient_bytes: Resident gradient bytes visible on the rank.
        trainable_parameter_count: Unsharded trainable parameter count.

    Returns:
        Explicit persistent backend-buffer bytes kept resident outside named
        model tensors for supported ZeRO full fine-tuning regimes.

    Example:
        ZeRO-2 full FT on a tested optimizer keeps one grad shard plus one
        parameter partition in backend-managed pools.
    """

    if config.tuning_mode != "full_ft" or not config.is_zero_mode():
        return 0
    buffer_count = config.persistent_backend_buffer_count()
    if buffer_count <= 0:
        return 0
    if (
        config.distributed_mode == "zero2"
        and config.persistent_backend_buffer_tensor_count is None
        and normalized_optimizer_name(config=config)
        in config.normalized_zero_tested_optimizer_names()
    ):
        parameter_partition_bytes = math.ceil(
            parameter_bytes / max(config.data_parallel_degree(), 1)
        )
        return _scaled_bytes(
            value=gradient_bytes + parameter_partition_bytes,
            factor=buffer_count,
        )
    return _scaled_bytes(
        value=(
            trainable_parameter_count
            * bytes_for_dtype(config.persistent_backend_buffer_resolved_dtype())
        ),
        factor=buffer_count,
    )


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
    parameter_bytes = _parameter_bytes(model_spec=model_spec, config=config)
    gradient_bytes = _gradient_bytes(model_spec=model_spec, config=config)
    optimizer_state_bytes = _optimizer_state_bytes(
        model_spec=model_spec,
        config=config,
    )
    master_weight_bytes = _master_weight_bytes(
        model_spec=model_spec,
        config=config,
    )
    runtime_support = runtime_support_bytes(config=config)
    persistent_backend_buffer_bytes = _persistent_backend_buffer_bytes(
        config=config,
        parameter_bytes=parameter_bytes,
        gradient_bytes=gradient_bytes,
        trainable_parameter_count=trainable_params,
    )
    return ResidentStateTerms(
        debug=ResidentStateDebug(
            parameter_bytes=parameter_bytes,
            gradient_bytes=gradient_bytes,
            optimizer_state_bytes=optimizer_state_bytes,
            master_weight_bytes=master_weight_bytes,
            runtime_support_bytes=runtime_support,
            persistent_backend_buffer_bytes=persistent_backend_buffer_bytes,
            trainable_parameter_bytes=local_trainable_bytes,
        ),
        trainable_parameter_count=trainable_params,
    )
