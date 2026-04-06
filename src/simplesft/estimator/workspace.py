"""Workspace and communication-window accounting for the simplified estimator."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Union

from .activation import CHECKPOINT_SEGMENT_LAYERS
from .optimizer import (
    optimizer_is_zero_tested,
    optimizer_update_numel,
    resolved_optimizer_update_dtype,
    zero_optimizer_update_is_sharded,
    zero_untested_optimizer_update_replica_floor_bytes,
)
from .parallelism import sequence_parallel_divisor, tensor_parallel_degree
from ..types import EstimatorConfig, MeasurementConfig, ModelSpec, WorkspaceDebug
from ..utils import bytes_for_dtype


ConfigLike = Union[EstimatorConfig, MeasurementConfig]
STANDARD_ATTENTION_TILE_SIZE = 1024


@dataclass(frozen=True)
class WorkspaceTerms:
    """Workspace terms and phase-local windows for one analytical estimate."""

    debug: WorkspaceDebug
    forward_workspace_bytes: int
    backward_workspace_bytes: int
    optimizer_workspace_bytes: int
    backward_end_state_bytes: int


def _token_count(*, model_spec: ModelSpec, config: ConfigLike) -> int:
    """Return the total token count processed by one layer."""

    return model_spec.tokens_per_layer(
        batch_size=config.micro_batch_size_per_gpu,
        sequence_length=config.max_seq_len,
    )


def _attention_forward_workspace_bytes(
    *, model_spec: ModelSpec, config: ConfigLike
) -> int:
    """Return local forward attention workspace bytes."""

    backend = config.normalized_attention_backend()
    local_token_count = math.ceil(
        _token_count(model_spec=model_spec, config=config)
        / sequence_parallel_divisor(config=config)
    )
    if backend == "standard":
        effective_window = model_spec.attention.effective_window(
            sequence_length=config.max_seq_len
        )
        tile_size = min(effective_window, STANDARD_ATTENTION_TILE_SIZE)
        local_head_count = model_spec.attention.local_query_heads(
            tensor_parallel_degree=tensor_parallel_degree(config=config)
        )
        return (
            local_token_count * local_head_count * tile_size * bytes_for_dtype("fp32")
        )
    local_attention_width = max(
        math.ceil(
            model_spec.attention.query_width / tensor_parallel_degree(config=config)
        ),
        math.ceil(
            model_spec.attention.value_width / tensor_parallel_degree(config=config)
        ),
        math.ceil(
            model_spec.attention.output_proj_input_width
            / tensor_parallel_degree(config=config)
        ),
    )
    return (
        local_token_count * local_attention_width * bytes_for_dtype(config.weight_dtype)
    )


def _backward_kernel_workspace_bytes(
    *, model_spec: ModelSpec, config: ConfigLike
) -> int:
    """Return local backward kernel workspace bytes."""

    mlp_workspace = (
        math.ceil(
            _token_count(model_spec=model_spec, config=config)
            / sequence_parallel_divisor(config=config)
        )
        * math.ceil(
            model_spec.intermediate_size / tensor_parallel_degree(config=config)
        )
        * bytes_for_dtype(config.weight_dtype)
    )
    return (
        _attention_forward_workspace_bytes(model_spec=model_spec, config=config)
        + mlp_workspace
    )


def _recompute_workspace_bytes(*, model_spec: ModelSpec, config: ConfigLike) -> int:
    """Return local recompute workspace bytes under checkpointing."""

    if not config.gradient_checkpointing:
        return 0
    token_count = _token_count(model_spec=model_spec, config=config)
    return (
        math.ceil(token_count / sequence_parallel_divisor(config=config))
        * CHECKPOINT_SEGMENT_LAYERS
        * (
            model_spec.hidden_size
            + math.ceil(
                model_spec.intermediate_size / tensor_parallel_degree(config=config)
            )
        )
        * bytes_for_dtype(config.weight_dtype)
    )


def _tensor_parallel_comm_window_bytes(
    *, model_spec: ModelSpec, config: ConfigLike
) -> int:
    """Return the TP communication window for one local hidden-state exchange."""

    if not config.uses_tensor_parallel():
        return 0
    exchange_width = max(
        model_spec.hidden_size,
        model_spec.attention.output_proj_input_width,
        model_spec.attention.query_width,
        model_spec.attention.key_width,
        model_spec.attention.value_width,
    )
    return (
        math.ceil(
            _token_count(model_spec=model_spec, config=config)
            / sequence_parallel_divisor(config=config)
        )
        * exchange_width
        * bytes_for_dtype(config.weight_dtype)
    )


def _sequence_parallel_comm_window_bytes(
    *, model_spec: ModelSpec, config: ConfigLike
) -> int:
    """Return the SP communication window for one sequence gather / scatter."""

    if not config.sequence_parallel or not config.uses_tensor_parallel():
        return 0
    return (
        math.ceil(
            _token_count(model_spec=model_spec, config=config)
            / sequence_parallel_divisor(config=config)
        )
        * model_spec.hidden_size
        * bytes_for_dtype(config.weight_dtype)
    )


def _optimizer_update_workspace_bytes(
    *, model_spec: ModelSpec, config: ConfigLike
) -> int:
    """Return local optimizer update workspace bytes.

    Args:
        model_spec: Inspected model summary.
        config: Estimator or measurement config.

    Returns:
        Local optimizer-step scratch bytes. In ZeRO modes, the scratch is
        sharded by world size rather than modeled as a full-model local buffer.
    """

    update_numel = optimizer_update_numel(model_spec=model_spec, config=config)
    if zero_optimizer_update_is_sharded(config=config):
        update_numel = math.ceil(update_numel / max(config.world_size(), 1))
    update_bytes = update_numel * bytes_for_dtype(
        resolved_optimizer_update_dtype(config=config)
    )
    if not config.is_zero_mode():
        return update_bytes
    if zero_optimizer_update_is_sharded(config=config):
        return update_bytes
    return max(
        update_bytes,
        zero_untested_optimizer_update_replica_floor_bytes(
            model_spec=model_spec,
            config=config,
        ),
    )


def _loss_workspace_bytes(*, model_spec: ModelSpec, config: ConfigLike) -> int:
    """Return logits-sized loss materialization workspace bytes.

    Args:
        model_spec: Inspected model summary.
        config: Estimator or measurement config.

    Returns:
        Local logits/loss workspace bytes for one forward/backward microstep.
    """

    local_vocab_size = math.ceil(
        model_spec.vocab_size
        / (tensor_parallel_degree(config=config) if config.vocab_parallel_logits else 1)
    )
    return math.ceil(
        _token_count(model_spec=model_spec, config=config)
        / sequence_parallel_divisor(config=config)
    ) * local_vocab_size * bytes_for_dtype(config.loss_output_resolved_dtype())


def _gradient_dtype_name(*, config: ConfigLike) -> str:
    """Return the dtype used for gradient-like communication buckets."""

    if config.tuning_mode == "lora":
        return config.adapter_gradient_dtype()
    return config.grad_dtype


def _trainable_parameter_dtype_name(*, config: ConfigLike) -> str:
    """Return the dtype used for trainable parameter fetch/prefetch buckets."""

    if config.tuning_mode == "lora":
        return config.adapter_parameter_dtype()
    return config.weight_dtype


def _bucket_bytes(*, element_count: int, dtype_name: str) -> int:
    """Return bytes for one configured bucket object."""

    return element_count * bytes_for_dtype(dtype_name)


def _active_bucket_bytes(*, payload_bytes: int, capacity_bytes: int) -> int:
    """Return the active bytes for one configured bucket object."""

    return min(payload_bytes, capacity_bytes)


def _ddp_reducer_bucket_capacity_bytes(*, config: ConfigLike) -> int:
    """Return the configured DDP reducer bucket capacity in bytes."""

    return _bucket_bytes(
        element_count=config.ddp_bucket_elements,
        dtype_name=_gradient_dtype_name(config=config),
    )


def _zero_allgather_bucket_capacity_bytes(*, config: ConfigLike) -> int:
    """Return the configured ZeRO all-gather bucket capacity in bytes."""

    return _bucket_bytes(
        element_count=math.ceil(
            config.zero_bucket_elements / max(config.data_parallel_degree(), 1)
        ),
        dtype_name=_trainable_parameter_dtype_name(config=config),
    )


def _zero_reduce_bucket_capacity_bytes(*, config: ConfigLike) -> int:
    """Return the configured ZeRO reduce bucket capacity in bytes."""

    return _bucket_bytes(
        element_count=math.ceil(
            config.zero_bucket_elements / max(config.data_parallel_degree(), 1)
        ),
        dtype_name=_gradient_dtype_name(config=config),
    )


def _zero_prefetch_bucket_capacity_bytes(*, config: ConfigLike) -> int:
    """Return the configured ZeRO prefetch bucket capacity in bytes."""

    return _bucket_bytes(
        element_count=math.ceil(
            config.zero_prefetch_elements / max(config.data_parallel_degree(), 1)
        ),
        dtype_name=_trainable_parameter_dtype_name(config=config),
    )


def _zero_parameter_partition_bytes(
    *,
    config: ConfigLike,
    trainable_parameter_bytes: int,
) -> int:
    """Return one ZeRO parameter-partition window in bytes."""

    return math.ceil(trainable_parameter_bytes / max(config.data_parallel_degree(), 1))


def _zero_master_shard_bytes(
    *,
    config: ConfigLike,
    trainable_parameter_bytes: int,
) -> int:
    """Return one ZeRO master-weight shard window in bytes."""

    if not config.use_master_weights:
        return 0
    trainable_numel = math.ceil(
        trainable_parameter_bytes / bytes_for_dtype(config.weight_dtype)
    )
    master_bytes = trainable_numel * bytes_for_dtype(config.master_weight_dtype)
    return math.ceil(master_bytes / max(config.data_parallel_degree(), 1))


def _zero2_full_ft_tested_backward_workspace_bytes(
    *, gradient_bytes: int, zero_reduce_bucket_bytes: int
) -> int:
    """Return explicit ZeRO-2 full-FT backward windows for tested optimizers."""

    return (2 * gradient_bytes) + zero_reduce_bucket_bytes


def _zero2_full_ft_tested_optimizer_workspace_bytes(
    *,
    config: ConfigLike,
    trainable_parameter_bytes: int,
    gradient_bytes: int,
    zero_allgather_bucket_bytes: int,
    zero_prefetch_bucket_bytes: int,
    zero_reduce_bucket_bytes: int,
    optimizer_update_workspace_bytes: int,
) -> int:
    """Return explicit ZeRO-2 full-FT optimizer windows for tested optimizers."""

    parameter_partition_bytes = _zero_parameter_partition_bytes(
        config=config,
        trainable_parameter_bytes=trainable_parameter_bytes,
    )
    return (
        _zero_master_shard_bytes(
            config=config,
            trainable_parameter_bytes=trainable_parameter_bytes,
        )
        + (2 * gradient_bytes)
        + zero_reduce_bucket_bytes
        + optimizer_update_workspace_bytes
        + parameter_partition_bytes
        + zero_allgather_bucket_bytes
        + zero_prefetch_bucket_bytes
    )


def _zero_incremental_comm_window_bytes(*, zero_reduce_bucket_bytes: int) -> int:
    """Return ZeRO communication bytes that are not already resident.

    Args:
        zero_reduce_bucket_bytes: Explicit reduce bucket kept live during the
            communication subphase.

    Returns:
        Extra communication bytes beyond resident gradients.
    """

    return zero_reduce_bucket_bytes


def _zero_optimizer_workspace_bytes(
    *,
    zero_fetch_window_bytes: int,
    optimizer_update_workspace_bytes: int,
    zero_reduce_bucket_bytes: int,
) -> int:
    """Return the incremental ZeRO optimizer workspace peak.

    Args:
        zero_fetch_window_bytes: Fetch/all-gather subphase window.
        optimizer_update_workspace_bytes: Local update scratch bytes.
        zero_reduce_bucket_bytes: Explicit reduce bucket kept live during the
            communication subphase.

    Returns:
        Largest transient ZeRO optimizer window after removing state already
        counted in the resident optimizer-step floor.
    """

    return max(
        zero_fetch_window_bytes,
        optimizer_update_workspace_bytes,
        _zero_incremental_comm_window_bytes(
            zero_reduce_bucket_bytes=zero_reduce_bucket_bytes
        ),
    )


def build_workspace_terms(
    *,
    model_spec: ModelSpec,
    config: ConfigLike,
    parameter_bytes: int,
    trainable_parameter_bytes: int,
    gradient_bytes: int,
    optimizer_state_bytes: int,
) -> WorkspaceTerms:
    """Build local workspace and communication-window terms.

    Args:
        model_spec: Inspected model summary.
        config: Estimator or measurement config.
        parameter_bytes: Resident parameter bytes visible on the rank.
        trainable_parameter_bytes: Local bytes that actually participate in the
            optimizer step.
        gradient_bytes: Resident gradient bytes visible on the rank.
        optimizer_state_bytes: Resident optimizer-state bytes visible on the rank.

    Returns:
        Workspace and communication-window terms for the three execution phases.
    """

    attention_forward_workspace_bytes = _attention_forward_workspace_bytes(
        model_spec=model_spec,
        config=config,
    )
    backward_kernel_workspace_bytes = _backward_kernel_workspace_bytes(
        model_spec=model_spec,
        config=config,
    )
    recompute_workspace_bytes = _recompute_workspace_bytes(
        model_spec=model_spec,
        config=config,
    )
    tensor_parallel_comm_window_bytes = _tensor_parallel_comm_window_bytes(
        model_spec=model_spec,
        config=config,
    )
    sequence_parallel_comm_window_bytes = _sequence_parallel_comm_window_bytes(
        model_spec=model_spec,
        config=config,
    )
    optimizer_update_workspace_bytes = _optimizer_update_workspace_bytes(
        model_spec=model_spec,
        config=config,
    )
    loss_workspace_bytes = _loss_workspace_bytes(
        model_spec=model_spec,
        config=config,
    )
    ddp_reducer_bucket_bytes = _active_bucket_bytes(
        payload_bytes=gradient_bytes,
        capacity_bytes=_ddp_reducer_bucket_capacity_bytes(config=config),
    )
    ddp_comm_overlap_bytes = ddp_reducer_bucket_bytes + gradient_bytes
    optimizer_parameter_window_bytes = max(trainable_parameter_bytes, 0)
    zero_allgather_bucket_bytes = _active_bucket_bytes(
        payload_bytes=optimizer_parameter_window_bytes,
        capacity_bytes=_zero_allgather_bucket_capacity_bytes(config=config),
    )
    zero_reduce_bucket_bytes = _active_bucket_bytes(
        payload_bytes=gradient_bytes,
        capacity_bytes=_zero_reduce_bucket_capacity_bytes(config=config),
    )
    zero_prefetch_bucket_bytes = _active_bucket_bytes(
        payload_bytes=optimizer_parameter_window_bytes,
        capacity_bytes=_zero_prefetch_bucket_capacity_bytes(config=config),
    )
    zero_fetch_window_bytes = (
        optimizer_parameter_window_bytes
        + zero_allgather_bucket_bytes
        + zero_prefetch_bucket_bytes
    )
    zero_update_window_bytes = (
        optimizer_state_bytes + gradient_bytes + optimizer_update_workspace_bytes
    )
    zero_comm_window_bytes = gradient_bytes + zero_reduce_bucket_bytes
    optimizer_workspace_bytes = optimizer_update_workspace_bytes
    backward_end_state_bytes = 0
    backward_workspace_bytes = (
        backward_kernel_workspace_bytes
        + loss_workspace_bytes
        + recompute_workspace_bytes
        + tensor_parallel_comm_window_bytes
        + sequence_parallel_comm_window_bytes
    )
    if config.distributed_mode == "ddp":
        backward_end_state_bytes = gradient_bytes + ddp_reducer_bucket_bytes
        optimizer_workspace_bytes = max(
            ddp_comm_overlap_bytes, optimizer_update_workspace_bytes
        )
    if config.distributed_mode == "zero2":
        backward_end_state_bytes = gradient_bytes + zero_reduce_bucket_bytes
        if (
            config.tuning_mode == "full_ft"
            and optimizer_is_zero_tested(config=config)
            and not config.gradient_checkpointing
        ):
            backward_workspace_bytes = max(
                backward_workspace_bytes,
                _zero2_full_ft_tested_backward_workspace_bytes(
                    gradient_bytes=gradient_bytes,
                    zero_reduce_bucket_bytes=zero_reduce_bucket_bytes,
                ),
            )
        optimizer_workspace_bytes = _zero_optimizer_workspace_bytes(
            zero_fetch_window_bytes=zero_fetch_window_bytes,
            optimizer_update_workspace_bytes=optimizer_update_workspace_bytes,
            zero_reduce_bucket_bytes=zero_reduce_bucket_bytes,
        )
        if (
            config.tuning_mode == "full_ft"
            and optimizer_is_zero_tested(config=config)
            and not config.gradient_checkpointing
        ):
            optimizer_workspace_bytes = max(
                optimizer_workspace_bytes,
                _zero2_full_ft_tested_optimizer_workspace_bytes(
                    config=config,
                    trainable_parameter_bytes=trainable_parameter_bytes,
                    gradient_bytes=gradient_bytes,
                    zero_allgather_bucket_bytes=zero_allgather_bucket_bytes,
                    zero_prefetch_bucket_bytes=zero_prefetch_bucket_bytes,
                    zero_reduce_bucket_bytes=zero_reduce_bucket_bytes,
                    optimizer_update_workspace_bytes=optimizer_update_workspace_bytes,
                ),
            )
    if config.distributed_mode == "zero3":
        backward_end_state_bytes = gradient_bytes + zero_allgather_bucket_bytes
        optimizer_workspace_bytes = _zero_optimizer_workspace_bytes(
            zero_fetch_window_bytes=zero_fetch_window_bytes,
            optimizer_update_workspace_bytes=optimizer_update_workspace_bytes,
            zero_reduce_bucket_bytes=zero_reduce_bucket_bytes,
        )
    return WorkspaceTerms(
        debug=WorkspaceDebug(
            attention_forward_workspace_bytes=attention_forward_workspace_bytes,
            backward_kernel_workspace_bytes=backward_kernel_workspace_bytes,
            recompute_workspace_bytes=recompute_workspace_bytes,
            loss_workspace_bytes=loss_workspace_bytes,
            optimizer_update_workspace_bytes=optimizer_update_workspace_bytes,
            ddp_reducer_bucket_bytes=ddp_reducer_bucket_bytes,
            ddp_comm_overlap_bytes=ddp_comm_overlap_bytes,
            zero_allgather_bucket_bytes=zero_allgather_bucket_bytes,
            zero_reduce_bucket_bytes=zero_reduce_bucket_bytes,
            zero_prefetch_bucket_bytes=zero_prefetch_bucket_bytes,
            zero_fetch_window_bytes=zero_fetch_window_bytes,
            zero_update_window_bytes=zero_update_window_bytes,
            zero_comm_window_bytes=zero_comm_window_bytes,
            tensor_parallel_comm_window_bytes=tensor_parallel_comm_window_bytes,
            sequence_parallel_comm_window_bytes=sequence_parallel_comm_window_bytes,
        ),
        forward_workspace_bytes=attention_forward_workspace_bytes
        + tensor_parallel_comm_window_bytes
        + sequence_parallel_comm_window_bytes,
        backward_workspace_bytes=backward_workspace_bytes,
        optimizer_workspace_bytes=optimizer_workspace_bytes,
        backward_end_state_bytes=backward_end_state_bytes,
    )
