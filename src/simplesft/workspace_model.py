"""Workspace and communication-window accounting for the simplified estimator."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Union

from .activation_model import CHECKPOINT_SEGMENT_LAYERS
from .optimizer_model import optimizer_update_numel, resolved_optimizer_update_dtype
from .parallelism_model import sequence_parallel_divisor, tensor_parallel_degree
from .types import EstimatorConfig, MeasurementConfig, ModelSpec, WorkspaceDebug
from .utils import bytes_for_dtype


ConfigLike = Union[EstimatorConfig, MeasurementConfig]
COMM_WINDOW_BYTES = 512 * 1024**2
PREFETCH_WINDOW_BYTES = 256 * 1024**2
ATTENTION_TILE_SIZE = {"standard": 1024, "sdpa": 256, "flash2": 128}


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
    tile_size = min(config.max_seq_len, ATTENTION_TILE_SIZE.get(backend, 256))
    local_head_count = math.ceil(
        model_spec.num_attention_heads / tensor_parallel_degree(config=config)
    )
    if backend == "standard":
        return (
            config.micro_batch_size_per_gpu
            * local_head_count
            * config.max_seq_len
            * tile_size
            * model_spec.num_layers
            * bytes_for_dtype("fp32")
        )
    local_token_count = math.ceil(
        _token_count(model_spec=model_spec, config=config)
        / sequence_parallel_divisor(config=config)
    )
    return (
        local_token_count
        * math.ceil(model_spec.hidden_size / tensor_parallel_degree(config=config))
        * model_spec.num_layers
        * bytes_for_dtype(config.weight_dtype)
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
    return (
        math.ceil(
            _token_count(model_spec=model_spec, config=config)
            / sequence_parallel_divisor(config=config)
        )
        * model_spec.hidden_size
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
    if config.is_zero_mode():
        update_numel = math.ceil(update_numel / max(config.world_size(), 1))
    return update_numel * bytes_for_dtype(
        resolved_optimizer_update_dtype(config=config)
    )


def _bucket_window_bytes(*, resident_bytes: int) -> int:
    """Return the communication window size for one resident tensor class."""

    return min(resident_bytes, COMM_WINDOW_BYTES)


def build_workspace_terms(
    *,
    model_spec: ModelSpec,
    config: ConfigLike,
    parameter_bytes: int,
    gradient_bytes: int,
    optimizer_state_bytes: int,
) -> WorkspaceTerms:
    """Build local workspace and communication-window terms."""

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
    ddp_reducer_bucket_bytes = _bucket_window_bytes(resident_bytes=gradient_bytes)
    ddp_comm_overlap_bytes = ddp_reducer_bucket_bytes + gradient_bytes
    zero_fetch_window_bytes = (
        parameter_bytes
        + _bucket_window_bytes(resident_bytes=parameter_bytes)
        + min(parameter_bytes, PREFETCH_WINDOW_BYTES)
    )
    zero_update_window_bytes = (
        optimizer_state_bytes + gradient_bytes + optimizer_update_workspace_bytes
    )
    zero_comm_window_bytes = gradient_bytes + _bucket_window_bytes(
        resident_bytes=gradient_bytes
    )
    optimizer_workspace_bytes = optimizer_update_workspace_bytes
    backward_end_state_bytes = 0
    if config.distributed_mode == "ddp":
        backward_end_state_bytes = gradient_bytes + ddp_reducer_bucket_bytes
        optimizer_workspace_bytes = max(
            ddp_comm_overlap_bytes, optimizer_update_workspace_bytes
        )
    if config.distributed_mode == "zero2":
        backward_end_state_bytes = gradient_bytes + _bucket_window_bytes(
            resident_bytes=gradient_bytes
        )
        optimizer_workspace_bytes = max(
            zero_fetch_window_bytes, zero_update_window_bytes, zero_comm_window_bytes
        )
    if config.distributed_mode == "zero3":
        backward_end_state_bytes = gradient_bytes + _bucket_window_bytes(
            resident_bytes=parameter_bytes
        )
        optimizer_workspace_bytes = max(
            zero_fetch_window_bytes, zero_update_window_bytes, zero_comm_window_bytes
        )
    return WorkspaceTerms(
        debug=WorkspaceDebug(
            attention_forward_workspace_bytes=attention_forward_workspace_bytes,
            backward_kernel_workspace_bytes=backward_kernel_workspace_bytes,
            recompute_workspace_bytes=recompute_workspace_bytes,
            loss_workspace_bytes=0,
            optimizer_update_workspace_bytes=optimizer_update_workspace_bytes,
            ddp_reducer_bucket_bytes=ddp_reducer_bucket_bytes,
            ddp_comm_overlap_bytes=ddp_comm_overlap_bytes,
            zero_fetch_window_bytes=zero_fetch_window_bytes,
            zero_update_window_bytes=zero_update_window_bytes,
            zero_comm_window_bytes=zero_comm_window_bytes,
            tensor_parallel_comm_window_bytes=tensor_parallel_comm_window_bytes,
            sequence_parallel_comm_window_bytes=sequence_parallel_comm_window_bytes,
        ),
        forward_workspace_bytes=attention_forward_workspace_bytes
        + tensor_parallel_comm_window_bytes
        + sequence_parallel_comm_window_bytes,
        backward_workspace_bytes=backward_kernel_workspace_bytes
        + recompute_workspace_bytes
        + tensor_parallel_comm_window_bytes
        + sequence_parallel_comm_window_bytes,
        optimizer_workspace_bytes=optimizer_workspace_bytes,
        backward_end_state_bytes=backward_end_state_bytes,
    )
