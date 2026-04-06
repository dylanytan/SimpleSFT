"""Runtime measurement helpers for DeepSpeed ZeRO stage 2/3."""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

from ..models.inspect import inspect_model
from .measure import (
    _activation_tracker,
    _build_measured_breakdown,
    _build_memory_result,
    _build_optimizer,
    _configure_model_for_measurement,
    _cuda_peak_snapshot,
    _cuda_snapshot,
    _gradient_bytes,
    _load_model,
    _make_synthetic_batch,
    _optimizer_state_bytes,
    _phase_record,
    _reset_cuda_peak_stats,
    _runtime_attention_implementation,
    _runtime_checkpointing_metadata,
)
from ..types import MemoryResult, ModelSpec, PhaseMemoryRecord, TrainingConfig
from ..utils import maybe_get_deepspeed


def _zero2_precision_config(*, config: TrainingConfig) -> dict[str, dict[str, bool]]:
    """Build DeepSpeed precision flags for ZeRO measurement."""

    if config.weight_dtype == "bf16":
        return {"bf16": {"enabled": True}, "fp16": {"enabled": False}}
    if config.weight_dtype == "fp16":
        return {"bf16": {"enabled": False}, "fp16": {"enabled": True}}
    raise AssertionError("ZeRO measurement currently supports bf16 and fp16 only.")


def _zero2_config_dict(*, config: TrainingConfig) -> dict[str, Any]:
    """Build the DeepSpeed ZeRO stage 2/3 config used for measurement."""

    deep_speed_config: dict[str, Any] = {
        "train_micro_batch_size_per_gpu": config.micro_batch_size_per_gpu,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "train_batch_size": (
            config.micro_batch_size_per_gpu
            * config.gradient_accumulation_steps
            * config.world_size()
        ),
        "zero_optimization": {
            "stage": config.resolved_zero_stage(),
            "allgather_partitions": config.zero_allgather_partitions,
            "reduce_scatter": config.zero_reduce_scatter,
            "overlap_comm": config.zero_overlap_comm,
            "contiguous_gradients": config.zero_contiguous_gradients,
            "reduce_bucket_size": config.zero_bucket_elements,
            "allgather_bucket_size": config.zero_bucket_elements,
            "prefetch_bucket_size": config.zero_prefetch_elements,
            "sub_group_size": config.zero_bucket_elements,
        },
        "zero_allow_untested_optimizer": config.zero_allow_untested_optimizer,
        "steps_per_print": config.zero_steps_per_print,
        "wall_clock_breakdown": config.zero_wall_clock_breakdown,
    }
    deep_speed_config.update(_zero2_precision_config(config=config))
    return deep_speed_config


def _maybe_zero2_barrier() -> None:
    """Synchronize ZeRO workers when a process group is initialized."""

    if dist.is_initialized():
        dist.barrier(device_ids=[torch.cuda.current_device()])


def _initialize_zero2_engine(
    *,
    model: str | ModelSpec,
    config: TrainingConfig,
    device_index: int,
) -> tuple[Any, ModelSpec, torch.device]:
    """Initialize a DeepSpeed engine for ZeRO stage 2/3 measurement."""

    deepspeed = maybe_get_deepspeed()
    model_spec = (
        inspect_model(
            model_ref=model,
            trust_remote_code=config.trust_remote_code,
            supported_model_types=config.supported_model_types,
            default_attention_type=config.default_attention_type,
            intermediate_size_fallback_multiplier=(
                config.intermediate_size_fallback_multiplier
            ),
        )
        if isinstance(model, str)
        else model
    )
    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device=device)
    model_ref = model if isinstance(model, str) else model.model_name
    model_instance = _load_model(
        model_ref=model_ref,
        model_spec=model_spec,
        config=config,
        device=device,
    )
    _configure_model_for_measurement(model=model_instance, config=config)
    optimizer = _build_optimizer(model=model_instance, config=config)
    trainable_parameters = [
        parameter
        for parameter in model_instance.parameters()
        if parameter.requires_grad
    ]
    engine, _, _, _ = deepspeed.initialize(
        model=model_instance,
        model_parameters=trainable_parameters,
        optimizer=optimizer,
        config=_zero2_config_dict(config=config),
        dist_init_required=False,
    )
    return engine, model_spec, device


def _run_zero2_warmup_steps(
    *,
    engine: Any,
    model_spec: ModelSpec,
    config: TrainingConfig,
    device: torch.device,
) -> None:
    """Run warmup steps through the DeepSpeed engine."""

    for _ in range(config.warmup_steps):
        batch = _make_synthetic_batch(
            model_spec=model_spec, config=config, device=device
        )
        outputs = engine(**batch)
        engine.backward(outputs.loss)
        engine.step()


def _capture_zero2_measurement_phases(
    *,
    engine: Any,
    model_spec: ModelSpec,
    config: TrainingConfig,
    device: torch.device,
) -> tuple[list[PhaseMemoryRecord], dict[str, int], dict[str, int]]:
    """Run one measured ZeRO microstep and capture phase records."""

    phase_records: list[PhaseMemoryRecord] = []
    previous_allocated = 0
    previous_reserved = 0
    state_snapshots: dict[str, int] = {}

    def capture_phase(phase_name: str, *notes: str) -> None:
        nonlocal previous_allocated, previous_reserved
        current_allocated, current_reserved = _cuda_snapshot(device=device)
        peak_allocated, peak_reserved = _cuda_peak_snapshot(device=device)
        phase_records.append(
            _phase_record(
                phase_name=phase_name,
                current_allocated=current_allocated,
                current_reserved=current_reserved,
                previous_allocated=previous_allocated,
                previous_reserved=previous_reserved,
                peak_allocated=peak_allocated,
                peak_reserved=peak_reserved,
                notes=tuple(notes) if notes else ("measured",),
            )
        )
        previous_allocated = current_allocated
        previous_reserved = current_reserved
        _reset_cuda_peak_stats(device=device)

    torch.cuda.empty_cache()
    _reset_cuda_peak_stats(device=device)
    capture_phase("model_load", "measured")
    capture_phase("optimizer_create", "measured")
    capture_phase("post_init_baseline", "measured")
    state_snapshots["baseline_reserved_bytes"] = phase_records[-1].reserved_bytes
    batch = _make_synthetic_batch(model_spec=model_spec, config=config, device=device)
    capture_phase("batch_materialization", "measured")
    with _activation_tracker(model=engine.module, config=config) as activations:
        outputs = engine(**batch)
        capture_phase("forward", "measured", "activation_summary")
        capture_phase("loss_materialization", "measured")
        engine.backward(outputs.loss)
        state_snapshots["gradient_bytes_after_backward"] = _gradient_bytes(
            model=engine.module
        )
        capture_phase("backward", "measured")
    engine.step()
    state_snapshots["gradient_bytes_after_step"] = _gradient_bytes(model=engine.module)
    state_snapshots["optimizer_state_bytes_after_step"] = _optimizer_state_bytes(
        optimizer=engine.optimizer
    )
    capture_phase("optimizer_step", "measured")
    engine.zero_grad()
    capture_phase("zero_grad", "measured")
    capture_phase("step_end", "measured")
    return phase_records, activations, state_snapshots


def measure_zero2_local_peak_memory(
    *,
    model: str | ModelSpec,
    config: TrainingConfig,
    device_index: int,
) -> MemoryResult:
    """Measure one ZeRO rank on its assigned CUDA device."""

    engine, model_spec, device = _initialize_zero2_engine(
        model=model,
        config=config,
        device_index=device_index,
    )
    _maybe_zero2_barrier()
    _run_zero2_warmup_steps(
        engine=engine,
        model_spec=model_spec,
        config=config,
        device=device,
    )
    _maybe_zero2_barrier()
    phase_records, activations, state_snapshots = _capture_zero2_measurement_phases(
        engine=engine,
        model_spec=model_spec,
        config=config,
        device=device,
    )
    breakdown, activation_metadata = _build_measured_breakdown(
        model=engine.module,
        config=config,
        activations=activations,
        phase_records=phase_records,
        state_snapshots=state_snapshots,
    )
    return _build_memory_result(
        model_spec=model_spec,
        config=config,
        phase_records=phase_records,
        breakdown=breakdown,
        activation_metadata=activation_metadata,
        state_snapshots=state_snapshots,
        extra_metadata={
            "runtime_attention_implementation": _runtime_attention_implementation(
                model=engine.module,
            ),
            **_runtime_checkpointing_metadata(model=engine.module),
        },
    )
