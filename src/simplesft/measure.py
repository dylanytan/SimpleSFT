"""Runtime memory measurement for SimpleSFT training microsteps."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterator, cast

import torch

from .attribution import build_workspace_proxy_metadata
from .constants import PHASE_PEAK_CANDIDATES
from .estimate import estimate_lora_parameter_count
from .inspect import inspect_model
from .runtime import load_pretrained_causal_lm
from .types import (
    MemoryComponentBreakdown,
    MemoryResult,
    ModelSpec,
    PhaseMemoryRecord,
    TrainingConfig,
)
from .utils import (
    canonical_torch_dtype,
    is_cuda_available,
    maybe_get_deepspeed,
    maybe_get_peft,
    optimizer_state_in_baseline,
)


@contextmanager
def _activation_tracker(model: torch.nn.Module) -> Iterator[dict[str, int]]:
    """Track per-block activation summaries via forward hooks.

    Args:
        model: Torch model to instrument.

    Returns:
        Context manager yielding activation bytes by block name.
    """

    activations: dict[str, int] = {}
    handles: list[Any] = []

    def hook(module_name: str):
        def _record(_module: torch.nn.Module, _inputs: Any, output: Any) -> None:
            if isinstance(output, torch.Tensor):
                activations[module_name] = activations.get(module_name, 0) + (
                    output.numel() * output.element_size()
                )
                return
            if isinstance(output, (tuple, list)):
                tensor_bytes = sum(
                    item.numel() * item.element_size()
                    for item in output
                    if isinstance(item, torch.Tensor)
                )
                activations[module_name] = activations.get(module_name, 0) + tensor_bytes

        return _record

    for module_name, module in model.named_modules():
        if any(token in module_name for token in ("layers.", "h.")):
            handles.append(module.register_forward_hook(hook(module_name=module_name)))
    try:
        yield activations
    finally:
        for handle in handles:
            handle.remove()


def _cuda_snapshot(device: torch.device) -> tuple[int, int]:
    """Return current allocated and reserved CUDA memory for a device."""

    return (
        torch.cuda.memory_allocated(device=device),
        torch.cuda.memory_reserved(device=device),
    )


def _phase_record(
    *,
    phase_name: str,
    current_allocated: int,
    current_reserved: int,
    previous_allocated: int,
    previous_reserved: int,
    peak_allocated: int,
    peak_reserved: int,
    notes: tuple[str, ...] = (),
) -> PhaseMemoryRecord:
    """Build one phase record from CUDA memory stats."""

    return PhaseMemoryRecord(
        phase_name=phase_name,
        allocated_bytes=current_allocated,
        reserved_bytes=current_reserved,
        peak_allocated_bytes=peak_allocated,
        peak_reserved_bytes=peak_reserved,
        delta_allocated_bytes=current_allocated - previous_allocated,
        delta_reserved_bytes=current_reserved - previous_reserved,
        notes=notes,
    )


def _parameter_bytes(model: torch.nn.Module) -> int:
    """Return live parameter memory in bytes."""

    return sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())


def _gradient_bytes(model: torch.nn.Module) -> int:
    """Return live gradient memory in bytes."""

    return sum(
        parameter.grad.numel() * parameter.grad.element_size()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def _optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    """Return optimizer state tensor bytes."""

    if hasattr(optimizer, "optimizer") and getattr(optimizer, "optimizer") is not optimizer:
        nested_optimizer = cast(torch.optim.Optimizer, getattr(optimizer, "optimizer"))
        return _optimizer_state_bytes(optimizer=nested_optimizer)
    tensor_bytes = 0
    state_mapping = getattr(optimizer, "state", {})
    for state in state_mapping.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                tensor_bytes += value.numel() * value.element_size()
    return tensor_bytes


def _activation_breakdown(activations: dict[str, int]) -> tuple[int, int]:
    """Return activation bytes split between attention and non-attention blocks."""

    attention_bytes = sum(
        num_bytes
        for module_name, num_bytes in activations.items()
        if "attn" in module_name or "attention" in module_name
    )
    return attention_bytes, sum(activations.values()) - attention_bytes


def _make_synthetic_batch(
    *,
    model_spec: ModelSpec,
    config: TrainingConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Create a synthetic causal LM batch for measurement."""

    input_ids = torch.randint(
        low=0,
        high=model_spec.vocab_size,
        size=(config.micro_batch_size_per_gpu, config.max_seq_len),
        device=device,
    )
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids, device=device),
        "labels": input_ids.clone(),
    }


def _apply_lora(
    *,
    model: torch.nn.Module,
    config: TrainingConfig,
) -> torch.nn.Module:
    """Apply LoRA adapters using `peft` when requested."""

    assert config.lora is not None, "LoRA config is required for tuning_mode='lora'."
    peft = maybe_get_peft()
    lora_config = peft.LoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=list(config.lora.target_modules),
        bias=config.lora.bias,
        task_type=peft.TaskType.CAUSAL_LM,
    )
    return peft.get_peft_model(model=model, peft_config=lora_config)


def _load_model(
    *,
    model_ref: str,
    config: TrainingConfig,
    device: torch.device,
) -> torch.nn.Module:
    """Load the model and apply the requested tuning mode."""

    loaded_model = load_pretrained_causal_lm(
        model_ref=model_ref,
        torch_dtype=canonical_torch_dtype(config.weight_dtype),
        attention_backend=config.attention_backend,
        low_cpu_mem_usage=True,
    )
    loaded_model_any = cast(Any, loaded_model)
    model = cast(torch.nn.Module, loaded_model_any.to(device))
    if config.tuning_mode == "lora":
        model = _apply_lora(model=model, config=config)
    else:
        assert config.tuning_mode == "full_ft", f"Unsupported tuning mode: {config.tuning_mode}"
    return model


def _build_optimizer(*, model: torch.nn.Module) -> torch.optim.Optimizer:
    """Build the AdamW optimizer for trainable model parameters."""

    return torch.optim.AdamW(
        params=[parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4,
    )


def _run_warmup_steps(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    model_spec: ModelSpec,
    config: TrainingConfig,
    device: torch.device,
) -> None:
    """Run warmup steps before the measured training step."""

    for _ in range(config.warmup_steps):
        warmup_batch = _make_synthetic_batch(
            model_spec=model_spec,
            config=config,
            device=device,
        )
        outputs = model(**warmup_batch)
        outputs.loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


def _capture_measurement_phases(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    model_spec: ModelSpec,
    config: TrainingConfig,
    device: torch.device,
) -> tuple[list[PhaseMemoryRecord], dict[str, int], dict[str, int]]:
    """Run one measured microstep and capture phase records."""

    phase_records: list[PhaseMemoryRecord] = []
    previous_allocated = 0
    previous_reserved = 0
    state_snapshots: dict[str, int] = {}

    def capture_phase(phase_name: str, *notes: str) -> None:
        nonlocal previous_allocated, previous_reserved
        current_allocated, current_reserved = _cuda_snapshot(device=device)
        phase_records.append(
            _phase_record(
                phase_name=phase_name,
                current_allocated=current_allocated,
                current_reserved=current_reserved,
                previous_allocated=previous_allocated,
                previous_reserved=previous_reserved,
                peak_allocated=torch.cuda.max_memory_allocated(device=device),
                peak_reserved=torch.cuda.max_memory_reserved(device=device),
                notes=tuple(notes) if notes else ("measured",),
            )
        )
        previous_allocated = current_allocated
        previous_reserved = current_reserved
        torch.cuda.reset_peak_memory_stats(device=device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device=device)
    capture_phase("model_load", "measured")
    capture_phase("optimizer_create", "measured")
    capture_phase("post_init_baseline", "measured")
    state_snapshots["baseline_reserved_bytes"] = phase_records[-1].reserved_bytes
    batch = _make_synthetic_batch(model_spec=model_spec, config=config, device=device)
    capture_phase("batch_materialization", "measured")
    with _activation_tracker(model=model) as activations:
        outputs = model(**batch)
        capture_phase("forward", "measured", "activation_summary")
        capture_phase("loss_materialization", "measured")
        outputs.loss.backward()
        state_snapshots["gradient_bytes_after_backward"] = _gradient_bytes(model=model)
        capture_phase("backward", "measured")
    optimizer.step()
    state_snapshots["gradient_bytes_after_step"] = _gradient_bytes(model=model)
    state_snapshots["optimizer_state_bytes_after_step"] = _optimizer_state_bytes(
        optimizer=optimizer
    )
    capture_phase("optimizer_step", "measured")
    optimizer.zero_grad(set_to_none=True)
    capture_phase("zero_grad", "measured")
    capture_phase("step_end", "measured")
    return phase_records, activations, state_snapshots


def _build_measured_breakdown(
    *,
    model: torch.nn.Module,
    config: TrainingConfig,
    activations: dict[str, int],
    phase_records: list[PhaseMemoryRecord],
    state_snapshots: dict[str, int],
) -> tuple[MemoryComponentBreakdown, dict[str, int]]:
    """Build a modular measured breakdown from live runtime state."""

    attention_activation_bytes, other_activation_bytes = _activation_breakdown(
        activations=activations
    )
    parameter_bytes = _parameter_bytes(model=model)
    gradient_bytes = max(
        state_snapshots.get("gradient_bytes_after_backward", 0),
        state_snapshots.get("gradient_bytes_after_step", 0),
    )
    optimizer_state_bytes = state_snapshots.get("optimizer_state_bytes_after_step", 0)
    activation_bytes = attention_activation_bytes + other_activation_bytes
    observed_peak_bytes = max(record.peak_reserved_bytes for record in phase_records)
    optimizer_in_baseline = optimizer_state_in_baseline(
        warmup_steps=config.warmup_steps,
        optimizer_name=config.optimizer_name,
    )
    runtime_reserve_bytes = max(
        0,
        state_snapshots.get("baseline_reserved_bytes", 0)
        - parameter_bytes
        - (optimizer_state_bytes if optimizer_in_baseline else 0),
    )
    peak_phase_record = max(
        (
            record
            for record in phase_records
            if record.phase_name in PHASE_PEAK_CANDIDATES
        ),
        key=lambda record: record.peak_reserved_bytes,
    )
    accounted_bytes = parameter_bytes + runtime_reserve_bytes
    if optimizer_in_baseline:
        accounted_bytes += optimizer_state_bytes
    if peak_phase_record.phase_name == "backward":
        accounted_bytes += gradient_bytes + activation_bytes
    if peak_phase_record.phase_name == "optimizer_step":
        accounted_bytes += gradient_bytes
        if not optimizer_in_baseline:
            accounted_bytes += optimizer_state_bytes
    if peak_phase_record.phase_name == "forward":
        accounted_bytes += activation_bytes
    return (
        MemoryComponentBreakdown(
            parameter_bytes=parameter_bytes,
            gradient_bytes=gradient_bytes,
            optimizer_state_bytes=optimizer_state_bytes,
            activation_bytes=activation_bytes,
            transient_bytes=max(0, observed_peak_bytes - accounted_bytes),
            residual_bytes=0,
            runtime_reserve_bytes=runtime_reserve_bytes,
        ),
        {
            "attention_activation_bytes": attention_activation_bytes,
            "other_activation_bytes": other_activation_bytes,
        },
    )


def _build_measurement_assumptions(
    *,
    model_spec: ModelSpec,
    config: TrainingConfig,
) -> tuple[str, ...]:
    """Build measurement assumptions and notes."""

    assumptions = [
        "Measurement uses module-level activation summaries rather than per-op tracing.",
        "Gradient and optimizer-state bytes are captured from phase-local snapshots.",
    ]
    if optimizer_state_in_baseline(
        warmup_steps=config.warmup_steps,
        optimizer_name=config.optimizer_name,
    ):
        assumptions.append("Warmup materializes optimizer state before the measured step.")
    if config.tuning_mode == "lora":
        assert config.lora is not None, "LoRA config is required for tuning_mode='lora'."
        assumptions.append(
            "Estimated trainable LoRA params: "
            f"{estimate_lora_parameter_count(model_spec=model_spec, lora_config=config.lora)}"
        )
    return tuple(assumptions)


def _measure_zero2_requires_multi_gpu() -> None:
    """Raise a clear error for unsupported single-rank ZeRO-2 measurement."""

    raise RuntimeError(
        "ZeRO-2 measurement requires at least two GPUs so optimizer states and gradients "
        "can actually be sharded."
    )


def _maybe_wrap_ddp_model(
    *,
    model: torch.nn.Module,
    config: TrainingConfig,
    device_index: int,
) -> torch.nn.Module:
    """Wrap a local model in DDP for distributed measurement.

    Args:
        model: Rank-local CUDA model instance.
        config: Training configuration that determines the runtime shape.
        device_index: Local CUDA device index for the current rank.

    Returns:
        The original model for single-GPU runs or a DDP-wrapped module.
    """

    if config.distributed_mode != "ddp" or config.world_size() <= 1:
        return model
    from torch.nn.parallel import DistributedDataParallel

    return DistributedDataParallel(
        module=model,
        device_ids=[device_index],
        output_device=device_index,
        broadcast_buffers=False,
        init_sync=False,
    )


def _maybe_barrier(*, config: TrainingConfig) -> None:
    """Synchronize distributed ranks when a process group is active."""

    if config.distributed_mode != "ddp" or config.world_size() <= 1:
        return
    import torch.distributed as dist

    if dist.is_initialized():
        dist.barrier(device_ids=[torch.cuda.current_device()])


def _build_memory_result(
    *,
    model_spec: ModelSpec,
    config: TrainingConfig,
    phase_records: list[PhaseMemoryRecord],
    breakdown: MemoryComponentBreakdown,
    activation_metadata: dict[str, int],
    state_snapshots: dict[str, int],
) -> MemoryResult:
    """Build a `MemoryResult` from measured phase records and breakdowns."""

    workspace_proxy_metadata = build_workspace_proxy_metadata(
        phase_records=phase_records,
        breakdown=breakdown,
    )
    peak_phase_record = max(
        (
            record
            for record in phase_records
            if record.phase_name in PHASE_PEAK_CANDIDATES
        ),
        key=lambda record: record.peak_reserved_bytes,
    )
    global_peak_bytes = max(record.peak_reserved_bytes for record in phase_records)
    return MemoryResult(
        mode="measure",
        model_name=model_spec.model_name,
        config=replace(config),
        breakdown=breakdown,
        phase_records=tuple(phase_records),
        peak_phase=peak_phase_record.phase_name,
        global_peak_bytes=global_peak_bytes,
        feasible=global_peak_bytes <= int(config.gpu_memory_gb * (1024**3)),
        metadata={
            "model_type": model_spec.model_type,
            "world_size": config.world_size(),
            "optimizer_state_in_baseline": optimizer_state_in_baseline(
                warmup_steps=config.warmup_steps,
                optimizer_name=config.optimizer_name,
            ),
            "baseline_reserved_bytes": state_snapshots.get("baseline_reserved_bytes", 0),
            **activation_metadata,
            **workspace_proxy_metadata,
        },
        assumptions=_build_measurement_assumptions(
            model_spec=model_spec,
            config=config,
        ),
    )


def _measure_local_peak_memory(
    *,
    model: str | ModelSpec,
    config: TrainingConfig,
    device_index: int,
) -> MemoryResult:
    """Measure one local rank on its assigned CUDA device."""

    model_spec = inspect_model(model_ref=model) if isinstance(model, str) else model
    device = torch.device("cuda", device_index)
    torch.cuda.set_device(device=device)
    model_ref = model if isinstance(model, str) else model.model_name
    model_instance = _load_model(
        model_ref=model_ref,
        config=config,
        device=device,
    )
    if config.gradient_checkpointing:
        gradient_checkpointable_model = cast(Any, model_instance)
        gradient_checkpointable_model.gradient_checkpointing_enable()
    model_instance = _maybe_wrap_ddp_model(
        model=model_instance,
        config=config,
        device_index=device_index,
    )
    optimizer = _build_optimizer(model=model_instance)
    _maybe_barrier(config=config)
    _run_warmup_steps(
        model=model_instance,
        optimizer=optimizer,
        model_spec=model_spec,
        config=config,
        device=device,
    )
    _maybe_barrier(config=config)
    phase_records, activations, state_snapshots = _capture_measurement_phases(
        model=model_instance,
        optimizer=optimizer,
        model_spec=model_spec,
        config=config,
        device=device,
    )
    breakdown, activation_metadata = _build_measured_breakdown(
        model=model_instance,
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
    )


def _aggregate_phase_records(
    *,
    results: list[MemoryResult],
    aggregation_tag: str,
) -> tuple[PhaseMemoryRecord, ...]:
    """Aggregate per-rank phase records into a conservative max timeline."""

    phase_names = [record.phase_name for record in results[0].phase_records]
    phase_records = []
    for phase_name in phase_names:
        phase_group = [
            next(record for record in result.phase_records if record.phase_name == phase_name)
            for result in results
        ]
        phase_records.append(
            PhaseMemoryRecord(
                phase_name=phase_name,
                allocated_bytes=max(record.allocated_bytes for record in phase_group),
                reserved_bytes=max(record.reserved_bytes for record in phase_group),
                peak_allocated_bytes=max(
                    record.peak_allocated_bytes for record in phase_group
                ),
                peak_reserved_bytes=max(
                    record.peak_reserved_bytes for record in phase_group
                ),
                delta_allocated_bytes=max(
                    record.delta_allocated_bytes for record in phase_group
                ),
                delta_reserved_bytes=max(
                    record.delta_reserved_bytes for record in phase_group
                ),
                notes=(aggregation_tag,),
            )
        )
    return tuple(phase_records)


def _aggregate_breakdown(*, results: list[MemoryResult]) -> MemoryComponentBreakdown:
    """Aggregate conservative max component values across ranks."""

    return MemoryComponentBreakdown(
        parameter_bytes=max(result.breakdown.parameter_bytes for result in results),
        gradient_bytes=max(result.breakdown.gradient_bytes for result in results),
        optimizer_state_bytes=max(result.breakdown.optimizer_state_bytes for result in results),
        activation_bytes=max(result.breakdown.activation_bytes for result in results),
        transient_bytes=max(result.breakdown.transient_bytes for result in results),
        residual_bytes=max(result.breakdown.residual_bytes for result in results),
        runtime_reserve_bytes=max(
            result.breakdown.runtime_reserve_bytes for result in results
        ),
    )


def aggregate_rank_results(
    *,
    results: list[MemoryResult],
    aggregation_tag: str = "ddp_aggregated",
    aggregation_assumption: str = "DDP result aggregates the max across ranks.",
) -> MemoryResult:
    """Aggregate per-rank results into a max-per-rank measurement result."""

    assert results, "At least one rank result is required."
    phase_records = _aggregate_phase_records(
        results=results,
        aggregation_tag=aggregation_tag,
    )
    aggregated_breakdown = _aggregate_breakdown(results=results)
    global_peak_bytes = max(result.global_peak_bytes for result in results)
    peak_phase = max(
        phase_records,
        key=lambda record: record.peak_reserved_bytes,
    ).phase_name
    return MemoryResult(
        mode="measure",
        model_name=results[0].model_name,
        config=replace(results[0].config),
        breakdown=aggregated_breakdown,
        phase_records=tuple(phase_records),
        peak_phase=peak_phase,
        global_peak_bytes=global_peak_bytes,
        feasible=global_peak_bytes <= int(results[0].config.gpu_memory_gb * (1024**3)),
        metadata={
            **results[0].metadata,
            "aggregated_across_ranks": True,
            "rank_aggregation_tag": aggregation_tag,
            "per_rank_global_peak_bytes": [
                result.global_peak_bytes for result in results
            ],
        },
        assumptions=results[0].assumptions + (aggregation_assumption,),
    )


def measure_peak_memory(
    model: str | ModelSpec,
    config: TrainingConfig,
) -> MemoryResult:
    """Measure per-rank training-step memory on a CUDA runtime.

    Args:
        model: Hugging Face model reference or precomputed `ModelSpec`.
        config: Training configuration to measure.

    Returns:
        MemoryResult containing measured phase records and component breakdown.

    Example:
        >>> from simplesft.types import TrainingConfig
        >>> measure_peak_memory(
        ...     model="sshleifer/tiny-gpt2",
        ...     config=TrainingConfig(tuning_mode="full_ft", max_seq_len=8),
        ... )
        Traceback (most recent call last):
        ...
        RuntimeError: CUDA is required for measurement.
    """

    if config.distributed_mode == "zero2":
        maybe_get_deepspeed()
        if config.world_size() <= 1:
            _measure_zero2_requires_multi_gpu()
        from .distributed_zero2_measure import run_zero2_measurement

        return run_zero2_measurement(model=model, config=config)
    assert config.distributed_mode in {"single_gpu", "ddp"}, (
        "Measurement currently supports single_gpu, ddp, and zero2 runtime shapes only."
    )
    if not is_cuda_available():
        raise RuntimeError("CUDA is required for measurement.")
    if config.distributed_mode == "ddp" and config.world_size() > 1:
        from .distributed_measure import run_ddp_measurement

        return run_ddp_measurement(model=model, config=config)
    return _measure_local_peak_memory(model=model, config=config, device_index=0)
