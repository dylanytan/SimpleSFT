"""Runtime memory measurement for SimpleSFT training microsteps."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Iterator, cast

import torch

from .allocator import (
    AllocatorPhaseState,
    build_allocator_metadata,
    build_allocator_phase_states,
    selected_allocator_peak_state,
)
from .attribution import (
    build_reserved_carryover_metadata,
    build_workspace_proxy_metadata,
)
from ..constants import PHASE_PEAK_CANDIDATES
from ..estimator.estimate import estimate_lora_parameter_count
from ..models.inspect import inspect_model
from ..runtime import load_pretrained_model
from ..types import (
    MeasurementConfig,
    MemoryComponentBreakdown,
    MemoryResult,
    ModelSpec,
    PhaseMemoryRecord,
    TrainingConfig,
)
from ..utils import (
    canonical_torch_dtype,
    is_cuda_available,
    maybe_get_deepspeed,
    maybe_get_peft,
    optimizer_state_in_baseline,
)


@contextmanager
def _activation_tracker(
    *,
    model: torch.nn.Module,
    config: TrainingConfig,
) -> Iterator[dict[str, int]]:
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
                activations[module_name] = (
                    activations.get(module_name, 0) + tensor_bytes
                )

        return _record

    for module_name, module in model.named_modules():
        if any(
            token in module_name
            for token in config.measurement_activation_module_tokens
        ):
            handles.append(module.register_forward_hook(hook(module_name=module_name)))
    try:
        yield activations
    finally:
        for handle in handles:
            handle.remove()


def _cuda_snapshot(device: torch.device) -> tuple[int, int]:
    """Return current allocated and reserved CUDA memory for a device."""

    _synchronize_cuda(device=device)
    return (
        torch.cuda.memory_allocated(device=device),
        torch.cuda.memory_reserved(device=device),
    )


def _cuda_peak_snapshot(device: torch.device) -> tuple[int, int]:
    """Return synchronized peak allocated and reserved CUDA memory."""

    _synchronize_cuda(device=device)
    return (
        torch.cuda.max_memory_allocated(device=device),
        torch.cuda.max_memory_reserved(device=device),
    )


def _reset_cuda_peak_stats(device: torch.device) -> None:
    """Synchronize pending work before resetting CUDA peak statistics."""

    _synchronize_cuda(device=device)
    torch.cuda.reset_peak_memory_stats(device=device)


def _synchronize_cuda(*, device: torch.device) -> None:
    """Synchronize the current CUDA stream for stable memory accounting."""

    torch.cuda.synchronize(device=device)


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

    return sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )


def _gradient_bytes(model: torch.nn.Module) -> int:
    """Return live gradient memory in bytes."""

    return sum(
        parameter.grad.numel() * parameter.grad.element_size()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def _optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    """Return optimizer state tensor bytes."""

    if (
        hasattr(optimizer, "optimizer")
        and getattr(optimizer, "optimizer") is not optimizer
    ):
        nested_optimizer = cast(torch.optim.Optimizer, getattr(optimizer, "optimizer"))
        return _optimizer_state_bytes(optimizer=nested_optimizer)
    tensor_bytes = 0
    state_mapping = getattr(optimizer, "state", {})
    for state in state_mapping.values():
        for value in state.values():
            if isinstance(value, torch.Tensor):
                tensor_bytes += value.numel() * value.element_size()
    return tensor_bytes


def _activation_breakdown(
    *,
    activations: dict[str, int],
    config: TrainingConfig,
) -> tuple[int, int]:
    """Return activation bytes split between attention and non-attention blocks."""

    attention_bytes = sum(
        num_bytes
        for module_name, num_bytes in activations.items()
        if any(
            token in module_name for token in config.measurement_attention_module_tokens
        )
    )
    return attention_bytes, sum(activations.values()) - attention_bytes


def _phase_by_name(
    *,
    phase_records: list[PhaseMemoryRecord],
) -> dict[str, PhaseMemoryRecord]:
    """Return phase records keyed by phase name."""

    return {record.phase_name: record for record in phase_records}


def _retained_activation_proxy_bytes(
    *,
    phase_records: list[PhaseMemoryRecord],
) -> int:
    """Return retained activations from the post-forward allocated delta."""

    phase_by_name = _phase_by_name(phase_records=phase_records)
    return max(
        0,
        phase_by_name["forward"].allocated_bytes
        - phase_by_name["batch_materialization"].allocated_bytes,
    )


def _make_synthetic_batch(
    *,
    model_spec: ModelSpec,
    config: TrainingConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Create a synthetic training batch for measurement."""

    if model_spec.supports_vision_inputs() and config.vision_images_per_sample > 0:
        return _make_qwen_vision_batch(
            model_spec=model_spec,
            config=config,
            device=device,
        )
    return _make_text_batch(
        model_spec=model_spec,
        config=config,
        device=device,
    )


def _make_text_batch(
    *,
    model_spec: ModelSpec,
    config: TrainingConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Create a text-only synthetic training batch."""

    input_ids = torch.randint(
        low=0,
        high=model_spec.vocab_size,
        size=(config.micro_batch_size_per_gpu, config.max_seq_len),
        device=device,
    )
    attention_mask = torch.full_like(
        input_ids,
        fill_value=config.synthetic_attention_mask_value,
        device=device,
    )
    if config.synthetic_labels_mode == "clone_input_ids":
        labels = input_ids.clone()
    elif config.synthetic_labels_mode == "zeros":
        labels = torch.zeros_like(input_ids, device=device)
    else:
        raise AssertionError(
            f"Unsupported synthetic_labels_mode: {config.synthetic_labels_mode}"
        )
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _make_qwen_vision_batch(
    *,
    model_spec: ModelSpec,
    config: TrainingConfig,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Create a Qwen-VL style synthetic batch with one or more images."""

    assert model_spec.vision is not None
    vision_spec = model_spec.vision
    assert (
        vision_spec.supports_images()
    ), "Vision metadata is incomplete for image inputs."
    assert vision_spec.image_token_id is not None
    assert vision_spec.vision_start_token_id is not None
    assert vision_spec.vision_end_token_id is not None
    batch_size = config.micro_batch_size_per_gpu
    image_token_count = vision_spec.image_token_count(
        image_size=config.vision_image_size,
    )
    image_span_count = config.vision_images_per_sample
    text_token_count = max(config.max_seq_len - (2 * image_span_count), 0)
    vision_start = torch.full(
        (batch_size, 1),
        fill_value=vision_spec.vision_start_token_id,
        device=device,
        dtype=torch.long,
    )
    image_tokens = torch.full(
        (batch_size, image_token_count),
        fill_value=vision_spec.image_token_id,
        device=device,
        dtype=torch.long,
    )
    vision_end = torch.full(
        (batch_size, 1),
        fill_value=vision_spec.vision_end_token_id,
        device=device,
        dtype=torch.long,
    )
    segments = [
        tensor
        for _ in range(image_span_count)
        for tensor in (vision_start, image_tokens, vision_end)
    ]
    text_tokens = torch.randint(
        low=0,
        high=model_spec.vocab_size,
        size=(batch_size, text_token_count),
        device=device,
    )
    input_ids = torch.cat((*segments, text_tokens), dim=1)
    attention_mask = torch.ones_like(input_ids, device=device)
    labels = input_ids.clone()
    labels[:, : image_span_count * (image_token_count + 2)] = -100
    grid_side = vision_spec.grid_side(image_size=config.vision_image_size)
    patch_rows = batch_size * image_span_count * grid_side * grid_side
    pixel_values = torch.zeros(
        (patch_rows, vision_spec.flattened_patch_dim()),
        dtype=torch.float32,
        device=device,
    )
    image_grid_thw = torch.tensor(
        [[1, grid_side, grid_side]],
        dtype=torch.long,
        device=device,
    ).repeat(batch_size * image_span_count, 1)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values,
        "image_grid_thw": image_grid_thw,
    }


def _resolve_lora_task_type(*, config: TrainingConfig, peft_module: Any) -> Any:
    """Return the PEFT task type requested by the config."""

    task_type_name = config.measurement_lora_task_type.upper()
    assert hasattr(
        peft_module.TaskType, task_type_name
    ), f"Unsupported PEFT task type: {config.measurement_lora_task_type}"
    return getattr(peft_module.TaskType, task_type_name)


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
        task_type=_resolve_lora_task_type(config=config, peft_module=peft),
    )
    return peft.get_peft_model(model=model, peft_config=lora_config)


def _load_model(
    *,
    model_ref: str,
    model_spec: ModelSpec,
    config: TrainingConfig,
    device: torch.device,
) -> torch.nn.Module:
    """Load the model and apply the requested tuning mode."""

    loaded_model = load_pretrained_model(
        model_ref=model_ref,
        model_type=model_spec.model_type,
        torch_dtype=canonical_torch_dtype(config.weight_dtype),
        attention_backend=config.attention_backend,
        low_cpu_mem_usage=config.measurement_low_cpu_mem_usage,
        trust_remote_code=config.trust_remote_code,
    )
    loaded_model_any = cast(Any, loaded_model)
    model = cast(torch.nn.Module, loaded_model_any.to(device))
    if config.tuning_mode == "lora":
        model = _apply_lora(model=model, config=config)
    else:
        assert (
            config.tuning_mode == "full_ft"
        ), f"Unsupported tuning mode: {config.tuning_mode}"
    return model


def _disable_use_cache_for_checkpointing(*, model: torch.nn.Module) -> None:
    """Disable model caches that conflict with gradient checkpointing.

    Args:
        model: Loaded model before distributed wrappers.

    Returns:
        None. Mutates config objects in-place when `use_cache` exists.
    """

    model_any = cast(Any, model)
    candidate_configs = (
        getattr(model_any, "config", None),
        getattr(getattr(model_any, "model", None), "config", None),
        getattr(getattr(model_any, "base_model", None), "config", None),
    )
    for config_obj in candidate_configs:
        if config_obj is None or not hasattr(config_obj, "use_cache"):
            continue
        setattr(config_obj, "use_cache", False)


def _enable_checkpoint_input_grads(
    *,
    model: torch.nn.Module,
    config: TrainingConfig,
) -> None:
    """Ensure checkpointed LoRA runs have a grad-carrying model input.

    Args:
        model: Loaded model before distributed wrappers.
        config: Measurement configuration.

    Returns:
        None. Mutates model input handling in-place.

    Example:
        >>> cfg = TrainingConfig(tuning_mode="lora", gradient_checkpointing=True)
        >>> _enable_checkpoint_input_grads(model=model, config=cfg)
    """

    if not config.gradient_checkpointing or config.tuning_mode != "lora":
        return
    model_any = cast(Any, model)
    if hasattr(model_any, "enable_input_require_grads"):
        model_any.enable_input_require_grads()
        setattr(model_any, "_simplesft_input_grads_enabled", True)
        return
    assert hasattr(model_any, "get_input_embeddings"), (
        "Checkpointed LoRA models must expose enable_input_require_grads "
        "or get_input_embeddings."
    )
    embedding_module = model_any.get_input_embeddings()
    assert embedding_module is not None, (
        "Checkpointed LoRA models must provide input embeddings."
    )

    def _require_grad_on_embedding_output(
        _module: torch.nn.Module, _inputs: Any, output: Any
    ) -> None:
        if isinstance(output, torch.Tensor):
            output.requires_grad_(True)
            return
        if isinstance(output, (tuple, list)):
            for item in output:
                if isinstance(item, torch.Tensor):
                    item.requires_grad_(True)

    if getattr(model_any, "_simplesft_input_require_grads_hook", None) is None:
        hook_handle = embedding_module.register_forward_hook(
            _require_grad_on_embedding_output
        )
        setattr(model_any, "_simplesft_input_require_grads_hook", hook_handle)
    setattr(model_any, "_simplesft_input_grads_enabled", True)


def _configure_model_for_measurement(
    *,
    model: torch.nn.Module,
    config: TrainingConfig,
) -> None:
    """Prepare the model for measured training execution.

    Args:
        model: Loaded model before distributed wrappers.
        config: Measurement configuration.

    Returns:
        None. Applies training mode and checkpointing in-place.
    """

    model.train()
    if not config.gradient_checkpointing:
        return
    model_any = cast(Any, model)
    assert hasattr(model_any, "gradient_checkpointing_enable"), (
        "Model does not expose gradient_checkpointing_enable."
    )
    model_any.gradient_checkpointing_enable()
    _disable_use_cache_for_checkpointing(model=model)
    _enable_checkpoint_input_grads(model=model, config=config)


def _runtime_checkpointing_metadata(
    *,
    model: torch.nn.Module,
) -> dict[str, Any]:
    """Return runtime checkpointing state for one model instance.

    Args:
        model: Model used for measurement.

    Returns:
        Metadata dictionary exposing runtime training and checkpoint flags.
    """

    model_any = cast(Any, model)
    config_obj = getattr(model_any, "config", None)
    use_cache = getattr(config_obj, "use_cache", None) if config_obj else None
    return {
        "runtime_model_training_mode": bool(model.training),
        "runtime_gradient_checkpointing_enabled": bool(
            getattr(model_any, "is_gradient_checkpointing", False)
        ),
        "runtime_use_cache": use_cache,
        "runtime_checkpoint_input_grads_enabled": bool(
            getattr(model_any, "_simplesft_input_grads_enabled", False)
        ),
    }


def _build_optimizer(
    *,
    model: torch.nn.Module,
    config: TrainingConfig,
) -> torch.optim.Optimizer:
    """Build the requested optimizer for trainable model parameters."""

    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer_name = config.optimizer_name.lower()
    if optimizer_name == "adamw":
        return torch.optim.AdamW(
            params=parameters,
            lr=config.optimizer_learning_rate,
            betas=(config.optimizer_beta1, config.optimizer_beta2),
            eps=config.optimizer_epsilon,
            weight_decay=config.optimizer_weight_decay,
        )
    if optimizer_name == "adam":
        return torch.optim.Adam(
            params=parameters,
            lr=config.optimizer_learning_rate,
            betas=(config.optimizer_beta1, config.optimizer_beta2),
            eps=config.optimizer_epsilon,
            weight_decay=config.optimizer_weight_decay,
        )
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            params=parameters,
            lr=config.optimizer_learning_rate,
            momentum=config.optimizer_momentum,
            weight_decay=config.optimizer_weight_decay,
        )
    if optimizer_name == "rmsprop":
        return torch.optim.RMSprop(
            params=parameters,
            lr=config.optimizer_learning_rate,
            alpha=config.optimizer_alpha,
            eps=config.optimizer_epsilon,
            momentum=config.optimizer_momentum,
            centered=config.optimizer_centered,
            weight_decay=config.optimizer_weight_decay,
        )
    if optimizer_name == "adagrad":
        return torch.optim.Adagrad(
            params=parameters,
            lr=config.optimizer_learning_rate,
            eps=config.optimizer_epsilon,
            weight_decay=config.optimizer_weight_decay,
        )
    if optimizer_name == "adafactor":
        from transformers.optimization import Adafactor

        return Adafactor(
            params=parameters,
            lr=config.optimizer_learning_rate,
            eps=(config.optimizer_epsilon, config.optimizer_epsilon),
            weight_decay=config.optimizer_weight_decay,
            scale_parameter=False,
            relative_step=False,
            warmup_init=False,
            beta1=config.optimizer_beta1 if config.optimizer_momentum > 0 else None,
        )
    raise AssertionError(
        f"Unsupported optimizer for measurement: {config.optimizer_name}"
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
    with _activation_tracker(model=model, config=config) as activations:
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
        activations=activations,
        config=config,
    )
    parameter_bytes = _parameter_bytes(model=model)
    gradient_bytes = max(
        state_snapshots.get("gradient_bytes_after_backward", 0),
        state_snapshots.get("gradient_bytes_after_step", 0),
    )
    optimizer_state_bytes = state_snapshots.get("optimizer_state_bytes_after_step", 0)
    hook_visible_activation_bytes = attention_activation_bytes + other_activation_bytes
    retained_activation_bytes = _retained_activation_proxy_bytes(
        phase_records=phase_records,
    )
    observed_peak_bytes = max(record.peak_reserved_bytes for record in phase_records)
    optimizer_in_baseline = optimizer_state_in_baseline(
        warmup_steps=config.warmup_steps,
        optimizer_state_in_baseline_after_warmup=(
            config.optimizer_state_in_baseline_after_warmup
        ),
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
        accounted_bytes += gradient_bytes + retained_activation_bytes
    if peak_phase_record.phase_name == "optimizer_step":
        accounted_bytes += gradient_bytes
        if not optimizer_in_baseline:
            accounted_bytes += optimizer_state_bytes
    if peak_phase_record.phase_name == "forward":
        accounted_bytes += retained_activation_bytes
    return (
        MemoryComponentBreakdown(
            parameter_bytes=parameter_bytes,
            gradient_bytes=gradient_bytes,
            optimizer_state_bytes=optimizer_state_bytes,
            activation_bytes=retained_activation_bytes,
            transient_bytes=max(0, observed_peak_bytes - accounted_bytes),
            residual_bytes=0,
            runtime_reserve_bytes=runtime_reserve_bytes,
        ),
        {
            "retained_activation_bytes": retained_activation_bytes,
            "hook_visible_activation_bytes": hook_visible_activation_bytes,
            "hook_visible_attention_activation_bytes": attention_activation_bytes,
            "hook_visible_other_activation_bytes": other_activation_bytes,
            "attention_activation_bytes": attention_activation_bytes,
            "other_activation_bytes": other_activation_bytes,
            "hook_visible_activation_gap_bytes": max(
                0,
                hook_visible_activation_bytes - retained_activation_bytes,
            ),
        },
    )


def _build_measurement_assumptions(
    *,
    model_spec: ModelSpec,
    config: TrainingConfig,
) -> tuple[str, ...]:
    """Build measurement assumptions and notes."""

    assumptions = [
        "Measurement reports retained activations from the post-forward allocated delta.",
        "Measurement also keeps hook-visible activation summaries separate from retained activations.",
        "Gradient and optimizer-state bytes are captured from phase-local snapshots.",
    ]
    if optimizer_state_in_baseline(
        warmup_steps=config.warmup_steps,
        optimizer_state_in_baseline_after_warmup=(
            config.optimizer_state_in_baseline_after_warmup
        ),
    ):
        assumptions.append(
            "Warmup materializes optimizer state before the measured step."
        )
    if config.tuning_mode == "lora":
        assert (
            config.lora is not None
        ), "LoRA config is required for tuning_mode='lora'."
        assumptions.append(
            "Estimated trainable LoRA params: "
            f"{estimate_lora_parameter_count(model_spec=model_spec, lora_config=config.lora)}"
        )
    if model_spec.supports_vision_inputs() and config.vision_images_per_sample > 0:
        assumptions.append(
            "Vision-language measurement uses synthetic Qwen-style image placeholders "
            "plus zero-valued pixel patches."
        )
    return tuple(assumptions)


def _measure_zero_requires_multi_gpu(*, distributed_mode: str) -> None:
    """Raise a clear error for unsupported single-rank ZeRO measurement."""

    raise RuntimeError(
        f"{distributed_mode} measurement requires at least two GPUs so optimizer states, "
        "parameters, and gradients can actually be sharded."
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
        broadcast_buffers=config.measurement_ddp_broadcast_buffers,
        init_sync=config.measurement_ddp_init_sync,
    )


def _maybe_barrier(*, config: TrainingConfig) -> None:
    """Synchronize distributed ranks when a process group is active."""

    if config.distributed_mode != "ddp" or config.world_size() <= 1:
        return
    import torch.distributed as dist

    if dist.is_initialized():
        dist.barrier(device_ids=[torch.cuda.current_device()])


def _runtime_attention_implementation(*, model: torch.nn.Module) -> str:
    """Return the actual runtime attention implementation for a loaded model."""

    model_config = getattr(getattr(model, "module", model), "config", None)
    if model_config is None:
        return "unknown"
    implementation = getattr(model_config, "_attn_implementation", None)
    if implementation is None:
        implementation = getattr(model_config, "attn_implementation", None)
    return str(implementation or "default")


def _build_memory_result(
    *,
    model_spec: ModelSpec,
    config: TrainingConfig,
    phase_records: list[PhaseMemoryRecord],
    breakdown: MemoryComponentBreakdown,
    activation_metadata: dict[str, int],
    state_snapshots: dict[str, int],
    extra_metadata: dict[str, Any] | None = None,
) -> MemoryResult:
    """Build a `MemoryResult` from measured phase records and breakdowns."""

    workspace_proxy_metadata = build_workspace_proxy_metadata(
        phase_records=phase_records,
        breakdown=breakdown,
    )
    allocator_phase_states = build_allocator_phase_states(
        phase_records=tuple(
            record
            for record in phase_records
            if record.phase_name in PHASE_PEAK_CANDIDATES
        ),
        runtime_reserved_only_bytes=breakdown.runtime_reserve_bytes,
        gpu_capacity_bytes=int(config.gpu_memory_gb * (1024**3)),
        stress_trigger_fraction=config.allocator_stress_trigger_fraction,
    )
    allocator_peak_state = selected_allocator_peak_state(
        allocator_phase_states=allocator_phase_states,
        allocator_peak_mode=config.normalized_allocator_peak_mode(),
    )
    global_peak_bytes = _selected_allocator_peak_bytes(
        allocator_peak_mode=config.normalized_allocator_peak_mode(),
        allocator_peak_state=allocator_peak_state,
    )
    return MemoryResult(
        mode="measure",
        model_name=model_spec.model_name,
        config=replace(config),
        breakdown=breakdown,
        phase_records=tuple(phase_records),
        peak_phase=allocator_peak_state.phase_name,
        global_peak_bytes=global_peak_bytes,
        feasible=global_peak_bytes <= int(config.gpu_memory_gb * (1024**3)),
        metadata={
            "model_type": model_spec.model_type,
            "world_size": config.world_size(),
            "optimizer_state_in_baseline": optimizer_state_in_baseline(
                warmup_steps=config.warmup_steps,
                optimizer_state_in_baseline_after_warmup=(
                    config.optimizer_state_in_baseline_after_warmup
                ),
            ),
            "baseline_reserved_bytes": state_snapshots.get(
                "baseline_reserved_bytes", 0
            ),
            **(extra_metadata or {}),
            **activation_metadata,
            **workspace_proxy_metadata,
            **build_allocator_metadata(
                allocator_phase_states=allocator_phase_states,
            ),
            **build_reserved_carryover_metadata(phase_records=phase_records),
        },
        assumptions=_build_measurement_assumptions(
            model_spec=model_spec,
            config=config,
        ),
    )


def _selected_allocator_peak_bytes(
    *,
    allocator_peak_mode: str,
    allocator_peak_state: AllocatorPhaseState,
) -> int:
    """Return the selected peak bytes for one allocator mode."""

    if allocator_peak_mode == "allocated":
        return allocator_peak_state.allocated_peak_bytes
    if allocator_peak_mode == "stressed_reserved":
        return allocator_peak_state.stressed_reserved_peak_bytes
    return allocator_peak_state.soft_reserved_peak_bytes


def _measure_local_peak_memory(
    *,
    model: str | ModelSpec,
    config: TrainingConfig,
    device_index: int,
) -> MemoryResult:
    """Measure one local rank on its assigned CUDA device."""

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
    model_instance = _maybe_wrap_ddp_model(
        model=model_instance,
        config=config,
        device_index=device_index,
    )
    optimizer = _build_optimizer(model=model_instance, config=config)
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
        extra_metadata={
            "runtime_attention_implementation": _runtime_attention_implementation(
                model=model_instance,
            ),
            **_runtime_checkpointing_metadata(model=model_instance),
        },
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
            next(
                record
                for record in result.phase_records
                if record.phase_name == phase_name
            )
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
        optimizer_state_bytes=max(
            result.breakdown.optimizer_state_bytes for result in results
        ),
        activation_bytes=max(result.breakdown.activation_bytes for result in results),
        transient_bytes=max(result.breakdown.transient_bytes for result in results),
        residual_bytes=max(result.breakdown.residual_bytes for result in results),
        runtime_reserve_bytes=max(
            result.breakdown.runtime_reserve_bytes for result in results
        ),
    )


def _aggregate_numeric_metadata(*, results: list[MemoryResult]) -> dict[str, int]:
    """Aggregate conservative max numeric metadata across ranks."""

    numeric_keys = {
        key
        for result in results
        for key, value in result.metadata.items()
        if isinstance(value, int)
    }
    return {
        key: max(int(result.metadata.get(key, 0)) for result in results)
        for key in numeric_keys
    }


def aggregate_rank_results(
    *,
    results: list[MemoryResult],
    aggregation_tag: str = "ddp_aggregated",
    aggregation_assumption: str = "DDP result aggregates the max across ranks.",
) -> MemoryResult:
    """Aggregate per-rank results into a max-per-rank measurement result."""

    assert results, "At least one rank result is required."
    config = cast(MeasurementConfig, replace(results[0].config))
    phase_records = _aggregate_phase_records(
        results=results,
        aggregation_tag=aggregation_tag,
    )
    aggregated_breakdown = _aggregate_breakdown(results=results)
    allocator_phase_states = build_allocator_phase_states(
        phase_records=phase_records,
        runtime_reserved_only_bytes=aggregated_breakdown.runtime_reserve_bytes,
        gpu_capacity_bytes=int(config.gpu_memory_gb * (1024**3)),
        stress_trigger_fraction=config.allocator_stress_trigger_fraction,
    )
    allocator_peak_state = selected_allocator_peak_state(
        allocator_phase_states=allocator_phase_states,
        allocator_peak_mode=config.normalized_allocator_peak_mode(),
    )
    global_peak_bytes = _selected_allocator_peak_bytes(
        allocator_peak_mode=config.normalized_allocator_peak_mode(),
        allocator_peak_state=allocator_peak_state,
    )
    return MemoryResult(
        mode="measure",
        model_name=results[0].model_name,
        config=config,
        breakdown=aggregated_breakdown,
        phase_records=tuple(phase_records),
        peak_phase=allocator_peak_state.phase_name,
        global_peak_bytes=global_peak_bytes,
        feasible=global_peak_bytes <= int(results[0].config.gpu_memory_gb * (1024**3)),
        metadata={
            **results[0].metadata,
            **_aggregate_numeric_metadata(results=results),
            **build_allocator_metadata(
                allocator_phase_states=allocator_phase_states,
            ),
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

    if config.uses_tensor_parallel() or config.sequence_parallel:
        raise RuntimeError(
            "Measurement does not yet support tensor_parallel_degree > 1 "
            "or sequence_parallel=True."
        )
    if config.distributed_mode in {"zero2", "zero3"}:
        maybe_get_deepspeed()
        if config.world_size() <= 1:
            _measure_zero_requires_multi_gpu(
                distributed_mode=config.distributed_mode,
            )
        from .distributed_zero2 import run_zero2_measurement

        return run_zero2_measurement(model=model, config=config)
    assert config.distributed_mode in {
        "single_gpu",
        "ddp",
    }, "Measurement currently supports single_gpu, ddp, zero2, and zero3 runtime shapes only."
    if not is_cuda_available():
        raise RuntimeError("CUDA is required for measurement.")
    if config.distributed_mode == "ddp" and config.world_size() > 1:
        from .distributed import run_ddp_measurement

        return run_ddp_measurement(model=model, config=config)
    return _measure_local_peak_memory(model=model, config=config, device_index=0)
