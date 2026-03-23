"""Analytical peak-memory estimation for SimpleSFT training configurations."""

from __future__ import annotations

from dataclasses import replace

from .attribution import build_workspace_proxy_metadata
from .constants import BYTES_PER_GB, PHASE_PEAK_CANDIDATES, TRAINING_PHASES
from .inspect import inspect_model
from .types import (
    LoRAConfig,
    MemoryComponentBreakdown,
    MemoryResult,
    ModelSpec,
    PhaseMemoryRecord,
    TrainingConfig,
)
from .utils import bytes_for_dtype, bytes_to_gb, optimizer_state_in_baseline


def _gradient_shard_factor(config: TrainingConfig) -> int:
    """Return the gradient sharding factor for the config."""

    if config.distributed_mode == "zero2":
        return max(config.world_size(), 1)
    if config.distributed_mode in {"ddp", "single_gpu"}:
        return 1
    raise AssertionError(f"Unsupported distributed mode: {config.distributed_mode}")


def _optimizer_shard_factor(config: TrainingConfig) -> int:
    """Return the optimizer-state sharding factor for the config."""

    if config.distributed_mode == "zero2":
        return max(config.world_size(), 1)
    if config.distributed_mode in {"ddp", "single_gpu"}:
        return 1
    raise AssertionError(f"Unsupported distributed mode: {config.distributed_mode}")


def _parameter_shard_factor(config: TrainingConfig) -> int:
    """Return the parameter residency sharding factor for the config."""

    if config.distributed_mode in {"ddp", "single_gpu", "zero2"}:
        return 1
    raise AssertionError(f"Unsupported distributed mode: {config.distributed_mode}")


def _resolved_optimizer_state_dtype(config: TrainingConfig) -> str:
    """Resolve the effective optimizer-state dtype for the config."""

    if config.optimizer_state_dtype == "auto":
        if config.tuning_mode == "lora":
            return "fp32"
        return config.weight_dtype
    return config.optimizer_state_dtype


def _effective_optimizer_state_dtype(config: TrainingConfig) -> str:
    """Resolve the optimizer-state dtype after backend-specific overrides."""

    if config.distributed_mode == "zero2" and config.tuning_mode == "full_ft":
        return "fp32"
    return _resolved_optimizer_state_dtype(config=config)


def _resolved_trainable_state_dtype(config: TrainingConfig) -> str:
    """Resolve the effective dtype for trainable adapter params and grads."""

    if config.tuning_mode == "lora":
        return "fp32"
    return config.grad_dtype


def _trainable_params(model_spec: ModelSpec, config: TrainingConfig) -> int:
    """Estimate the trainable parameter count for the selected tuning mode."""

    if config.tuning_mode == "full_ft":
        return model_spec.total_params
    assert config.tuning_mode == "lora", f"Unsupported tuning mode: {config.tuning_mode}"
    assert config.lora is not None, "LoRA config is required for tuning_mode='lora'."
    return estimate_lora_parameter_count(
        model_spec=model_spec,
        lora_config=config.lora,
    )


def estimate_lora_parameter_count(
    *,
    model_spec: ModelSpec,
    lora_config: LoRAConfig,
) -> int:
    """Estimate the total trainable LoRA parameter count.

    Args:
        model_spec: Inspected model architecture summary.
        lora_config: LoRA adapter configuration.

    Returns:
        Estimated total number of trainable LoRA parameters.

    Example:
        >>> from simplesft.types import LoRAConfig, ModelSpec
        >>> estimate_lora_parameter_count(
        ...     model_spec=ModelSpec(
        ...         model_name="toy",
        ...         model_type="llama",
        ...         num_layers=1,
        ...         hidden_size=8,
        ...         num_attention_heads=1,
        ...         intermediate_size=16,
        ...         vocab_size=32,
        ...         max_position_embeddings=64,
        ...         total_params=100,
        ...         trainable_linear_layers=(),
        ...     ),
        ...     lora_config=LoRAConfig(rank=4),
        ... )
        0
    """

    total_params = 0
    for layer in model_spec.trainable_linear_layers:
        if not layer.module_name.endswith(lora_config.target_modules):
            continue
        total_params += lora_config.rank * (layer.input_dim + layer.output_dim)
    return total_params


def _parameter_component_bytes(model_spec: ModelSpec, config: TrainingConfig) -> int:
    """Estimate persistent parameter bytes resident on one rank."""

    weight_bytes = bytes_for_dtype(config.weight_dtype)
    shard_factor = _parameter_shard_factor(config=config)
    parameter_count = model_spec.total_params
    parameter_bytes = (parameter_count * weight_bytes) // shard_factor
    if config.tuning_mode == "lora":
        parameter_bytes += (
            _trainable_params(model_spec=model_spec, config=config)
            * bytes_for_dtype(_resolved_trainable_state_dtype(config=config))
        )
    return parameter_bytes


def _gradient_component_bytes(model_spec: ModelSpec, config: TrainingConfig) -> int:
    """Estimate gradient bytes resident on one rank."""

    if config.distributed_mode == "zero2":
        return 0
    trainable_params = _trainable_params(model_spec=model_spec, config=config)
    grad_bytes = bytes_for_dtype(_resolved_trainable_state_dtype(config=config))
    shard_factor = _gradient_shard_factor(config=config)
    return (trainable_params * grad_bytes) // shard_factor


def _master_weight_component_bytes(model_spec: ModelSpec, config: TrainingConfig) -> int:
    """Estimate extra master-weight bytes resident on one rank."""

    if not config.use_master_weights:
        return 0
    if config.master_weight_dtype.lower() == config.weight_dtype.lower():
        return 0
    trainable_params = _trainable_params(model_spec=model_spec, config=config)
    master_bytes = bytes_for_dtype(config.master_weight_dtype)
    shard_factor = _optimizer_shard_factor(config=config)
    if config.distributed_mode == "zero2":
        return (trainable_params * master_bytes) // shard_factor
    return trainable_params * master_bytes


def _optimizer_component_bytes(model_spec: ModelSpec, config: TrainingConfig) -> int:
    """Estimate AdamW optimizer-state bytes resident on one rank."""

    assert config.optimizer_name.lower() == "adamw", "Only AdamW is supported in v1."
    trainable_params = _trainable_params(model_spec=model_spec, config=config)
    state_bytes = bytes_for_dtype(_effective_optimizer_state_dtype(config=config))
    shard_factor = _optimizer_shard_factor(config=config)
    total_state_bytes = trainable_params * state_bytes * 2
    return total_state_bytes // shard_factor


def _activation_component_bytes(model_spec: ModelSpec, config: TrainingConfig) -> int:
    """Estimate retained activation bytes for one microstep."""

    token_count = model_spec.tokens_per_layer(
        batch_size=config.micro_batch_size_per_gpu,
        sequence_length=config.max_seq_len,
    )
    activation_bytes = (
        token_count
        * model_spec.hidden_size
        * model_spec.num_layers
        * bytes_for_dtype(config.weight_dtype)
        * 24
    )
    if config.tuning_mode == "lora":
        activation_bytes = activation_bytes * 8 // 5
    if config.attention_backend.lower().startswith("flash"):
        activation_bytes = activation_bytes * 7 // 10
    return activation_bytes


def _is_large_single_gpu_model(*, parameter_bytes: int, config: TrainingConfig) -> bool:
    """Return whether a config falls into the large-model single-GPU regime."""

    return config.distributed_mode == "single_gpu" and parameter_bytes >= 2 * BYTES_PER_GB


def _is_large_ddp_model(*, parameter_bytes: int, config: TrainingConfig) -> bool:
    """Return whether a config falls into the large-model DDP regime."""

    return config.distributed_mode == "ddp" and parameter_bytes >= 2 * BYTES_PER_GB


def _ddp_lora_transient_scale(
    *,
    parameter_bytes: int,
    config: TrainingConfig,
) -> float:
    """Return a calibrated transient scale for LoRA DDP activation traffic."""

    parameter_gb = bytes_to_gb(parameter_bytes)
    sequence_ratio = min(config.max_seq_len / 512, 2.0)
    scale = 0.2 + (1.3 * sequence_ratio)
    scale += 0.25 * max(config.micro_batch_size_per_gpu - 1, 0) * sequence_ratio
    scale -= 0.1 * max(parameter_gb - 1.0, 0.0) * max(
        config.micro_batch_size_per_gpu - 1,
        0,
    )
    return max(scale, 0.25)


def _zero2_full_ft_transient_bytes(
    *,
    parameter_bytes: int,
    activation_bytes: int,
) -> int:
    """Estimate ZeRO-2 full-FT transient bytes from params and activations."""

    parameter_gb = bytes_to_gb(parameter_bytes)
    base_scale = max(4.4, 5.5 - (0.35 * parameter_gb))
    return int((parameter_bytes * base_scale) + (activation_bytes * 1.6))


def _zero2_lora_transient_bytes(
    *,
    parameter_bytes: int,
    activation_bytes: int,
    config: TrainingConfig,
) -> int:
    """Estimate ZeRO-2 LoRA transient bytes from params and activations."""

    parameter_gb = bytes_to_gb(parameter_bytes)
    parameter_scale = max(0.5, 2.9 - (0.8 * parameter_gb))
    activation_scale = max(
        0.45,
        0.95 - (0.16 * parameter_gb) + (0.15 * max(config.micro_batch_size_per_gpu - 1, 0)),
    )
    return int((parameter_bytes * parameter_scale) + (activation_bytes * activation_scale))


def _transient_component_bytes(model_spec: ModelSpec, config: TrainingConfig) -> int:
    """Estimate communication and temporary transient bytes."""

    token_count = model_spec.tokens_per_layer(
        batch_size=config.micro_batch_size_per_gpu,
        sequence_length=config.max_seq_len,
    )
    parameter_bytes = _parameter_component_bytes(model_spec=model_spec, config=config)
    activation_bytes = _activation_component_bytes(model_spec=model_spec, config=config)
    transient_bytes = token_count * model_spec.hidden_size * bytes_for_dtype(config.weight_dtype)
    if config.distributed_mode == "single_gpu" and config.tuning_mode == "full_ft":
        if _is_large_single_gpu_model(parameter_bytes=parameter_bytes, config=config):
            return parameter_bytes + max(
                parameter_bytes // 8,
                max(0, (activation_bytes * 14) // 10 - parameter_bytes // 10),
            )
        return max((parameter_bytes * 94) // 100, (activation_bytes * 9) // 5)
    if config.distributed_mode == "single_gpu" and config.tuning_mode == "lora":
        sequence_ratio = min(config.max_seq_len, 512) / 512
        lora_workspace_scale = 0.35 + 0.6 * sequence_ratio
        transient_bytes += int(activation_bytes * lora_workspace_scale)
        if _is_large_single_gpu_model(parameter_bytes=parameter_bytes, config=config):
            large_model_scale = 1.0 + 0.3 * max(0.0, bytes_to_gb(parameter_bytes) - 1.0) * sequence_ratio
            transient_bytes = int(transient_bytes * large_model_scale)
    if config.distributed_mode == "ddp":
        if config.tuning_mode == "full_ft":
            if _is_large_ddp_model(parameter_bytes=parameter_bytes, config=config):
                return parameter_bytes + (parameter_bytes // 4)
            return (parameter_bytes * 7) // 8
        transient_bytes += int(
            activation_bytes
            * _ddp_lora_transient_scale(
                parameter_bytes=parameter_bytes,
                config=config,
            )
        )
        return transient_bytes
    if config.distributed_mode == "zero2":
        if config.tuning_mode == "full_ft":
            return _zero2_full_ft_transient_bytes(
                parameter_bytes=parameter_bytes,
                activation_bytes=activation_bytes,
            )
        return _zero2_lora_transient_bytes(
            parameter_bytes=parameter_bytes,
            activation_bytes=activation_bytes,
            config=config,
        )
    return transient_bytes


def _zero2_full_ft_reserve_gb(*, parameter_bytes: int) -> float:
    """Return the calibrated ZeRO-2 full-FT reserve in GiB."""

    return 0.08 + (0.99 * bytes_to_gb(parameter_bytes))


def _zero2_lora_reserve_gb(
    *,
    parameter_bytes: int,
    config: TrainingConfig,
) -> float:
    """Return the calibrated ZeRO-2 LoRA reserve in GiB."""

    parameter_gb = bytes_to_gb(parameter_bytes)
    sequence_ratio = min(config.max_seq_len / 512, 1.0)
    return (0.03 * parameter_gb) + (
        0.04 * max(config.micro_batch_size_per_gpu - 1, 0) * sequence_ratio
    )


def _runtime_reserve_bytes(
    *,
    parameter_bytes: int,
    activation_bytes: int,
    config: TrainingConfig,
) -> int:
    """Return the fixed runtime reserve bytes for one rank."""

    if config.reserved_vram_gb_per_gpu is not None:
        reserve_gb = config.reserved_vram_gb_per_gpu
    elif config.distributed_mode == "ddp" and config.tuning_mode == "full_ft":
        reserve_gb = 0.35 + (1.1 * bytes_to_gb(parameter_bytes))
    elif config.distributed_mode == "ddp" and config.tuning_mode == "lora":
        reserve_gb = 0.0
    elif (
        config.distributed_mode == "single_gpu"
        and config.tuning_mode == "full_ft"
        and optimizer_state_in_baseline(
            warmup_steps=config.warmup_steps,
            optimizer_name=config.optimizer_name,
        )
    ):
        reserve_gb = 0.05 + (0.55 * bytes_to_gb(activation_bytes))
    elif config.distributed_mode == "zero2" and config.tuning_mode == "full_ft":
        reserve_gb = _zero2_full_ft_reserve_gb(parameter_bytes=parameter_bytes)
    elif config.distributed_mode == "zero2":
        reserve_gb = _zero2_lora_reserve_gb(
            parameter_bytes=parameter_bytes,
            config=config,
        )
    elif config.distributed_mode == "ddp":
        reserve_gb = 2.0
    elif config.tuning_mode == "lora":
        reserve_gb = 0.0
    else:
        reserve_gb = 2.0
    reserve_gb += config.activation_safety_margin_gb
    return int(reserve_gb * (1024**3))


def _build_phase_records(
    *,
    config: TrainingConfig,
    breakdown: MemoryComponentBreakdown,
) -> tuple[PhaseMemoryRecord, ...]:
    """Build synthetic phase records from an analytical breakdown."""

    optimizer_in_baseline = optimizer_state_in_baseline(
        warmup_steps=config.warmup_steps,
        optimizer_name=config.optimizer_name,
    )
    baseline_bytes = breakdown.parameter_bytes + breakdown.runtime_reserve_bytes
    if optimizer_in_baseline:
        baseline_bytes += breakdown.optimizer_state_bytes
    post_step_baseline_bytes = baseline_bytes
    if not optimizer_in_baseline:
        post_step_baseline_bytes += breakdown.optimizer_state_bytes

    parameter_gb = bytes_to_gb(breakdown.parameter_bytes)
    seq_ratio = min(config.max_seq_len / 512, 1.0)

    if config.distributed_mode == "single_gpu" and config.tuning_mode == "full_ft":
        optimizer_transient_bytes = breakdown.transient_bytes
        phase_allocated = {
            "model_load": breakdown.parameter_bytes + breakdown.runtime_reserve_bytes,
            "optimizer_create": baseline_bytes,
            "post_init_baseline": baseline_bytes,
            "batch_materialization": baseline_bytes,
            "forward": baseline_bytes + breakdown.activation_bytes,
            "loss_materialization": baseline_bytes + breakdown.activation_bytes,
            "backward": baseline_bytes + breakdown.gradient_bytes,
            "optimizer_step": baseline_bytes + breakdown.gradient_bytes,
            "zero_grad": post_step_baseline_bytes,
            "step_end": post_step_baseline_bytes,
        }
        phase_reserved = {
            "model_load": phase_allocated["model_load"],
            "optimizer_create": baseline_bytes,
            "post_init_baseline": baseline_bytes,
            "batch_materialization": baseline_bytes,
            "forward": baseline_bytes + (breakdown.activation_bytes * 2),
            "loss_materialization": baseline_bytes + (breakdown.activation_bytes * 2),
            "backward": (
                baseline_bytes
                + breakdown.gradient_bytes
                + breakdown.activation_bytes
                + max(breakdown.parameter_bytes // 4, optimizer_transient_bytes // 4)
            ),
            "optimizer_step": (
                baseline_bytes
                + breakdown.gradient_bytes
                + optimizer_transient_bytes
            ),
            "zero_grad": max(
                post_step_baseline_bytes,
                baseline_bytes
                + breakdown.gradient_bytes
                + optimizer_transient_bytes,
            ),
            "step_end": max(
                post_step_baseline_bytes,
                baseline_bytes
                + breakdown.gradient_bytes
                + optimizer_transient_bytes,
            ),
        }
    elif config.distributed_mode == "ddp":
        forward_peak = (
            baseline_bytes
            + breakdown.activation_bytes
            + breakdown.transient_bytes // 3
        )
        backward_peak = (
            baseline_bytes
            + breakdown.gradient_bytes
            + breakdown.activation_bytes // 2
            + breakdown.transient_bytes // 2
        )
        optimizer_peak = (
            baseline_bytes
            + breakdown.gradient_bytes
            + breakdown.transient_bytes
        )
        optimizer_peak = max(optimizer_peak, backward_peak)
        phase_allocated = {
            "model_load": breakdown.parameter_bytes + breakdown.runtime_reserve_bytes,
            "optimizer_create": baseline_bytes,
            "post_init_baseline": baseline_bytes,
            "batch_materialization": baseline_bytes,
            "forward": forward_peak,
            "loss_materialization": forward_peak,
            "backward": backward_peak,
            "optimizer_step": optimizer_peak,
            "zero_grad": post_step_baseline_bytes,
            "step_end": post_step_baseline_bytes,
        }
        phase_reserved = {
            **phase_allocated,
            "zero_grad": max(post_step_baseline_bytes, optimizer_peak),
            "step_end": max(post_step_baseline_bytes, optimizer_peak),
        }
    elif config.distributed_mode == "zero2" and config.tuning_mode == "full_ft":
        forward_ratio = max(
            0.01,
            0.03
            + (0.04 * seq_ratio)
            + (0.08 * max(config.micro_batch_size_per_gpu - 1, 0) * seq_ratio)
            - (0.015 * max(parameter_gb - 1.0, 0.0)),
        )
        backward_ratio = min(
            0.85,
            max(
                0.42,
                0.88
                - (0.15 * parameter_gb)
                + (0.02 * max(config.micro_batch_size_per_gpu - 1, 0))
                + (0.04 * seq_ratio),
            ),
        )
        forward_peak = (
            baseline_bytes
            + breakdown.activation_bytes
            + int(breakdown.transient_bytes * forward_ratio)
        )
        backward_peak = baseline_bytes + int(breakdown.transient_bytes * backward_ratio)
        optimizer_peak = baseline_bytes + breakdown.transient_bytes
        phase_allocated = {
            "model_load": breakdown.parameter_bytes + breakdown.runtime_reserve_bytes,
            "optimizer_create": baseline_bytes,
            "post_init_baseline": baseline_bytes,
            "batch_materialization": baseline_bytes,
            "forward": forward_peak,
            "loss_materialization": forward_peak,
            "backward": backward_peak,
            "optimizer_step": optimizer_peak,
            "zero_grad": post_step_baseline_bytes,
            "step_end": post_step_baseline_bytes,
        }
        phase_reserved = {
            **phase_allocated,
            "zero_grad": max(post_step_baseline_bytes, optimizer_peak),
            "step_end": max(post_step_baseline_bytes, optimizer_peak),
        }
    elif config.distributed_mode == "zero2":
        forward_peak = baseline_bytes + int(breakdown.activation_bytes * 1.25)
        backward_peak = (
            baseline_bytes
            + breakdown.transient_bytes
            + int(breakdown.activation_bytes * 0.75)
        )
        optimizer_peak = backward_peak
        phase_allocated = {
            "model_load": breakdown.parameter_bytes + breakdown.runtime_reserve_bytes,
            "optimizer_create": baseline_bytes,
            "post_init_baseline": baseline_bytes,
            "batch_materialization": baseline_bytes,
            "forward": forward_peak,
            "loss_materialization": forward_peak,
            "backward": backward_peak,
            "optimizer_step": optimizer_peak,
            "zero_grad": post_step_baseline_bytes,
            "step_end": post_step_baseline_bytes,
        }
        phase_reserved = {
            **phase_allocated,
            "zero_grad": max(post_step_baseline_bytes, optimizer_peak),
            "step_end": max(post_step_baseline_bytes, optimizer_peak),
        }
    else:
        forward_peak = (
            baseline_bytes + breakdown.activation_bytes + breakdown.transient_bytes // 2
        )
        if (
            config.tuning_mode == "lora"
            and _is_large_single_gpu_model(
                parameter_bytes=breakdown.parameter_bytes,
                config=config,
            )
        ):
            forward_peak = (
                baseline_bytes
                + breakdown.activation_bytes
                + breakdown.transient_bytes // 6
            )
            backward_peak = (
                baseline_bytes
                + breakdown.gradient_bytes
                + breakdown.transient_bytes
            )
            optimizer_peak = backward_peak
        else:
            backward_peak = (
                baseline_bytes
                + breakdown.gradient_bytes
                + breakdown.activation_bytes
                + breakdown.transient_bytes
            )
            optimizer_peak = (
                post_step_baseline_bytes
                + breakdown.gradient_bytes
                + breakdown.transient_bytes
            )
        optimizer_peak = max(optimizer_peak, backward_peak)
        phase_allocated = {
            "model_load": breakdown.parameter_bytes + breakdown.runtime_reserve_bytes,
            "optimizer_create": baseline_bytes,
            "post_init_baseline": baseline_bytes,
            "batch_materialization": baseline_bytes + breakdown.transient_bytes // 4,
            "forward": forward_peak,
            "loss_materialization": forward_peak,
            "backward": backward_peak,
            "optimizer_step": optimizer_peak,
            "zero_grad": post_step_baseline_bytes,
            "step_end": post_step_baseline_bytes,
        }
        phase_reserved = {
            **phase_allocated,
            "optimizer_step": optimizer_peak,
            "zero_grad": max(post_step_baseline_bytes, optimizer_peak),
            "step_end": max(post_step_baseline_bytes, optimizer_peak),
        }

    previous_allocated = 0
    previous_reserved = 0
    records: list[PhaseMemoryRecord] = []
    for phase_name in TRAINING_PHASES:
        allocated_bytes = phase_allocated[phase_name]
        reserved_bytes = phase_reserved[phase_name]
        records.append(
            PhaseMemoryRecord(
                phase_name=phase_name,
                allocated_bytes=allocated_bytes,
                reserved_bytes=reserved_bytes,
                peak_allocated_bytes=allocated_bytes,
                peak_reserved_bytes=reserved_bytes,
                delta_allocated_bytes=allocated_bytes - previous_allocated,
                delta_reserved_bytes=reserved_bytes - previous_reserved,
                notes=("estimated",),
            )
        )
        previous_allocated = allocated_bytes
        previous_reserved = reserved_bytes
    return tuple(records)


def estimate_peak_memory(
    model: str | ModelSpec,
    config: TrainingConfig,
) -> MemoryResult:
    """Estimate per-rank peak memory for one training configuration.

    Args:
        model: Hugging Face model reference or precomputed `ModelSpec`.
        config: Training configuration to evaluate.

    Returns:
        MemoryResult containing the estimate and modular breakdown.

    Example:
        >>> from simplesft.types import TrainingConfig
        >>> result = estimate_peak_memory(
        ...     model="sshleifer/tiny-gpt2",
        ...     config=TrainingConfig(tuning_mode="full_ft", max_seq_len=16),
        ... )
        >>> result.mode
        'estimate'
    """

    model_spec = inspect_model(model_ref=model) if isinstance(model, str) else model
    parameter_bytes = _parameter_component_bytes(model_spec=model_spec, config=config) + (
        _master_weight_component_bytes(model_spec=model_spec, config=config)
    )
    activation_bytes = _activation_component_bytes(model_spec=model_spec, config=config)
    breakdown = MemoryComponentBreakdown(
        parameter_bytes=parameter_bytes,
        gradient_bytes=_gradient_component_bytes(model_spec=model_spec, config=config),
        optimizer_state_bytes=_optimizer_component_bytes(
            model_spec=model_spec,
            config=config,
        ),
        activation_bytes=activation_bytes,
        transient_bytes=_transient_component_bytes(model_spec=model_spec, config=config),
        residual_bytes=0,
        runtime_reserve_bytes=_runtime_reserve_bytes(
            parameter_bytes=parameter_bytes,
            activation_bytes=activation_bytes,
            config=config,
        ),
    )
    phase_records = _build_phase_records(config=config, breakdown=breakdown)
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
        mode="estimate",
        model_name=model_spec.model_name,
        config=replace(config),
        breakdown=breakdown,
        phase_records=phase_records,
        peak_phase=peak_phase_record.phase_name,
        global_peak_bytes=global_peak_bytes,
        feasible=global_peak_bytes <= int(config.gpu_memory_gb * (1024**3)),
        metadata={
            "model_type": model_spec.model_type,
            "world_size": config.world_size(),
            "trainable_params": _trainable_params(model_spec=model_spec, config=config),
            "optimizer_state_in_baseline": optimizer_state_in_baseline(
                warmup_steps=config.warmup_steps,
                optimizer_name=config.optimizer_name,
            ),
            "optimizer_state_dtype_resolved": _effective_optimizer_state_dtype(
                config=config
            ),
            **build_workspace_proxy_metadata(
                phase_records=phase_records,
                breakdown=breakdown,
            ),
        },
        assumptions=(
            "Estimator is analytical and uses heuristic activation/transient terms.",
            "Gradient checkpointing currently does not reduce the activation component directly; peak impact is modeled through phase heuristics.",
            "AdamW optimizer-state dtype defaults to weight dtype when optimizer_state_dtype='auto'.",
            "LoRA trainable adapters, gradients, and optimizer states default to fp32 in the current runtime model.",
            "Auto reserve defaults are backend- and tuning-mode-specific.",
            "ZeRO-2 estimates keep parameters replicated, model gradients as transient, and shard optimizer states across world size.",
        ),
    )
