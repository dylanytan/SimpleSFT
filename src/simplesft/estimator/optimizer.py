"""Explicit optimizer-state accounting for memory estimation."""

from __future__ import annotations
from typing import Union

from ..types import (
    EstimatorConfig,
    LoRAConfig,
    MeasurementConfig,
    ModelParameterSpec,
    ModelSpec,
)
from ..utils import bytes_for_dtype


ConfigLike = Union[EstimatorConfig, MeasurementConfig]


def normalized_optimizer_name(*, config: ConfigLike) -> str:
    """Return the normalized optimizer name.

    Args:
        config: Training configuration with optimizer settings.

    Returns:
        Lowercase optimizer name.
    """

    return config.optimizer_name.lower()


def optimizer_is_zero_tested(*, config: ConfigLike) -> bool:
    """Return whether one optimizer is explicitly treated as ZeRO-tested.

    Args:
        config: Training configuration with optimizer settings.

    Returns:
        True when the normalized optimizer name is listed in the config's
        ZeRO-tested optimizer names.
    """

    return normalized_optimizer_name(
        config=config
    ) in config.normalized_zero_tested_optimizer_names()


def _zero_untested_full_ft_update_copy_count(*, config: ConfigLike) -> int:
    """Return update-buffer copies for untested ZeRO full fine-tuning.

    Args:
        config: Training configuration with optimizer settings.

    Returns:
        Number of full-parameter update buffers materialized in the untested
        ZeRO full fine-tuning path.
    """

    if (
        (not config.is_zero_mode())
        or config.tuning_mode != "full_ft"
        or optimizer_is_zero_tested(config=config)
    ):
        return 1
    optimizer_name = normalized_optimizer_name(config=config)
    if optimizer_name in {"adagrad", "adafactor"}:
        return 2
    return 1


def zero_optimizer_state_is_sharded(*, config: ConfigLike) -> bool:
    """Return whether ZeRO optimizer-state residency should be sharded.

    Args:
        config: Training configuration with optimizer settings.

    Returns:
        True when ZeRO state bytes should be sharded across the data-parallel
        group for the selected optimizer.
    """

    if not config.is_zero_mode():
        return False
    if optimizer_is_zero_tested(config=config):
        return True
    return config.zero_untested_optimizer_state_is_sharded


def zero_optimizer_update_is_sharded(*, config: ConfigLike) -> bool:
    """Return whether ZeRO optimizer-update scratch should be sharded.

    Args:
        config: Training configuration with optimizer settings.

    Returns:
        True when ZeRO update scratch bytes should be sharded across the
        data-parallel group for the selected optimizer.
    """

    if not config.is_zero_mode():
        return False
    if optimizer_is_zero_tested(config=config):
        return True
    return config.zero_untested_optimizer_update_is_sharded


def trainable_parameter_specs(
    *,
    model_spec: ModelSpec,
    config: ConfigLike,
) -> tuple[ModelParameterSpec, ...]:
    """Return the trainable parameter shapes for the selected tuning mode.

    Args:
        model_spec: Inspected model summary.
        config: Training configuration being estimated.

    Returns:
        Tuple of trainable parameter summaries.
    """

    if config.tuning_mode == "full_ft":
        if model_spec.parameter_specs:
            return model_spec.parameter_specs
        return (
            ModelParameterSpec(
                parameter_name="model",
                shape=(model_spec.total_params,),
                category="other",
            ),
        )
    assert config.lora is not None, "LoRA config is required for tuning_mode='lora'."
    return _lora_parameter_specs(model_spec=model_spec, lora_config=config.lora)


def trainable_parameter_numel(*, model_spec: ModelSpec, config: ConfigLike) -> int:
    """Return trainable parameter elements for one config.

    Args:
        model_spec: Inspected model summary.
        config: Training configuration being estimated.

    Returns:
        Trainable parameter element count before dtype conversion.
    """

    return sum(
        parameter_spec.numel()
        for parameter_spec in trainable_parameter_specs(
            model_spec=model_spec,
            config=config,
        )
    )


def _lora_parameter_specs(
    *,
    model_spec: ModelSpec,
    lora_config: LoRAConfig,
) -> tuple[ModelParameterSpec, ...]:
    """Return synthetic LoRA adapter parameter shapes for targeted layers."""

    specs: list[ModelParameterSpec] = []
    for layer in model_spec.trainable_linear_layers:
        if not layer.module_name.endswith(lora_config.target_modules):
            continue
        specs.append(
            ModelParameterSpec(
                parameter_name=f"{layer.module_name}.lora_A",
                shape=(lora_config.rank, layer.input_dim),
                category=layer.category,
            )
        )
        specs.append(
            ModelParameterSpec(
                parameter_name=f"{layer.module_name}.lora_B",
                shape=(layer.output_dim, lora_config.rank),
                category=layer.category,
            )
        )
        if lora_config.bias != "none":
            specs.append(
                ModelParameterSpec(
                    parameter_name=f"{layer.module_name}.bias",
                    shape=(layer.output_dim,),
                    category=layer.category,
                )
            )
    return tuple(specs)


def optimizer_state_numel(
    *,
    model_spec: ModelSpec,
    config: ConfigLike,
) -> int:
    """Return the persistent optimizer-state element count.

    Args:
        model_spec: Inspected model summary.
        config: Training configuration being estimated.

    Returns:
        Total optimizer-state elements before dtype conversion and sharding.
    """

    return sum(
        optimizer_state_numel_for_parameter(
            parameter_spec=parameter_spec,
            config=config,
        )
        for parameter_spec in trainable_parameter_specs(
            model_spec=model_spec,
            config=config,
        )
    )


def optimizer_update_numel(
    *,
    model_spec: ModelSpec,
    config: ConfigLike,
) -> int:
    """Return the optimizer-step update element count.

    Args:
        model_spec: Inspected model summary.
        config: Training configuration being estimated.

    Returns:
        Total update-buffer elements touched during optimizer step.

    Example:
        >>> from simplesft.types import EstimatorConfig, ModelSpec
        >>> optimizer_update_numel(
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
        64
    """

    return sum(
        optimizer_update_numel_for_parameter(
            parameter_spec=parameter_spec,
            config=config,
        )
        for parameter_spec in trainable_parameter_specs(
            model_spec=model_spec,
            config=config,
        )
    )


def resolved_optimizer_update_dtype(*, config: ConfigLike) -> str:
    """Return the dtype used for optimizer-step scratch buffers.

    Args:
        config: Training configuration being estimated.

    Returns:
        Canonical dtype string for optimizer-step scratch buffers.
    """

    if config.optimizer_update_dtype != "auto":
        return config.optimizer_update_dtype
    optimizer_name = normalized_optimizer_name(config=config)
    state_like_optimizers = {"adam", "adamw", "adagrad", "rmsprop", "adafactor"}
    if optimizer_name in state_like_optimizers:
        if config.is_zero_mode():
            if config.tuning_mode == "lora":
                return config.adapter_state_dtype()
            if config.optimizer_state_dtype != "auto":
                return config.optimizer_state_dtype
            return "fp32"
        if config.tuning_mode == "lora":
            return config.adapter_parameter_dtype()
        return config.weight_dtype
    if config.tuning_mode == "lora":
        return config.adapter_gradient_dtype()
    return config.grad_dtype


def optimizer_state_numel_for_parameter(
    *,
    parameter_spec: ModelParameterSpec,
    config: ConfigLike,
) -> int:
    """Return optimizer-state elements for one trainable parameter tensor."""

    optimizer_name = normalized_optimizer_name(config=config)
    if optimizer_name in {"adam", "adamw"}:
        return 2 * parameter_spec.numel()
    if optimizer_name == "sgd":
        return 0
    if optimizer_name == "rmsprop":
        return parameter_spec.numel()
    if optimizer_name == "adagrad":
        return parameter_spec.numel()
    if optimizer_name == "adafactor":
        return _adafactor_state_numel(parameter_spec=parameter_spec)
    raise AssertionError(
        f"Unsupported optimizer for estimation: {config.optimizer_name}"
    )


def optimizer_update_numel_for_parameter(
    *,
    parameter_spec: ModelParameterSpec,
    config: ConfigLike,
) -> int:
    """Return optimizer-step scratch elements for one trainable parameter tensor."""

    optimizer_name = normalized_optimizer_name(config=config)
    if optimizer_name in {"adam", "adamw", "sgd", "rmsprop", "adagrad", "adafactor"}:
        return (
            _zero_untested_full_ft_update_copy_count(config=config)
            * parameter_spec.numel()
        )
    raise AssertionError(
        f"Unsupported optimizer for estimation: {config.optimizer_name}"
    )


def zero_untested_optimizer_state_replica_floor_bytes(
    *, model_spec: ModelSpec, config: ConfigLike
) -> int:
    """Return fallback unsharded ZeRO state bytes for untested optimizers.

    Args:
        model_spec: Inspected model summary.
        config: Training configuration with ZeRO settings.

    Returns:
        Lower-bound state residency bytes for untested optimizers. Returns zero
        for non-ZeRO or ZeRO-tested optimizers.
    """

    if (not config.is_zero_mode()) or optimizer_is_zero_tested(config=config):
        return 0
    replica_numel = int(
        round(
            trainable_parameter_numel(model_spec=model_spec, config=config)
            * config.zero_untested_optimizer_replica_tensor_count
        )
    )
    return replica_numel * bytes_for_dtype(config.zero_untested_replica_dtype())


def zero_untested_optimizer_update_replica_floor_bytes(
    *, model_spec: ModelSpec, config: ConfigLike
) -> int:
    """Return fallback unsharded ZeRO update bytes for untested optimizers.

    Args:
        model_spec: Inspected model summary.
        config: Training configuration with ZeRO settings.

    Returns:
        Lower-bound update residency bytes for untested optimizers. Returns zero
        for non-ZeRO or ZeRO-tested optimizers.
    """

    if (not config.is_zero_mode()) or optimizer_is_zero_tested(config=config):
        return 0
    replica_numel = int(
        round(
            trainable_parameter_numel(model_spec=model_spec, config=config)
            * config.zero_untested_optimizer_update_replica_tensor_count
        )
    )
    return replica_numel * bytes_for_dtype(config.zero_untested_update_dtype())


def _adafactor_state_numel(
    *,
    parameter_spec: ModelParameterSpec,
) -> int:
    """Return Adafactor state elements for one parameter tensor."""

    if not parameter_spec.is_matrix():
        return parameter_spec.numel()
    assert (
        len(parameter_spec.shape) >= 2
    ), "Matrix-like Adafactor state requires 2D shape."
    return parameter_spec.shape[0] + parameter_spec.shape[1]
