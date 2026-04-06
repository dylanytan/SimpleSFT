"""Model inspection helpers for officially supported dense language models."""

from __future__ import annotations

from typing import Any, Iterable, cast

import torch

from .architecture_registry import (
    ArchitectureManifest,
    SyntheticLinearRule,
    manifest_for_model_type,
)
from .architecture_types import (
    ArchitectureFamilySpec,
    AttentionSpec,
    TensorLayoutSpec,
    LINEAR_ROLE_ATTENTION_KEY,
    LINEAR_ROLE_ATTENTION_OUTPUT,
    LINEAR_ROLE_ATTENTION_QKV,
    LINEAR_ROLE_ATTENTION_QUERY,
    LINEAR_ROLE_ATTENTION_VALUE,
)
from ..constants import SUPPORTED_MODEL_TYPES, model_type_is_supported
from ..runtime import build_empty_model, load_auto_config
from ..types import ModelLinearLayerSpec, ModelParameterSpec, ModelSpec


def _get_config_value(config: Any, *names: str) -> int:
    """Return the first available integer config attribute.

    Args:
        config: Loaded Hugging Face config.
        names: Candidate attribute names to read.

    Returns:
        First present integer value.
    """

    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    raise AssertionError(f"Config is missing required attributes: {names}")


def _maybe_get_config_value(config: Any, *names: str) -> int | None:
    """Return the first optional integer config attribute when present."""

    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    return None


def _get_intermediate_size(
    *,
    config: Any,
    manifest: ArchitectureManifest,
    fallback_multiplier: int,
) -> int:
    """Return the MLP intermediate size with manifest-aware fallbacks."""

    value = _maybe_get_config_value(config, *manifest.intermediate_size_fields)
    if value is not None:
        return value
    hidden_size = manifest.get_first_config_value(
        config=config,
        field_names=manifest.hidden_size_fields,
    )
    return fallback_multiplier * hidden_size


def _build_linear_like_spec(
    *,
    module_name: str,
    module: torch.nn.Module,
    manifest: ArchitectureManifest,
) -> ModelLinearLayerSpec | None:
    """Build a layer summary for one linear-like torch module."""

    input_dim: int
    output_dim: int
    if isinstance(module, torch.nn.Linear):
        input_dim = module.in_features
        output_dim = module.out_features
    elif module.__class__.__name__ == "Conv1D" and hasattr(module, "weight"):
        weight_shape = cast(torch.Tensor, module.weight).shape
        input_dim = int(weight_shape[0])
        output_dim = int(weight_shape[1])
    else:
        return None
    classification = manifest.classify_linear(module_name=module_name)
    return ModelLinearLayerSpec(
        module_name=module_name,
        input_dim=input_dim,
        output_dim=output_dim,
        category=classification.category,
        role=classification.role,
        tensor_parallel_role=classification.tensor_parallel_role,
    )


def _synthetic_linear_spec_from_parameter(
    *,
    parameter_name: str,
    parameter: torch.nn.Parameter,
    rule: SyntheticLinearRule,
    manifest: ArchitectureManifest,
) -> ModelLinearLayerSpec:
    """Build one pseudo-linear layer from a structured parameter tensor."""

    shape = tuple(int(dim_size) for dim_size in parameter.shape)
    assert rule.input_axis < len(shape), f"Missing input axis for {parameter_name}"
    assert rule.output_axis < len(shape), f"Missing output axis for {parameter_name}"
    classification = manifest.classify_linear(module_name=parameter_name)
    return ModelLinearLayerSpec(
        module_name=parameter_name,
        input_dim=shape[rule.input_axis],
        output_dim=shape[rule.output_axis],
        category=classification.category,
        role=classification.role,
        tensor_parallel_role=classification.tensor_parallel_role,
    )


def _iter_linear_layers(
    *,
    model: torch.nn.Module,
    manifest: ArchitectureManifest,
) -> Iterable[ModelLinearLayerSpec]:
    """Yield explicit linear-layer summaries from one model."""

    seen_module_names: set[str] = set()
    for module_name, module in model.named_modules():
        layer_spec = _build_linear_like_spec(
            module_name=module_name,
            module=module,
            manifest=manifest,
        )
        if layer_spec is None:
            continue
        seen_module_names.add(module_name)
        yield layer_spec
    for parameter_name, parameter in model.named_parameters():
        if parameter_name in seen_module_names:
            continue
        for rule in manifest.synthetic_linear_rules:
            if not rule.matches(parameter_name=parameter_name):
                continue
            yield _synthetic_linear_spec_from_parameter(
                parameter_name=parameter_name,
                parameter=parameter,
                rule=rule,
                manifest=manifest,
            )
            break


def _iter_parameter_specs(
    *,
    model: torch.nn.Module,
    manifest: ArchitectureManifest,
) -> Iterable[ModelParameterSpec]:
    """Yield explicit parameter summaries from one model."""

    for parameter_name, parameter in model.named_parameters():
        classification = manifest.classify_parameter(parameter_name=parameter_name)
        yield ModelParameterSpec(
            parameter_name=parameter_name,
            shape=tuple(int(dim_size) for dim_size in parameter.shape),
            category=classification.category,
            role=classification.role,
            tensor_parallel_role=classification.tensor_parallel_role,
        )


def _attention_projection_width(
    *,
    linear_layers: tuple[ModelLinearLayerSpec, ...],
    role: str,
    default_width: int,
) -> int:
    """Return the representative width for one attention projection role."""

    for layer in linear_layers:
        if layer.role != role:
            continue
        if role == LINEAR_ROLE_ATTENTION_OUTPUT:
            return layer.input_dim
        return layer.output_dim
    return default_width


def _sliding_window_size(*, config: Any, manifest: ArchitectureManifest) -> int | None:
    """Return the configured sliding-window size when present."""

    window_size = _maybe_get_config_value(config, *manifest.sliding_window_fields)
    if window_size is None or window_size <= 0:
        return None
    return window_size


def _rope_metadata(
    *, config: Any, family_spec: ArchitectureFamilySpec
) -> tuple[str, bool]:
    """Return normalized rope metadata from a config."""

    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is None:
        return ("rope" if family_spec.supports_rope else "none", False)
    if isinstance(rope_scaling, dict):
        rope_variant = str(
            rope_scaling.get("rope_type", rope_scaling.get("type", "rope"))
        )
        return (rope_variant, True)
    return ("rope", True)


def _build_attention_spec(
    *,
    config: Any,
    manifest: ArchitectureManifest,
    hidden_size: int,
    num_attention_heads: int,
    linear_layers: tuple[ModelLinearLayerSpec, ...],
) -> AttentionSpec:
    """Build explicit attention metadata for one inspected model."""

    num_key_value_heads = _maybe_get_config_value(
        config,
        *manifest.num_key_value_heads_fields,
    )
    num_key_value_heads = num_key_value_heads or num_attention_heads
    head_dim = _maybe_get_config_value(config, *manifest.head_dim_fields)
    head_dim = head_dim or max(1, hidden_size // max(num_attention_heads, 1))
    rope_variant, rope_scaling_present = _rope_metadata(
        config=config,
        family_spec=manifest.family_spec,
    )
    return AttentionSpec(
        num_query_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        query_width=_attention_projection_width(
            linear_layers=linear_layers,
            role=LINEAR_ROLE_ATTENTION_QUERY,
            default_width=head_dim * num_attention_heads,
        ),
        key_width=_attention_projection_width(
            linear_layers=linear_layers,
            role=LINEAR_ROLE_ATTENTION_KEY,
            default_width=head_dim * num_key_value_heads,
        ),
        value_width=_attention_projection_width(
            linear_layers=linear_layers,
            role=LINEAR_ROLE_ATTENTION_VALUE,
            default_width=head_dim * num_key_value_heads,
        ),
        output_proj_input_width=_attention_projection_width(
            linear_layers=linear_layers,
            role=LINEAR_ROLE_ATTENTION_OUTPUT,
            default_width=head_dim * num_attention_heads,
        ),
        uses_grouped_query=num_attention_heads > num_key_value_heads > 1,
        uses_multi_query=num_attention_heads > 1 and num_key_value_heads == 1,
        sliding_window_size=_sliding_window_size(config=config, manifest=manifest),
        rope_variant=rope_variant,
        rope_scaling_present=rope_scaling_present,
    )


def inspect_model(
    model_ref: str,
    *,
    trust_remote_code: bool = False,
    supported_model_types: tuple[str, ...] = tuple(sorted(SUPPORTED_MODEL_TYPES)),
    default_attention_type: str = "causal",
    intermediate_size_fallback_multiplier: int = 4,
) -> ModelSpec:
    """Inspect one officially supported dense Hugging Face model.

    Args:
        model_ref: Hugging Face model id or local model path.
        trust_remote_code: Whether to allow custom remote model code.
        supported_model_types: Explicit model-type allow-list.
        default_attention_type: Normalized attention label.
        intermediate_size_fallback_multiplier: Fallback MLP expansion ratio.

    Returns:
        Structured `ModelSpec` with explicit architecture metadata.

    Example:
        >>> spec = inspect_model("openai-community/gpt2")
        >>> spec.architecture_family.family_label
        'gpt2_dense'
    """

    config = load_auto_config(
        model_ref=model_ref,
        trust_remote_code=trust_remote_code,
    )
    assert model_type_is_supported(
        model_type=config.model_type,
        supported_model_types=supported_model_types,
    ), (
        f"Unsupported model_type `{config.model_type}`. "
        "Missing explicit dense architecture manifest for this family."
    )
    manifest = manifest_for_model_type(model_type=config.model_type)
    assert manifest is not None, (
        f"Unsupported model_type `{config.model_type}`. "
        "No architecture manifest is registered for this family."
    )
    model = build_empty_model(
        config=config,
        trust_remote_code=trust_remote_code,
    )
    linear_layers = tuple(_iter_linear_layers(model=model, manifest=manifest))
    hidden_size = manifest.get_first_config_value(
        config=config,
        field_names=manifest.hidden_size_fields,
    )
    num_attention_heads = manifest.get_first_config_value(
        config=config,
        field_names=manifest.num_attention_heads_fields,
    )
    return ModelSpec(
        model_name=model_ref,
        model_type=config.model_type,
        num_layers=manifest.get_first_config_value(
            config=config,
            field_names=manifest.num_layers_fields,
        ),
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        intermediate_size=_get_intermediate_size(
            config=config,
            manifest=manifest,
            fallback_multiplier=intermediate_size_fallback_multiplier,
        ),
        vocab_size=_get_config_value(config, "vocab_size"),
        max_position_embeddings=manifest.get_first_config_value(
            config=config,
            field_names=manifest.max_position_fields,
        ),
        total_params=sum(parameter.numel() for parameter in model.parameters()),
        trainable_linear_layers=linear_layers,
        parameter_specs=tuple(_iter_parameter_specs(model=model, manifest=manifest)),
        attention_type=default_attention_type,
        architecture_family=manifest.family_spec,
        attention=_build_attention_spec(
            config=config,
            manifest=manifest,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            linear_layers=linear_layers,
        ),
        tensor_layout=manifest.tensor_layout,
    )
