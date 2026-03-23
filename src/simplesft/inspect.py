"""Model inspection helpers for supported Hugging Face decoder-only models."""

from __future__ import annotations

from typing import Any, Iterable, cast

import torch

from .constants import SUPPORTED_MODEL_TYPES
from .runtime import build_empty_causal_lm, load_auto_config
from .types import ModelLinearLayerSpec, ModelSpec


def _classify_module(module_name: str) -> str:
    """Return a coarse module category for a linear layer."""

    lower_name = module_name.lower()
    if "attn" in lower_name and any(
        token in lower_name for token in ("c_proj", "o_proj", "c_attn")
    ):
        return "attention"
    if "mlp" in lower_name and any(
        token in lower_name for token in ("c_proj", "c_fc", "up_proj", "down_proj")
    ):
        return "mlp"
    if any(
        token in lower_name
        for token in ("q_proj", "k_proj", "v_proj", "o_proj", "c_attn")
    ):
        return "attention"
    if any(
        token in lower_name
        for token in ("gate_proj", "up_proj", "down_proj", "fc", "c_fc")
    ):
        return "mlp"
    return "other"


def _build_linear_like_spec(
    *,
    module_name: str,
    module: torch.nn.Module,
) -> ModelLinearLayerSpec | None:
    """Build a layer summary for linear or linear-like modules."""

    if isinstance(module, torch.nn.Linear):
        return ModelLinearLayerSpec(
            module_name=module_name,
            input_dim=module.in_features,
            output_dim=module.out_features,
            category=_classify_module(module_name=module_name),
        )
    if module.__class__.__name__ == "Conv1D" and hasattr(module, "weight"):
        weight_shape = cast(torch.Tensor, module.weight).shape
        input_dim = int(weight_shape[0])
        output_dim = int(weight_shape[1])
        return ModelLinearLayerSpec(
            module_name=module_name,
            input_dim=input_dim,
            output_dim=output_dim,
            category=_classify_module(module_name=module_name),
        )
    return None


def _iter_linear_layers(model: torch.nn.Module) -> Iterable[ModelLinearLayerSpec]:
    """Yield linear-layer summaries from a torch model."""

    for module_name, module in model.named_modules():
        layer_spec = _build_linear_like_spec(module_name=module_name, module=module)
        if layer_spec is not None:
            yield layer_spec


def _get_config_value(config: Any, *names: str) -> int:
    """Return the first available integer config attribute."""

    for name in names:
        value = getattr(config, name, None)
        if value is not None:
            return int(value)
    raise AssertionError(f"Config is missing required attributes: {names}")


def _get_intermediate_size(config: Any) -> int:
    """Return the MLP intermediate size with architecture-aware fallbacks."""

    value = getattr(config, "intermediate_size", None)
    if value is not None:
        return int(value)
    value = getattr(config, "n_inner", None)
    if value is not None:
        return int(value)
    hidden_size = _get_config_value(config, "hidden_size", "n_embd")
    return 4 * hidden_size


def inspect_model(
    model_ref: str,
    *,
    trust_remote_code: bool = False,
) -> ModelSpec:
    """Inspect a Hugging Face causal LM and return a compact model summary.

    Args:
        model_ref: Hugging Face model id or local model path.
        trust_remote_code: Whether to allow custom remote model code.

    Returns:
        ModelSpec describing the model structure.

    Example:
        >>> spec = inspect_model("sshleifer/tiny-gpt2")
        >>> spec.model_type
        'gpt2'
    """

    config = load_auto_config(
        model_ref=model_ref,
        trust_remote_code=trust_remote_code,
    )
    assert (
        config.model_type in SUPPORTED_MODEL_TYPES
    ), f"Unsupported model_type `{config.model_type}`. Only dense decoder-only models are supported."
    model = build_empty_causal_lm(
        config=config,
        trust_remote_code=trust_remote_code,
    )
    total_params = sum(parameter.numel() for parameter in model.parameters())
    linear_layers = tuple(_iter_linear_layers(model=model))
    return ModelSpec(
        model_name=model_ref,
        model_type=config.model_type,
        num_layers=_get_config_value(config, "num_hidden_layers", "n_layer"),
        hidden_size=_get_config_value(config, "hidden_size", "n_embd"),
        num_attention_heads=_get_config_value(
            config,
            "num_attention_heads",
            "n_head",
        ),
        intermediate_size=_get_intermediate_size(config),
        vocab_size=_get_config_value(config, "vocab_size"),
        max_position_embeddings=getattr(config, "max_position_embeddings", 0),
        total_params=total_params,
        trainable_linear_layers=linear_layers,
        attention_type="causal",
    )
