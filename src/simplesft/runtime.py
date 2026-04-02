"""Runtime helpers for working around environment-specific HF import issues."""

from __future__ import annotations

import importlib.util
from typing import Any

from .constants import model_type_uses_image_text_runtime


def prepare_transformers_runtime() -> bool:
    """Disable broken optional torchvision paths in `transformers`.

    Args:
        None.

    Returns:
        Whether the runtime was patched.

    Example:
        >>> isinstance(prepare_transformers_runtime(), bool)
        True
    """

    from transformers.utils import import_utils

    if not import_utils._torchvision_available:
        return False
    try:
        import torchvision  # noqa: F401
    except Exception:
        import_utils._torchvision_available = False
        return True
    return False


def load_auto_config(
    *,
    model_ref: str,
    trust_remote_code: bool = False,
) -> Any:
    """Load a Hugging Face config after preparing the runtime."""

    prepare_transformers_runtime()
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(
        pretrained_model_name_or_path=model_ref,
        trust_remote_code=trust_remote_code,
    )


def _model_auto_class(*, model_type: str) -> Any:
    """Return the Hugging Face auto-model class for one supported model type."""

    if model_type_uses_image_text_runtime(model_type=model_type):
        from transformers import AutoModelForImageTextToText

        return AutoModelForImageTextToText
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM


def build_empty_model(
    *,
    config: Any,
    trust_remote_code: bool = False,
) -> Any:
    """Instantiate a supported model on empty weights for safe inspection."""

    prepare_transformers_runtime()
    from accelerate import init_empty_weights

    auto_model_class = _model_auto_class(model_type=config.model_type)

    with init_empty_weights():
        model = auto_model_class.from_config(
            config=config,
            trust_remote_code=trust_remote_code,
        )
    model.tie_weights()
    return model


def build_empty_causal_lm(
    *,
    config: Any,
    trust_remote_code: bool = False,
) -> Any:
    """Instantiate a supported model on empty weights for safe inspection."""

    return build_empty_model(
        config=config,
        trust_remote_code=trust_remote_code,
    )


def load_pretrained_model(
    *,
    model_ref: str,
    model_type: str,
    torch_dtype: Any,
    attention_backend: str = "standard",
    low_cpu_mem_usage: bool = True,
    trust_remote_code: bool = False,
) -> Any:
    """Load a supported model after preparing the runtime.

    Args:
        model_ref: Hugging Face model reference.
        model_type: Resolved model type used to pick the auto-model family.
        torch_dtype: Torch dtype used for weight loading.
        attention_backend: Requested attention backend.
        low_cpu_mem_usage: Whether to minimize CPU RAM during load.
        trust_remote_code: Whether to allow remote code execution.

    Returns:
        Loaded causal language model.
    """

    prepare_transformers_runtime()
    auto_model_class = _model_auto_class(model_type=model_type)

    load_kwargs: dict[str, Any] = dict(
        pretrained_model_name_or_path=model_ref,
        dtype=torch_dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
        trust_remote_code=trust_remote_code,
    )
    attention_implementation = resolve_attention_implementation(
        attention_backend=attention_backend
    )
    if attention_implementation is not None:
        load_kwargs["attn_implementation"] = attention_implementation
    return auto_model_class.from_pretrained(**load_kwargs)


def load_pretrained_causal_lm(
    *,
    model_ref: str,
    torch_dtype: Any,
    attention_backend: str = "standard",
    low_cpu_mem_usage: bool = True,
    trust_remote_code: bool = False,
) -> Any:
    """Load a supported model after preparing the runtime."""

    config = load_auto_config(
        model_ref=model_ref,
        trust_remote_code=trust_remote_code,
    )
    return load_pretrained_model(
        model_ref=model_ref,
        model_type=config.model_type,
        torch_dtype=torch_dtype,
        attention_backend=attention_backend,
        low_cpu_mem_usage=low_cpu_mem_usage,
        trust_remote_code=trust_remote_code,
    )


def resolve_attention_implementation(*, attention_backend: str) -> str | None:
    """Resolve a user backend into a Hugging Face attention implementation.

    Args:
        attention_backend: Backend label such as `standard`, `sdpa`, or `flash2`.

    Returns:
        Hugging Face `attn_implementation` string, or `None` for the default path.
    """

    normalized_backend = attention_backend.lower()
    if normalized_backend in {"standard", "auto"}:
        return None
    if normalized_backend == "eager":
        return "eager"
    if normalized_backend == "sdpa":
        return "sdpa"
    if normalized_backend in {
        "flash",
        "flash2",
        "flashattention2",
        "flash_attention_2",
    }:
        if importlib.util.find_spec("flash_attn") is None:
            raise RuntimeError(
                "FlashAttention measurement requires the optional dependency `flash-attn`."
            )
        return "flash_attention_2"
    raise AssertionError(f"Unsupported attention backend: {attention_backend}")
