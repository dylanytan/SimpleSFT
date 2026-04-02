"""Constants used by the SimpleSFT estimators and measurement flow."""

BYTES_PER_GB = 1024**3
DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
SUPPORTED_MODEL_TYPES = {
    "flex_olmo",
    "gpt2",
    "gpt_neo",
    "gpt_neox",
    "llama",
    "mistral",
    "falcon",
    "molmo",
    "olmo",
    "olmo2",
    "olmo3",
    "olmoe",
    "qwen2",
    "qwen2_5_omni",
    "qwen2_5_vl",
    "qwen2_5_vl_text",
    "qwen2_audio",
    "qwen2_audio_encoder",
    "qwen2_moe",
    "qwen2_vl",
    "qwen2_vl_text",
    "qwen3",
    "qwen3_moe",
    "qwen3_next",
    "qwen3_omni_moe",
    "qwen3_vl",
    "qwen3_vl_moe",
    "qwen3_vl_moe_text",
    "qwen3_vl_text",
}
DEFAULT_SDPA_MODEL_TYPES = {
    model_type
    for model_type in SUPPORTED_MODEL_TYPES
    if model_type in {"falcon", "gpt_neox", "llama", "mistral"}
    or model_type.startswith(("qwen", "olmo"))
    or model_type == "molmo"
}
IMAGE_TEXT_TO_TEXT_MODEL_TYPES = {
    "qwen2_5_vl",
    "qwen2_vl",
    "qwen3_vl",
    "qwen3_vl_moe",
}
VISION_LANGUAGE_MODEL_TYPES = IMAGE_TEXT_TO_TEXT_MODEL_TYPES | {"molmo"}
ZERO_DISTRIBUTED_MODES = ("zero2", "zero3")
TRAINING_PHASES = (
    "model_load",
    "optimizer_create",
    "post_init_baseline",
    "batch_materialization",
    "forward",
    "loss_materialization",
    "backward",
    "optimizer_step",
    "zero_grad",
    "step_end",
)
PHASE_PEAK_CANDIDATES = ("forward", "backward", "optimizer_step")


def model_type_is_supported(
    *,
    model_type: str,
    supported_model_types: tuple[str, ...],
) -> bool:
    """Return whether a model type is within the supported family surface.

    Args:
        model_type: Hugging Face model type from the loaded config.
        supported_model_types: Explicit allow-list passed through config.

    Returns:
        Whether the model type should be accepted by inspection/runtime helpers.
    """

    normalized_type = model_type.lower()
    normalized_supported = {item.lower() for item in supported_model_types}
    if normalized_type in normalized_supported:
        return True
    return normalized_type.startswith(("qwen", "olmo")) or normalized_type == "molmo"


def model_type_uses_sdpa(*, model_type: str) -> bool:
    """Return whether the model family defaults to an SDPA-backed path."""

    normalized_type = model_type.lower()
    if normalized_type in DEFAULT_SDPA_MODEL_TYPES:
        return True
    return normalized_type.startswith(("qwen", "olmo")) or normalized_type == "molmo"


def model_type_uses_image_text_runtime(*, model_type: str) -> bool:
    """Return whether the runtime should use image-text auto-model classes."""

    normalized_type = model_type.lower()
    if normalized_type in IMAGE_TEXT_TO_TEXT_MODEL_TYPES:
        return True
    return normalized_type.startswith("qwen") and "_vl" in normalized_type


def model_type_supports_vision_inputs(*, model_type: str) -> bool:
    """Return whether the model family can accept image-conditioned inputs."""

    normalized_type = model_type.lower()
    return (
        normalized_type in VISION_LANGUAGE_MODEL_TYPES
        or model_type_uses_image_text_runtime(model_type=normalized_type)
    )
