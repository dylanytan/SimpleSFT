"""Constants used by the SimpleSFT estimator and measurement flow."""

from .models.architecture_registry import supported_dense_model_types


BYTES_PER_GB = 1024**3
DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
SUPPORTED_MODEL_TYPES = set(supported_dense_model_types())
IMAGE_TEXT_TO_TEXT_MODEL_TYPES = {
    "qwen2_5_vl",
    "qwen2_vl",
    "qwen3_vl",
    "qwen3_vl_moe",
}
VISION_LANGUAGE_MODEL_TYPES = IMAGE_TEXT_TO_TEXT_MODEL_TYPES | {"molmo"}
ZERO_DISTRIBUTED_MODES = ("zero2", "zero3")
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
    return normalized_type in normalized_supported

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
