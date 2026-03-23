"""Constants used by the SimpleSFT estimators and measurement flow."""

BYTES_PER_GB = 1024**3
DEFAULT_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")
SUPPORTED_MODEL_TYPES = {
    "gpt2",
    "gpt_neo",
    "gpt_neox",
    "llama",
    "mistral",
    "falcon",
    "qwen2",
}
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

