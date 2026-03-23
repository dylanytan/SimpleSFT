"""Unit tests for analytical memory estimation."""

from simplesft.estimate import estimate_lora_parameter_count, estimate_peak_memory
from simplesft.types import LoRAConfig, ModelLinearLayerSpec, ModelSpec, TrainingConfig


def _toy_model_spec() -> ModelSpec:
    """Return a compact model spec for unit tests."""

    return ModelSpec(
        model_name="toy",
        model_type="llama",
        num_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        intermediate_size=64,
        vocab_size=128,
        max_position_embeddings=128,
        total_params=10_000,
        trainable_linear_layers=(
            ModelLinearLayerSpec("layers.0.self_attn.q_proj", 32, 32, "attention"),
            ModelLinearLayerSpec("layers.0.self_attn.k_proj", 32, 32, "attention"),
            ModelLinearLayerSpec("layers.0.mlp.up_proj", 32, 64, "mlp"),
        ),
    )


def test_estimate_lora_parameter_count_uses_target_modules() -> None:
    """LoRA parameter estimates should include only targeted modules."""

    count = estimate_lora_parameter_count(
        model_spec=_toy_model_spec(),
        lora_config=LoRAConfig(rank=8, target_modules=("q_proj", "k_proj")),
    )
    assert count == 8 * (32 + 32) * 2


def test_estimate_peak_memory_for_zero2_shards_gradients_and_optimizer() -> None:
    """ZeRO-2 should reduce gradient and optimizer-state residency."""

    model_spec = _toy_model_spec()
    single_result = estimate_peak_memory(
        model=model_spec,
        config=TrainingConfig(tuning_mode="full_ft", distributed_mode="single_gpu"),
    )
    single_fp32_optim_result = estimate_peak_memory(
        model=model_spec,
        config=TrainingConfig(
            tuning_mode="full_ft",
            distributed_mode="single_gpu",
            optimizer_state_dtype="fp32",
        ),
    )
    zero2_result = estimate_peak_memory(
        model=model_spec,
        config=TrainingConfig(
            tuning_mode="full_ft",
            distributed_mode="zero2",
            gpus_per_node=2,
        ),
    )
    assert zero2_result.breakdown.parameter_bytes == single_result.breakdown.parameter_bytes
    assert zero2_result.breakdown.gradient_bytes < single_result.breakdown.gradient_bytes
    assert (
        zero2_result.breakdown.optimizer_state_bytes
        < single_fp32_optim_result.breakdown.optimizer_state_bytes
    )


def test_estimate_peak_memory_lora_has_smaller_trainable_state() -> None:
    """LoRA estimates should reduce trainable-state memory."""

    model_spec = _toy_model_spec()
    full_result = estimate_peak_memory(
        model=model_spec,
        config=TrainingConfig(tuning_mode="full_ft"),
    )
    lora_result = estimate_peak_memory(
        model=model_spec,
        config=TrainingConfig(
            tuning_mode="lora",
            lora=LoRAConfig(rank=4, target_modules=("q_proj", "k_proj")),
        ),
    )
    assert lora_result.breakdown.gradient_bytes < full_result.breakdown.gradient_bytes
    assert (
        lora_result.breakdown.optimizer_state_bytes
        < full_result.breakdown.optimizer_state_bytes
    )


def test_estimate_warmup_puts_optimizer_state_in_baseline() -> None:
    """Warmup-aware estimates should raise the pre-step baseline."""

    model_spec = _toy_model_spec()
    cold_result = estimate_peak_memory(
        model=model_spec,
        config=TrainingConfig(tuning_mode="full_ft", warmup_steps=0),
    )
    warm_result = estimate_peak_memory(
        model=model_spec,
        config=TrainingConfig(tuning_mode="full_ft", warmup_steps=1),
    )
    cold_baseline = next(
        record
        for record in cold_result.phase_records
        if record.phase_name == "post_init_baseline"
    )
    warm_baseline = next(
        record
        for record in warm_result.phase_records
        if record.phase_name == "post_init_baseline"
    )
    assert warm_result.metadata["optimizer_state_in_baseline"] is True
    assert cold_baseline.peak_reserved_bytes == (
        cold_result.breakdown.parameter_bytes + cold_result.breakdown.runtime_reserve_bytes
    )
    assert warm_baseline.peak_reserved_bytes == (
        warm_result.breakdown.parameter_bytes
        + warm_result.breakdown.runtime_reserve_bytes
        + warm_result.breakdown.optimizer_state_bytes
    )
