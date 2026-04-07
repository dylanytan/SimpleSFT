"""Structural tests for the simplified analytical estimator."""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any, cast

from pytest import raises

from simplesft.estimator.activation import build_activation_terms
from simplesft.estimator.estimate import (
    _optimizer_reserved_carryover_bytes,
    estimate_lora_parameter_count,
    estimate_peak_memory,
)
from simplesft.models.precomputed_model_specs import load_precomputed_model_spec_snapshot
from simplesft.estimator.phase import PhaseInputs, build_phase_records
from simplesft.estimator.optimizer import (
    optimizer_update_numel,
    resolved_optimizer_update_dtype,
)
from simplesft.estimator.resident_state import (
    build_resident_state_terms,
    resolved_optimizer_state_dtype,
)
from simplesft.utils import bytes_for_dtype
from simplesft.types import (
    EstimatorConfig,
    LoRAConfig,
    ModelLinearLayerSpec,
    ModelParameterSpec,
    ModelSpec,
    TrainingConfig,
)
from simplesft.models.architecture_types import AttentionSpec
from simplesft.estimator.workspace import build_workspace_terms


def _precomputed_model_spec(*, model_id: str) -> ModelSpec:
    """Return one checked-in precomputed model spec for tests.

    Args:
        model_id: Model id present in the checked-in snapshot.

    Returns:
        Matching precomputed `ModelSpec`.
    """

    snapshot = load_precomputed_model_spec_snapshot()
    for model_spec in snapshot.model_specs:
        if model_spec.model_name == model_id:
            return model_spec
    raise AssertionError(f"Missing precomputed model spec for {model_id}.")


def _toy_model_spec() -> ModelSpec:
    """Return a compact dense decoder model for estimator tests."""

    return ModelSpec(
        model_name="toy",
        model_type="llama",
        num_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        intermediate_size=64,
        vocab_size=128,
        max_position_embeddings=128,
        total_params=8_224,
        trainable_linear_layers=(
            ModelLinearLayerSpec(
                "layers.0.self_attn.q_proj",
                32,
                32,
                "attention",
                role="attention_query",
                tensor_parallel_role="column_parallel",
            ),
            ModelLinearLayerSpec(
                "layers.0.self_attn.k_proj",
                32,
                32,
                "attention",
                role="attention_key",
                tensor_parallel_role="column_parallel",
            ),
            ModelLinearLayerSpec(
                "layers.0.self_attn.v_proj",
                32,
                32,
                "attention",
                role="attention_value",
                tensor_parallel_role="column_parallel",
            ),
            ModelLinearLayerSpec(
                "layers.0.self_attn.o_proj",
                32,
                32,
                "attention",
                role="attention_output",
                tensor_parallel_role="row_parallel",
            ),
            ModelLinearLayerSpec(
                "layers.0.mlp.up_proj",
                32,
                64,
                "mlp",
                role="mlp_up",
                tensor_parallel_role="column_parallel",
            ),
        ),
        parameter_specs=(
            ModelParameterSpec(
                "embed.weight",
                (128, 32),
                "embedding",
                role="embedding",
                tensor_parallel_role="vocab_parallel",
            ),
            ModelParameterSpec(
                "layers.0.self_attn.q_proj.weight",
                (32, 32),
                "attention",
                role="attention_query",
                tensor_parallel_role="column_parallel",
            ),
            ModelParameterSpec(
                "layers.0.self_attn.k_proj.weight",
                (32, 32),
                "attention",
                role="attention_key",
                tensor_parallel_role="column_parallel",
            ),
            ModelParameterSpec(
                "layers.0.self_attn.v_proj.weight",
                (32, 32),
                "attention",
                role="attention_value",
                tensor_parallel_role="column_parallel",
            ),
            ModelParameterSpec(
                "layers.0.self_attn.o_proj.weight",
                (32, 32),
                "attention",
                role="attention_output",
                tensor_parallel_role="row_parallel",
            ),
            ModelParameterSpec(
                "layers.0.mlp.up_proj.weight",
                (64, 32),
                "mlp",
                role="mlp_up",
                tensor_parallel_role="column_parallel",
            ),
        ),
    )


def _expanded_query_model_spec() -> ModelSpec:
    """Return a toy model whose query width exceeds hidden size."""

    return ModelSpec(
        model_name="toy-expanded",
        model_type="qwen3",
        num_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        intermediate_size=64,
        vocab_size=128,
        max_position_embeddings=128,
        total_params=10_272,
        trainable_linear_layers=(
            ModelLinearLayerSpec(
                "layers.0.self_attn.q_proj",
                32,
                64,
                "attention",
                role="attention_query",
                tensor_parallel_role="column_parallel",
            ),
            ModelLinearLayerSpec(
                "layers.0.self_attn.k_proj",
                32,
                32,
                "attention",
                role="attention_key",
                tensor_parallel_role="column_parallel",
            ),
            ModelLinearLayerSpec(
                "layers.0.self_attn.v_proj",
                32,
                32,
                "attention",
                role="attention_value",
                tensor_parallel_role="column_parallel",
            ),
            ModelLinearLayerSpec(
                "layers.0.self_attn.o_proj",
                64,
                32,
                "attention",
                role="attention_output",
                tensor_parallel_role="row_parallel",
            ),
            ModelLinearLayerSpec(
                "layers.0.mlp.up_proj",
                32,
                64,
                "mlp",
                role="mlp_up",
                tensor_parallel_role="column_parallel",
            ),
        ),
        parameter_specs=(
            ModelParameterSpec("embed.weight", (128, 32), "embedding"),
            ModelParameterSpec(
                "layers.0.self_attn.q_proj.weight", (64, 32), "attention"
            ),
            ModelParameterSpec(
                "layers.0.self_attn.k_proj.weight", (32, 32), "attention"
            ),
            ModelParameterSpec(
                "layers.0.self_attn.v_proj.weight", (32, 32), "attention"
            ),
            ModelParameterSpec(
                "layers.0.self_attn.o_proj.weight", (32, 64), "attention"
            ),
            ModelParameterSpec("layers.0.mlp.up_proj.weight", (64, 32), "mlp"),
        ),
        attention=AttentionSpec(
            num_query_heads=4,
            num_key_value_heads=4,
            head_dim=8,
            query_width=64,
            key_width=32,
            value_width=32,
            output_proj_input_width=64,
        ),
    )


def _sliding_window_model_spec() -> ModelSpec:
    """Return a toy model with explicit local-attention metadata."""

    return ModelSpec(
        model_name="toy-windowed",
        model_type="gpt_oss",
        num_layers=2,
        hidden_size=64,
        num_attention_heads=8,
        intermediate_size=128,
        vocab_size=128,
        max_position_embeddings=256,
        total_params=16_000,
        trainable_linear_layers=(),
        attention=AttentionSpec(
            num_query_heads=8,
            num_key_value_heads=1,
            head_dim=8,
            query_width=64,
            key_width=8,
            value_width=8,
            output_proj_input_width=64,
            uses_multi_query=True,
            sliding_window_size=32,
        ),
    )


def test_estimate_peak_memory_requires_estimator_config() -> None:
    """Estimator entrypoint should reject the measurement config type."""

    with raises(AssertionError):
        estimate_peak_memory(
            model=_toy_model_spec(),
            config=cast(EstimatorConfig, TrainingConfig(tuning_mode="full_ft")),
        )


def test_runtime_support_bytes_sum_named_components() -> None:
    """Runtime support should be the sum of named runtime support objects."""

    resident_terms = build_resident_state_terms(
        model_spec=_toy_model_spec(),
        config=EstimatorConfig(
            tuning_mode="full_ft",
            runtime_cuda_context_gb=0.25,
            runtime_allocator_pool_gb=0.10,
            runtime_nccl_gb=0.15,
            runtime_deepspeed_gb=0.20,
        ),
    )
    expected_bytes = int((0.25 + 0.10 + 0.15 + 0.20) * (1024**3))
    assert resident_terms.debug.runtime_support_bytes == expected_bytes


def test_ddp_reducer_bucket_bytes_use_configured_element_count() -> None:
    """DDP reducer buckets should come from configured element counts."""

    config = EstimatorConfig(
        tuning_mode="full_ft",
        distributed_mode="ddp",
        gpus_per_node=2,
        ddp_bucket_elements=1024,
        grad_dtype="fp32",
    )
    resident_terms = build_resident_state_terms(
        model_spec=_toy_model_spec(),
        config=config,
    )
    workspace_terms = build_workspace_terms(
        model_spec=_toy_model_spec(),
        config=config,
        parameter_bytes=resident_terms.debug.parameter_bytes,
        trainable_parameter_bytes=resident_terms.debug.trainable_parameter_bytes,
        gradient_bytes=resident_terms.debug.gradient_bytes,
        optimizer_state_bytes=resident_terms.debug.optimizer_state_bytes,
    )
    assert workspace_terms.debug.ddp_reducer_bucket_bytes == (
        1024 * bytes_for_dtype("fp32")
    )


def test_estimate_lora_parameter_count_uses_target_modules() -> None:
    """LoRA parameter estimates should include only targeted modules."""

    count = estimate_lora_parameter_count(
        model_spec=_toy_model_spec(),
        lora_config=LoRAConfig(rank=8, target_modules=("q_proj", "k_proj")),
    )
    assert count == 8 * (32 + 32) * 2


def test_lora_config_normalizes_target_modules_lists() -> None:
    """LoRA configs loaded from JSON should normalize target modules to tuples."""

    config = LoRAConfig(rank=8, target_modules=cast(Any, ["q_proj", "k_proj"]))
    assert config.target_modules == ("q_proj", "k_proj")


def test_zero2_shards_gradients_and_optimizer_state() -> None:
    """ZeRO-2 should reduce trainable resident state but keep full params."""

    model_spec = _toy_model_spec()
    single_terms = build_resident_state_terms(
        model_spec=model_spec,
        config=EstimatorConfig(tuning_mode="full_ft", distributed_mode="single_gpu"),
    )
    single_fp32_terms = build_resident_state_terms(
        model_spec=model_spec,
        config=EstimatorConfig(
            tuning_mode="full_ft",
            distributed_mode="single_gpu",
            optimizer_state_dtype="fp32",
        ),
    )
    zero2_terms = build_resident_state_terms(
        model_spec=model_spec,
        config=EstimatorConfig(
            tuning_mode="full_ft",
            distributed_mode="zero2",
            gpus_per_node=2,
        ),
    )
    assert zero2_terms.debug.parameter_bytes == single_terms.debug.parameter_bytes
    assert zero2_terms.debug.gradient_bytes < single_terms.debug.gradient_bytes
    assert (
        zero2_terms.debug.optimizer_state_bytes
        <= single_fp32_terms.debug.optimizer_state_bytes
    )


def test_zero2_untested_optimizer_state_can_stay_unsharded() -> None:
    """Untested ZeRO optimizers should honor the unsharded-state fallback."""

    model_spec = _toy_model_spec()
    unsharded_config = EstimatorConfig(
        tuning_mode="full_ft",
        distributed_mode="zero2",
        gpus_per_node=2,
        optimizer_name="adafactor",
        zero_untested_optimizer_state_is_sharded=False,
        zero_untested_optimizer_replica_tensor_count=1.0,
        zero_untested_optimizer_replica_dtype="weight_dtype",
    )
    sharded_config = replace(
        unsharded_config,
        zero_untested_optimizer_state_is_sharded=True,
    )
    unsharded_terms = build_resident_state_terms(
        model_spec=model_spec,
        config=unsharded_config,
    )
    sharded_terms = build_resident_state_terms(
        model_spec=model_spec,
        config=sharded_config,
    )
    assert (
        unsharded_terms.debug.optimizer_state_bytes
        == unsharded_terms.debug.trainable_parameter_bytes
    )
    assert (
        unsharded_terms.debug.optimizer_state_bytes
        > sharded_terms.debug.optimizer_state_bytes
    )


def test_zero3_shards_parameters_below_zero2() -> None:
    """ZeRO-3 should shard parameter residency below ZeRO-2."""

    model_spec = _toy_model_spec()
    zero2_terms = build_resident_state_terms(
        model_spec=model_spec,
        config=EstimatorConfig(
            tuning_mode="full_ft",
            distributed_mode="zero2",
            gpus_per_node=2,
        ),
    )
    zero3_terms = build_resident_state_terms(
        model_spec=model_spec,
        config=EstimatorConfig(
            tuning_mode="full_ft",
            distributed_mode="zero3",
            gpus_per_node=2,
        ),
    )
    assert zero3_terms.debug.parameter_bytes < zero2_terms.debug.parameter_bytes


def test_zero_full_ft_exposes_persistent_backend_buffers() -> None:
    """ZeRO full FT should expose explicit persistent backend support buffers."""

    zero2_terms = build_resident_state_terms(
        model_spec=_toy_model_spec(),
        config=EstimatorConfig(
            tuning_mode="full_ft",
            distributed_mode="zero2",
            gpus_per_node=2,
        ),
    )
    zero3_terms = build_resident_state_terms(
        model_spec=_toy_model_spec(),
        config=EstimatorConfig(
            tuning_mode="full_ft",
            distributed_mode="zero3",
            gpus_per_node=2,
        ),
    )
    zero3_lora_terms = build_resident_state_terms(
        model_spec=_toy_model_spec(),
        config=EstimatorConfig(
            tuning_mode="lora",
            distributed_mode="zero3",
            gpus_per_node=2,
            lora=LoRAConfig(rank=4, target_modules=("q_proj", "k_proj")),
        ),
    )
    assert zero2_terms.debug.persistent_backend_buffer_bytes == (
        zero2_terms.debug.gradient_bytes
        + math.ceil(zero2_terms.debug.parameter_bytes / 2)
    )
    assert zero3_terms.debug.persistent_backend_buffer_bytes == (
        zero3_terms.debug.trainable_parameter_bytes
    )
    assert zero3_lora_terms.debug.persistent_backend_buffer_bytes == 0


def test_tensor_parallel_shards_attention_and_mlp_resident_bytes() -> None:
    """Tensor parallelism should reduce local resident parameter bytes."""

    model_spec = _toy_model_spec()
    dense_terms = build_resident_state_terms(
        model_spec=model_spec,
        config=EstimatorConfig(
            tuning_mode="full_ft",
            distributed_mode="ddp",
            gpus_per_node=2,
        ),
    )
    tp_terms = build_resident_state_terms(
        model_spec=model_spec,
        config=EstimatorConfig(
            tuning_mode="full_ft",
            distributed_mode="ddp",
            gpus_per_node=2,
            tensor_parallel_degree=2,
        ),
    )
    assert tp_terms.debug.parameter_bytes < dense_terms.debug.parameter_bytes


def test_checkpointing_reduces_backward_retained_activation_bytes() -> None:
    """Checkpointing should shrink backward retained activation bytes."""

    model_spec = _toy_model_spec()
    eager_terms = build_activation_terms(
        model_spec=model_spec,
        config=EstimatorConfig(tuning_mode="full_ft"),
    )
    checkpoint_terms = build_activation_terms(
        model_spec=model_spec,
        config=EstimatorConfig(
            tuning_mode="full_ft",
            gradient_checkpointing=True,
        ),
    )
    assert (
        checkpoint_terms.debug.backward_phase_activation_bytes
        < eager_terms.debug.backward_phase_activation_bytes
    )


def test_saved_linear_input_bytes_deduplicate_shared_block_sources() -> None:
    """Shared Q/K/V and MLP inputs should be counted once per unique source."""

    model_spec = _toy_model_spec()
    config = EstimatorConfig(
        tuning_mode="full_ft",
        max_seq_len=4,
        micro_batch_size_per_gpu=1,
    )
    activation_terms = build_activation_terms(
        model_spec=model_spec,
        config=config,
    )
    expected_group_width = 32 + 32 + 32
    expected_bytes = (
        config.max_seq_len
        * model_spec.num_layers
        * expected_group_width
        * bytes_for_dtype(config.weight_dtype)
    )
    assert activation_terms.debug.saved_linear_input_bytes == expected_bytes


def test_sequence_parallel_reduces_checkpointed_activation_bytes_under_tp() -> None:
    """Sequence parallelism should shrink sequence-local checkpointed activations."""

    model_spec = _toy_model_spec()
    tp_terms = build_activation_terms(
        model_spec=model_spec,
        config=EstimatorConfig(
            tuning_mode="full_ft",
            distributed_mode="ddp",
            gpus_per_node=2,
            tensor_parallel_degree=2,
            gradient_checkpointing=True,
        ),
    )
    sp_terms = build_activation_terms(
        model_spec=model_spec,
        config=EstimatorConfig(
            tuning_mode="full_ft",
            distributed_mode="ddp",
            gpus_per_node=2,
            tensor_parallel_degree=2,
            sequence_parallel=True,
            gradient_checkpointing=True,
        ),
    )
    assert (
        sp_terms.debug.backward_phase_activation_bytes
        < tp_terms.debug.backward_phase_activation_bytes
    )


def test_standard_attention_saved_bytes_respect_sliding_window() -> None:
    """Sliding-window attention should reduce eager score retention."""

    dense_terms = build_activation_terms(
        model_spec=_toy_model_spec(),
        config=EstimatorConfig(tuning_mode="full_ft", attention_backend="standard"),
    )
    windowed_terms = build_activation_terms(
        model_spec=_sliding_window_model_spec(),
        config=EstimatorConfig(tuning_mode="full_ft", attention_backend="standard"),
    )
    assert windowed_terms.debug.attention_saved_bytes < (
        dense_terms.debug.attention_saved_bytes
    )


def test_expanded_query_models_add_saved_context() -> None:
    """Expanded-query attention should contribute explicit saved context."""

    base_terms = build_activation_terms(
        model_spec=_toy_model_spec(),
        config=EstimatorConfig(tuning_mode="full_ft"),
    )
    expanded_terms = build_activation_terms(
        model_spec=_expanded_query_model_spec(),
        config=EstimatorConfig(tuning_mode="full_ft"),
    )
    assert expanded_terms.debug.expanded_query_saved_bytes > 0
    assert (
        expanded_terms.debug.backward_phase_activation_bytes
        > base_terms.debug.backward_phase_activation_bytes
    )


def test_widened_context_bytes_shard_with_tensor_parallelism() -> None:
    """Expanded-path context terms should shard with tensor parallelism."""

    dense_terms = build_activation_terms(
        model_spec=_expanded_query_model_spec(),
        config=EstimatorConfig(tuning_mode="full_ft"),
    )
    tp_terms = build_activation_terms(
        model_spec=_expanded_query_model_spec(),
        config=EstimatorConfig(
            tuning_mode="full_ft",
            distributed_mode="ddp",
            gpus_per_node=2,
            tensor_parallel_degree=2,
        ),
    )
    assert tp_terms.debug.query_output_context_bytes < (
        dense_terms.debug.query_output_context_bytes
    )
    assert tp_terms.debug.expanded_query_saved_bytes == sum(
        (
            tp_terms.debug.query_output_context_bytes,
            tp_terms.debug.key_output_context_bytes,
            tp_terms.debug.value_output_context_bytes,
            tp_terms.debug.output_proj_input_context_bytes,
            tp_terms.debug.output_proj_output_context_bytes,
        )
    )


def test_lora_low_rank_bytes_scale_per_block_not_full_model_depth() -> None:
    """LoRA low-rank bytes should count first-block targets once per layer."""

    lora_config = LoRAConfig(rank=4, target_modules=("q_proj", "k_proj"))
    config = EstimatorConfig(
        tuning_mode="lora",
        max_seq_len=4,
        micro_batch_size_per_gpu=1,
        lora=lora_config,
    )
    activation_terms = build_activation_terms(
        model_spec=_toy_model_spec(),
        config=config,
    )
    expected_bytes = (
        config.max_seq_len
        * _toy_model_spec().num_layers
        * 2
        * lora_config.rank
        * bytes_for_dtype(config.adapter_parameter_dtype())
    )
    assert activation_terms.debug.lora_low_rank_bytes == expected_bytes


def test_reference_models_expose_expected_activation_terms() -> None:
    """Checked-in reference specs should expose stable retained-object terms."""

    qwen_terms = build_activation_terms(
        model_spec=_precomputed_model_spec(model_id="Qwen/Qwen3-0.6B"),
        config=EstimatorConfig(
            tuning_mode="full_ft",
            attention_backend="flash2",
            max_seq_len=2048,
        ),
    )
    olmo_terms = build_activation_terms(
        model_spec=_precomputed_model_spec(model_id="allenai/OLMo-1B-hf"),
        config=EstimatorConfig(
            tuning_mode="full_ft",
            attention_backend="sdpa",
            max_seq_len=2048,
        ),
    )
    tiny_terms = build_activation_terms(
        model_spec=_precomputed_model_spec(
            model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        ),
        config=EstimatorConfig(
            tuning_mode="full_ft",
            attention_backend="sdpa",
            max_seq_len=2048,
        ),
    )
    assert qwen_terms.debug.visible_propagation_bytes > 0
    assert qwen_terms.debug.mlp_intermediate_bytes > 0
    assert qwen_terms.debug.expanded_query_saved_bytes > 0
    assert olmo_terms.debug.visible_propagation_bytes > 0
    assert olmo_terms.debug.mlp_intermediate_bytes > 0
    assert olmo_terms.debug.expanded_query_saved_bytes == 0
    assert tiny_terms.debug.visible_propagation_bytes > 0
    assert tiny_terms.debug.mlp_intermediate_bytes > 0
    assert tiny_terms.debug.expanded_query_saved_bytes == 0


def test_qwen_lora_single_gpu_retains_hook_visible_outputs() -> None:
    """Single-GPU LoRA should retain visible outputs for widened Qwen paths."""

    activation_terms = build_activation_terms(
        model_spec=_precomputed_model_spec(model_id="Qwen/Qwen3-0.6B"),
        config=EstimatorConfig(
            tuning_mode="lora",
            attention_backend="flash2",
            max_seq_len=2048,
            lora=LoRAConfig(
                rank=16,
                target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            ),
        ),
    )
    assert activation_terms.debug.visible_propagation_bytes == (
        activation_terms.debug.hook_visible_activation_bytes
    )
    assert activation_terms.debug.expanded_query_saved_bytes == 0


def test_non_widened_lora_single_gpu_scales_visible_extras_for_full_kv_heads() -> None:
    """Non-widened single-GPU LoRA should scale extras for full-KV attention."""

    model_spec = _toy_model_spec()
    activation_terms = build_activation_terms(
        model_spec=model_spec,
        config=EstimatorConfig(
            tuning_mode="lora",
            attention_backend="sdpa",
            max_seq_len=2048,
            lora=LoRAConfig(
                rank=16,
                target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            ),
        ),
    )
    extra_visible_bytes = (
        activation_terms.debug.hook_visible_activation_bytes
        - activation_terms.debug.base_hook_visible_activation_bytes
    )
    expected_visible_bytes = activation_terms.debug.base_hook_visible_activation_bytes + (
        int(round(extra_visible_bytes * (1.0 / 3.0)))
    )
    assert activation_terms.debug.visible_propagation_bytes == expected_visible_bytes


def test_non_widened_lora_single_gpu_scales_visible_extras_for_grouped_query() -> None:
    """Grouped-query non-widened LoRA should retain half of visible extras."""

    base_model_spec = _toy_model_spec()
    grouped_query_model_spec = replace(
        base_model_spec,
        attention=replace(base_model_spec.attention, num_key_value_heads=2),
    )
    activation_terms = build_activation_terms(
        model_spec=grouped_query_model_spec,
        config=EstimatorConfig(
            tuning_mode="lora",
            attention_backend="sdpa",
            max_seq_len=2048,
            lora=LoRAConfig(
                rank=16,
                target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            ),
        ),
    )
    extra_visible_bytes = (
        activation_terms.debug.hook_visible_activation_bytes
        - activation_terms.debug.base_hook_visible_activation_bytes
    )
    expected_visible_bytes = activation_terms.debug.base_hook_visible_activation_bytes + (
        int(round(extra_visible_bytes * 0.5))
    )
    assert activation_terms.debug.visible_propagation_bytes == expected_visible_bytes


def test_checkpointing_replaces_visible_proxy_with_boundaries() -> None:
    """Checkpointing should collapse the retained-forward proxy."""

    eager_terms = build_activation_terms(
        model_spec=_precomputed_model_spec(model_id="Qwen/Qwen3-0.6B"),
        config=EstimatorConfig(
            tuning_mode="full_ft",
            attention_backend="flash2",
            max_seq_len=2048,
        ),
    )
    checkpoint_terms = build_activation_terms(
        model_spec=_precomputed_model_spec(model_id="Qwen/Qwen3-0.6B"),
        config=EstimatorConfig(
            tuning_mode="full_ft",
            attention_backend="flash2",
            max_seq_len=2048,
            gradient_checkpointing=True,
        ),
    )
    assert checkpoint_terms.debug.retained_forward_proxy_bytes == (
        checkpoint_terms.debug.checkpoint_boundary_bytes
        + checkpoint_terms.debug.attention_saved_bytes
    )
    assert checkpoint_terms.debug.retained_forward_proxy_bytes < (
        eager_terms.debug.retained_forward_proxy_bytes
    )


def test_checkpointed_single_gpu_lora_keeps_block_input_and_loss_proxy() -> None:
    """Checkpointed single-GPU LoRA should keep an explicit forward proxy."""

    activation_terms = build_activation_terms(
        model_spec=_precomputed_model_spec(model_id="Qwen/Qwen2.5-7B-Instruct"),
        config=EstimatorConfig(
            tuning_mode="lora",
            attention_backend="sdpa",
            max_seq_len=4096,
            gradient_checkpointing=True,
            lora=LoRAConfig(
                rank=16,
                target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            ),
        ),
    )
    assert activation_terms.debug.checkpoint_resident_block_input_bytes > 0
    assert activation_terms.debug.retained_forward_proxy_bytes == (
        activation_terms.debug.checkpoint_boundary_bytes
        + activation_terms.debug.loss_state_bytes
        + activation_terms.debug.lora_low_rank_bytes
        + activation_terms.debug.checkpoint_resident_block_input_bytes
        + activation_terms.debug.attention_saved_bytes
    )
    assert (
        activation_terms.debug.forward_phase_activation_bytes
        == activation_terms.debug.retained_forward_proxy_bytes
    )


def test_checkpointed_ddp_lora_uses_boundary_loss_proxy() -> None:
    """Checkpointed DDP LoRA should keep the distributed boundary-loss proxy."""

    activation_terms = build_activation_terms(
        model_spec=_precomputed_model_spec(model_id="Qwen/Qwen2.5-7B-Instruct"),
        config=EstimatorConfig(
            tuning_mode="lora",
            distributed_mode="ddp",
            gpus_per_node=2,
            attention_backend="sdpa",
            max_seq_len=4096,
            gradient_checkpointing=True,
            lora=LoRAConfig(
                rank=16,
                target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            ),
        ),
    )
    assert activation_terms.debug.retained_forward_proxy_bytes == (
        activation_terms.debug.checkpoint_boundary_bytes
        + activation_terms.debug.loss_state_bytes
        + activation_terms.debug.lora_low_rank_bytes
        + activation_terms.debug.attention_saved_bytes
    )
    assert activation_terms.debug.backward_phase_activation_bytes == (
        activation_terms.debug.retained_forward_proxy_bytes
        + activation_terms.debug.saved_linear_input_bytes
        + activation_terms.debug.residual_norm_bytes
    )


def test_checkpointed_zero3_lora_uses_boundary_loss_proxy_without_visible_slice() -> None:
    """Checkpointed ZeRO-3 LoRA should not add the legacy visible-gap term."""

    activation_terms = build_activation_terms(
        model_spec=_precomputed_model_spec(model_id="Qwen/Qwen2.5-7B-Instruct"),
        config=EstimatorConfig(
            tuning_mode="lora",
            distributed_mode="zero3",
            gpus_per_node=2,
            attention_backend="sdpa",
            max_seq_len=4096,
            gradient_checkpointing=True,
            lora=LoRAConfig(
                rank=16,
                target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            ),
        ),
    )
    assert activation_terms.debug.retained_forward_proxy_bytes == (
        activation_terms.debug.checkpoint_boundary_bytes
        + activation_terms.debug.loss_state_bytes
        + activation_terms.debug.lora_low_rank_bytes
        + activation_terms.debug.attention_saved_bytes
    )
    assert (
        activation_terms.debug.checkpointed_sharded_lora_backward_visible_bytes
        == 0
    )
    assert activation_terms.debug.backward_phase_activation_bytes == (
        activation_terms.debug.retained_forward_proxy_bytes
        + activation_terms.debug.saved_linear_input_bytes
        + activation_terms.debug.residual_norm_bytes
    )
    assert activation_terms.debug.retained_forward_proxy_bytes < (
        activation_terms.debug.visible_propagation_bytes
    )


def test_checkpointed_zero2_lora_keeps_backward_visible_fraction_disabled() -> None:
    """Checkpointed ZeRO-2 LoRA should not add the ZeRO-3 visible slice."""

    activation_terms = build_activation_terms(
        model_spec=_precomputed_model_spec(model_id="Qwen/Qwen2.5-7B-Instruct"),
        config=EstimatorConfig(
            tuning_mode="lora",
            distributed_mode="zero2",
            gpus_per_node=2,
            attention_backend="sdpa",
            max_seq_len=4096,
            gradient_checkpointing=True,
            lora=LoRAConfig(
                rank=16,
                target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            ),
        ),
    )
    assert activation_terms.debug.checkpointed_sharded_lora_backward_visible_bytes == 0


def test_non_eager_backward_includes_parameter_gradient_context_for_full_ft() -> None:
    """Non-eager backward should overlap forward retention and parameter-grad context."""

    activation_terms = build_activation_terms(
        model_spec=_precomputed_model_spec(model_id="Qwen/Qwen3-0.6B"),
        config=EstimatorConfig(
            tuning_mode="full_ft",
            attention_backend="flash2",
            max_seq_len=2048,
        ),
    )
    assert activation_terms.debug.parameter_gradient_context_bytes == (
        activation_terms.debug.saved_linear_input_bytes
        + activation_terms.debug.mlp_intermediate_bytes
    )
    assert activation_terms.debug.backward_phase_activation_bytes == (
        activation_terms.debug.retained_forward_proxy_bytes
        + activation_terms.debug.parameter_gradient_context_bytes
        + activation_terms.debug.residual_norm_bytes
        + activation_terms.debug.lora_low_rank_bytes
        + activation_terms.debug.attention_saved_bytes
    )


def test_non_eager_backward_includes_parameter_gradient_context_for_lora() -> None:
    """Non-eager LoRA backward should overlap retained outputs and adapter inputs."""

    activation_terms = build_activation_terms(
        model_spec=_precomputed_model_spec(model_id="Qwen/Qwen3-0.6B"),
        config=EstimatorConfig(
            tuning_mode="lora",
            attention_backend="flash2",
            max_seq_len=2048,
            lora=LoRAConfig(
                rank=16,
                target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            ),
        ),
    )
    assert activation_terms.debug.parameter_gradient_context_bytes == (
        activation_terms.debug.saved_linear_input_bytes
    )
    assert activation_terms.debug.backward_phase_activation_bytes > (
        activation_terms.debug.retained_forward_proxy_bytes
    )


def test_flash2_single_gpu_lora_adds_backward_logits_context() -> None:
    """Flash2 single-GPU LoRA should retain a backward-local logits context."""

    activation_terms = build_activation_terms(
        model_spec=_precomputed_model_spec(model_id="Qwen/Qwen3-0.6B"),
        config=EstimatorConfig(
            tuning_mode="lora",
            attention_backend="flash2",
            max_seq_len=8192,
            lora=LoRAConfig(
                rank=16,
                target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            ),
        ),
    )
    expected_logits_context = int(round(activation_terms.debug.loss_state_bytes * 0.5))
    assert (
        activation_terms.debug.lora_backward_logits_context_bytes
        == expected_logits_context
    )
    assert activation_terms.debug.backward_phase_activation_bytes == (
        activation_terms.debug.retained_forward_proxy_bytes
        + activation_terms.debug.parameter_gradient_context_bytes
        + activation_terms.debug.residual_norm_bytes
        + activation_terms.debug.lora_low_rank_bytes
        + activation_terms.debug.lora_backward_logits_context_bytes
        + activation_terms.debug.attention_saved_bytes
    )


def test_sdpa_large_single_gpu_lora_deduplicates_saved_input_overlap() -> None:
    """Large single-GPU LoRA SDPA should deduplicate saved-input overlap."""

    qwen14_terms = build_activation_terms(
        model_spec=_precomputed_model_spec(model_id="Qwen/Qwen2.5-14B-Instruct"),
        config=EstimatorConfig(
            tuning_mode="lora",
            attention_backend="sdpa",
            max_seq_len=3328,
            lora=LoRAConfig(
                rank=16,
                target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            ),
        ),
    )
    qwen7_terms = build_activation_terms(
        model_spec=_precomputed_model_spec(model_id="Qwen/Qwen2.5-7B-Instruct"),
        config=EstimatorConfig(
            tuning_mode="lora",
            attention_backend="sdpa",
            max_seq_len=8192,
            lora=LoRAConfig(
                rank=16,
                target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            ),
        ),
    )
    assert qwen14_terms.debug.saved_input_overlap_bytes > 0
    assert (
        qwen14_terms.debug.saved_input_overlap_bytes
        == qwen14_terms.debug.saved_linear_input_bytes
    )
    assert qwen7_terms.debug.saved_input_overlap_bytes == 0


def test_loss_state_uses_resolved_loss_output_dtype() -> None:
    """Loss-state bytes should use the explicit loss-output dtype."""

    activation_terms = build_activation_terms(
        model_spec=_toy_model_spec(),
        config=EstimatorConfig(
            tuning_mode="full_ft",
            max_seq_len=4,
            micro_batch_size_per_gpu=1,
            weight_dtype="bf16",
            loss_output_dtype="fp32",
        ),
    )
    expected_bytes = 4 * _toy_model_spec().vocab_size * bytes_for_dtype("fp32")
    assert activation_terms.debug.loss_state_bytes == expected_bytes


def test_loss_workspace_uses_resolved_loss_output_dtype() -> None:
    """Backward loss workspace should materialize one logits-sized buffer."""

    workspace_terms = build_workspace_terms(
        model_spec=_toy_model_spec(),
        config=EstimatorConfig(
            tuning_mode="full_ft",
            max_seq_len=4,
            micro_batch_size_per_gpu=1,
            weight_dtype="bf16",
            loss_output_dtype="fp32",
        ),
        parameter_bytes=0,
        trainable_parameter_bytes=0,
        gradient_bytes=0,
        optimizer_state_bytes=0,
    )
    expected_bytes = 4 * _toy_model_spec().vocab_size * bytes_for_dtype("fp32")
    assert workspace_terms.debug.loss_workspace_bytes == expected_bytes
    assert workspace_terms.backward_workspace_bytes >= expected_bytes


def test_hook_visible_outputs_are_diagnostic_only() -> None:
    """Hook-visible outputs should not directly define retained activations."""

    result = estimate_peak_memory(
        model=_toy_model_spec(),
        config=EstimatorConfig(
            tuning_mode="full_ft",
            gradient_checkpointing=True,
        ),
    )
    assert result.debug is not None
    assert result.debug.activations.hook_visible_activation_bytes > 0
    assert (
        result.debug.activations.hook_visible_activation_bytes
        != result.debug.activations.backward_phase_activation_bytes
    )


def test_zero_bucket_bytes_follow_configured_element_counts() -> None:
    """Active ZeRO bucket bytes should derive from configured capacities."""

    full_result = estimate_peak_memory(
        model=_toy_model_spec(),
        config=EstimatorConfig(
            tuning_mode="full_ft",
            distributed_mode="zero3",
            gpus_per_node=2,
            gradient_checkpointing=True,
            zero_bucket_elements=1024,
            zero_prefetch_elements=256,
        ),
    )
    lora_result = estimate_peak_memory(
        model=_toy_model_spec(),
        config=EstimatorConfig(
            tuning_mode="lora",
            distributed_mode="zero3",
            gpus_per_node=2,
            gradient_checkpointing=True,
            zero_bucket_elements=1024,
            zero_prefetch_elements=256,
            lora=LoRAConfig(rank=4, target_modules=("q_proj", "k_proj")),
        ),
    )
    assert full_result.debug is not None
    assert lora_result.debug is not None
    assert full_result.debug.workspace.zero_allgather_bucket_bytes == (
        512 * bytes_for_dtype("bf16")
    )
    assert full_result.debug.workspace.zero_reduce_bucket_bytes == (
        512 * bytes_for_dtype("bf16")
    )
    assert full_result.debug.workspace.zero_prefetch_bucket_bytes == (
        128 * bytes_for_dtype("bf16")
    )
    assert lora_result.debug.workspace.zero_allgather_bucket_bytes == (
        lora_result.debug.resident_state.trainable_parameter_bytes
    )
    assert lora_result.debug.workspace.zero_reduce_bucket_bytes == (
        lora_result.debug.resident_state.gradient_bytes
    )
    assert lora_result.debug.workspace.zero_prefetch_bucket_bytes == (
        128 * bytes_for_dtype("fp32")
    )


def test_workspace_zero2_optimizer_peak_uses_explicit_subphase_max() -> None:
    """Tested ZeRO-2 full FT should use explicit optimizer subphase windows."""

    resident_terms = build_resident_state_terms(
        model_spec=_toy_model_spec(),
        config=EstimatorConfig(
            tuning_mode="full_ft",
            distributed_mode="zero2",
            gpus_per_node=2,
        ),
    )
    workspace_terms = build_workspace_terms(
        model_spec=_toy_model_spec(),
        config=EstimatorConfig(
            tuning_mode="full_ft",
            distributed_mode="zero2",
            gpus_per_node=2,
        ),
        parameter_bytes=resident_terms.debug.parameter_bytes,
        trainable_parameter_bytes=resident_terms.debug.trainable_parameter_bytes,
        gradient_bytes=resident_terms.debug.gradient_bytes,
        optimizer_state_bytes=resident_terms.debug.optimizer_state_bytes,
    )
    parameter_partition_bytes = math.ceil(
        resident_terms.debug.trainable_parameter_bytes / 2
    )
    expected_peak = (
        (2 * resident_terms.debug.gradient_bytes)
        + workspace_terms.debug.zero_reduce_bucket_bytes
        + workspace_terms.debug.optimizer_update_workspace_bytes
        + parameter_partition_bytes
        + workspace_terms.debug.zero_allgather_bucket_bytes
        + workspace_terms.debug.zero_prefetch_bucket_bytes
    )
    assert workspace_terms.optimizer_workspace_bytes == expected_peak


def test_workspace_zero2_full_ft_backward_uses_explicit_grad_windows() -> None:
    """Tested ZeRO-2 full FT should expose explicit backward grad windows."""

    resident_terms = build_resident_state_terms(
        model_spec=_toy_model_spec(),
        config=EstimatorConfig(
            tuning_mode="full_ft",
            distributed_mode="zero2",
            gpus_per_node=2,
        ),
    )
    workspace_terms = build_workspace_terms(
        model_spec=_toy_model_spec(),
        config=EstimatorConfig(
            tuning_mode="full_ft",
            distributed_mode="zero2",
            gpus_per_node=2,
        ),
        parameter_bytes=resident_terms.debug.parameter_bytes,
        trainable_parameter_bytes=resident_terms.debug.trainable_parameter_bytes,
        gradient_bytes=resident_terms.debug.gradient_bytes,
        optimizer_state_bytes=resident_terms.debug.optimizer_state_bytes,
    )
    expected_backward = (
        (2 * resident_terms.debug.gradient_bytes)
        + workspace_terms.debug.zero_reduce_bucket_bytes
    )
    assert workspace_terms.backward_workspace_bytes == max(
        expected_backward,
        workspace_terms.debug.backward_kernel_workspace_bytes
        + workspace_terms.debug.loss_workspace_bytes
        + workspace_terms.debug.recompute_workspace_bytes,
    )


def test_workspace_zero2_checkpointed_full_ft_uses_generic_backward() -> None:
    """Checkpointed ZeRO-2 full FT should avoid the explicit grad-window stack."""

    config = EstimatorConfig(
        tuning_mode="full_ft",
        distributed_mode="zero2",
        gpus_per_node=2,
        gradient_checkpointing=True,
    )
    resident_terms = build_resident_state_terms(
        model_spec=_toy_model_spec(),
        config=config,
    )
    workspace_terms = build_workspace_terms(
        model_spec=_toy_model_spec(),
        config=config,
        parameter_bytes=resident_terms.debug.parameter_bytes,
        trainable_parameter_bytes=resident_terms.debug.trainable_parameter_bytes,
        gradient_bytes=resident_terms.debug.gradient_bytes,
        optimizer_state_bytes=resident_terms.debug.optimizer_state_bytes,
    )
    expected_backward = (
        workspace_terms.debug.backward_kernel_workspace_bytes
        + workspace_terms.debug.loss_workspace_bytes
        + workspace_terms.debug.recompute_workspace_bytes
    )
    assert workspace_terms.backward_workspace_bytes == expected_backward


def test_workspace_zero2_checkpointed_full_ft_uses_incremental_optimizer() -> None:
    """Checkpointed ZeRO-2 full FT should use the generic optimizer windows."""

    config = EstimatorConfig(
        tuning_mode="full_ft",
        distributed_mode="zero2",
        gpus_per_node=2,
        gradient_checkpointing=True,
    )
    resident_terms = build_resident_state_terms(
        model_spec=_toy_model_spec(),
        config=config,
    )
    workspace_terms = build_workspace_terms(
        model_spec=_toy_model_spec(),
        config=config,
        parameter_bytes=resident_terms.debug.parameter_bytes,
        trainable_parameter_bytes=resident_terms.debug.trainable_parameter_bytes,
        gradient_bytes=resident_terms.debug.gradient_bytes,
        optimizer_state_bytes=resident_terms.debug.optimizer_state_bytes,
    )
    expected_optimizer = max(
        workspace_terms.debug.zero_fetch_window_bytes,
        workspace_terms.debug.optimizer_update_workspace_bytes,
        min(resident_terms.debug.gradient_bytes, 512 * 1024**2),
    )
    assert workspace_terms.optimizer_workspace_bytes == expected_optimizer


def test_workspace_zero3_full_ft_backward_uses_generic_workspace() -> None:
    """ZeRO-3 full FT should use the generic backward workspace model."""

    config = EstimatorConfig(
        tuning_mode="full_ft",
        distributed_mode="zero3",
        gpus_per_node=2,
    )
    resident_terms = build_resident_state_terms(
        model_spec=_toy_model_spec(),
        config=config,
    )
    workspace_terms = build_workspace_terms(
        model_spec=_toy_model_spec(),
        config=config,
        parameter_bytes=resident_terms.debug.parameter_bytes,
        trainable_parameter_bytes=resident_terms.debug.trainable_parameter_bytes,
        gradient_bytes=resident_terms.debug.gradient_bytes,
        optimizer_state_bytes=resident_terms.debug.optimizer_state_bytes,
    )
    expected_backward = (
        workspace_terms.debug.backward_kernel_workspace_bytes
        + workspace_terms.debug.loss_workspace_bytes
        + workspace_terms.debug.recompute_workspace_bytes
    )
    assert workspace_terms.backward_workspace_bytes == expected_backward


def test_workspace_zero3_full_ft_optimizer_uses_incremental_windows() -> None:
    """ZeRO-3 full FT optimizer peak should use incremental ZeRO windows."""

    config = EstimatorConfig(
        tuning_mode="full_ft",
        distributed_mode="zero3",
        gpus_per_node=2,
    )
    resident_terms = build_resident_state_terms(
        model_spec=_toy_model_spec(),
        config=config,
    )
    workspace_terms = build_workspace_terms(
        model_spec=_toy_model_spec(),
        config=config,
        parameter_bytes=resident_terms.debug.parameter_bytes,
        trainable_parameter_bytes=resident_terms.debug.trainable_parameter_bytes,
        gradient_bytes=resident_terms.debug.gradient_bytes,
        optimizer_state_bytes=resident_terms.debug.optimizer_state_bytes,
    )
    expected_optimizer = max(
        workspace_terms.debug.zero_fetch_window_bytes,
        workspace_terms.debug.optimizer_update_workspace_bytes,
        min(resident_terms.debug.gradient_bytes, 512 * 1024**2),
    )
    assert workspace_terms.optimizer_workspace_bytes == expected_optimizer


def test_phase_model_zero_optimizer_step_does_not_restack_resident_state() -> None:
    """Optimizer-step transient should add only incremental ZeRO workspace."""

    result = estimate_peak_memory(
        model=_precomputed_model_spec(model_id="Qwen/Qwen2.5-7B-Instruct"),
        config=EstimatorConfig(
            tuning_mode="full_ft",
            distributed_mode="zero2",
            gpus_per_node=2,
            attention_backend="sdpa",
            max_seq_len=2048,
            gradient_checkpointing=True,
        ),
    )
    assert result.debug is not None
    resident_debug = result.debug.resident_state
    resident_floor_bytes = (
        resident_debug.parameter_bytes
        + resident_debug.gradient_bytes
        + resident_debug.optimizer_state_bytes
        + resident_debug.master_weight_bytes
        + resident_debug.runtime_support_bytes
        + resident_debug.persistent_backend_buffer_bytes
    )
    optimizer_peak_bytes = result.debug.phase_peaks.optimizer_peak_bytes
    assert optimizer_peak_bytes - resident_floor_bytes == result.breakdown.transient_bytes
    derived_optimizer_workspace_bytes = (
        result.breakdown.transient_bytes
        - result.debug.phase_peaks.backward_end_state_bytes
        - result.debug.activations.backward_phase_activation_bytes
    )
    assert derived_optimizer_workspace_bytes < result.debug.workspace.zero_update_window_bytes


def test_zero_full_ft_breakdown_includes_persistent_backend_buffers() -> None:
    """Public runtime reserve should include explicit backend support buffers."""

    result = estimate_peak_memory(
        model=_toy_model_spec(),
        config=EstimatorConfig(
            tuning_mode="full_ft",
            distributed_mode="zero2",
            gpus_per_node=2,
        ),
    )
    assert result.debug is not None
    assert result.breakdown.runtime_reserve_bytes == (
        result.debug.resident_state.runtime_support_bytes
        + result.debug.resident_state.persistent_backend_buffer_bytes
        + result.debug.resident_state.master_weight_bytes
    )


def test_tp_sharding_uses_explicit_tensor_roles() -> None:
    """TP sharding should follow explicit tensor roles instead of suffixes."""

    model_spec = ModelSpec(
        model_name="role-only",
        model_type="llama",
        num_layers=1,
        hidden_size=64,
        num_attention_heads=8,
        intermediate_size=128,
        vocab_size=256,
        max_position_embeddings=128,
        total_params=10_000,
        trainable_linear_layers=(),
        parameter_specs=(
            ModelParameterSpec(
                parameter_name="tensor.alpha",
                shape=(64, 64),
                category="attention",
                role="attention_query",
                tensor_parallel_role="column_parallel",
            ),
            ModelParameterSpec(
                parameter_name="tensor.beta",
                shape=(64, 64),
                category="attention",
                role="attention_output",
                tensor_parallel_role="row_parallel",
            ),
            ModelParameterSpec(
                parameter_name="tensor.gamma",
                shape=(256, 64),
                category="embedding",
                role="embedding",
                tensor_parallel_role="vocab_parallel",
            ),
        ),
    )
    tp_terms = build_resident_state_terms(
        model_spec=model_spec,
        config=EstimatorConfig(
            tuning_mode="full_ft",
            distributed_mode="ddp",
            gpus_per_node=2,
            tensor_parallel_degree=2,
        ),
    )
    dense_terms = build_resident_state_terms(
        model_spec=model_spec,
        config=EstimatorConfig(
            tuning_mode="full_ft",
            distributed_mode="ddp",
            gpus_per_node=2,
        ),
    )
    assert tp_terms.debug.parameter_bytes < dense_terms.debug.parameter_bytes


def test_zero_optimizer_update_workspace_is_sharded_by_world_size() -> None:
    """ZeRO optimizer scratch should divide by world size."""

    model_spec = _toy_model_spec()
    single_terms = build_resident_state_terms(
        model_spec=model_spec,
        config=EstimatorConfig(tuning_mode="full_ft"),
    )
    zero_terms = build_resident_state_terms(
        model_spec=model_spec,
        config=EstimatorConfig(
            tuning_mode="full_ft",
            distributed_mode="zero3",
            gpus_per_node=2,
        ),
    )
    single_workspace = build_workspace_terms(
        model_spec=model_spec,
        config=EstimatorConfig(tuning_mode="full_ft"),
        parameter_bytes=single_terms.debug.parameter_bytes,
        trainable_parameter_bytes=single_terms.debug.trainable_parameter_bytes,
        gradient_bytes=single_terms.debug.gradient_bytes,
        optimizer_state_bytes=single_terms.debug.optimizer_state_bytes,
    )
    zero_workspace = build_workspace_terms(
        model_spec=model_spec,
        config=EstimatorConfig(
            tuning_mode="full_ft",
            distributed_mode="zero3",
            gpus_per_node=2,
        ),
        parameter_bytes=zero_terms.debug.parameter_bytes,
        trainable_parameter_bytes=zero_terms.debug.trainable_parameter_bytes,
        gradient_bytes=zero_terms.debug.gradient_bytes,
        optimizer_state_bytes=zero_terms.debug.optimizer_state_bytes,
    )
    expected_numel = math.ceil(
        optimizer_update_numel(
            model_spec=model_spec,
            config=EstimatorConfig(
                tuning_mode="full_ft",
                distributed_mode="zero3",
                gpus_per_node=2,
            ),
        )
        / 2
    )
    assert zero_workspace.debug.optimizer_update_workspace_bytes == (
        expected_numel
        * bytes_for_dtype(
            resolved_optimizer_update_dtype(
                config=EstimatorConfig(
                    tuning_mode="full_ft",
                    distributed_mode="zero3",
                    gpus_per_node=2,
                )
            )
        )
    )


def test_zero_untested_optimizer_update_workspace_can_stay_unsharded() -> None:
    """Untested ZeRO optimizers should honor the unsharded-update fallback."""

    model_spec = _toy_model_spec()
    unsharded_config = EstimatorConfig(
        tuning_mode="full_ft",
        distributed_mode="zero2",
        gpus_per_node=2,
        optimizer_name="sgd",
        zero_untested_optimizer_update_is_sharded=False,
        zero_untested_optimizer_update_replica_tensor_count=1.0,
        zero_untested_optimizer_update_dtype="weight_dtype",
    )
    sharded_config = replace(
        unsharded_config,
        zero_untested_optimizer_update_is_sharded=True,
    )
    unsharded_resident_terms = build_resident_state_terms(
        model_spec=model_spec,
        config=unsharded_config,
    )
    sharded_resident_terms = build_resident_state_terms(
        model_spec=model_spec,
        config=sharded_config,
    )
    unsharded_workspace_terms = build_workspace_terms(
        model_spec=model_spec,
        config=unsharded_config,
        parameter_bytes=unsharded_resident_terms.debug.parameter_bytes,
        trainable_parameter_bytes=unsharded_resident_terms.debug.trainable_parameter_bytes,
        gradient_bytes=unsharded_resident_terms.debug.gradient_bytes,
        optimizer_state_bytes=unsharded_resident_terms.debug.optimizer_state_bytes,
    )
    sharded_workspace_terms = build_workspace_terms(
        model_spec=model_spec,
        config=sharded_config,
        parameter_bytes=sharded_resident_terms.debug.parameter_bytes,
        trainable_parameter_bytes=sharded_resident_terms.debug.trainable_parameter_bytes,
        gradient_bytes=sharded_resident_terms.debug.gradient_bytes,
        optimizer_state_bytes=sharded_resident_terms.debug.optimizer_state_bytes,
    )
    assert (
        unsharded_workspace_terms.debug.optimizer_update_workspace_bytes
        == unsharded_resident_terms.debug.trainable_parameter_bytes
    )
    assert (
        unsharded_workspace_terms.debug.optimizer_update_workspace_bytes
        > sharded_workspace_terms.debug.optimizer_update_workspace_bytes
    )


def test_zero2_untested_accumulator_update_numel_uses_two_copies() -> None:
    """Accumulator optimizers should materialize two update copies on untested ZeRO."""

    model_spec = _toy_model_spec()
    config = EstimatorConfig(
        tuning_mode="full_ft",
        distributed_mode="zero2",
        gpus_per_node=2,
        optimizer_name="adagrad",
    )
    adamw_numel = optimizer_update_numel(
        model_spec=model_spec,
        config=replace(config, optimizer_name="adamw"),
    )
    adagrad_numel = optimizer_update_numel(
        model_spec=model_spec,
        config=config,
    )
    assert adagrad_numel == 2 * adamw_numel


def test_zero2_untested_full_ft_carries_backward_activation_into_step() -> None:
    """Untested ZeRO-2 full FT should retain backward activation into optimizer step."""

    config = EstimatorConfig(
        tuning_mode="full_ft",
        distributed_mode="zero2",
        gpus_per_node=2,
        optimizer_name="sgd",
        attention_backend="sdpa",
    )
    result = estimate_peak_memory(
        model=_toy_model_spec(),
        config=config,
    )
    assert result.debug is not None
    resident_debug = result.debug.resident_state
    resident_floor_bytes = (
        resident_debug.parameter_bytes
        + resident_debug.gradient_bytes
        + resident_debug.optimizer_state_bytes
        + resident_debug.master_weight_bytes
        + resident_debug.runtime_support_bytes
        + resident_debug.persistent_backend_buffer_bytes
    )
    optimizer_peak_term = (
        result.debug.phase_peaks.optimizer_peak_bytes
        - resident_floor_bytes
        - result.debug.phase_peaks.backward_end_state_bytes
    )
    optimizer_workspace_bytes = max(
        result.debug.workspace.zero_fetch_window_bytes,
        result.debug.workspace.optimizer_update_workspace_bytes,
        result.debug.workspace.zero_reduce_bucket_bytes,
    )
    expected_term = (
        optimizer_workspace_bytes
        + result.debug.activations.backward_phase_activation_bytes
    )
    assert optimizer_peak_term == expected_term


def test_zero2_tested_full_ft_carries_retained_forward_without_loss() -> None:
    """Tested ZeRO-2 full FT should carry retained-forward state into step."""

    model_spec = _precomputed_model_spec(model_id="Qwen/Qwen3-0.6B")
    config = EstimatorConfig(
        tuning_mode="full_ft",
        distributed_mode="zero2",
        gpus_per_node=2,
        optimizer_name="adamw",
        attention_backend="flash2",
        max_seq_len=2048,
    )
    resident_terms = build_resident_state_terms(
        model_spec=model_spec,
        config=config,
    )
    activation_terms = build_activation_terms(
        model_spec=model_spec,
        config=config,
    )
    workspace_terms = build_workspace_terms(
        model_spec=model_spec,
        config=config,
        parameter_bytes=resident_terms.debug.parameter_bytes,
        trainable_parameter_bytes=resident_terms.debug.trainable_parameter_bytes,
        gradient_bytes=resident_terms.debug.gradient_bytes,
        optimizer_state_bytes=resident_terms.debug.optimizer_state_bytes,
    )
    carryover_bytes = _optimizer_reserved_carryover_bytes(
        config=config,
        activation_terms=activation_terms,
        workspace_terms=workspace_terms,
    )
    retained_forward_without_loss_bytes = max(
        0,
        activation_terms.debug.retained_forward_proxy_bytes
        - activation_terms.debug.loss_state_bytes,
    )
    expected_carryover_bytes = max(
        workspace_terms.backward_workspace_bytes,
        retained_forward_without_loss_bytes,
    )
    assert carryover_bytes == expected_carryover_bytes


def test_checkpointed_distributed_lora_carries_soft_reserved_stack() -> None:
    """Checkpointed distributed LoRA should model the long-context reserve stack."""

    model_spec = _precomputed_model_spec(model_id="Qwen/Qwen2.5-7B-Instruct")
    config = EstimatorConfig(
        tuning_mode="lora",
        distributed_mode="zero3",
        gpus_per_node=2,
        optimizer_name="adamw",
        attention_backend="flash2",
        max_seq_len=4096,
        gradient_checkpointing=True,
        lora=LoRAConfig(
            rank=16,
            target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        ),
    )
    resident_terms = build_resident_state_terms(
        model_spec=model_spec,
        config=config,
    )
    activation_terms = build_activation_terms(
        model_spec=model_spec,
        config=config,
    )
    workspace_terms = build_workspace_terms(
        model_spec=model_spec,
        config=config,
        parameter_bytes=resident_terms.debug.parameter_bytes,
        trainable_parameter_bytes=resident_terms.debug.trainable_parameter_bytes,
        gradient_bytes=resident_terms.debug.gradient_bytes,
        optimizer_state_bytes=resident_terms.debug.optimizer_state_bytes,
    )
    carryover_bytes = _optimizer_reserved_carryover_bytes(
        config=config,
        activation_terms=activation_terms,
        workspace_terms=workspace_terms,
    )
    assert carryover_bytes == (
        activation_terms.debug.forward_phase_activation_bytes
        + workspace_terms.debug.loss_workspace_bytes
        + workspace_terms.backward_workspace_bytes
        + activation_terms.debug.checkpoint_boundary_bytes
        + activation_terms.debug.saved_linear_input_bytes
        + activation_terms.debug.residual_norm_bytes
    )


def test_attention_workspace_is_per_layer_not_full_depth() -> None:
    """Attention workspace should not scale with the total layer count."""

    shallow_model = _toy_model_spec()
    deep_model = replace(shallow_model, num_layers=8)
    shallow_workspace = build_workspace_terms(
        model_spec=shallow_model,
        config=EstimatorConfig(tuning_mode="full_ft", attention_backend="standard"),
        parameter_bytes=1,
        trainable_parameter_bytes=1,
        gradient_bytes=1,
        optimizer_state_bytes=1,
    )
    deep_workspace = build_workspace_terms(
        model_spec=deep_model,
        config=EstimatorConfig(tuning_mode="full_ft", attention_backend="standard"),
        parameter_bytes=1,
        trainable_parameter_bytes=1,
        gradient_bytes=1,
        optimizer_state_bytes=1,
    )
    assert (
        deep_workspace.debug.attention_forward_workspace_bytes
        == shallow_workspace.debug.attention_forward_workspace_bytes
    )
    assert (
        deep_workspace.debug.backward_kernel_workspace_bytes
        == shallow_workspace.debug.backward_kernel_workspace_bytes
    )


def test_non_eager_attention_workspace_ignores_tile_constants() -> None:
    """SDPA and flash2 should share the width-based attention workspace path."""

    model_spec = _toy_model_spec()
    sdpa_workspace = build_workspace_terms(
        model_spec=model_spec,
        config=EstimatorConfig(
            tuning_mode="full_ft",
            attention_backend="sdpa",
            max_seq_len=32,
            micro_batch_size_per_gpu=2,
        ),
        parameter_bytes=0,
        trainable_parameter_bytes=0,
        gradient_bytes=0,
        optimizer_state_bytes=0,
    )
    flash2_workspace = build_workspace_terms(
        model_spec=model_spec,
        config=EstimatorConfig(
            tuning_mode="full_ft",
            attention_backend="flash2",
            max_seq_len=32,
            micro_batch_size_per_gpu=2,
        ),
        parameter_bytes=0,
        trainable_parameter_bytes=0,
        gradient_bytes=0,
        optimizer_state_bytes=0,
    )
    standard_workspace = build_workspace_terms(
        model_spec=model_spec,
        config=EstimatorConfig(
            tuning_mode="full_ft",
            attention_backend="standard",
            max_seq_len=32,
            micro_batch_size_per_gpu=2,
        ),
        parameter_bytes=0,
        trainable_parameter_bytes=0,
        gradient_bytes=0,
        optimizer_state_bytes=0,
    )
    assert (
        sdpa_workspace.debug.attention_forward_workspace_bytes
        == flash2_workspace.debug.attention_forward_workspace_bytes
    )
    assert (
        standard_workspace.debug.attention_forward_workspace_bytes
        != sdpa_workspace.debug.attention_forward_workspace_bytes
    )


def test_zero2_lora_optimizer_fetch_window_uses_trainable_bytes() -> None:
    """LoRA ZeRO-2 optimizer fetch should scale with adapter bytes, not full params."""

    config = EstimatorConfig(
        tuning_mode="lora",
        distributed_mode="zero2",
        gpus_per_node=2,
        lora=LoRAConfig(rank=4, target_modules=("q_proj", "k_proj")),
    )
    resident_terms = build_resident_state_terms(
        model_spec=_toy_model_spec(),
        config=config,
    )
    workspace_terms = build_workspace_terms(
        model_spec=_toy_model_spec(),
        config=config,
        parameter_bytes=resident_terms.debug.parameter_bytes,
        trainable_parameter_bytes=resident_terms.debug.trainable_parameter_bytes,
        gradient_bytes=resident_terms.debug.gradient_bytes,
        optimizer_state_bytes=resident_terms.debug.optimizer_state_bytes,
    )
    expected_fetch = (
        resident_terms.debug.trainable_parameter_bytes
        + workspace_terms.debug.zero_allgather_bucket_bytes
        + workspace_terms.debug.zero_prefetch_bucket_bytes
    )
    assert workspace_terms.debug.zero_fetch_window_bytes == expected_fetch
    assert workspace_terms.debug.zero_fetch_window_bytes != (
        resident_terms.debug.parameter_bytes
        + workspace_terms.debug.zero_allgather_bucket_bytes
        + workspace_terms.debug.zero_prefetch_bucket_bytes
    )


def test_phase_model_uses_backward_end_state_for_optimizer_step() -> None:
    """Optimizer peak should use resident state plus backward end state."""

    phase_records = build_phase_records(
        inputs=PhaseInputs(
            parameter_bytes=10,
            gradient_bytes=20,
            optimizer_state_bytes=30,
            master_weight_bytes=40,
            runtime_support_bytes=50,
            persistent_backend_buffer_bytes=0,
            forward_activation_bytes=60,
            backward_activation_bytes=70,
            forward_workspace_bytes=80,
            backward_workspace_bytes=90,
            optimizer_workspace_bytes=15,
            backward_end_state_bytes=7,
            optimizer_reserved_carryover_bytes=0,
        )
    )
    by_name = {record.phase_name: record for record in phase_records}
    assert by_name["optimizer_step"].peak_allocated_bytes == 172
    assert by_name["backward"].peak_allocated_bytes == 310


def test_checkpointed_single_gpu_lora_peak_moves_to_optimizer_step() -> None:
    """Checkpointed single-GPU LoRA should carry reserve into optimizer step."""

    result = estimate_peak_memory(
        model=_precomputed_model_spec(model_id="Qwen/Qwen2.5-7B-Instruct"),
        config=EstimatorConfig(
            tuning_mode="lora",
            distributed_mode="single_gpu",
            attention_backend="sdpa",
            max_seq_len=4096,
            gradient_checkpointing=True,
            lora=LoRAConfig(
                rank=16,
                target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
            ),
        ),
    )
    assert result.debug is not None
    assert result.peak_phase == "optimizer_step"
    assert result.breakdown.transient_bytes > (
        result.debug.workspace.optimizer_update_workspace_bytes
    )


def test_optimizer_step_breakdown_zeroes_activation_bytes() -> None:
    """Optimizer-step peaks should not report retained activations."""

    result = estimate_peak_memory(
        model=_toy_model_spec(),
        config=EstimatorConfig(
            tuning_mode="full_ft",
            distributed_mode="zero3",
            gpus_per_node=2,
            gradient_checkpointing=True,
        ),
    )
    assert result.debug is not None
    if result.peak_phase != "optimizer_step":
        return
    assert result.breakdown.activation_bytes == 0


def test_estimate_result_exposes_typed_debug_metadata() -> None:
    """Estimate results should expose grouped debug internals and flat metadata."""

    result = estimate_peak_memory(
        model=_toy_model_spec(),
        config=EstimatorConfig(tuning_mode="full_ft"),
    )
    assert result.debug is not None
    metadata = result.comparable_metadata()
    assert metadata["saved_linear_input_bytes"] > 0
    assert metadata["attention_forward_workspace_bytes"] > 0
    assert metadata["allocated_global_peak_bytes"] == result.global_peak_bytes


def test_zero_sharded_optimizer_state_uses_fp32_by_default() -> None:
    """ZeRO modes should resolve optimizer state to fp32 by default."""

    resolved_dtype = resolved_optimizer_state_dtype(
        config=EstimatorConfig(
            tuning_mode="full_ft",
            distributed_mode="zero2",
            gpus_per_node=2,
        )
    )
    assert resolved_dtype == "fp32"
