"""Structural tests for the simplified analytical estimator."""

from __future__ import annotations

import math
from typing import Any, cast

from pytest import raises

from simplesft.activation_model import build_activation_terms
from simplesft.estimate import estimate_lora_parameter_count, estimate_peak_memory
from simplesft.phase_model import PhaseInputs, build_phase_records
from simplesft.optimizer_model import (
    optimizer_update_numel,
    resolved_optimizer_update_dtype,
)
from simplesft.resident_state_model import (
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
from simplesft.workspace_model import build_workspace_terms


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
            ModelLinearLayerSpec("layers.0.self_attn.q_proj", 32, 32, "attention"),
            ModelLinearLayerSpec("layers.0.self_attn.k_proj", 32, 32, "attention"),
            ModelLinearLayerSpec("layers.0.self_attn.v_proj", 32, 32, "attention"),
            ModelLinearLayerSpec("layers.0.self_attn.o_proj", 32, 32, "attention"),
            ModelLinearLayerSpec("layers.0.mlp.up_proj", 32, 64, "mlp"),
        ),
        parameter_specs=(
            ModelParameterSpec("embed.weight", (128, 32), "embedding"),
            ModelParameterSpec(
                "layers.0.self_attn.q_proj.weight", (32, 32), "attention"
            ),
            ModelParameterSpec(
                "layers.0.self_attn.k_proj.weight", (32, 32), "attention"
            ),
            ModelParameterSpec(
                "layers.0.self_attn.v_proj.weight", (32, 32), "attention"
            ),
            ModelParameterSpec(
                "layers.0.self_attn.o_proj.weight", (32, 32), "attention"
            ),
            ModelParameterSpec("layers.0.mlp.up_proj.weight", (64, 32), "mlp"),
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
            ModelLinearLayerSpec("layers.0.self_attn.q_proj", 32, 64, "attention"),
            ModelLinearLayerSpec("layers.0.self_attn.k_proj", 32, 32, "attention"),
            ModelLinearLayerSpec("layers.0.self_attn.v_proj", 32, 32, "attention"),
            ModelLinearLayerSpec("layers.0.self_attn.o_proj", 64, 32, "attention"),
            ModelLinearLayerSpec("layers.0.mlp.up_proj", 32, 64, "mlp"),
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
    )


def test_estimate_peak_memory_requires_estimator_config() -> None:
    """Estimator entrypoint should reject the measurement config type."""

    with raises(AssertionError):
        estimate_peak_memory(
            model=_toy_model_spec(),
            config=cast(EstimatorConfig, TrainingConfig(tuning_mode="full_ft")),
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


def test_checkpointed_lora_peak_stays_below_full_ft_peak() -> None:
    """Checkpointed LoRA should not exceed checkpointed full FT for one setup."""

    full_result = estimate_peak_memory(
        model=_toy_model_spec(),
        config=EstimatorConfig(
            tuning_mode="full_ft",
            distributed_mode="zero3",
            gpus_per_node=2,
            gradient_checkpointing=True,
        ),
    )
    lora_result = estimate_peak_memory(
        model=_toy_model_spec(),
        config=EstimatorConfig(
            tuning_mode="lora",
            distributed_mode="zero3",
            gpus_per_node=2,
            gradient_checkpointing=True,
            lora=LoRAConfig(rank=4, target_modules=("q_proj", "k_proj")),
        ),
    )
    assert lora_result.global_peak_bytes < full_result.global_peak_bytes


def test_workspace_zero2_optimizer_peak_uses_explicit_subphase_max() -> None:
    """ZeRO-2 optimizer workspace should be the max of named subphases."""

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
        gradient_bytes=resident_terms.debug.gradient_bytes,
        optimizer_state_bytes=resident_terms.debug.optimizer_state_bytes,
    )
    expected_peak = max(
        workspace_terms.debug.zero_fetch_window_bytes,
        workspace_terms.debug.zero_update_window_bytes,
        workspace_terms.debug.zero_comm_window_bytes,
    )
    assert workspace_terms.optimizer_workspace_bytes == expected_peak


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


def test_phase_model_uses_backward_end_state_for_optimizer_step() -> None:
    """Optimizer peak should use resident state plus backward end state."""

    phase_records = build_phase_records(
        inputs=PhaseInputs(
            parameter_bytes=10,
            gradient_bytes=20,
            optimizer_state_bytes=30,
            master_weight_bytes=40,
            runtime_floor_bytes=50,
            forward_activation_bytes=60,
            backward_activation_bytes=70,
            forward_workspace_bytes=80,
            backward_workspace_bytes=90,
            optimizer_workspace_bytes=15,
            backward_end_state_bytes=7,
        )
    )
    by_name = {record.phase_name: record for record in phase_records}
    assert by_name["optimizer_step"].peak_allocated_bytes == 172
    assert by_name["backward"].peak_allocated_bytes == 310


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
