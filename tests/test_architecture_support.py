"""Tests for explicit architecture manifests and inspection metadata."""

from __future__ import annotations

from simplesft.models.architecture_registry import (
    manifest_for_model_type,
    supported_dense_model_types,
)
from simplesft.models.inspect import inspect_model


def test_supported_dense_manifest_surface_matches_expected() -> None:
    """Official dense support should be driven by explicit registered manifests."""

    expected_model_types = {
        "bloom",
        "gpt2",
        "gpt_neox",
        "gpt_oss",
        "llama",
        "mistral",
        "olmo",
        "olmo2",
        "olmo3",
        "opt",
        "phi",
        "phi3",
        "qwen2",
        "qwen3",
        "qwen3_next",
        "smollm3",
    }
    assert set(supported_dense_model_types()) == expected_model_types
    assert manifest_for_model_type(model_type="qwen3_moe") is None
    assert manifest_for_model_type(model_type="deepseek_v3") is None


def test_inspect_model_extracts_qwen2_attention_metadata() -> None:
    """Public Qwen2 inspection should populate explicit GQA metadata."""

    spec = inspect_model("Qwen/Qwen2.5-0.5B-Instruct")
    assert spec.architecture_family.family_label == "qwen2_dense"
    assert spec.attention.num_query_heads == 14
    assert spec.attention.num_key_value_heads == 2
    assert spec.attention.uses_grouped_query is True


def test_inspect_model_extracts_phi3_fused_tensor_roles() -> None:
    """Phi-3 inspection should expose fused QKV and fused MLP metadata."""

    spec = inspect_model("microsoft/Phi-3-mini-4k-instruct")
    role_names = {layer.role for layer in spec.trainable_linear_layers}
    assert spec.architecture_family.family_label == "phi3_dense"
    assert "attention_qkv" in role_names
    assert "mlp_gate_up" in role_names
    assert spec.attention.query_width > 0


def test_inspect_model_supports_public_opt_repo() -> None:
    """OPT should be officially supported through the manifest path."""

    spec = inspect_model("facebook/opt-125m")
    assert spec.architecture_family.family_label == "opt_dense"
    assert spec.model_type == "opt"


def test_inspect_model_supports_public_olmo3_repo() -> None:
    """OLMo 3 should populate explicit sliding-window attention metadata."""

    spec = inspect_model("allenai/Olmo-3-1025-7B")
    assert spec.architecture_family.family_label == "olmo3_dense"
    assert spec.attention.sliding_window_size == 4096
    assert spec.attention.rope_scaling_present is True


def test_inspect_model_supports_public_olmo2_repo() -> None:
    """OLMo 2 should be officially supported through the manifest path."""

    spec = inspect_model("allenai/OLMo-2-1124-7B")
    assert spec.architecture_family.family_label == "olmo2_dense"
    assert spec.model_type == "olmo2"


def test_gpt_oss_synthetic_linear_rules_ignore_bias_tensors() -> None:
    """GPT-OSS synthetic expert-linears should match weights, not bias tensors."""

    manifest = manifest_for_model_type(model_type="gpt_oss")
    assert manifest is not None
    synthetic_rule = manifest.synthetic_linear_rules[0]
    assert synthetic_rule.matches(
        parameter_name="model.layers.0.mlp.experts.gate_up_proj.weight"
    )
    assert not synthetic_rule.matches(
        parameter_name="model.layers.0.mlp.experts.gate_up_proj_bias"
    )
