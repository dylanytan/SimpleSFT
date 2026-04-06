"""Explicit dense-architecture manifests for model inspection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .architecture_types import (
    ArchitectureFamilySpec,
    LINEAR_ROLE_ATTENTION_KEY,
    LINEAR_ROLE_ATTENTION_OUTPUT,
    LINEAR_ROLE_ATTENTION_QKV,
    LINEAR_ROLE_ATTENTION_QUERY,
    LINEAR_ROLE_ATTENTION_VALUE,
    LINEAR_ROLE_LM_HEAD,
    LINEAR_ROLE_MLP_DOWN,
    LINEAR_ROLE_MLP_GATE,
    LINEAR_ROLE_MLP_GATE_UP,
    LINEAR_ROLE_MLP_UP,
    LINEAR_ROLE_OTHER,
    LINEAR_ROLE_ROUTER,
    PARAMETER_ROLE_BIAS,
    PARAMETER_ROLE_EMBEDDING,
    PARAMETER_ROLE_LM_HEAD,
    PARAMETER_ROLE_NORM,
    PARAMETER_ROLE_OTHER,
    PARAMETER_ROLE_POSITION_EMBEDDING,
    TensorLayoutSpec,
)


@dataclass(frozen=True)
class ClassifiedTensor:
    """Explicit classification for one inspected tensor-like object."""

    category: str
    role: str
    tensor_parallel_role: str


@dataclass(frozen=True)
class RoleRule:
    """String-matching rule used to classify modules and parameters."""

    category: str
    role: str
    match_fragments: tuple[str, ...]
    requires_substrings: tuple[str, ...] = ()
    forbids_substrings: tuple[str, ...] = ()

    def matches(self, *, qualified_name: str) -> bool:
        """Return whether the rule matches one qualified name."""

        lower_name = qualified_name.lower()
        if self.match_fragments and not any(
            fragment in lower_name for fragment in self.match_fragments
        ):
            return False
        if self.requires_substrings and not all(
            fragment in lower_name for fragment in self.requires_substrings
        ):
            return False
        if any(fragment in lower_name for fragment in self.forbids_substrings):
            return False
        return True


@dataclass(frozen=True)
class SyntheticLinearRule:
    """Parameter-based pseudo-linear rule for non-`Linear` expert tensors."""

    category: str
    role: str
    match_fragments: tuple[str, ...]
    input_axis: int
    output_axis: int
    required_suffixes: tuple[str, ...] = ()

    def matches(self, *, parameter_name: str) -> bool:
        """Return whether one parameter should be surfaced as a linear layer."""

        lower_name = parameter_name.lower()
        if not any(fragment in lower_name for fragment in self.match_fragments):
            return False
        if not self.required_suffixes:
            return True
        return lower_name.endswith(self.required_suffixes)


@dataclass(frozen=True)
class ArchitectureManifest:
    """Explicit extraction and tensor-layout rules for one model family."""

    model_type: str
    family_spec: ArchitectureFamilySpec
    tensor_layout: TensorLayoutSpec
    num_layers_fields: tuple[str, ...]
    hidden_size_fields: tuple[str, ...]
    num_attention_heads_fields: tuple[str, ...]
    intermediate_size_fields: tuple[str, ...]
    max_position_fields: tuple[str, ...]
    num_key_value_heads_fields: tuple[str, ...] = ()
    head_dim_fields: tuple[str, ...] = ()
    sliding_window_fields: tuple[str, ...] = ()
    linear_rules: tuple[RoleRule, ...] = ()
    parameter_rules: tuple[RoleRule, ...] = ()
    synthetic_linear_rules: tuple[SyntheticLinearRule, ...] = ()

    def classify_linear(self, *, module_name: str) -> ClassifiedTensor:
        """Return explicit linear classification for one module name."""

        for rule in self.linear_rules:
            if rule.matches(qualified_name=module_name):
                return ClassifiedTensor(
                    category=rule.category,
                    role=rule.role,
                    tensor_parallel_role=self.tensor_layout.linear_tensor_parallel_role(
                        linear_role=rule.role
                    ),
                )
        return ClassifiedTensor(
            category="other",
            role=LINEAR_ROLE_OTHER,
            tensor_parallel_role=self.tensor_layout.linear_tensor_parallel_role(
                linear_role=LINEAR_ROLE_OTHER
            ),
        )

    def classify_parameter(self, *, parameter_name: str) -> ClassifiedTensor:
        """Return explicit parameter classification for one parameter name."""

        for rule in self.parameter_rules:
            if rule.matches(qualified_name=parameter_name):
                return ClassifiedTensor(
                    category=rule.category,
                    role=rule.role,
                    tensor_parallel_role=(
                        self.tensor_layout.parameter_tensor_parallel_role(
                            parameter_role=rule.role
                        )
                    ),
                )
        return ClassifiedTensor(
            category="other",
            role=PARAMETER_ROLE_OTHER,
            tensor_parallel_role=self.tensor_layout.parameter_tensor_parallel_role(
                parameter_role=PARAMETER_ROLE_OTHER
            ),
        )

    def get_first_config_value(
        self, *, config: Any, field_names: tuple[str, ...]
    ) -> int:
        """Return the first present integer config field from a manifest list."""

        for field_name in field_names:
            field_value = getattr(config, field_name, None)
            if field_value is not None:
                return int(field_value)
        raise AssertionError(
            f"{self.model_type} config is missing required fields: {field_names}"
        )


COMMON_LAYOUT = TensorLayoutSpec()

COMMON_DECODER_LINEAR_RULES = (
    RoleRule(
        category="attention",
        role=LINEAR_ROLE_ATTENTION_QUERY,
        match_fragments=("q_proj", "wq"),
    ),
    RoleRule(
        category="attention",
        role=LINEAR_ROLE_ATTENTION_KEY,
        match_fragments=("k_proj", "wk"),
    ),
    RoleRule(
        category="attention",
        role=LINEAR_ROLE_ATTENTION_VALUE,
        match_fragments=("v_proj", "wv"),
    ),
    RoleRule(
        category="attention",
        role=LINEAR_ROLE_ATTENTION_OUTPUT,
        match_fragments=("o_proj", "out_proj", "wo"),
    ),
    RoleRule(
        category="mlp", role=LINEAR_ROLE_MLP_GATE_UP, match_fragments=("gate_up_proj",)
    ),
    RoleRule(category="mlp", role=LINEAR_ROLE_MLP_GATE, match_fragments=("gate_proj",)),
    RoleRule(
        category="mlp",
        role=LINEAR_ROLE_MLP_UP,
        match_fragments=("up_proj", "fc1", "w1", "w3"),
    ),
    RoleRule(
        category="mlp",
        role=LINEAR_ROLE_MLP_DOWN,
        match_fragments=("down_proj", "fc2", "w2"),
    ),
    RoleRule(category="other", role=LINEAR_ROLE_LM_HEAD, match_fragments=("lm_head",)),
    RoleRule(category="other", role=LINEAR_ROLE_ROUTER, match_fragments=("router",)),
)

COMMON_DECODER_PARAMETER_RULES = (
    RoleRule(
        category="embedding", role=PARAMETER_ROLE_LM_HEAD, match_fragments=("lm_head",)
    ),
    RoleRule(
        category="embedding",
        role=PARAMETER_ROLE_POSITION_EMBEDDING,
        match_fragments=("wpe", "embed_positions"),
    ),
    RoleRule(
        category="norm",
        role=PARAMETER_ROLE_NORM,
        match_fragments=("norm", "ln_", "layernorm"),
    ),
    RoleRule(
        category="embedding",
        role=PARAMETER_ROLE_EMBEDDING,
        match_fragments=("embed_tokens", "embed_in", "word_embeddings", "wte"),
        forbids_substrings=("layernorm", "norm"),
    ),
    RoleRule(
        category="attention",
        role=LINEAR_ROLE_ATTENTION_QUERY,
        match_fragments=("q_proj", "wq"),
    ),
    RoleRule(
        category="attention",
        role=LINEAR_ROLE_ATTENTION_KEY,
        match_fragments=("k_proj", "wk"),
    ),
    RoleRule(
        category="attention",
        role=LINEAR_ROLE_ATTENTION_VALUE,
        match_fragments=("v_proj", "wv"),
    ),
    RoleRule(
        category="attention",
        role=LINEAR_ROLE_ATTENTION_OUTPUT,
        match_fragments=("o_proj", "out_proj", "wo"),
    ),
    RoleRule(
        category="mlp", role=LINEAR_ROLE_MLP_GATE_UP, match_fragments=("gate_up_proj",)
    ),
    RoleRule(category="mlp", role=LINEAR_ROLE_MLP_GATE, match_fragments=("gate_proj",)),
    RoleRule(
        category="mlp",
        role=LINEAR_ROLE_MLP_UP,
        match_fragments=("up_proj", "fc1", "w1", "w3"),
    ),
    RoleRule(
        category="mlp",
        role=LINEAR_ROLE_MLP_DOWN,
        match_fragments=("down_proj", "fc2", "w2"),
    ),
    RoleRule(category="other", role=LINEAR_ROLE_ROUTER, match_fragments=("router",)),
    RoleRule(
        category="other", role=PARAMETER_ROLE_BIAS, match_fragments=(".bias", "_bias")
    ),
)

GPT2_LINEAR_RULES = (
    RoleRule(
        category="attention",
        role=LINEAR_ROLE_ATTENTION_QKV,
        match_fragments=("c_attn",),
        requires_substrings=("attn",),
    ),
    RoleRule(
        category="attention",
        role=LINEAR_ROLE_ATTENTION_OUTPUT,
        match_fragments=("c_proj",),
        requires_substrings=("attn",),
    ),
    RoleRule(
        category="mlp",
        role=LINEAR_ROLE_MLP_UP,
        match_fragments=("c_fc",),
        requires_substrings=("mlp",),
    ),
    RoleRule(
        category="mlp",
        role=LINEAR_ROLE_MLP_DOWN,
        match_fragments=("c_proj",),
        requires_substrings=("mlp",),
    ),
    RoleRule(category="other", role=LINEAR_ROLE_LM_HEAD, match_fragments=("lm_head",)),
)

GPT2_PARAMETER_RULES = (
    RoleRule(
        category="embedding", role=PARAMETER_ROLE_LM_HEAD, match_fragments=("lm_head",)
    ),
    RoleRule(
        category="embedding",
        role=PARAMETER_ROLE_POSITION_EMBEDDING,
        match_fragments=("wpe",),
    ),
    RoleRule(
        category="embedding", role=PARAMETER_ROLE_EMBEDDING, match_fragments=("wte",)
    ),
    RoleRule(category="norm", role=PARAMETER_ROLE_NORM, match_fragments=("ln_",)),
    RoleRule(
        category="attention",
        role=LINEAR_ROLE_ATTENTION_QKV,
        match_fragments=("c_attn",),
        requires_substrings=("attn",),
    ),
    RoleRule(
        category="attention",
        role=LINEAR_ROLE_ATTENTION_OUTPUT,
        match_fragments=("c_proj",),
        requires_substrings=("attn",),
    ),
    RoleRule(
        category="mlp",
        role=LINEAR_ROLE_MLP_UP,
        match_fragments=("c_fc",),
        requires_substrings=("mlp",),
    ),
    RoleRule(
        category="mlp",
        role=LINEAR_ROLE_MLP_DOWN,
        match_fragments=("c_proj",),
        requires_substrings=("mlp",),
    ),
    RoleRule(category="other", role=PARAMETER_ROLE_BIAS, match_fragments=(".bias",)),
)

GPT_NEOX_LINEAR_RULES = (
    RoleRule(
        category="attention",
        role=LINEAR_ROLE_ATTENTION_QKV,
        match_fragments=("query_key_value",),
    ),
    RoleRule(
        category="attention",
        role=LINEAR_ROLE_ATTENTION_OUTPUT,
        match_fragments=("dense",),
        requires_substrings=("attention",),
    ),
    RoleRule(
        category="mlp", role=LINEAR_ROLE_MLP_UP, match_fragments=("dense_h_to_4h",)
    ),
    RoleRule(
        category="mlp", role=LINEAR_ROLE_MLP_DOWN, match_fragments=("dense_4h_to_h",)
    ),
)

GPT_NEOX_PARAMETER_RULES = (
    RoleRule(
        category="embedding",
        role=PARAMETER_ROLE_LM_HEAD,
        match_fragments=("embed_out",),
    ),
    RoleRule(
        category="embedding",
        role=PARAMETER_ROLE_EMBEDDING,
        match_fragments=("embed_in",),
    ),
    RoleRule(
        category="norm",
        role=PARAMETER_ROLE_NORM,
        match_fragments=("layer_norm", "norm"),
    ),
    RoleRule(
        category="attention",
        role=LINEAR_ROLE_ATTENTION_QKV,
        match_fragments=("query_key_value",),
    ),
    RoleRule(
        category="attention",
        role=LINEAR_ROLE_ATTENTION_OUTPUT,
        match_fragments=("dense",),
        requires_substrings=("attention",),
    ),
    RoleRule(
        category="mlp", role=LINEAR_ROLE_MLP_UP, match_fragments=("dense_h_to_4h",)
    ),
    RoleRule(
        category="mlp", role=LINEAR_ROLE_MLP_DOWN, match_fragments=("dense_4h_to_h",)
    ),
    RoleRule(category="other", role=PARAMETER_ROLE_BIAS, match_fragments=(".bias",)),
)

PHI3_LINEAR_RULES = (
    RoleRule(
        category="attention",
        role=LINEAR_ROLE_ATTENTION_QKV,
        match_fragments=("qkv_proj",),
    ),
    RoleRule(
        category="attention",
        role=LINEAR_ROLE_ATTENTION_OUTPUT,
        match_fragments=("o_proj",),
    ),
    RoleRule(
        category="mlp", role=LINEAR_ROLE_MLP_GATE_UP, match_fragments=("gate_up_proj",)
    ),
    RoleRule(category="mlp", role=LINEAR_ROLE_MLP_DOWN, match_fragments=("down_proj",)),
)

PHI3_PARAMETER_RULES = (
    RoleRule(
        category="embedding", role=PARAMETER_ROLE_LM_HEAD, match_fragments=("lm_head",)
    ),
    RoleRule(
        category="embedding",
        role=PARAMETER_ROLE_EMBEDDING,
        match_fragments=("embed_tokens",),
    ),
    RoleRule(category="norm", role=PARAMETER_ROLE_NORM, match_fragments=("norm",)),
    RoleRule(
        category="attention",
        role=LINEAR_ROLE_ATTENTION_QKV,
        match_fragments=("qkv_proj",),
    ),
    RoleRule(
        category="attention",
        role=LINEAR_ROLE_ATTENTION_OUTPUT,
        match_fragments=("o_proj",),
    ),
    RoleRule(
        category="mlp", role=LINEAR_ROLE_MLP_GATE_UP, match_fragments=("gate_up_proj",)
    ),
    RoleRule(category="mlp", role=LINEAR_ROLE_MLP_DOWN, match_fragments=("down_proj",)),
    RoleRule(category="other", role=PARAMETER_ROLE_BIAS, match_fragments=(".bias",)),
)

PHI_PARAMETER_RULES = (
    RoleRule(
        category="embedding", role=PARAMETER_ROLE_LM_HEAD, match_fragments=("lm_head",)
    ),
    RoleRule(
        category="embedding",
        role=PARAMETER_ROLE_EMBEDDING,
        match_fragments=("embed_tokens",),
    ),
    RoleRule(
        category="norm", role=PARAMETER_ROLE_NORM, match_fragments=("norm", "layernorm")
    ),
    RoleRule(
        category="attention",
        role=LINEAR_ROLE_ATTENTION_QUERY,
        match_fragments=("q_proj",),
    ),
    RoleRule(
        category="attention",
        role=LINEAR_ROLE_ATTENTION_KEY,
        match_fragments=("k_proj",),
    ),
    RoleRule(
        category="attention",
        role=LINEAR_ROLE_ATTENTION_VALUE,
        match_fragments=("v_proj",),
    ),
    RoleRule(
        category="attention",
        role=LINEAR_ROLE_ATTENTION_OUTPUT,
        match_fragments=("self_attn.dense",),
    ),
    RoleRule(category="mlp", role=LINEAR_ROLE_MLP_UP, match_fragments=("mlp.fc1",)),
    RoleRule(category="mlp", role=LINEAR_ROLE_MLP_DOWN, match_fragments=("mlp.fc2",)),
    RoleRule(category="other", role=PARAMETER_ROLE_BIAS, match_fragments=(".bias",)),
)

BLOOM_LINEAR_RULES = (
    RoleRule(
        category="attention",
        role=LINEAR_ROLE_ATTENTION_QKV,
        match_fragments=("query_key_value",),
    ),
    RoleRule(
        category="attention",
        role=LINEAR_ROLE_ATTENTION_OUTPUT,
        match_fragments=("self_attention.dense",),
    ),
    RoleRule(
        category="mlp", role=LINEAR_ROLE_MLP_UP, match_fragments=("dense_h_to_4h",)
    ),
    RoleRule(
        category="mlp", role=LINEAR_ROLE_MLP_DOWN, match_fragments=("dense_4h_to_h",)
    ),
)

BLOOM_PARAMETER_RULES = (
    RoleRule(
        category="embedding", role=PARAMETER_ROLE_LM_HEAD, match_fragments=("lm_head",)
    ),
    RoleRule(
        category="norm",
        role=PARAMETER_ROLE_NORM,
        match_fragments=("layernorm", "ln_", "norm"),
    ),
    RoleRule(
        category="embedding",
        role=PARAMETER_ROLE_EMBEDDING,
        match_fragments=("word_embeddings",),
        forbids_substrings=("layernorm", "norm"),
    ),
    RoleRule(
        category="attention",
        role=LINEAR_ROLE_ATTENTION_QKV,
        match_fragments=("query_key_value",),
    ),
    RoleRule(
        category="attention",
        role=LINEAR_ROLE_ATTENTION_OUTPUT,
        match_fragments=("self_attention.dense",),
    ),
    RoleRule(
        category="mlp", role=LINEAR_ROLE_MLP_UP, match_fragments=("dense_h_to_4h",)
    ),
    RoleRule(
        category="mlp", role=LINEAR_ROLE_MLP_DOWN, match_fragments=("dense_4h_to_h",)
    ),
    RoleRule(category="other", role=PARAMETER_ROLE_BIAS, match_fragments=(".bias",)),
)

GPT_OSS_SYNTHETIC_RULES = (
    SyntheticLinearRule(
        category="mlp",
        role=LINEAR_ROLE_MLP_GATE_UP,
        match_fragments=("mlp.experts.gate_up_proj",),
        input_axis=1,
        output_axis=2,
        required_suffixes=(".weight",),
    ),
    SyntheticLinearRule(
        category="mlp",
        role=LINEAR_ROLE_MLP_DOWN,
        match_fragments=("mlp.experts.down_proj",),
        input_axis=1,
        output_axis=2,
        required_suffixes=(".weight",),
    ),
)


def _family(
    *,
    family_label: str,
    display_name: str,
    supports_grouped_query_attention: bool = False,
    supports_sliding_window_attention: bool = False,
    supports_rope: bool = False,
    uses_fused_qkv_projection: bool = False,
    notes: str = "",
) -> ArchitectureFamilySpec:
    """Return a normalized architecture-family object."""

    return ArchitectureFamilySpec(
        family_label=family_label,
        display_name=display_name,
        supports_grouped_query_attention=supports_grouped_query_attention,
        supports_sliding_window_attention=supports_sliding_window_attention,
        supports_rope=supports_rope,
        uses_fused_qkv_projection=uses_fused_qkv_projection,
        notes=notes,
    )


_MANIFESTS = (
    ArchitectureManifest(
        model_type="qwen2",
        family_spec=_family(
            family_label="qwen2_dense",
            display_name="Qwen2 dense",
            supports_grouped_query_attention=True,
            supports_rope=True,
        ),
        tensor_layout=COMMON_LAYOUT,
        num_layers_fields=("num_hidden_layers",),
        hidden_size_fields=("hidden_size",),
        num_attention_heads_fields=("num_attention_heads",),
        intermediate_size_fields=("intermediate_size",),
        max_position_fields=("max_position_embeddings",),
        num_key_value_heads_fields=("num_key_value_heads",),
        head_dim_fields=("head_dim",),
        linear_rules=COMMON_DECODER_LINEAR_RULES,
        parameter_rules=COMMON_DECODER_PARAMETER_RULES,
    ),
    ArchitectureManifest(
        model_type="qwen3",
        family_spec=_family(
            family_label="qwen3_dense",
            display_name="Qwen3 dense",
            supports_grouped_query_attention=True,
            supports_rope=True,
        ),
        tensor_layout=COMMON_LAYOUT,
        num_layers_fields=("num_hidden_layers",),
        hidden_size_fields=("hidden_size",),
        num_attention_heads_fields=("num_attention_heads",),
        intermediate_size_fields=("intermediate_size",),
        max_position_fields=("max_position_embeddings",),
        num_key_value_heads_fields=("num_key_value_heads",),
        head_dim_fields=("head_dim",),
        linear_rules=COMMON_DECODER_LINEAR_RULES,
        parameter_rules=COMMON_DECODER_PARAMETER_RULES,
    ),
    ArchitectureManifest(
        model_type="qwen3_next",
        family_spec=_family(
            family_label="qwen3_next_dense",
            display_name="Qwen3 Next dense",
            supports_grouped_query_attention=True,
            supports_rope=True,
        ),
        tensor_layout=COMMON_LAYOUT,
        num_layers_fields=("num_hidden_layers",),
        hidden_size_fields=("hidden_size",),
        num_attention_heads_fields=("num_attention_heads",),
        intermediate_size_fields=("intermediate_size",),
        max_position_fields=("max_position_embeddings",),
        num_key_value_heads_fields=("num_key_value_heads",),
        head_dim_fields=("head_dim",),
        linear_rules=COMMON_DECODER_LINEAR_RULES,
        parameter_rules=COMMON_DECODER_PARAMETER_RULES,
    ),
    ArchitectureManifest(
        model_type="llama",
        family_spec=_family(
            family_label="llama_dense",
            display_name="Llama dense",
            supports_grouped_query_attention=True,
            supports_rope=True,
        ),
        tensor_layout=COMMON_LAYOUT,
        num_layers_fields=("num_hidden_layers",),
        hidden_size_fields=("hidden_size",),
        num_attention_heads_fields=("num_attention_heads",),
        intermediate_size_fields=("intermediate_size",),
        max_position_fields=("max_position_embeddings",),
        num_key_value_heads_fields=("num_key_value_heads",),
        head_dim_fields=("head_dim",),
        linear_rules=COMMON_DECODER_LINEAR_RULES,
        parameter_rules=COMMON_DECODER_PARAMETER_RULES,
    ),
    ArchitectureManifest(
        model_type="mistral",
        family_spec=_family(
            family_label="mistral_dense",
            display_name="Mistral dense",
            supports_grouped_query_attention=True,
            supports_sliding_window_attention=True,
            supports_rope=True,
        ),
        tensor_layout=COMMON_LAYOUT,
        num_layers_fields=("num_hidden_layers",),
        hidden_size_fields=("hidden_size",),
        num_attention_heads_fields=("num_attention_heads",),
        intermediate_size_fields=("intermediate_size",),
        max_position_fields=("max_position_embeddings",),
        num_key_value_heads_fields=("num_key_value_heads",),
        head_dim_fields=("head_dim",),
        sliding_window_fields=("sliding_window",),
        linear_rules=COMMON_DECODER_LINEAR_RULES,
        parameter_rules=COMMON_DECODER_PARAMETER_RULES,
    ),
    ArchitectureManifest(
        model_type="olmo",
        family_spec=_family(
            family_label="olmo_dense",
            display_name="OLMo dense",
            supports_rope=True,
        ),
        tensor_layout=COMMON_LAYOUT,
        num_layers_fields=("num_hidden_layers",),
        hidden_size_fields=("hidden_size",),
        num_attention_heads_fields=("num_attention_heads",),
        intermediate_size_fields=("intermediate_size",),
        max_position_fields=("max_position_embeddings",),
        num_key_value_heads_fields=("num_key_value_heads",),
        linear_rules=COMMON_DECODER_LINEAR_RULES,
        parameter_rules=COMMON_DECODER_PARAMETER_RULES,
    ),
    ArchitectureManifest(
        model_type="olmo2",
        family_spec=_family(
            family_label="olmo2_dense",
            display_name="OLMo 2 dense",
            supports_rope=True,
        ),
        tensor_layout=COMMON_LAYOUT,
        num_layers_fields=("num_hidden_layers",),
        hidden_size_fields=("hidden_size",),
        num_attention_heads_fields=("num_attention_heads",),
        intermediate_size_fields=("intermediate_size",),
        max_position_fields=("max_position_embeddings",),
        num_key_value_heads_fields=("num_key_value_heads",),
        linear_rules=COMMON_DECODER_LINEAR_RULES,
        parameter_rules=COMMON_DECODER_PARAMETER_RULES,
    ),
    ArchitectureManifest(
        model_type="olmo3",
        family_spec=_family(
            family_label="olmo3_dense",
            display_name="OLMo 3 dense",
            supports_grouped_query_attention=True,
            supports_sliding_window_attention=True,
            supports_rope=True,
        ),
        tensor_layout=COMMON_LAYOUT,
        num_layers_fields=("num_hidden_layers",),
        hidden_size_fields=("hidden_size",),
        num_attention_heads_fields=("num_attention_heads",),
        intermediate_size_fields=("intermediate_size",),
        max_position_fields=("max_position_embeddings",),
        num_key_value_heads_fields=("num_key_value_heads",),
        sliding_window_fields=("sliding_window",),
        linear_rules=COMMON_DECODER_LINEAR_RULES,
        parameter_rules=COMMON_DECODER_PARAMETER_RULES,
    ),
    ArchitectureManifest(
        model_type="gpt2",
        family_spec=_family(
            family_label="gpt2_dense",
            display_name="GPT-2 dense",
            uses_fused_qkv_projection=True,
        ),
        tensor_layout=COMMON_LAYOUT,
        num_layers_fields=("num_hidden_layers", "n_layer"),
        hidden_size_fields=("hidden_size", "n_embd"),
        num_attention_heads_fields=("num_attention_heads", "n_head"),
        intermediate_size_fields=("intermediate_size", "n_inner"),
        max_position_fields=("n_positions", "max_position_embeddings"),
        linear_rules=GPT2_LINEAR_RULES,
        parameter_rules=GPT2_PARAMETER_RULES,
    ),
    ArchitectureManifest(
        model_type="gpt_neox",
        family_spec=_family(
            family_label="gpt_neox_dense",
            display_name="GPT-NeoX dense",
            supports_rope=True,
            uses_fused_qkv_projection=True,
        ),
        tensor_layout=COMMON_LAYOUT,
        num_layers_fields=("num_hidden_layers",),
        hidden_size_fields=("hidden_size",),
        num_attention_heads_fields=("num_attention_heads",),
        intermediate_size_fields=("intermediate_size",),
        max_position_fields=("max_position_embeddings",),
        linear_rules=GPT_NEOX_LINEAR_RULES,
        parameter_rules=GPT_NEOX_PARAMETER_RULES,
    ),
    ArchitectureManifest(
        model_type="opt",
        family_spec=_family(
            family_label="opt_dense",
            display_name="OPT dense",
        ),
        tensor_layout=COMMON_LAYOUT,
        num_layers_fields=("num_hidden_layers",),
        hidden_size_fields=("hidden_size",),
        num_attention_heads_fields=("num_attention_heads",),
        intermediate_size_fields=("ffn_dim", "intermediate_size"),
        max_position_fields=("max_position_embeddings",),
        linear_rules=COMMON_DECODER_LINEAR_RULES,
        parameter_rules=COMMON_DECODER_PARAMETER_RULES,
    ),
    ArchitectureManifest(
        model_type="phi",
        family_spec=_family(
            family_label="phi_dense",
            display_name="Phi dense",
            supports_rope=True,
        ),
        tensor_layout=COMMON_LAYOUT,
        num_layers_fields=("num_hidden_layers",),
        hidden_size_fields=("hidden_size",),
        num_attention_heads_fields=("num_attention_heads",),
        intermediate_size_fields=("intermediate_size",),
        max_position_fields=("max_position_embeddings",),
        num_key_value_heads_fields=("num_key_value_heads",),
        linear_rules=(
            RoleRule(
                category="attention",
                role=LINEAR_ROLE_ATTENTION_QUERY,
                match_fragments=("q_proj",),
            ),
            RoleRule(
                category="attention",
                role=LINEAR_ROLE_ATTENTION_KEY,
                match_fragments=("k_proj",),
            ),
            RoleRule(
                category="attention",
                role=LINEAR_ROLE_ATTENTION_VALUE,
                match_fragments=("v_proj",),
            ),
            RoleRule(
                category="attention",
                role=LINEAR_ROLE_ATTENTION_OUTPUT,
                match_fragments=("self_attn.dense",),
            ),
            RoleRule(
                category="mlp", role=LINEAR_ROLE_MLP_UP, match_fragments=("mlp.fc1",)
            ),
            RoleRule(
                category="mlp", role=LINEAR_ROLE_MLP_DOWN, match_fragments=("mlp.fc2",)
            ),
        ),
        parameter_rules=PHI_PARAMETER_RULES,
    ),
    ArchitectureManifest(
        model_type="phi3",
        family_spec=_family(
            family_label="phi3_dense",
            display_name="Phi-3 / Phi-4 dense",
            supports_rope=True,
            uses_fused_qkv_projection=True,
        ),
        tensor_layout=COMMON_LAYOUT,
        num_layers_fields=("num_hidden_layers",),
        hidden_size_fields=("hidden_size",),
        num_attention_heads_fields=("num_attention_heads",),
        intermediate_size_fields=("intermediate_size",),
        max_position_fields=(
            "max_position_embeddings",
            "original_max_position_embeddings",
        ),
        num_key_value_heads_fields=("num_key_value_heads",),
        head_dim_fields=("head_dim",),
        linear_rules=PHI3_LINEAR_RULES,
        parameter_rules=PHI3_PARAMETER_RULES,
    ),
    ArchitectureManifest(
        model_type="bloom",
        family_spec=_family(
            family_label="bloom_dense",
            display_name="BLOOM dense",
            uses_fused_qkv_projection=True,
        ),
        tensor_layout=COMMON_LAYOUT,
        num_layers_fields=("num_hidden_layers", "n_layer"),
        hidden_size_fields=("hidden_size", "n_embed", "n_embd"),
        num_attention_heads_fields=("num_attention_heads", "n_head"),
        intermediate_size_fields=("intermediate_size", "n_inner"),
        max_position_fields=("seq_length", "max_position_embeddings"),
        linear_rules=BLOOM_LINEAR_RULES,
        parameter_rules=BLOOM_PARAMETER_RULES,
    ),
    ArchitectureManifest(
        model_type="smollm3",
        family_spec=_family(
            family_label="smollm3_dense",
            display_name="SmolLM3 dense",
            supports_grouped_query_attention=True,
            supports_rope=True,
        ),
        tensor_layout=COMMON_LAYOUT,
        num_layers_fields=("num_hidden_layers",),
        hidden_size_fields=("hidden_size",),
        num_attention_heads_fields=("num_attention_heads",),
        intermediate_size_fields=("intermediate_size",),
        max_position_fields=("max_position_embeddings",),
        num_key_value_heads_fields=("num_key_value_heads",),
        linear_rules=COMMON_DECODER_LINEAR_RULES,
        parameter_rules=COMMON_DECODER_PARAMETER_RULES,
    ),
    ArchitectureManifest(
        model_type="gpt_oss",
        family_spec=_family(
            family_label="gpt_oss_dense",
            display_name="GPT-OSS",
            supports_grouped_query_attention=True,
            supports_sliding_window_attention=True,
            supports_rope=True,
            notes="Expert projections are approximated from tensor metadata.",
        ),
        tensor_layout=COMMON_LAYOUT,
        num_layers_fields=("num_hidden_layers",),
        hidden_size_fields=("hidden_size",),
        num_attention_heads_fields=("num_attention_heads",),
        intermediate_size_fields=("intermediate_size",),
        max_position_fields=(
            "max_position_embeddings",
            "original_max_position_embeddings",
        ),
        num_key_value_heads_fields=("num_key_value_heads",),
        head_dim_fields=("head_dim",),
        sliding_window_fields=("sliding_window",),
        linear_rules=COMMON_DECODER_LINEAR_RULES,
        parameter_rules=COMMON_DECODER_PARAMETER_RULES,
        synthetic_linear_rules=GPT_OSS_SYNTHETIC_RULES,
    ),
)

ARCHITECTURE_MANIFESTS = {manifest.model_type: manifest for manifest in _MANIFESTS}
SUPPORTED_DENSE_MODEL_TYPES = tuple(sorted(ARCHITECTURE_MANIFESTS))


def manifest_for_model_type(*, model_type: str) -> ArchitectureManifest | None:
    """Return the explicit manifest for one supported dense model type.

    Args:
        model_type: Hugging Face config model type.

    Returns:
        Matching manifest, or `None` when the family is not officially supported.
    """

    return ARCHITECTURE_MANIFESTS.get(model_type.lower())


def supported_dense_model_types() -> tuple[str, ...]:
    """Return the sorted official dense-model support surface."""

    return SUPPORTED_DENSE_MODEL_TYPES
