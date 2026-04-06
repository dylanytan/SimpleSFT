"""Typed architecture metadata used by the structural estimator."""

from __future__ import annotations

from dataclasses import dataclass


LINEAR_ROLE_OTHER = "other"
LINEAR_ROLE_ATTENTION_QUERY = "attention_query"
LINEAR_ROLE_ATTENTION_KEY = "attention_key"
LINEAR_ROLE_ATTENTION_VALUE = "attention_value"
LINEAR_ROLE_ATTENTION_QKV = "attention_qkv"
LINEAR_ROLE_ATTENTION_OUTPUT = "attention_output"
LINEAR_ROLE_MLP_GATE = "mlp_gate"
LINEAR_ROLE_MLP_UP = "mlp_up"
LINEAR_ROLE_MLP_GATE_UP = "mlp_gate_up"
LINEAR_ROLE_MLP_DOWN = "mlp_down"
LINEAR_ROLE_LM_HEAD = "lm_head"
LINEAR_ROLE_ROUTER = "router"

PARAMETER_ROLE_OTHER = "other"
PARAMETER_ROLE_EMBEDDING = "embedding"
PARAMETER_ROLE_POSITION_EMBEDDING = "position_embedding"
PARAMETER_ROLE_LM_HEAD = "lm_head"
PARAMETER_ROLE_NORM = "norm"
PARAMETER_ROLE_BIAS = "bias"

TENSOR_PARALLEL_REPLICATED = "replicated"
TENSOR_PARALLEL_COLUMN = "column_parallel"
TENSOR_PARALLEL_ROW = "row_parallel"
TENSOR_PARALLEL_VOCAB = "vocab_parallel"


@dataclass(frozen=True)
class AttentionSpec:
    """Architecture-level attention metadata for one model family.

    Args:
        num_query_heads: Full query-head count before TP sharding.
        num_key_value_heads: Full KV-head count before TP sharding.
        head_dim: Per-head width.
        query_width: Query projection output width.
        key_width: Key projection output width.
        value_width: Value projection output width.
        output_proj_input_width: Attention output-projection input width.
        uses_grouped_query: Whether query heads outnumber KV heads.
        uses_multi_query: Whether one KV head fans out to many query heads.
        sliding_window_size: Optional local-attention window size.
        rope_variant: Rope implementation label, if known.
        rope_scaling_present: Whether rope scaling config is present.

    Returns:
        Frozen dataclass describing the inspected attention path.

    Example:
        >>> spec = AttentionSpec(num_query_heads=16, num_key_value_heads=4, head_dim=128)
        >>> spec.uses_grouped_query
        True
    """

    num_query_heads: int = 0
    num_key_value_heads: int = 0
    head_dim: int = 0
    query_width: int = 0
    key_width: int = 0
    value_width: int = 0
    output_proj_input_width: int = 0
    uses_grouped_query: bool = False
    uses_multi_query: bool = False
    sliding_window_size: int | None = None
    rope_variant: str = "unknown"
    rope_scaling_present: bool = False

    def local_query_heads(self, *, tensor_parallel_degree: int) -> int:
        """Return TP-local query heads.

        Args:
            tensor_parallel_degree: Megatron tensor-parallel degree.

        Returns:
            Query-head count local to one TP rank.
        """

        divisor = max(tensor_parallel_degree, 1)
        return max(1, -(-self.num_query_heads // divisor))

    def local_key_value_heads(self, *, tensor_parallel_degree: int) -> int:
        """Return TP-local KV heads.

        Args:
            tensor_parallel_degree: Megatron tensor-parallel degree.

        Returns:
            Key/value-head count local to one TP rank.
        """

        divisor = max(tensor_parallel_degree, 1)
        return max(1, -(-max(self.num_key_value_heads, 1) // divisor))

    def effective_window(self, *, sequence_length: int) -> int:
        """Return the effective attention window.

        Args:
            sequence_length: Requested sequence length.

        Returns:
            Either the full sequence length or the configured sliding window.
        """

        if self.sliding_window_size is None or self.sliding_window_size <= 0:
            return sequence_length
        return min(sequence_length, self.sliding_window_size)

    def expanded_query_width(self, *, hidden_size: int) -> int:
        """Return the extra query/output width beyond hidden size.

        Args:
            hidden_size: Transformer hidden width.

        Returns:
            Extra saved-context width implied by expanded-query attention.
        """

        return max(self.query_width - hidden_size, 0) + max(
            self.output_proj_input_width - hidden_size,
            0,
        )


@dataclass(frozen=True)
class TensorLayoutSpec:
    """Explicit tensor-parallel role layout for one supported architecture.

    Args:
        column_parallel_linear_roles: Linear roles sharded on output columns.
        row_parallel_linear_roles: Linear roles sharded on input rows.
        replicated_linear_roles: Linear roles not TP sharded.
        column_parallel_parameter_roles: Parameter roles sharded on columns.
        row_parallel_parameter_roles: Parameter roles sharded on rows.
        vocab_parallel_parameter_roles: Parameter roles sharded over vocab.
        replicated_parameter_roles: Parameter roles left replicated.

    Returns:
        Frozen dataclass describing the family TP layout.
    """

    column_parallel_linear_roles: tuple[str, ...] = (
        LINEAR_ROLE_ATTENTION_QUERY,
        LINEAR_ROLE_ATTENTION_KEY,
        LINEAR_ROLE_ATTENTION_VALUE,
        LINEAR_ROLE_ATTENTION_QKV,
        LINEAR_ROLE_MLP_GATE,
        LINEAR_ROLE_MLP_UP,
        LINEAR_ROLE_MLP_GATE_UP,
    )
    row_parallel_linear_roles: tuple[str, ...] = (
        LINEAR_ROLE_ATTENTION_OUTPUT,
        LINEAR_ROLE_MLP_DOWN,
    )
    replicated_linear_roles: tuple[str, ...] = (
        LINEAR_ROLE_LM_HEAD,
        LINEAR_ROLE_ROUTER,
        LINEAR_ROLE_OTHER,
    )
    column_parallel_parameter_roles: tuple[str, ...] = (
        LINEAR_ROLE_ATTENTION_QUERY,
        LINEAR_ROLE_ATTENTION_KEY,
        LINEAR_ROLE_ATTENTION_VALUE,
        LINEAR_ROLE_ATTENTION_QKV,
        LINEAR_ROLE_MLP_GATE,
        LINEAR_ROLE_MLP_UP,
        LINEAR_ROLE_MLP_GATE_UP,
    )
    row_parallel_parameter_roles: tuple[str, ...] = (
        LINEAR_ROLE_ATTENTION_OUTPUT,
        LINEAR_ROLE_MLP_DOWN,
    )
    vocab_parallel_parameter_roles: tuple[str, ...] = (
        PARAMETER_ROLE_EMBEDDING,
        PARAMETER_ROLE_LM_HEAD,
    )
    replicated_parameter_roles: tuple[str, ...] = (
        PARAMETER_ROLE_POSITION_EMBEDDING,
        PARAMETER_ROLE_NORM,
        PARAMETER_ROLE_BIAS,
        LINEAR_ROLE_ROUTER,
        PARAMETER_ROLE_OTHER,
    )

    def linear_tensor_parallel_role(self, *, linear_role: str) -> str:
        """Return the TP role for one linear-role label."""

        if linear_role in self.column_parallel_linear_roles:
            return TENSOR_PARALLEL_COLUMN
        if linear_role in self.row_parallel_linear_roles:
            return TENSOR_PARALLEL_ROW
        return TENSOR_PARALLEL_REPLICATED

    def parameter_tensor_parallel_role(self, *, parameter_role: str) -> str:
        """Return the TP role for one parameter-role label."""

        if parameter_role in self.column_parallel_parameter_roles:
            return TENSOR_PARALLEL_COLUMN
        if parameter_role in self.row_parallel_parameter_roles:
            return TENSOR_PARALLEL_ROW
        if parameter_role in self.vocab_parallel_parameter_roles:
            return TENSOR_PARALLEL_VOCAB
        return TENSOR_PARALLEL_REPLICATED

    def summary_label(self) -> str:
        """Return a compact human-readable TP layout summary."""

        return "col(qkv/up) · row(o/down) · vocab(embed/lm_head)"


@dataclass(frozen=True)
class ArchitectureFamilySpec:
    """Normalized architecture-family metadata for supported dense models.

    Args:
        family_label: Stable internal family label.
        display_name: Short UI/report label.
        supports_grouped_query_attention: Whether GQA/MQA is expected.
        supports_sliding_window_attention: Whether local attention may appear.
        supports_rope: Whether rope metadata is meaningful for the family.
        uses_fused_qkv_projection: Whether QKV is commonly fused into one layer.
        notes: Short family note for reports/debugging.

    Returns:
        Frozen dataclass describing the estimator path for one family.
    """

    family_label: str = "unknown_dense"
    display_name: str = "Unknown dense LM"
    supports_grouped_query_attention: bool = False
    supports_sliding_window_attention: bool = False
    supports_rope: bool = False
    uses_fused_qkv_projection: bool = False
    notes: str = ""

    def display_label(self) -> str:
        """Return the user-facing architecture-family label."""

        return self.display_name
