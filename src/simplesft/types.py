"""Typed data objects used across measurement, estimation, and comparison."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .constants import BYTES_PER_GB


def bytes_to_gb(num_bytes: int) -> float:
    """Convert bytes to gibibytes for reporting.

    Args:
        num_bytes: Memory quantity in bytes.

    Returns:
        Memory quantity in GiB.
    """

    return num_bytes / BYTES_PER_GB


@dataclass(frozen=True)
class LoRAConfig:
    """Configuration for LoRA fine-tuning adapters.

    Args:
        rank: Adapter rank `r`.
        alpha: LoRA scaling factor.
        dropout: Adapter dropout probability.
        target_modules: Module-name suffixes to target.
        bias: Bias handling mode.

    Returns:
        Frozen dataclass describing LoRA settings.

    Example:
        >>> LoRAConfig(rank=16, alpha=32, dropout=0.05)
        LoRAConfig(rank=16, alpha=32.0, dropout=0.05, ...)
    """

    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.0
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    bias: str = "none"


@dataclass(frozen=True)
class ModelLinearLayerSpec:
    """Linear-layer summary used for LoRA parameter estimation.

    Args:
        module_name: Qualified module name.
        input_dim: Layer input width.
        output_dim: Layer output width.
        category: High-level grouping such as attention or mlp.

    Returns:
        Frozen dataclass representing one linear layer.
    """

    module_name: str
    input_dim: int
    output_dim: int
    category: str

    def parameter_count(self) -> int:
        """Return the dense parameter count for the layer."""

        return self.input_dim * self.output_dim


@dataclass(frozen=True)
class ModelSpec:
    """Model architecture summary used by measurement and estimation.

    Args:
        model_name: User-facing model identifier.
        model_type: Hugging Face architecture type.
        num_layers: Transformer block count.
        hidden_size: Transformer hidden width.
        num_attention_heads: Attention head count.
        intermediate_size: MLP expansion width.
        vocab_size: Vocabulary size.
        max_position_embeddings: Maximum context length.
        total_params: Total parameter count.
        trainable_linear_layers: Linear layers relevant for LoRA accounting.
        attention_type: Normalized attention type label.

    Returns:
        Frozen dataclass describing model structure.

    Example:
        >>> spec = ModelSpec(
        ...     model_name="tiny",
        ...     model_type="llama",
        ...     num_layers=2,
        ...     hidden_size=64,
        ...     num_attention_heads=4,
        ...     intermediate_size=256,
        ...     vocab_size=32000,
        ...     max_position_embeddings=2048,
        ...     total_params=123456,
        ...     trainable_linear_layers=(),
        ...     attention_type="causal",
        ... )
        >>> spec.tokens_per_layer(batch_size=2, sequence_length=16)
        32
    """

    model_name: str
    model_type: str
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int
    total_params: int
    trainable_linear_layers: tuple[ModelLinearLayerSpec, ...]
    attention_type: str = "causal"

    def tokens_per_layer(self, batch_size: int, sequence_length: int) -> int:
        """Return the token count processed by one layer."""

        return batch_size * sequence_length


@dataclass(frozen=True)
class TrainingConfig:
    """Training-memory configuration for one measurement or estimate run.

    Args:
        tuning_mode: One of `full_ft` or `lora`.
        optimizer_name: Optimizer name, currently `adamw`.
        weight_dtype: Model forward/backward dtype.
        grad_dtype: Gradient dtype.
        master_weight_dtype: Master-weight dtype when master weights are enabled.
        optimizer_state_dtype: Optimizer-state dtype or `auto`.
        micro_batch_size_per_gpu: Micro-batch size per rank.
        gradient_accumulation_steps: Number of accumulation steps.
        max_seq_len: Sequence length in tokens.
        gradient_checkpointing: Whether checkpointing is enabled.
        attention_backend: Attention kernel family.
        distributed_mode: `single_gpu`, `ddp`, or `zero2`.
        num_nodes: Number of nodes.
        gpus_per_node: GPUs per node.
        gpu_memory_gb: VRAM capacity per GPU.
        lora: Optional LoRA configuration.
        use_master_weights: Whether to include a separate master-weight copy.
        reserved_vram_gb_per_gpu: Fixed reserve for runtime overhead, or `None` for auto.
        activation_safety_margin_gb: Extra safety margin added to final peak.
        warmup_steps: Number of warmup steps before measurement.

    Returns:
        Frozen dataclass describing a training configuration.

    Example:
        >>> TrainingConfig(
        ...     tuning_mode="full_ft",
        ...     micro_batch_size_per_gpu=1,
        ...     max_seq_len=128,
        ... )
    """

    tuning_mode: str
    optimizer_name: str = "adamw"
    weight_dtype: str = "bf16"
    grad_dtype: str = "bf16"
    master_weight_dtype: str = "fp32"
    optimizer_state_dtype: str = "auto"
    micro_batch_size_per_gpu: int = 1
    gradient_accumulation_steps: int = 1
    max_seq_len: int = 512
    gradient_checkpointing: bool = False
    attention_backend: str = "standard"
    distributed_mode: str = "single_gpu"
    num_nodes: int = 1
    gpus_per_node: int = 1
    gpu_memory_gb: float = 24.0
    lora: LoRAConfig | None = None
    use_master_weights: bool = False
    reserved_vram_gb_per_gpu: float | None = None
    activation_safety_margin_gb: float = 0.25
    warmup_steps: int = 1

    def world_size(self) -> int:
        """Return the total number of ranks implied by the config."""

        return self.num_nodes * self.gpus_per_node


@dataclass(frozen=True)
class MemoryComponentBreakdown:
    """Named memory components for one estimate or measurement result.

    Args:
        parameter_bytes: Persistent parameter memory.
        gradient_bytes: Persistent gradient memory.
        optimizer_state_bytes: Persistent optimizer-state memory.
        activation_bytes: Retained activation memory.
        transient_bytes: Communication and temporary spike memory.
        residual_bytes: Unattributed remainder.
        runtime_reserve_bytes: Fixed reserve for allocator/runtime overhead.

    Returns:
        Frozen dataclass of memory components.
    """

    parameter_bytes: int = 0
    gradient_bytes: int = 0
    optimizer_state_bytes: int = 0
    activation_bytes: int = 0
    transient_bytes: int = 0
    residual_bytes: int = 0
    runtime_reserve_bytes: int = 0

    def total_bytes(self) -> int:
        """Return the sum of all component bytes."""

        return (
            self.parameter_bytes
            + self.gradient_bytes
            + self.optimizer_state_bytes
            + self.activation_bytes
            + self.transient_bytes
            + self.residual_bytes
            + self.runtime_reserve_bytes
        )


@dataclass(frozen=True)
class PhaseMemoryRecord:
    """Memory values recorded at one training phase boundary.

    Args:
        phase_name: Human-readable phase name.
        allocated_bytes: Current allocated CUDA bytes.
        reserved_bytes: Current reserved CUDA bytes.
        peak_allocated_bytes: Peak allocated bytes since last reset.
        peak_reserved_bytes: Peak reserved bytes since last reset.
        delta_allocated_bytes: Change in allocated bytes from the prior phase.
        delta_reserved_bytes: Change in reserved bytes from the prior phase.
        notes: Optional tags or measurement notes.

    Returns:
        Frozen dataclass for one phase snapshot.
    """

    phase_name: str
    allocated_bytes: int
    reserved_bytes: int
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    delta_allocated_bytes: int
    delta_reserved_bytes: int
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class MemoryResult:
    """Combined memory result produced by measurement or estimation.

    Args:
        mode: `measure` or `estimate`.
        model_name: Model identifier.
        config: Training configuration used.
        breakdown: Modular memory breakdown.
        phase_records: Phase-wise memory timeline.
        peak_phase: Phase with highest peak.
        global_peak_bytes: Peak memory per rank in bytes.
        feasible: Whether the config fits in available GPU memory.
        metadata: Extra runtime metadata.
        assumptions: Human-readable assumptions and warnings.

    Returns:
        Frozen result dataclass.
    """

    mode: str
    model_name: str
    config: TrainingConfig
    breakdown: MemoryComponentBreakdown
    phase_records: tuple[PhaseMemoryRecord, ...]
    peak_phase: str
    global_peak_bytes: int
    feasible: bool
    metadata: dict[str, Any] = field(default_factory=dict)
    assumptions: tuple[str, ...] = ()

    def global_peak_gb(self) -> float:
        """Return the peak memory per rank in GiB."""

        return bytes_to_gb(self.global_peak_bytes)

    def headroom_gb(self) -> float:
        """Return per-rank VRAM headroom in GiB."""

        return self.config.gpu_memory_gb - self.global_peak_gb()


@dataclass(frozen=True)
class ComparisonResult:
    """Comparison between measured and estimated memory results.

    Args:
        model_name: Model identifier.
        measured: Ground-truth measured result.
        estimated: Analytical estimate.
        global_peak_error_bytes: Absolute peak-memory error.
        phase_peak_error_bytes: Error by phase name.
        component_error_bytes: Error by component name.
        global_peak_relative_error: Relative peak-memory error.
        phase_peak_relative_error: Relative error by phase name.
        component_relative_error: Relative error by component name.
        workspace_proxy_error_bytes: Error by workspace-proxy name.
        workspace_proxy_relative_error: Relative error by workspace-proxy name.
        benchmark_metadata: Benchmark metadata helpful for iteration reports.
        notes: Human-readable summary notes.

    Returns:
        Frozen comparison dataclass.
    """

    model_name: str
    measured: MemoryResult
    estimated: MemoryResult
    global_peak_error_bytes: int
    global_peak_relative_error: float
    phase_peak_error_bytes: dict[str, int]
    phase_peak_relative_error: dict[str, float]
    component_error_bytes: dict[str, int]
    component_relative_error: dict[str, float]
    workspace_proxy_error_bytes: dict[str, int] = field(default_factory=dict)
    workspace_proxy_relative_error: dict[str, float] = field(default_factory=dict)
    benchmark_metadata: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class SearchResult:
    """Ranked result for configuration search.

    Args:
        candidates: Ranked candidate estimates.
        infeasible_candidates: Candidate estimates that exceed VRAM.

    Returns:
        Frozen search result with feasible and infeasible estimates.
    """

    candidates: tuple[MemoryResult, ...]
    infeasible_candidates: tuple[MemoryResult, ...]


@dataclass(frozen=True)
class BenchmarkCase:
    """One benchmark configuration in the developer iteration corpus.

    Args:
        name: Stable benchmark case name.
        model: Model reference or precomputed model spec.
        config: Training config for the case.
        tags: Optional grouping tags.

    Returns:
        Frozen benchmark case dataclass.
    """

    name: str
    model: str | ModelSpec
    config: TrainingConfig
    tags: tuple[str, ...] = ()

    def artifact_slug(self) -> str:
        """Return a filesystem-safe artifact slug for the case."""

        normalized_name = self.name.lower().replace(" ", "-").replace("/", "-")
        return "".join(
            character
            for character in normalized_name
            if character.isalnum() or character == "-"
        )


@dataclass(frozen=True)
class BenchmarkCaseResult:
    """Artifact paths and status for one benchmark case.

    Args:
        case: Benchmark case definition.
        estimate_path: Path to the saved estimate artifact.
        measurement_path: Path to the saved measurement artifact, if any.
        comparison_path: Path to the saved comparison artifact, if any.
        error_message: Benchmark failure message, if any.

    Returns:
        Frozen benchmark case-result dataclass.
    """

    case: BenchmarkCase
    estimate_path: str
    measurement_path: str | None = None
    comparison_path: str | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class BenchmarkSuiteResult:
    """Saved benchmark-suite summary for one developer iteration.

    Args:
        output_dir: Root artifact directory.
        case_results: Saved results for each benchmark case.
        notes: Optional suite-level notes.

    Returns:
        Frozen benchmark suite-result dataclass.
    """

    output_dir: str
    case_results: tuple[BenchmarkCaseResult, ...]
    notes: tuple[str, ...] = ()
