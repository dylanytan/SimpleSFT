"""Typed data objects used across measurement, estimation, and comparison."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .models.architecture_types import (
    ArchitectureFamilySpec,
    AttentionSpec,
    TensorLayoutSpec,
)
from .constants import BYTES_PER_GB, DEFAULT_TARGET_MODULES, SUPPORTED_MODEL_TYPES


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
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES
    bias: str = "none"

    def __post_init__(self) -> None:
        """Normalize list-like target modules to a tuple of suffix strings."""

        object.__setattr__(self, "target_modules", tuple(self.target_modules))


@dataclass(frozen=True)
class VisionSpec:
    """Vision-input metadata for multimodal inspection and synthetic batches.

    Args:
        default_image_size: Default square image side length in pixels.
        patch_size: Spatial patch size used by the vision encoder.
        temporal_patch_size: Temporal patch size for flattened image patches.
        spatial_merge_size: Patch merge factor applied before image tokens.
        channels: Input image channel count.
        image_token_id: Placeholder token id used for image slots.
        vision_start_token_id: Vision-span start token id.
        vision_end_token_id: Vision-span end token id.

    Returns:
        Frozen dataclass describing one supported image-input path.

    Example:
        >>> vision = VisionSpec(default_image_size=448, patch_size=14, spatial_merge_size=2)
        >>> vision.image_token_count()
        256
    """

    default_image_size: int = 448
    patch_size: int = 0
    temporal_patch_size: int = 1
    spatial_merge_size: int = 1
    channels: int = 3
    image_token_id: int | None = None
    vision_start_token_id: int | None = None
    vision_end_token_id: int | None = None

    def supports_images(self) -> bool:
        """Return whether the spec exposes enough metadata for image inputs."""

        return (
            self.patch_size > 0
            and self.image_token_id is not None
            and self.vision_start_token_id is not None
            and self.vision_end_token_id is not None
        )

    def resolved_image_size(self, *, image_size: int | None = None) -> int:
        """Return the image size used for estimates and synthetic batches."""

        return image_size or self.default_image_size

    def grid_side(self, *, image_size: int | None = None) -> int:
        """Return the per-side patch grid before token merging."""

        resolved_image_size = self.resolved_image_size(image_size=image_size)
        return max(1, math.ceil(resolved_image_size / max(self.patch_size, 1)))

    def image_token_count(self, *, image_size: int | None = None) -> int:
        """Return the image-token count contributed by one square image."""

        if not self.supports_images():
            return 0
        merged_side = math.ceil(
            self.grid_side(image_size=image_size) / max(self.spatial_merge_size, 1)
        )
        return merged_side * merged_side

    def flattened_patch_dim(self) -> int:
        """Return the flattened patch width used by Qwen-style pixel inputs."""

        return (
            self.temporal_patch_size * self.patch_size * self.patch_size * self.channels
        )


@dataclass(frozen=True)
class ModelLinearLayerSpec:
    """Linear-layer summary used for LoRA parameter estimation.

    Args:
        module_name: Qualified module name.
        input_dim: Layer input width.
        output_dim: Layer output width.
        category: High-level grouping such as attention or mlp.
        role: Explicit architectural linear role.
        tensor_parallel_role: Explicit TP layout role.

    Returns:
        Frozen dataclass representing one linear layer.
    """

    module_name: str
    input_dim: int
    output_dim: int
    category: str
    role: str = "other"
    tensor_parallel_role: str = "replicated"

    def parameter_count(self) -> int:
        """Return the dense parameter count for the layer."""

        return self.input_dim * self.output_dim


@dataclass(frozen=True)
class ModelParameterSpec:
    """Named parameter summary used for optimizer-state estimation.

    Args:
        parameter_name: Qualified parameter name.
        shape: Parameter tensor shape.
        category: Coarse parameter category.
        role: Explicit architectural parameter role.
        tensor_parallel_role: Explicit TP layout role.

    Returns:
        Frozen dataclass representing one parameter tensor.
    """

    parameter_name: str
    shape: tuple[int, ...]
    category: str
    role: str = "other"
    tensor_parallel_role: str = "replicated"

    def numel(self) -> int:
        """Return the total parameter count for this tensor."""

        total = 1
        for dim_size in self.shape:
            total *= dim_size
        return total

    def is_matrix(self) -> bool:
        """Return whether the tensor is matrix-like for factored optimizers."""

        return len(self.shape) >= 2


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
        parameter_specs: Named parameter shapes for optimizer-state accounting.
        attention_type: Normalized attention type label.
        architecture_family: Explicit normalized architecture family metadata.
        attention: Explicit attention-layout metadata.
        tensor_layout: Explicit TP layout summary.
        vision: Optional image-input metadata for vision-language models.

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
    parameter_specs: tuple[ModelParameterSpec, ...] = ()
    attention_type: str = "causal"
    architecture_family: ArchitectureFamilySpec = field(
        default_factory=ArchitectureFamilySpec
    )
    attention: AttentionSpec = field(default_factory=AttentionSpec)
    tensor_layout: TensorLayoutSpec = field(default_factory=TensorLayoutSpec)
    vision: VisionSpec | None = None

    def __post_init__(self) -> None:
        """Backfill architecture metadata for incomplete model specs."""

        if self.attention.num_query_heads <= 0:
            object.__setattr__(
                self,
                "attention",
                AttentionSpec(
                    num_query_heads=self.num_attention_heads,
                    num_key_value_heads=max(1, self.num_attention_heads),
                    head_dim=max(
                        1, self.hidden_size // max(self.num_attention_heads, 1)
                    ),
                    query_width=self.hidden_size,
                    key_width=self.hidden_size,
                    value_width=self.hidden_size,
                    output_proj_input_width=self.hidden_size,
                ),
            )
        if self.architecture_family.family_label == "unknown_dense":
            object.__setattr__(
                self,
                "architecture_family",
                ArchitectureFamilySpec(
                    family_label=f"{self.model_type}_dense",
                    display_name=self.model_type,
                ),
            )

    def tokens_per_layer(self, batch_size: int, sequence_length: int) -> int:
        """Return the token count processed by one layer."""

        return batch_size * sequence_length

    def effective_tokens_per_layer(
        self,
        *,
        batch_size: int,
        sequence_length: int,
        image_tokens_per_sample: int = 0,
    ) -> int:
        """Return the total text-plus-image token count processed by one layer."""

        return batch_size * (sequence_length + image_tokens_per_sample)

    def supports_vision_inputs(self) -> bool:
        """Return whether the inspected model exposes a supported image path."""

        return self.vision is not None and self.vision.supports_images()


@dataclass(frozen=True)
class EstimatorConfig:
    """Structural configuration for analytical memory estimation.

    Args:
        tuning_mode: One of `full_ft` or `lora`.
        distributed_mode: `single_gpu`, `ddp`, `zero2`, or `zero3`.
        optimizer_name: Optimizer family name such as `adamw` or `sgd`.
        attention_backend: Attention kernel family label.
        gradient_checkpointing: Whether checkpointing is enabled.
        tensor_parallel_degree: Megatron-style tensor-parallel degree.
        sequence_parallel: Whether sequence-parallel activation sharding is enabled.
        vocab_parallel_logits: Whether embeddings / LM head shard with TP.
        max_seq_len: Sequence length in tokens.
        micro_batch_size_per_gpu: Micro-batch size per rank.
        gpus_per_node: GPUs visible to each node.
        num_nodes: Number of nodes.
        gpu_memory_gb: VRAM capacity per GPU.
        weight_dtype: Main forward/backward parameter dtype.
        grad_dtype: Gradient dtype.
        optimizer_state_dtype: Optimizer-state dtype or `auto`.
        optimizer_update_dtype: Optimizer-update scratch dtype or `auto`.
        master_weight_dtype: Master-weight dtype.
        adapter_weight_dtype: LoRA adapter parameter dtype.
        adapter_grad_dtype: LoRA adapter gradient dtype.
        adapter_optimizer_state_dtype: LoRA optimizer-state dtype.
        use_master_weights: Whether a separate master-weight copy is present.
        lora: Optional LoRA adapter configuration.
        runtime_cuda_context_gb: Runtime CUDA-context support assumption.
        runtime_allocator_pool_gb: Runtime allocator-pool support assumption.
        runtime_nccl_gb: Runtime NCCL support assumption.
        runtime_deepspeed_gb: Runtime DeepSpeed support assumption.
        runtime_support_gb_override: Optional total runtime-support override.
        ddp_bucket_elements: DDP reducer bucket size in gradient elements.
        zero_bucket_elements: ZeRO reduce/all-gather bucket size in elements.
        zero_prefetch_elements: ZeRO prefetch bucket size in elements.
        zero_tested_optimizer_names: Optimizers treated as ZeRO-tested.
        zero_untested_optimizer_state_is_sharded: Whether untested optimizer
            state is assumed sharded in ZeRO.
        zero_untested_optimizer_replica_tensor_count: Fallback state replica
            tensor copies for untested ZeRO optimizers.
        zero_untested_optimizer_replica_dtype: Dtype alias for fallback state
            replicas.
        zero_untested_optimizer_update_is_sharded: Whether untested optimizer
            update scratch is assumed sharded in ZeRO.
        zero_untested_optimizer_update_replica_tensor_count: Fallback update
            replica tensor copies for untested ZeRO optimizers.
        zero_untested_optimizer_update_dtype: Dtype alias for fallback update
            replicas.

    Returns:
        Frozen estimator config with only structural memory inputs.

    Example:
        >>> EstimatorConfig(tuning_mode="full_ft", max_seq_len=128).world_size()
        1
    """

    tuning_mode: str
    distributed_mode: str = "single_gpu"
    optimizer_name: str = "adamw"
    attention_backend: str = "standard"
    gradient_checkpointing: bool = False
    tensor_parallel_degree: int = 1
    sequence_parallel: bool = False
    vocab_parallel_logits: bool = True
    max_seq_len: int = 512
    micro_batch_size_per_gpu: int = 1
    gpus_per_node: int = 1
    num_nodes: int = 1
    gpu_memory_gb: float = 24.0
    weight_dtype: str = "bf16"
    grad_dtype: str = "bf16"
    optimizer_state_dtype: str = "auto"
    optimizer_update_dtype: str = "auto"
    master_weight_dtype: str = "fp32"
    adapter_weight_dtype: str = "fp32"
    adapter_grad_dtype: str = "fp32"
    adapter_optimizer_state_dtype: str = "fp32"
    runtime_cuda_context_gb: float = 0.25
    runtime_allocator_pool_gb: float = 0.05
    runtime_nccl_gb: float = 0.0
    runtime_deepspeed_gb: float = 0.0
    runtime_support_gb_override: float | None = None
    ddp_bucket_elements: int = 268_435_456
    zero_bucket_elements: int = 500_000_000
    zero_prefetch_elements: int = 50_000_000
    loss_output_dtype: str = "fp32"
    loss_output_logits_fraction: float = 1.0
    single_lora_flash2_backward_logits_fraction: float = 0.5
    single_lora_sdpa_saved_input_overlap_fraction: float = 1.0
    single_lora_sdpa_saved_input_overlap_min_hidden_size: int = 4096
    single_lora_sdpa_saved_input_overlap_min_layers: int = 40
    zero2_lora_backward_activation_fraction: float = 1.0 / 6.0
    zero3_lora_backward_activation_fraction: float = 1.0 / 6.0
    use_master_weights: bool = False
    lora: LoRAConfig | None = None
    persistent_backend_buffer_tensor_count: float | None = None
    persistent_backend_buffer_dtype: str = "weight_dtype"
    zero_tested_optimizer_names: tuple[str, ...] = ("adam", "adamw")
    zero_untested_optimizer_state_is_sharded: bool = False
    zero_untested_optimizer_replica_tensor_count: float = 1.0
    zero_untested_optimizer_replica_dtype: str = "weight_dtype"
    zero_untested_optimizer_update_is_sharded: bool = False
    zero_untested_optimizer_update_replica_tensor_count: float = 1.0
    zero_untested_optimizer_update_dtype: str = "weight_dtype"

    def _resolve_configured_dtype(self, dtype_name: str) -> str:
        """Resolve one configured dtype alias against estimator dtypes."""

        normalized_name = dtype_name.lower()
        if normalized_name == "auto":
            return self.weight_dtype
        if normalized_name == "weight_dtype":
            return self.weight_dtype
        if normalized_name == "grad_dtype":
            return self.grad_dtype
        if normalized_name == "master_weight_dtype":
            return self.master_weight_dtype
        if normalized_name == "optimizer_state_dtype":
            if self.optimizer_state_dtype != "auto":
                return self.optimizer_state_dtype
            return self.weight_dtype
        if normalized_name == "adapter_weight_dtype":
            return self.adapter_parameter_dtype()
        if normalized_name == "adapter_grad_dtype":
            return self.adapter_gradient_dtype()
        if normalized_name in {"adapter_state_dtype", "adapter_optimizer_state_dtype"}:
            return self.adapter_state_dtype()
        return dtype_name

    def available_gpu_count(self) -> int:
        """Return the number of GPUs available to the strategy search."""

        return max(1, self.num_nodes * self.gpus_per_node)

    def world_size(self) -> int:
        """Return the total rank count implied by the config."""

        if self.distributed_mode == "single_gpu":
            return 1
        return self.available_gpu_count()

    def tensor_parallel_degree_resolved(self) -> int:
        """Return the validated tensor-parallel degree."""

        assert self.tensor_parallel_degree >= 1, "Tensor parallel degree must be >= 1."
        if self.distributed_mode == "single_gpu":
            assert (
                self.tensor_parallel_degree == 1
            ), "single_gpu mode cannot use tensor parallelism."
            return 1
        assert (
            self.available_gpu_count() % self.tensor_parallel_degree == 0
        ), "Tensor parallel degree must divide the available GPU count."
        return self.tensor_parallel_degree

    def data_parallel_degree(self) -> int:
        """Return the data-parallel replica count after tensor parallelism."""

        if self.distributed_mode == "single_gpu":
            return 1
        return self.available_gpu_count() // self.tensor_parallel_degree_resolved()

    def uses_tensor_parallel(self) -> bool:
        """Return whether tensor parallelism is enabled."""

        return self.tensor_parallel_degree_resolved() > 1

    def is_zero_mode(self) -> bool:
        """Return whether the config uses ZeRO sharding."""

        return self.distributed_mode in {"zero2", "zero3"}

    def normalized_attention_backend(self) -> str:
        """Return the normalized attention-backend label."""

        return self.attention_backend.lower()

    def normalized_zero_tested_optimizer_names(self) -> tuple[str, ...]:
        """Return normalized ZeRO-tested optimizer names."""

        return tuple(
            optimizer_name.lower()
            for optimizer_name in self.zero_tested_optimizer_names
        )

    def adapter_parameter_dtype(self) -> str:
        """Return the dtype used for LoRA adapter parameters."""

        return self.adapter_weight_dtype

    def adapter_gradient_dtype(self) -> str:
        """Return the dtype used for LoRA adapter gradients."""

        return self.adapter_grad_dtype

    def adapter_state_dtype(self) -> str:
        """Return the dtype used for LoRA optimizer-state tensors."""

        return self.adapter_optimizer_state_dtype

    def loss_output_resolved_dtype(self) -> str:
        """Return the dtype used for persistent loss/logit outputs."""

        return self._resolve_configured_dtype(self.loss_output_dtype)

    def resolved_runtime_nccl_gb(self) -> float:
        """Return NCCL runtime support in GiB.

        Returns:
            Explicit NCCL support when configured, otherwise the default
            distributed-runtime assumption for the selected mode.
        """

        if self.runtime_nccl_gb > 0.0:
            return self.runtime_nccl_gb
        if self.distributed_mode == "single_gpu":
            return 0.0
        return 0.30

    def resolved_runtime_deepspeed_gb(self) -> float:
        """Return DeepSpeed runtime support in GiB.

        Returns:
            Explicit DeepSpeed support when configured, otherwise the default
            ZeRO-runtime assumption for the selected mode.
        """

        if self.runtime_deepspeed_gb > 0.0:
            return self.runtime_deepspeed_gb
        if self.distributed_mode == "zero2":
            return 0.20
        if self.distributed_mode == "zero3":
            return 0.40
        return 0.0

    def runtime_support_gb(self) -> float:
        """Return total runtime support in GiB.

        Returns:
            Configured total runtime support, or the sum of named runtime
            support components when no total override is present.
        """

        if self.runtime_support_gb_override is not None:
            return self.runtime_support_gb_override
        return (
            self.runtime_cuda_context_gb
            + self.runtime_allocator_pool_gb
            + self.resolved_runtime_nccl_gb()
            + self.resolved_runtime_deepspeed_gb()
        )

    def persistent_backend_buffer_resolved_dtype(self) -> str:
        """Return the dtype used for persistent backend support buffers."""

        return self._resolve_configured_dtype(self.persistent_backend_buffer_dtype)

    def persistent_backend_buffer_count(self) -> float:
        """Return persistent backend-buffer copies in trainable-tensor units."""

        if self.persistent_backend_buffer_tensor_count is not None:
            return self.persistent_backend_buffer_tensor_count
        optimizer_name = self.optimizer_name.lower()
        if self.distributed_mode == "ddp" and self.tuning_mode == "full_ft":
            return 1.0
        if not self.is_zero_mode() or self.tuning_mode != "full_ft":
            return 0.0
        if optimizer_name == "adagrad":
            return 2.0
        return 1.0

    def sharded_lora_backward_activation_fraction(self) -> float:
        """Return backward-visible retention for checkpointed ZeRO LoRA.

        Returns:
            Fraction of sharded LoRA visible propagation that survives into the
            backward peak after checkpoint rematerialization.
        """

        if self.tuning_mode != "lora" or not self.is_zero_mode():
            return 0.0
        if self.distributed_mode == "zero3":
            return self.zero3_lora_backward_activation_fraction
        return 0.0

    def zero_untested_replica_dtype(self) -> str:
        """Return the fallback dtype used for ZeRO untested state replicas."""

        return self._resolve_configured_dtype(
            self.zero_untested_optimizer_replica_dtype
        )

    def zero_untested_update_dtype(self) -> str:
        """Return the fallback dtype used for ZeRO untested update replicas."""

        return self._resolve_configured_dtype(self.zero_untested_optimizer_update_dtype)


@dataclass(frozen=True)
class ResidentStateDebug:
    """Resident memory objects used by the analytical estimator."""

    parameter_bytes: int
    gradient_bytes: int
    optimizer_state_bytes: int
    master_weight_bytes: int
    runtime_support_bytes: int
    persistent_backend_buffer_bytes: int
    trainable_parameter_bytes: int


@dataclass(frozen=True)
class ActivationDebug:
    """Retained activation objects used by the analytical estimator."""

    base_hook_visible_activation_bytes: int
    visible_propagation_bytes: int
    checkpoint_resident_block_input_bytes: int
    saved_linear_input_bytes: int
    saved_input_overlap_bytes: int
    mlp_intermediate_bytes: int
    parameter_gradient_context_bytes: int
    residual_norm_bytes: int
    checkpoint_boundary_bytes: int
    attention_saved_bytes: int
    loss_state_bytes: int
    lora_low_rank_bytes: int
    lora_backward_logits_context_bytes: int
    expanded_query_saved_bytes: int
    query_output_context_bytes: int
    key_output_context_bytes: int
    value_output_context_bytes: int
    output_proj_input_context_bytes: int
    output_proj_output_context_bytes: int
    retained_forward_proxy_bytes: int
    forward_phase_activation_bytes: int
    backward_phase_activation_bytes: int
    hook_visible_activation_bytes: int
    checkpointed_sharded_lora_backward_visible_bytes: int = 0


@dataclass(frozen=True)
class WorkspaceDebug:
    """Phase-local workspaces and communication windows for the estimator."""

    attention_forward_workspace_bytes: int
    backward_kernel_workspace_bytes: int
    recompute_workspace_bytes: int
    loss_workspace_bytes: int
    optimizer_update_workspace_bytes: int
    ddp_reducer_bucket_bytes: int
    ddp_comm_overlap_bytes: int
    zero_allgather_bucket_bytes: int
    zero_reduce_bucket_bytes: int
    zero_prefetch_bucket_bytes: int
    zero_fetch_window_bytes: int
    zero_update_window_bytes: int
    zero_comm_window_bytes: int
    tensor_parallel_comm_window_bytes: int
    sequence_parallel_comm_window_bytes: int


@dataclass(frozen=True)
class PhasePeakDebug:
    """Peak bytes by phase and reserve model for the estimator."""

    forward_peak_bytes: int
    backward_peak_bytes: int
    optimizer_peak_bytes: int
    global_peak_bytes: int
    global_peak_phase: str
    soft_reserved_global_peak_bytes: int
    stressed_reserved_global_peak_bytes: int
    backward_end_state_bytes: int


@dataclass(frozen=True)
class EstimatorDebugInfo:
    """Grouped analytical debug information for one estimate result."""

    resident_state: ResidentStateDebug
    activations: ActivationDebug
    workspace: WorkspaceDebug
    phase_peaks: PhasePeakDebug

    def flat_numeric_metadata(self) -> dict[str, int | str]:
        """Return a flat metadata mapping for comparison and reporting."""

        return {
            "parameter_bytes": self.resident_state.parameter_bytes,
            "gradient_bytes": self.resident_state.gradient_bytes,
            "optimizer_state_bytes": self.resident_state.optimizer_state_bytes,
            "master_weight_bytes": self.resident_state.master_weight_bytes,
            "runtime_support_bytes": self.resident_state.runtime_support_bytes,
            "persistent_backend_buffer_bytes": (
                self.resident_state.persistent_backend_buffer_bytes
            ),
            "trainable_parameter_bytes": self.resident_state.trainable_parameter_bytes,
            "base_hook_visible_activation_bytes": (
                self.activations.base_hook_visible_activation_bytes
            ),
            "visible_propagation_bytes": (
                self.activations.visible_propagation_bytes
            ),
            "checkpoint_resident_block_input_bytes": (
                self.activations.checkpoint_resident_block_input_bytes
            ),
            "saved_linear_input_bytes": self.activations.saved_linear_input_bytes,
            "saved_input_overlap_bytes": self.activations.saved_input_overlap_bytes,
            "mlp_intermediate_bytes": self.activations.mlp_intermediate_bytes,
            "parameter_gradient_context_bytes": (
                self.activations.parameter_gradient_context_bytes
            ),
            "residual_norm_bytes": self.activations.residual_norm_bytes,
            "checkpoint_boundary_bytes": self.activations.checkpoint_boundary_bytes,
            "attention_saved_bytes": self.activations.attention_saved_bytes,
            "loss_state_bytes": self.activations.loss_state_bytes,
            "lora_low_rank_bytes": self.activations.lora_low_rank_bytes,
            "lora_backward_logits_context_bytes": (
                self.activations.lora_backward_logits_context_bytes
            ),
            "expanded_query_saved_bytes": self.activations.expanded_query_saved_bytes,
            "query_output_context_bytes": (
                self.activations.query_output_context_bytes
            ),
            "key_output_context_bytes": self.activations.key_output_context_bytes,
            "value_output_context_bytes": (
                self.activations.value_output_context_bytes
            ),
            "output_proj_input_context_bytes": (
                self.activations.output_proj_input_context_bytes
            ),
            "output_proj_output_context_bytes": (
                self.activations.output_proj_output_context_bytes
            ),
            "checkpointed_sharded_lora_backward_visible_bytes": (
                self.activations.checkpointed_sharded_lora_backward_visible_bytes
            ),
            "retained_forward_proxy_bytes": (
                self.activations.retained_forward_proxy_bytes
            ),
            "forward_phase_activation_bytes": (
                self.activations.forward_phase_activation_bytes
            ),
            "backward_phase_activation_bytes": (
                self.activations.backward_phase_activation_bytes
            ),
            "hook_visible_activation_bytes": (
                self.activations.hook_visible_activation_bytes
            ),
            "attention_forward_workspace_bytes": (
                self.workspace.attention_forward_workspace_bytes
            ),
            "backward_kernel_workspace_bytes": (
                self.workspace.backward_kernel_workspace_bytes
            ),
            "recompute_workspace_bytes": self.workspace.recompute_workspace_bytes,
            "loss_workspace_bytes": self.workspace.loss_workspace_bytes,
            "optimizer_update_workspace_bytes": (
                self.workspace.optimizer_update_workspace_bytes
            ),
            "ddp_reducer_bucket_bytes": self.workspace.ddp_reducer_bucket_bytes,
            "ddp_comm_overlap_bytes": self.workspace.ddp_comm_overlap_bytes,
            "zero_allgather_bucket_bytes": (
                self.workspace.zero_allgather_bucket_bytes
            ),
            "zero_reduce_bucket_bytes": self.workspace.zero_reduce_bucket_bytes,
            "zero_prefetch_bucket_bytes": self.workspace.zero_prefetch_bucket_bytes,
            "zero_fetch_window_bytes": self.workspace.zero_fetch_window_bytes,
            "zero_update_window_bytes": self.workspace.zero_update_window_bytes,
            "zero_comm_window_bytes": self.workspace.zero_comm_window_bytes,
            "tensor_parallel_comm_window_bytes": (
                self.workspace.tensor_parallel_comm_window_bytes
            ),
            "sequence_parallel_comm_window_bytes": (
                self.workspace.sequence_parallel_comm_window_bytes
            ),
            "forward_peak_bytes": self.phase_peaks.forward_peak_bytes,
            "backward_peak_bytes": self.phase_peaks.backward_peak_bytes,
            "optimizer_peak_bytes": self.phase_peaks.optimizer_peak_bytes,
            "soft_global_peak_bytes": self.phase_peaks.soft_reserved_global_peak_bytes,
            "stressed_global_peak_bytes": (
                self.phase_peaks.stressed_reserved_global_peak_bytes
            ),
            "allocated_global_peak_bytes": self.phase_peaks.global_peak_bytes,
            "allocated_peak_phase": self.phase_peaks.global_peak_phase,
            "soft_peak_phase": self.phase_peaks.global_peak_phase,
            "stressed_peak_phase": self.phase_peaks.global_peak_phase,
            "backward_end_state_bytes": self.phase_peaks.backward_end_state_bytes,
        }


@dataclass(frozen=True)
class TrainingConfig:
    """Training-memory configuration for one measurement or estimate run.

    Args:
        tuning_mode: One of `full_ft` or `lora`.
        optimizer_name: Optimizer name such as `adamw`, `adam`, `sgd`,
            `rmsprop`, `adagrad`, or `adafactor`.
        weight_dtype: Model forward/backward dtype.
        grad_dtype: Gradient dtype.
        master_weight_dtype: Master-weight dtype when master weights are enabled.
        optimizer_state_dtype: Optimizer-state dtype or `auto`.
        micro_batch_size_per_gpu: Micro-batch size per rank.
        gradient_accumulation_steps: Number of accumulation steps.
        max_seq_len: Sequence length in tokens.
        gradient_checkpointing: Whether checkpointing is enabled.
        tensor_parallel_degree: Megatron-style tensor-parallel degree.
        sequence_parallel: Whether sequence-parallel activation sharding is enabled.
        vocab_parallel_logits: Whether embeddings / LM head shard with TP.
        attention_backend: Attention kernel family.
        allocator_peak_mode: Peak selector: `soft_reserved`, `stressed_reserved`,
            or `allocated`.
        allocator_stress_trigger_fraction: Fraction of GPU capacity above which
            reclaimable idle cache is dropped in the stressed allocator model.
        distributed_mode: `single_gpu`, `ddp`, or `zero2`.
        num_nodes: Number of nodes.
        gpus_per_node: GPUs per node.
        gpu_memory_gb: VRAM capacity per GPU.
        lora: Optional LoRA configuration.
        use_master_weights: Whether to include a separate master-weight copy.
        reserved_vram_gb_per_gpu: Fixed reserve for runtime overhead, or `None` for auto.
        activation_safety_margin_gb: Extra safety margin added to final peak.
        warmup_steps: Number of warmup steps before measurement.
        vision_images_per_sample: Number of images attached to each sample.
        vision_image_size: Square image size used for synthetic image inputs.

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
    optimizer_learning_rate: float = 1e-4
    optimizer_beta1: float = 0.9
    optimizer_beta2: float = 0.999
    optimizer_momentum: float = 0.0
    optimizer_alpha: float = 0.99
    optimizer_centered: bool = False
    optimizer_use_factored_state: bool = True
    optimizer_epsilon: float = 1e-8
    optimizer_weight_decay: float = 0.0
    optimizer_state_tensor_count: int | None = None
    optimizer_update_tensor_count: float | None = None
    optimizer_update_dtype: str = "auto"
    optimizer_state_in_baseline_after_warmup: bool = True
    weight_dtype: str = "bf16"
    grad_dtype: str = "bf16"
    master_weight_dtype: str = "fp32"
    optimizer_state_dtype: str = "auto"
    adapter_weight_dtype: str = "fp32"
    adapter_grad_dtype: str = "fp32"
    adapter_optimizer_state_dtype: str = "fp32"
    micro_batch_size_per_gpu: int = 1
    gradient_accumulation_steps: int = 1
    max_seq_len: int = 512
    gradient_checkpointing: bool = False
    tensor_parallel_degree: int = 1
    sequence_parallel: bool = False
    vocab_parallel_logits: bool = True
    attention_backend: str = "standard"
    allocator_peak_mode: str = "soft_reserved"
    allocator_stress_trigger_fraction: float = 1.0
    attention_score_dtype: str = "fp32"
    standard_attention_tile_size: int = 2048
    sdpa_attention_tile_size: int = 512
    flash_attention_tile_size: int = 128
    standard_attention_workspace_factor: float = 1.0
    nonstandard_attention_workspace_factor: float = 0.0
    standard_attention_activation_factor: float = 1.0
    nonstandard_attention_activation_factor: float = 1.0
    forward_to_loss_reserved_carryover_fraction: float = 1.0
    loss_to_backward_reserved_carryover_fraction: float = 1.0
    backward_to_optimizer_reserved_carryover_fraction: float = 1.0
    optimizer_to_zero_grad_reserved_carryover_fraction: float = 1.0
    zero_grad_to_step_end_reserved_carryover_fraction: float = 1.0
    default_optimizer_reserved_stack_fraction: float = 0.0
    distributed_mode: str = "single_gpu"
    num_nodes: int = 1
    gpus_per_node: int = 1
    gpu_memory_gb: float = 24.0
    lora: LoRAConfig | None = None
    vision_images_per_sample: int = 1
    vision_image_size: int = 448
    use_master_weights: bool = False
    reserved_vram_gb_per_gpu: float | None = None
    runtime_cuda_context_gb: float = 0.25
    runtime_allocator_pool_gb: float = 0.05
    runtime_nccl_gb: float = 0.0
    runtime_deepspeed_gb: float = 0.0
    runtime_cuda_context_allocated_fraction: float = 0.0
    runtime_allocator_pool_allocated_fraction: float = 0.0
    runtime_nccl_allocated_fraction: float = 0.0
    runtime_deepspeed_allocated_fraction: float = 0.0
    persistent_backend_buffer_tensor_count: float | None = None
    persistent_backend_buffer_dtype: str = "weight_dtype"
    persistent_backend_buffer_backward_overlap_fraction: float = 1.0
    persistent_backend_buffer_optimizer_overlap_fraction: float = 1.0
    ddp_bucket_elements: int = 268_435_456
    zero_bucket_elements: int = 500_000_000
    zero_prefetch_elements: int = 50_000_000
    loss_output_dtype: str = "fp32"
    loss_output_logits_fraction: float = 1.0
    single_lora_flash2_backward_logits_fraction: float = 0.5
    single_lora_sdpa_saved_input_overlap_fraction: float = 1.0
    single_lora_sdpa_saved_input_overlap_min_hidden_size: int = 4096
    single_lora_sdpa_saved_input_overlap_min_layers: int = 40
    single_full_forward_logits_copies: float = 2.0
    single_lora_forward_logits_fraction: float = 0.5
    ddp_full_forward_logits_copies: float = 2.0
    ddp_lora_forward_logits_fraction: float = 0.5
    ddp_backward_reduce_bucket_copies: float = 1.0
    zero2_full_forward_logits_copies: float = 2.0
    zero2_full_backward_grad_shard_copies: float = 2.0
    zero2_full_backward_reduce_bucket_copies: float = 1.0
    zero2_full_optimizer_master_shard_copies: float = 2.0
    zero2_full_optimizer_grad_shard_copies: float = 2.0
    zero2_full_optimizer_parameter_partition_copies: float = 1.0
    zero2_full_optimizer_allgather_bucket_copies: float = 1.0
    zero2_full_optimizer_prefetch_copies: float = 1.0
    zero2_full_optimizer_reduce_bucket_copies: float = 1.0
    zero2_full_optimizer_reserved_stack_fraction: float = 1.0
    zero2_full_checkpoint_expanded_query_overlap_fraction: float = 1.0
    zero2_lora_forward_logits_fraction: float = 0.5
    zero2_lora_backward_model_bucket_copies: float = 2.0
    zero2_lora_backward_grad_shard_copies: float = 1.0
    zero2_lora_backward_activation_fraction: float = 1.0 / 6.0
    zero2_lora_optimizer_grad_shard_copies: float = 1.0
    zero2_lora_visible_activation_extra_fraction: float = 1.0 / 6.0
    zero2_lora_optimizer_reserved_stack_fraction: float = 0.0
    zero3_full_forward_logits_copies: float = 2.0
    zero3_full_forward_allgather_bucket_copies: float = 1.0
    zero3_full_forward_parameter_partition_copies: float = 2.0
    zero3_full_backward_grad_shard_copies: float = 2.0
    zero3_full_backward_reduce_bucket_copies: float = 1.0
    zero3_full_backward_allgather_bucket_copies: float = 1.0
    zero3_full_backward_parameter_partition_copies: float = 2.0
    zero3_full_optimizer_master_shard_copies: float = 2.0
    zero3_full_optimizer_grad_shard_copies: float = 2.0
    zero3_full_optimizer_parameter_partition_copies: float = 2.0
    zero3_full_optimizer_allgather_bucket_copies: float = 1.0
    zero3_full_optimizer_prefetch_copies: float = 1.0
    zero3_full_optimizer_reduce_bucket_copies: float = 1.0
    zero3_full_optimizer_reserved_stack_fraction: float = 0.0
    zero3_full_checkpoint_expanded_query_overlap_fraction: float = 1.0
    zero3_lora_forward_logits_fraction: float = 0.5
    zero3_lora_forward_allgather_bucket_copies: float = 1.0
    zero3_lora_backward_grad_shard_copies: float = 1.0
    zero3_lora_backward_activation_fraction: float = 1.0 / 6.0
    zero3_lora_backward_allgather_bucket_copies: float = 1.0
    zero3_lora_optimizer_grad_shard_copies: float = 1.0
    zero3_lora_visible_activation_extra_fraction: float = 1.0
    zero3_lora_optimizer_reserved_stack_fraction: float = 0.0
    forward_end_allocated_transient_fraction: float = 0.0
    backward_end_allocated_transient_fraction: float = 0.0
    optimizer_step_end_allocated_transient_fraction: float = 0.0
    forward_end_reserved_transient_fraction: float = 1.0
    backward_end_reserved_transient_fraction: float = 1.0
    optimizer_step_end_reserved_transient_fraction: float = 1.0
    measurement_low_cpu_mem_usage: bool = True
    measurement_lora_task_type: str = "CAUSAL_LM"
    measurement_ddp_broadcast_buffers: bool = False
    measurement_ddp_init_sync: bool = False
    measurement_activation_module_tokens: tuple[str, ...] = ("layers.", "h.")
    measurement_attention_module_tokens: tuple[str, ...] = ("attn", "attention")
    zero_stage: int = 2
    zero_allgather_partitions: bool = True
    zero_reduce_scatter: bool = True
    zero_overlap_comm: bool = True
    zero_contiguous_gradients: bool = True
    zero_allow_untested_optimizer: bool = True
    zero_tested_optimizer_names: tuple[str, ...] = ("adam", "adamw")
    zero_untested_optimizer_state_is_sharded: bool = False
    zero_untested_optimizer_replica_tensor_count: float = 1.0
    zero_untested_optimizer_replica_dtype: str = "weight_dtype"
    zero_untested_optimizer_update_is_sharded: bool = False
    zero_untested_optimizer_update_replica_tensor_count: float = 1.0
    zero_untested_optimizer_update_dtype: str = "weight_dtype"
    zero_steps_per_print: int = 10**9
    zero_wall_clock_breakdown: bool = False
    synthetic_attention_mask_value: int = 1
    synthetic_labels_mode: str = "clone_input_ids"
    trust_remote_code: bool = False
    supported_model_types: tuple[str, ...] = field(
        default_factory=lambda: tuple(sorted(SUPPORTED_MODEL_TYPES))
    )
    default_attention_type: str = "causal"
    intermediate_size_fallback_multiplier: int = 4
    activation_safety_margin_gb: float = 0.25
    warmup_steps: int = 1

    def world_size(self) -> int:
        """Return the total number of active ranks implied by the config.

        Returns:
            Rank count used by the selected distributed mode.
        """

        if self.distributed_mode == "single_gpu":
            return 1
        return self.num_nodes * self.gpus_per_node

    def available_gpu_count(self) -> int:
        """Return the GPU count available to the runtime."""

        return max(1, self.num_nodes * self.gpus_per_node)

    def tensor_parallel_degree_resolved(self) -> int:
        """Return the validated tensor-parallel degree."""

        assert self.tensor_parallel_degree >= 1, "Tensor parallel degree must be >= 1."
        if self.distributed_mode == "single_gpu":
            assert (
                self.tensor_parallel_degree == 1
            ), "single_gpu mode cannot use tensor parallelism."
            return 1
        assert (
            self.available_gpu_count() % self.tensor_parallel_degree == 0
        ), "Tensor parallel degree must divide the available GPU count."
        return self.tensor_parallel_degree

    def data_parallel_degree(self) -> int:
        """Return the data-parallel replica count after tensor parallelism."""

        if self.distributed_mode == "single_gpu":
            return 1
        return self.available_gpu_count() // self.tensor_parallel_degree_resolved()

    def uses_tensor_parallel(self) -> bool:
        """Return whether tensor parallelism is enabled."""

        return self.tensor_parallel_degree_resolved() > 1

    def is_zero_mode(self) -> bool:
        """Return whether the config uses a ZeRO-sharded distributed mode."""

        return self.distributed_mode in {"zero2", "zero3"}

    def resolved_zero_stage(self) -> int:
        """Return the DeepSpeed ZeRO stage implied by the distributed mode."""

        if self.distributed_mode == "zero3":
            return 3
        if self.distributed_mode == "zero2":
            return 2
        return self.zero_stage

    def adapter_parameter_dtype(self) -> str:
        """Return the dtype used for LoRA adapter parameters."""

        return self.adapter_weight_dtype

    def adapter_gradient_dtype(self) -> str:
        """Return the dtype used for LoRA adapter gradients."""

        return self.adapter_grad_dtype

    def adapter_state_dtype(self) -> str:
        """Return the dtype used for LoRA optimizer-state tensors."""

        return self.adapter_optimizer_state_dtype

    def normalized_attention_backend(self) -> str:
        """Return a normalized attention-backend label."""

        return self.attention_backend.lower()

    def normalized_allocator_peak_mode(self) -> str:
        """Return the configured allocator peak mode in lowercase form."""

        return self.allocator_peak_mode.lower()

    def normalized_zero_tested_optimizer_names(self) -> tuple[str, ...]:
        """Return the normalized optimizer names treated as ZeRO-tested."""

        return tuple(
            optimizer_name.lower()
            for optimizer_name in self.zero_tested_optimizer_names
        )

    def _resolve_configured_dtype(self, dtype_name: str) -> str:
        """Resolve a configurable dtype alias to a concrete dtype name."""

        normalized_name = dtype_name.lower()
        if normalized_name == "weight_dtype":
            return self.weight_dtype
        if normalized_name == "grad_dtype":
            return self.grad_dtype
        if normalized_name == "optimizer_state_dtype":
            if self.optimizer_state_dtype != "auto":
                return self.optimizer_state_dtype
            return self.weight_dtype
        if normalized_name == "adapter_weight_dtype":
            return self.adapter_parameter_dtype()
        if normalized_name == "adapter_grad_dtype":
            return self.adapter_gradient_dtype()
        if normalized_name in {"adapter_state_dtype", "adapter_optimizer_state_dtype"}:
            return self.adapter_state_dtype()
        return dtype_name

    def zero_untested_replica_dtype(self) -> str:
        """Return the dtype used for fallback ZeRO optimizer replicas."""

        return self._resolve_configured_dtype(
            self.zero_untested_optimizer_replica_dtype
        )

    def zero_untested_update_dtype(self) -> str:
        """Return the dtype used for fallback ZeRO optimizer update replicas."""

        return self._resolve_configured_dtype(self.zero_untested_optimizer_update_dtype)

    def loss_output_resolved_dtype(self) -> str:
        """Return the dtype used for persistent model outputs."""

        return self._resolve_configured_dtype(self.loss_output_dtype)

    def persistent_backend_buffer_resolved_dtype(self) -> str:
        """Return the dtype used for persistent backend support buffers."""

        return self._resolve_configured_dtype(self.persistent_backend_buffer_dtype)

    def resolved_runtime_nccl_gb(self) -> float:
        """Return NCCL runtime support in GiB.

        Returns:
            Explicit NCCL support when configured, otherwise the default
            distributed-runtime assumption for the selected mode.
        """

        if self.runtime_nccl_gb > 0.0:
            return self.runtime_nccl_gb
        if self.distributed_mode == "single_gpu":
            return 0.0
        return 0.30

    def resolved_runtime_deepspeed_gb(self) -> float:
        """Return DeepSpeed runtime support in GiB.

        Returns:
            Explicit DeepSpeed support when configured, otherwise the default
            ZeRO-runtime assumption for the selected mode.
        """

        if self.runtime_deepspeed_gb > 0.0:
            return self.runtime_deepspeed_gb
        if self.distributed_mode == "zero2":
            return 0.20
        if self.distributed_mode == "zero3":
            return 0.40
        return 0.0

    def runtime_support_gb(self) -> float:
        """Return total runtime support in GiB.

        Returns:
            Explicit reserved support when configured, otherwise the sum of
            named runtime support components.
        """

        if self.reserved_vram_gb_per_gpu is not None:
            return self.reserved_vram_gb_per_gpu
        return (
            self.runtime_cuda_context_gb
            + self.runtime_allocator_pool_gb
            + self.resolved_runtime_nccl_gb()
            + self.resolved_runtime_deepspeed_gb()
        )

    def persistent_backend_buffer_count(self) -> float:
        """Return the number of trainable-sized backend buffers held persistently.

        Returns:
            Buffer count expressed in trainable-parameter tensor copies.
        """

        if self.persistent_backend_buffer_tensor_count is not None:
            return self.persistent_backend_buffer_tensor_count
        optimizer_name = self.optimizer_name.lower()
        if self.distributed_mode == "ddp" and self.tuning_mode == "full_ft":
            return 1.0
        if not self.is_zero_mode() or self.tuning_mode != "full_ft":
            return 0.0
        if optimizer_name == "adagrad":
            return 2.0
        return 1.0

    def lora_visible_activation_extra_fraction(self) -> float:
        """Return the retained fraction of LoRA-only visible activation extras."""

        if self.tuning_mode != "lora":
            return 0.0
        if self.distributed_mode == "zero2":
            return self.zero2_lora_visible_activation_extra_fraction
        if self.distributed_mode == "zero3":
            return self.zero3_lora_visible_activation_extra_fraction
        return 0.0

    def sharded_lora_backward_activation_fraction(self) -> float:
        """Return backward-visible retention for checkpointed ZeRO LoRA.

        Returns:
            Fraction of sharded LoRA visible propagation that survives into the
            backward peak after checkpoint rematerialization.
        """

        if self.tuning_mode != "lora" or not self.is_zero_mode():
            return 0.0
        if self.distributed_mode == "zero3":
            return self.zero3_lora_backward_activation_fraction
        return 0.0

    def optimizer_reserved_stack_fraction(self) -> float:
        """Return the reserved-pool stacking fraction used at optimizer step."""

        if self.distributed_mode == "zero2" and self.tuning_mode == "full_ft":
            return self.zero2_full_optimizer_reserved_stack_fraction
        if self.distributed_mode == "zero2" and self.tuning_mode == "lora":
            return self.zero2_lora_optimizer_reserved_stack_fraction
        if self.distributed_mode == "zero3" and self.tuning_mode == "full_ft":
            return self.zero3_full_optimizer_reserved_stack_fraction
        if self.distributed_mode == "zero3" and self.tuning_mode == "lora":
            return self.zero3_lora_optimizer_reserved_stack_fraction
        return self.default_optimizer_reserved_stack_fraction

    def checkpoint_expanded_query_overlap_fraction(self) -> float:
        """Return the retained overlap fraction for checkpointed expanded-query full FT.

        Returns:
            Fraction of the smaller visible/hidden forward object retained as a
            distinct overlap term in checkpointed sharded full fine-tuning.
        """

        if (
            not self.gradient_checkpointing
            or self.tuning_mode != "full_ft"
            or not self.is_zero_mode()
        ):
            return 0.0
        if self.distributed_mode == "zero2":
            return self.zero2_full_checkpoint_expanded_query_overlap_fraction
        if self.distributed_mode == "zero3":
            return self.zero3_full_checkpoint_expanded_query_overlap_fraction
        return 0.0

    def attention_workspace_factor(self) -> float:
        """Return the workspace multiplier for the selected attention backend."""

        if self.attention_backend == "standard":
            return self.standard_attention_workspace_factor
        return self.nonstandard_attention_workspace_factor

    def attention_activation_factor(self) -> float:
        """Return the retained-activation multiplier for the attention backend."""

        if self.attention_backend == "standard":
            return self.standard_attention_activation_factor
        return self.nonstandard_attention_activation_factor

    def reserved_carryover_fraction(
        self,
        *,
        previous_phase: str,
        next_phase: str,
    ) -> float:
        """Return the configured reserved-memory carry-over fraction.

        Args:
            previous_phase: Source phase for allocator carry-over.
            next_phase: Destination phase receiving carried reserve.

        Returns:
            Fraction of releasable bytes kept reserved across the transition.
        """

        phase_pair = (previous_phase, next_phase)
        if phase_pair == ("forward", "loss_materialization"):
            return self.forward_to_loss_reserved_carryover_fraction
        if phase_pair == ("loss_materialization", "backward"):
            return self.loss_to_backward_reserved_carryover_fraction
        if phase_pair == ("backward", "optimizer_step"):
            return self.backward_to_optimizer_reserved_carryover_fraction
        if phase_pair == ("optimizer_step", "zero_grad"):
            return self.optimizer_to_zero_grad_reserved_carryover_fraction
        if phase_pair == ("zero_grad", "step_end"):
            return self.zero_grad_to_step_end_reserved_carryover_fraction
        return 0.0

    def phase_end_allocated_transient_fraction(self, *, phase_name: str) -> float:
        """Return the local transient fraction that remains allocated at phase end."""

        if phase_name == "forward":
            return self.forward_end_allocated_transient_fraction
        if phase_name == "backward":
            return self.backward_end_allocated_transient_fraction
        if phase_name == "optimizer_step":
            return self.optimizer_step_end_allocated_transient_fraction
        return 0.0

    def phase_end_reserved_transient_fraction(self, *, phase_name: str) -> float:
        """Return the local transient fraction that remains reserved at phase end."""

        if phase_name == "forward":
            return self.forward_end_reserved_transient_fraction
        if phase_name == "backward":
            return self.backward_end_reserved_transient_fraction
        if phase_name == "optimizer_step":
            return self.optimizer_step_end_reserved_transient_fraction
        return 0.0

    def to_estimator_config(self) -> EstimatorConfig:
        """Project the richer measurement config onto the estimator surface.

        Returns:
            `EstimatorConfig` containing only structural memory inputs.
        """

        return EstimatorConfig(
            tuning_mode=self.tuning_mode,
            distributed_mode=self.distributed_mode,
            optimizer_name=self.optimizer_name,
            attention_backend=self.attention_backend,
            gradient_checkpointing=self.gradient_checkpointing,
            tensor_parallel_degree=self.tensor_parallel_degree,
            sequence_parallel=self.sequence_parallel,
            vocab_parallel_logits=self.vocab_parallel_logits,
            max_seq_len=self.max_seq_len,
            micro_batch_size_per_gpu=self.micro_batch_size_per_gpu,
            gpus_per_node=self.gpus_per_node,
            num_nodes=self.num_nodes,
            gpu_memory_gb=self.gpu_memory_gb,
            weight_dtype=self.weight_dtype,
            grad_dtype=self.grad_dtype,
            optimizer_state_dtype=self.optimizer_state_dtype,
            optimizer_update_dtype=self.optimizer_update_dtype,
            master_weight_dtype=self.master_weight_dtype,
            adapter_weight_dtype=self.adapter_weight_dtype,
            adapter_grad_dtype=self.adapter_grad_dtype,
            adapter_optimizer_state_dtype=self.adapter_optimizer_state_dtype,
            runtime_support_gb_override=self.reserved_vram_gb_per_gpu,
            ddp_bucket_elements=self.ddp_bucket_elements,
            zero_bucket_elements=self.zero_bucket_elements,
            zero_prefetch_elements=self.zero_prefetch_elements,
            loss_output_dtype=self.loss_output_dtype,
            loss_output_logits_fraction=self.loss_output_logits_fraction,
            single_lora_flash2_backward_logits_fraction=(
                self.single_lora_flash2_backward_logits_fraction
            ),
            single_lora_sdpa_saved_input_overlap_fraction=(
                self.single_lora_sdpa_saved_input_overlap_fraction
            ),
            single_lora_sdpa_saved_input_overlap_min_hidden_size=(
                self.single_lora_sdpa_saved_input_overlap_min_hidden_size
            ),
            single_lora_sdpa_saved_input_overlap_min_layers=(
                self.single_lora_sdpa_saved_input_overlap_min_layers
            ),
            zero2_lora_backward_activation_fraction=(
                self.zero2_lora_backward_activation_fraction
            ),
            zero3_lora_backward_activation_fraction=(
                self.zero3_lora_backward_activation_fraction
            ),
            use_master_weights=self.use_master_weights,
            lora=self.lora,
            persistent_backend_buffer_tensor_count=(
                self.persistent_backend_buffer_tensor_count
            ),
            persistent_backend_buffer_dtype=self.persistent_backend_buffer_dtype,
            zero_tested_optimizer_names=self.zero_tested_optimizer_names,
            zero_untested_optimizer_state_is_sharded=(
                self.zero_untested_optimizer_state_is_sharded
            ),
            zero_untested_optimizer_replica_tensor_count=(
                self.zero_untested_optimizer_replica_tensor_count
            ),
            zero_untested_optimizer_replica_dtype=(
                self.zero_untested_optimizer_replica_dtype
            ),
            zero_untested_optimizer_update_is_sharded=(
                self.zero_untested_optimizer_update_is_sharded
            ),
            zero_untested_optimizer_update_replica_tensor_count=(
                self.zero_untested_optimizer_update_replica_tensor_count
            ),
            zero_untested_optimizer_update_dtype=(
                self.zero_untested_optimizer_update_dtype
            ),
        )


MeasurementConfig = TrainingConfig


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
        config: Estimator or measurement configuration used.
        breakdown: Modular memory breakdown.
        phase_records: Phase-wise memory timeline.
        peak_phase: Phase with highest peak.
        global_peak_bytes: Peak memory per rank in bytes.
        feasible: Whether the config fits in available GPU memory.
        metadata: Extra runtime metadata retained for reporting and comparisons.
        debug: Optional typed debug payload for analytical estimates.
        assumptions: Human-readable assumptions and warnings.

    Returns:
        Frozen result dataclass.
    """

    mode: str
    model_name: str
    config: EstimatorConfig | MeasurementConfig
    breakdown: MemoryComponentBreakdown
    phase_records: tuple[PhaseMemoryRecord, ...]
    peak_phase: str
    global_peak_bytes: int
    feasible: bool
    metadata: dict[str, Any] = field(default_factory=dict)
    debug: EstimatorDebugInfo | None = None
    assumptions: tuple[str, ...] = ()

    def global_peak_gb(self) -> float:
        """Return the peak memory per rank in GiB."""

        return bytes_to_gb(self.global_peak_bytes)

    def headroom_gb(self) -> float:
        """Return per-rank VRAM headroom in GiB."""

        return self.config.gpu_memory_gb - self.global_peak_gb()

    def comparable_metadata(self) -> dict[str, Any]:
        """Return a flat metadata mapping used by comparison/reporting helpers."""

        if self.debug is None:
            return dict(self.metadata)
        merged_metadata = self.debug.flat_numeric_metadata()
        merged_metadata.update(self.metadata)
        return merged_metadata


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
        phase_aligned_component_error_bytes: Error by component name after
            projecting the estimate into the measured peak phase. This excludes
            retained-forward activation proxies, which are reported separately.
        retained_forward_proxy_error_bytes: Error for the retained-forward
            activation proxy compared independently from phase-local activation.
        global_peak_relative_error: Relative peak-memory error.
        phase_peak_relative_error: Relative error by phase name.
        component_relative_error: Relative error by component name.
        phase_aligned_component_relative_error: Relative error by component
            name after measured-phase projection.
        retained_forward_proxy_relative_error: Relative error for the
            retained-forward activation proxy.
        workspace_proxy_error_bytes: Error by workspace-proxy name.
        workspace_proxy_relative_error: Relative error by workspace-proxy name.
        intermediate_term_error_bytes: Error by non-additive metadata term name.
        intermediate_term_relative_error: Relative error by non-additive metadata term.
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
    phase_aligned_component_error_bytes: dict[str, int] = field(default_factory=dict)
    phase_aligned_component_relative_error: dict[str, float] = field(
        default_factory=dict
    )
    retained_forward_proxy_error_bytes: int = 0
    retained_forward_proxy_relative_error: float = 0.0
    workspace_proxy_error_bytes: dict[str, int] = field(default_factory=dict)
    workspace_proxy_relative_error: dict[str, float] = field(default_factory=dict)
    intermediate_term_error_bytes: dict[str, int] = field(default_factory=dict)
    intermediate_term_relative_error: dict[str, float] = field(default_factory=dict)
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
        config: Measurement config for the case.
        tags: Optional grouping tags.

    Returns:
        Frozen benchmark case dataclass.
    """

    name: str
    model: str | ModelSpec
    config: MeasurementConfig
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
