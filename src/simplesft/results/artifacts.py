"""Artifact persistence for memory results, comparisons, and benchmark runs."""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from ..types import (
    BenchmarkCase,
    BenchmarkCaseResult,
    BenchmarkSuiteResult,
    ComparisonResult,
    EstimatorConfig,
    EstimatorDebugInfo,
    ResidentStateDebug,
    ActivationDebug,
    WorkspaceDebug,
    PhasePeakDebug,
    LoRAConfig,
    MeasurementConfig,
    MemoryComponentBreakdown,
    MemoryResult,
    ModelLinearLayerSpec,
    ModelParameterSpec,
    ModelSpec,
    PhaseMemoryRecord,
    SearchResult,
    VisionSpec,
)
from ..models.architecture_types import (
    ArchitectureFamilySpec,
    AttentionSpec,
    TensorLayoutSpec,
)


def _ensure_parent_dir(path: Path) -> None:
    """Create the parent directory for an artifact path."""

    path.parent.mkdir(parents=True, exist_ok=True)


def write_json_artifact(*, path: str | Path, payload: dict[str, Any]) -> None:
    """Write a JSON artifact to disk."""

    artifact_path = Path(path)
    _ensure_parent_dir(artifact_path)
    artifact_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_text_artifact(*, path: str | Path, content: str) -> None:
    """Write a plain-text artifact to disk."""

    artifact_path = Path(path)
    _ensure_parent_dir(artifact_path)
    artifact_path.write_text(content, encoding="utf-8")


def save_memory_result(*, result: MemoryResult, path: str | Path) -> None:
    """Serialize a `MemoryResult` to JSON."""

    write_json_artifact(path=path, payload=asdict(result))


def save_comparison_result(*, result: ComparisonResult, path: str | Path) -> None:
    """Serialize a `ComparisonResult` to JSON."""

    write_json_artifact(path=path, payload=asdict(result))


def save_benchmark_suite_result(
    *, result: BenchmarkSuiteResult, path: str | Path
) -> None:
    """Serialize a `BenchmarkSuiteResult` to JSON."""

    write_json_artifact(path=path, payload=asdict(result))


def save_search_result(*, result: SearchResult, path: str | Path) -> None:
    """Serialize a `SearchResult` to JSON."""

    write_json_artifact(path=path, payload=asdict(result))


def _load_lora_config(raw: dict[str, Any] | None) -> LoRAConfig | None:
    """Deserialize a LoRA config payload."""

    if raw is None:
        return None
    return LoRAConfig(
        rank=raw["rank"],
        alpha=raw["alpha"],
        dropout=raw["dropout"],
        target_modules=tuple(raw["target_modules"]),
        bias=raw["bias"],
    )


def _load_estimator_config(raw: dict[str, Any]) -> EstimatorConfig:
    """Deserialize an `EstimatorConfig` payload."""

    payload = {
        key: value
        for key, value in raw.items()
        if key in {field_info.name for field_info in fields(EstimatorConfig)}
    }
    if "lora" in payload:
        payload["lora"] = _load_lora_config(raw=payload["lora"])
    return EstimatorConfig(**payload)


def _load_measurement_config(raw: dict[str, Any]) -> MeasurementConfig:
    """Deserialize a `MeasurementConfig` payload."""

    payload = {
        key: value
        for key, value in raw.items()
        if key in {field_info.name for field_info in fields(MeasurementConfig)}
    }
    if "lora" in payload:
        payload["lora"] = _load_lora_config(raw=payload["lora"])
    return MeasurementConfig(**payload)


def _load_breakdown(raw: dict[str, Any]) -> MemoryComponentBreakdown:
    """Deserialize a memory-breakdown payload."""

    return MemoryComponentBreakdown(**raw)


def _load_phase_record(raw: dict[str, Any]) -> PhaseMemoryRecord:
    """Deserialize a phase-memory record payload."""

    return PhaseMemoryRecord(
        phase_name=raw["phase_name"],
        allocated_bytes=raw["allocated_bytes"],
        reserved_bytes=raw["reserved_bytes"],
        peak_allocated_bytes=raw["peak_allocated_bytes"],
        peak_reserved_bytes=raw["peak_reserved_bytes"],
        delta_allocated_bytes=raw["delta_allocated_bytes"],
        delta_reserved_bytes=raw["delta_reserved_bytes"],
        notes=tuple(raw["notes"]),
    )


def _filtered_payload(
    *,
    payload: dict[str, Any],
    allowed_keys: set[str],
) -> dict[str, Any]:
    """Return one payload filtered down to dataclass-supported keys.

    Args:
        payload: Raw artifact payload that may contain legacy keys.
        allowed_keys: Dataclass field names supported by the current loader.

    Returns:
        Copy of the payload containing only recognized keys.
    """

    return {key: value for key, value in payload.items() if key in allowed_keys}


def _load_estimator_debug(raw: dict[str, Any] | None) -> EstimatorDebugInfo | None:
    """Deserialize the estimator debug payload when present."""

    if raw is None:
        return None
    resident_state_keys = {field_info.name for field_info in fields(ResidentStateDebug)}
    activation_keys = {field_info.name for field_info in fields(ActivationDebug)}
    workspace_keys = {field_info.name for field_info in fields(WorkspaceDebug)}
    phase_peak_keys = {field_info.name for field_info in fields(PhasePeakDebug)}
    resident_state_payload = {
        "parameter_bytes": 0,
        "gradient_bytes": 0,
        "optimizer_state_bytes": 0,
        "master_weight_bytes": 0,
        "runtime_support_bytes": 0,
        "persistent_backend_buffer_bytes": 0,
        "trainable_parameter_bytes": 0,
    }
    resident_state_payload.update(
        _filtered_payload(
            payload=raw["resident_state"],
            allowed_keys=resident_state_keys,
        )
    )
    workspace_payload = {
        "zero_allgather_bucket_bytes": 0,
        "zero_reduce_bucket_bytes": 0,
        "zero_prefetch_bucket_bytes": 0,
    }
    workspace_payload.update(
        _filtered_payload(
            payload=raw["workspace"],
            allowed_keys=workspace_keys,
        )
    )
    activation_payload = {
        "base_hook_visible_activation_bytes": 0,
        "visible_propagation_bytes": 0,
        "checkpoint_resident_block_input_bytes": 0,
        "saved_linear_input_bytes": 0,
        "saved_input_overlap_bytes": 0,
        "mlp_intermediate_bytes": 0,
        "parameter_gradient_context_bytes": 0,
        "residual_norm_bytes": 0,
        "checkpoint_boundary_bytes": 0,
        "attention_saved_bytes": 0,
        "loss_state_bytes": 0,
        "lora_low_rank_bytes": 0,
        "lora_backward_logits_context_bytes": 0,
        "expanded_query_saved_bytes": 0,
        "query_output_context_bytes": 0,
        "key_output_context_bytes": 0,
        "value_output_context_bytes": 0,
        "output_proj_input_context_bytes": 0,
        "output_proj_output_context_bytes": 0,
        "retained_forward_proxy_bytes": 0,
        "forward_phase_activation_bytes": 0,
        "backward_phase_activation_bytes": 0,
        "hook_visible_activation_bytes": 0,
    }
    activation_payload.update(
        _filtered_payload(
            payload=raw["activations"],
            allowed_keys=activation_keys,
        )
    )
    return EstimatorDebugInfo(
        resident_state=ResidentStateDebug(**resident_state_payload),
        activations=ActivationDebug(**activation_payload),
        workspace=WorkspaceDebug(**workspace_payload),
        phase_peaks=PhasePeakDebug(
            **_filtered_payload(
                payload=raw["phase_peaks"],
                allowed_keys=phase_peak_keys,
            )
        ),
    )


def load_memory_result_from_raw(*, raw: dict[str, Any]) -> MemoryResult:
    """Load a `MemoryResult` from a raw dictionary."""

    config_loader = _load_estimator_config
    if raw["mode"] == "measure":
        config_loader = _load_measurement_config
    return MemoryResult(
        mode=raw["mode"],
        model_name=raw["model_name"],
        config=config_loader(raw=raw["config"]),
        breakdown=_load_breakdown(raw=raw["breakdown"]),
        phase_records=tuple(
            _load_phase_record(raw=item) for item in raw["phase_records"]
        ),
        peak_phase=raw["peak_phase"],
        global_peak_bytes=raw["global_peak_bytes"],
        feasible=raw["feasible"],
        metadata=raw.get("metadata", {}),
        debug=_load_estimator_debug(raw=raw.get("debug")),
        assumptions=tuple(raw.get("assumptions", ())),
    )


def load_memory_result(*, path: str | Path) -> MemoryResult:
    """Load a `MemoryResult` artifact from JSON."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return load_memory_result_from_raw(raw=raw)


def load_comparison_result(*, path: str | Path) -> ComparisonResult:
    """Load a `ComparisonResult` artifact from JSON."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return ComparisonResult(
        model_name=raw["model_name"],
        measured=load_memory_result_from_raw(raw=raw["measured"]),
        estimated=load_memory_result_from_raw(raw=raw["estimated"]),
        global_peak_error_bytes=raw["global_peak_error_bytes"],
        global_peak_relative_error=raw["global_peak_relative_error"],
        phase_peak_error_bytes=raw["phase_peak_error_bytes"],
        phase_peak_relative_error=raw["phase_peak_relative_error"],
        component_error_bytes=raw["component_error_bytes"],
        component_relative_error=raw["component_relative_error"],
        phase_aligned_component_error_bytes=raw.get(
            "phase_aligned_component_error_bytes",
            {},
        ),
        phase_aligned_component_relative_error=raw.get(
            "phase_aligned_component_relative_error",
            {},
        ),
        retained_forward_proxy_error_bytes=raw.get(
            "retained_forward_proxy_error_bytes",
            0,
        ),
        retained_forward_proxy_relative_error=raw.get(
            "retained_forward_proxy_relative_error",
            0.0,
        ),
        workspace_proxy_error_bytes=raw.get("workspace_proxy_error_bytes", {}),
        workspace_proxy_relative_error=raw.get("workspace_proxy_relative_error", {}),
        intermediate_term_error_bytes=raw.get("intermediate_term_error_bytes", {}),
        intermediate_term_relative_error=raw.get(
            "intermediate_term_relative_error",
            {},
        ),
        benchmark_metadata=raw.get("benchmark_metadata", {}),
        notes=tuple(raw.get("notes", ())),
    )


def _load_model_linear_layer(raw: dict[str, Any]) -> ModelLinearLayerSpec:
    """Deserialize a linear-layer summary payload."""

    return ModelLinearLayerSpec(
        module_name=raw["module_name"],
        input_dim=raw["input_dim"],
        output_dim=raw["output_dim"],
        category=raw["category"],
        role=raw.get("role", "other"),
        tensor_parallel_role=raw.get("tensor_parallel_role", "replicated"),
    )


def _load_model_spec(raw: dict[str, Any]) -> ModelSpec:
    """Deserialize a model-spec payload."""

    return ModelSpec(
        model_name=raw["model_name"],
        model_type=raw["model_type"],
        num_layers=raw["num_layers"],
        hidden_size=raw["hidden_size"],
        num_attention_heads=raw["num_attention_heads"],
        intermediate_size=raw["intermediate_size"],
        vocab_size=raw["vocab_size"],
        max_position_embeddings=raw["max_position_embeddings"],
        total_params=raw["total_params"],
        trainable_linear_layers=tuple(
            _load_model_linear_layer(item) for item in raw["trainable_linear_layers"]
        ),
        parameter_specs=tuple(
            _load_model_parameter(item) for item in raw.get("parameter_specs", ())
        ),
        attention_type=raw["attention_type"],
        architecture_family=_load_architecture_family(
            raw=raw.get("architecture_family")
        ),
        attention=_load_attention_spec(raw=raw.get("attention")),
        tensor_layout=_load_tensor_layout(raw=raw.get("tensor_layout")),
        vision=_load_vision_spec(raw=raw.get("vision")),
    )


def model_spec_to_raw(*, model_spec: ModelSpec) -> dict[str, Any]:
    """Serialize one `ModelSpec` into a JSON-friendly dictionary.

    Args:
        model_spec: Structured model specification to serialize.

    Returns:
        Plain dictionary suitable for JSON persistence.

    Example:
        >>> raw = model_spec_to_raw(model_spec=ModelSpec(
        ...     model_name="toy",
        ...     model_type="llama",
        ...     num_layers=1,
        ...     hidden_size=8,
        ...     num_attention_heads=1,
        ...     intermediate_size=16,
        ...     vocab_size=32,
        ...     max_position_embeddings=64,
        ...     total_params=128,
        ...     trainable_linear_layers=(),
        ... ))
        >>> raw["model_name"]
        'toy'
    """

    return asdict(model_spec)


def load_model_spec_from_raw(*, raw: dict[str, Any]) -> ModelSpec:
    """Deserialize one `ModelSpec` from a raw JSON dictionary.

    Args:
        raw: JSON-loaded model-spec payload.

    Returns:
        Structured `ModelSpec`.
    """

    return _load_model_spec(raw)


def _load_model_parameter(raw: dict[str, Any]) -> ModelParameterSpec:
    """Deserialize a parameter-summary payload."""

    return ModelParameterSpec(
        parameter_name=raw["parameter_name"],
        shape=tuple(raw["shape"]),
        category=raw["category"],
        role=raw.get("role", "other"),
        tensor_parallel_role=raw.get("tensor_parallel_role", "replicated"),
    )


def _load_architecture_family(
    raw: dict[str, Any] | None,
) -> ArchitectureFamilySpec:
    """Deserialize an architecture-family payload or return a default."""

    if raw is None:
        return ArchitectureFamilySpec()
    return ArchitectureFamilySpec(**raw)


def _load_attention_spec(raw: dict[str, Any] | None) -> AttentionSpec:
    """Deserialize an attention-spec payload or return a default."""

    if raw is None:
        return AttentionSpec()
    return AttentionSpec(**raw)


def _load_tensor_layout(raw: dict[str, Any] | None) -> TensorLayoutSpec:
    """Deserialize a tensor-layout payload or return a default."""

    if raw is None:
        return TensorLayoutSpec()
    return TensorLayoutSpec(**raw)


def _load_vision_spec(raw: dict[str, Any] | None) -> VisionSpec | None:
    """Deserialize a vision-spec payload."""

    if raw is None:
        return None
    return VisionSpec(
        default_image_size=raw["default_image_size"],
        patch_size=raw["patch_size"],
        temporal_patch_size=raw["temporal_patch_size"],
        spatial_merge_size=raw["spatial_merge_size"],
        channels=raw.get("channels", 3),
        image_token_id=raw.get("image_token_id"),
        vision_start_token_id=raw.get("vision_start_token_id"),
        vision_end_token_id=raw.get("vision_end_token_id"),
    )


def _load_benchmark_case(raw: dict[str, Any]) -> BenchmarkCase:
    """Deserialize a benchmark case payload."""

    model_payload = raw["model"]
    model: str | ModelSpec = model_payload
    if isinstance(model_payload, dict):
        model = _load_model_spec(raw=model_payload)
    return BenchmarkCase(
        name=raw["name"],
        model=model,
        config=_load_measurement_config(raw=raw["config"]),
        tags=tuple(raw["tags"]),
    )


def _load_benchmark_case_result(raw: dict[str, Any]) -> BenchmarkCaseResult:
    """Deserialize a benchmark case-result payload."""

    return BenchmarkCaseResult(
        case=_load_benchmark_case(raw=raw["case"]),
        estimate_path=raw["estimate_path"],
        measurement_path=raw["measurement_path"],
        comparison_path=raw["comparison_path"],
        error_message=raw["error_message"],
    )


def load_benchmark_suite_result(*, path: str | Path) -> BenchmarkSuiteResult:
    """Load a `BenchmarkSuiteResult` artifact from JSON."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return BenchmarkSuiteResult(
        output_dir=raw["output_dir"],
        case_results=tuple(
            _load_benchmark_case_result(item) for item in raw["case_results"]
        ),
        notes=tuple(raw["notes"]),
    )


def find_comparison_artifacts(*, root_dir: str | Path) -> list[Path]:
    """Return all comparison JSON artifacts under a root directory."""

    return sorted(Path(root_dir).glob("**/comparison.json"))
