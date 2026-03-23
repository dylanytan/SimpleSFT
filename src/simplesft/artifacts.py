"""Artifact persistence for memory results, comparisons, and benchmark runs."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .types import (
    BenchmarkCase,
    BenchmarkCaseResult,
    BenchmarkSuiteResult,
    ComparisonResult,
    LoRAConfig,
    MemoryComponentBreakdown,
    MemoryResult,
    ModelLinearLayerSpec,
    ModelSpec,
    PhaseMemoryRecord,
    SearchResult,
    TrainingConfig,
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


def save_benchmark_suite_result(*, result: BenchmarkSuiteResult, path: str | Path) -> None:
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


def _load_training_config(raw: dict[str, Any]) -> TrainingConfig:
    """Deserialize a `TrainingConfig` payload."""

    return TrainingConfig(
        tuning_mode=raw["tuning_mode"],
        optimizer_name=raw["optimizer_name"],
        weight_dtype=raw["weight_dtype"],
        grad_dtype=raw["grad_dtype"],
        master_weight_dtype=raw["master_weight_dtype"],
        optimizer_state_dtype=raw["optimizer_state_dtype"],
        micro_batch_size_per_gpu=raw["micro_batch_size_per_gpu"],
        gradient_accumulation_steps=raw["gradient_accumulation_steps"],
        max_seq_len=raw["max_seq_len"],
        gradient_checkpointing=raw["gradient_checkpointing"],
        attention_backend=raw["attention_backend"],
        distributed_mode=raw["distributed_mode"],
        num_nodes=raw["num_nodes"],
        gpus_per_node=raw["gpus_per_node"],
        gpu_memory_gb=raw["gpu_memory_gb"],
        lora=_load_lora_config(raw=raw["lora"]),
        use_master_weights=raw["use_master_weights"],
        reserved_vram_gb_per_gpu=raw["reserved_vram_gb_per_gpu"],
        activation_safety_margin_gb=raw["activation_safety_margin_gb"],
        warmup_steps=raw["warmup_steps"],
    )


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


def load_memory_result_from_raw(*, raw: dict[str, Any]) -> MemoryResult:
    """Load a `MemoryResult` from a raw dictionary."""

    return MemoryResult(
        mode=raw["mode"],
        model_name=raw["model_name"],
        config=_load_training_config(raw=raw["config"]),
        breakdown=_load_breakdown(raw=raw["breakdown"]),
        phase_records=tuple(_load_phase_record(raw=item) for item in raw["phase_records"]),
        peak_phase=raw["peak_phase"],
        global_peak_bytes=raw["global_peak_bytes"],
        feasible=raw["feasible"],
        metadata=raw["metadata"],
        assumptions=tuple(raw["assumptions"]),
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
        workspace_proxy_error_bytes=raw.get("workspace_proxy_error_bytes", {}),
        workspace_proxy_relative_error=raw.get("workspace_proxy_relative_error", {}),
        benchmark_metadata=raw["benchmark_metadata"],
        notes=tuple(raw["notes"]),
    )


def _load_model_linear_layer(raw: dict[str, Any]) -> ModelLinearLayerSpec:
    """Deserialize a linear-layer summary payload."""

    return ModelLinearLayerSpec(
        module_name=raw["module_name"],
        input_dim=raw["input_dim"],
        output_dim=raw["output_dim"],
        category=raw["category"],
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
        attention_type=raw["attention_type"],
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
        config=_load_training_config(raw=raw["config"]),
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
