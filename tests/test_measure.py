"""Tests for measurement-path behavior that do not require CUDA."""

from dataclasses import dataclass

import pytest

import simplesft.measurement.distributed_zero2 as zero2_measure_module
import simplesft.measurement.measure as measure_module
import torch
from simplesft.measurement.measure import measure_peak_memory
from simplesft.types import MemoryComponentBreakdown, MemoryResult, PhaseMemoryRecord
from simplesft.types import ModelSpec, TrainingConfig


def _toy_model_spec() -> ModelSpec:
    """Return a compact model spec for measurement tests."""

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
        trainable_linear_layers=(),
    )


@dataclass
class _DummyConfig:
    """Typed config object used by unit-test doubles."""

    use_cache: bool = True


def test_measure_peak_memory_requires_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    """Measurement should fail clearly when CUDA is unavailable."""

    monkeypatch.setattr(measure_module, "is_cuda_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is required"):
        measure_peak_memory(
            model=_toy_model_spec(),
            config=TrainingConfig(tuning_mode="full_ft"),
        )


def test_measure_peak_memory_rejects_tp_and_sp_runtime_requests() -> None:
    """Measurement should fail clearly for unsupported TP/SP runtime shapes."""

    with pytest.raises(RuntimeError, match="Measurement does not yet support"):
        measure_peak_memory(
            model=_toy_model_spec(),
            config=TrainingConfig(
                tuning_mode="full_ft",
                distributed_mode="ddp",
                gpus_per_node=2,
                tensor_parallel_degree=2,
            ),
        )
    with pytest.raises(RuntimeError, match="Measurement does not yet support"):
        measure_peak_memory(
            model=_toy_model_spec(),
            config=TrainingConfig(
                tuning_mode="full_ft",
                distributed_mode="ddp",
                gpus_per_node=2,
                tensor_parallel_degree=2,
                sequence_parallel=True,
            ),
        )


def test_measure_peak_memory_zero2_requires_optional_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ZeRO-2 measurement should fail clearly without its runtime stack."""

    monkeypatch.setattr(
        measure_module,
        "maybe_get_deepspeed",
        lambda: (_ for _ in ()).throw(RuntimeError("missing deepspeed")),
    )
    with pytest.raises(RuntimeError):
        measure_peak_memory(
            model=_toy_model_spec(),
            config=TrainingConfig(tuning_mode="full_ft", distributed_mode="zero2"),
        )


def test_measure_peak_memory_delegates_zero2_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Top-level measurement should delegate multi-rank ZeRO-2 runs."""

    expected_result = MemoryResult(
        mode="measure",
        model_name="toy",
        config=TrainingConfig(
            tuning_mode="full_ft",
            distributed_mode="zero2",
            gpus_per_node=2,
            gpu_memory_gb=80.0,
        ),
        breakdown=MemoryComponentBreakdown(parameter_bytes=1),
        phase_records=(
            PhaseMemoryRecord(
                phase_name="optimizer_step",
                allocated_bytes=12,
                reserved_bytes=12,
                peak_allocated_bytes=12,
                peak_reserved_bytes=12,
                delta_allocated_bytes=12,
                delta_reserved_bytes=12,
            ),
        ),
        peak_phase="optimizer_step",
        global_peak_bytes=12,
        feasible=True,
    )
    monkeypatch.setattr(measure_module, "is_cuda_available", lambda: True)
    monkeypatch.setattr(measure_module, "maybe_get_deepspeed", lambda: object())
    monkeypatch.setattr(
        zero2_measure_module,
        "run_zero2_measurement",
        lambda model, config: expected_result,
    )
    result = measure_peak_memory(
        model="toy-model",
        config=TrainingConfig(
            tuning_mode="full_ft",
            distributed_mode="zero2",
            gpus_per_node=2,
        ),
    )
    assert result.global_peak_bytes == expected_result.global_peak_bytes


def test_measure_peak_memory_delegates_zero3_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Top-level measurement should delegate multi-rank ZeRO-3 runs."""

    expected_result = MemoryResult(
        mode="measure",
        model_name="toy",
        config=TrainingConfig(
            tuning_mode="full_ft",
            distributed_mode="zero3",
            gpus_per_node=2,
            gpu_memory_gb=80.0,
        ),
        breakdown=MemoryComponentBreakdown(parameter_bytes=1),
        phase_records=(
            PhaseMemoryRecord(
                phase_name="optimizer_step",
                allocated_bytes=12,
                reserved_bytes=12,
                peak_allocated_bytes=12,
                peak_reserved_bytes=12,
                delta_allocated_bytes=12,
                delta_reserved_bytes=12,
            ),
        ),
        peak_phase="optimizer_step",
        global_peak_bytes=12,
        feasible=True,
    )
    monkeypatch.setattr(measure_module, "is_cuda_available", lambda: True)
    monkeypatch.setattr(measure_module, "maybe_get_deepspeed", lambda: object())
    monkeypatch.setattr(
        zero2_measure_module,
        "run_zero2_measurement",
        lambda model, config: expected_result,
    )
    result = measure_peak_memory(
        model="toy-model",
        config=TrainingConfig(
            tuning_mode="full_ft",
            distributed_mode="zero3",
            gpus_per_node=2,
        ),
    )
    assert result.global_peak_bytes == expected_result.global_peak_bytes


def test_cuda_snapshot_synchronizes_before_reading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CUDA memory snapshots should synchronize before reading counters."""

    call_order: list[str] = []
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda *, device: call_order.append(f"sync:{device}"),
    )
    monkeypatch.setattr(
        torch.cuda,
        "memory_allocated",
        lambda *, device: call_order.append(f"allocated:{device}") or 11,
    )
    monkeypatch.setattr(
        torch.cuda,
        "memory_reserved",
        lambda *, device: call_order.append(f"reserved:{device}") or 13,
    )
    allocated, reserved = measure_module._cuda_snapshot(device=torch.device("cuda", 0))
    assert (allocated, reserved) == (11, 13)
    assert call_order == [
        "sync:cuda:0",
        "allocated:cuda:0",
        "reserved:cuda:0",
    ]


def test_build_memory_result_honors_allocator_peak_mode() -> None:
    """Measured peak selection should respect allocated vs reserved mode."""

    phase_records = [
        PhaseMemoryRecord(
            phase_name="batch_materialization",
            allocated_bytes=0,
            reserved_bytes=20,
            peak_allocated_bytes=0,
            peak_reserved_bytes=20,
            delta_allocated_bytes=0,
            delta_reserved_bytes=20,
        ),
        PhaseMemoryRecord(
            phase_name="forward",
            allocated_bytes=80,
            reserved_bytes=120,
            peak_allocated_bytes=90,
            peak_reserved_bytes=120,
            delta_allocated_bytes=80,
            delta_reserved_bytes=120,
        ),
        PhaseMemoryRecord(
            phase_name="backward",
            allocated_bytes=95,
            reserved_bytes=110,
            peak_allocated_bytes=100,
            peak_reserved_bytes=110,
            delta_allocated_bytes=15,
            delta_reserved_bytes=-10,
        ),
        PhaseMemoryRecord(
            phase_name="optimizer_step",
            allocated_bytes=70,
            reserved_bytes=105,
            peak_allocated_bytes=75,
            peak_reserved_bytes=105,
            delta_allocated_bytes=-25,
            delta_reserved_bytes=-5,
        ),
    ]
    breakdown = MemoryComponentBreakdown(runtime_reserve_bytes=10)
    allocated_result = measure_module._build_memory_result(
        model_spec=_toy_model_spec(),
        config=TrainingConfig(
            tuning_mode="full_ft",
            allocator_peak_mode="allocated",
        ),
        phase_records=phase_records,
        breakdown=breakdown,
        activation_metadata={},
        state_snapshots={},
    )
    soft_reserved_result = measure_module._build_memory_result(
        model_spec=_toy_model_spec(),
        config=TrainingConfig(
            tuning_mode="full_ft",
            allocator_peak_mode="soft_reserved",
        ),
        phase_records=phase_records,
        breakdown=breakdown,
        activation_metadata={},
        state_snapshots={},
    )
    assert allocated_result.peak_phase == "backward"
    assert allocated_result.global_peak_bytes == 100
    assert soft_reserved_result.peak_phase == "forward"
    assert soft_reserved_result.global_peak_bytes == 120


def test_configure_model_for_measurement_enables_lora_checkpoint_input_grads() -> None:
    """Checkpointed LoRA setup should enable input gradients when supported."""

    class DummyLoRAModel(torch.nn.Module):
        """Minimal model exposing the HF checkpointing toggles."""

        def __init__(self) -> None:
            super().__init__()
            self.config = _DummyConfig()
            self._checkpointing_enabled = False
            self.input_grads_enabled = False

        @property
        def is_gradient_checkpointing(self) -> bool:
            return self._checkpointing_enabled

        def gradient_checkpointing_enable(self) -> None:
            self._checkpointing_enabled = True

        def enable_input_require_grads(self) -> None:
            self.input_grads_enabled = True

    model = DummyLoRAModel()
    measure_module._configure_model_for_measurement(
        model=model,
        config=TrainingConfig(tuning_mode="lora", gradient_checkpointing=True),
    )
    metadata = measure_module._runtime_checkpointing_metadata(model=model)
    assert model.training
    assert model.is_gradient_checkpointing
    assert model.input_grads_enabled
    assert model.config.use_cache is False
    assert metadata["runtime_checkpoint_input_grads_enabled"] is True


def test_configure_model_for_measurement_hooks_embeddings_when_lora_lacks_helper() -> None:
    """Fallback hook should mark embedding outputs trainable for checkpointing."""

    class DummyFallbackLoRAModel(torch.nn.Module):
        """LoRA-like module with frozen embeddings and no helper method."""

        def __init__(self) -> None:
            super().__init__()
            self.config = _DummyConfig()
            self.embedding = torch.nn.Embedding(num_embeddings=16, embedding_dim=8)
            self.embedding.weight.requires_grad_(False)
            self._checkpointing_enabled = False

        @property
        def is_gradient_checkpointing(self) -> bool:
            return self._checkpointing_enabled

        def gradient_checkpointing_enable(self) -> None:
            self._checkpointing_enabled = True

        def get_input_embeddings(self) -> torch.nn.Embedding:
            return self.embedding

    model = DummyFallbackLoRAModel()
    baseline = model.get_input_embeddings()(torch.tensor([1, 2], dtype=torch.long))
    assert baseline.requires_grad is False
    measure_module._configure_model_for_measurement(
        model=model,
        config=TrainingConfig(tuning_mode="lora", gradient_checkpointing=True),
    )
    hooked_output = model.get_input_embeddings()(torch.tensor([1, 2], dtype=torch.long))
    metadata = measure_module._runtime_checkpointing_metadata(model=model)
    assert model.training
    assert model.is_gradient_checkpointing
    assert model.config.use_cache is False
    assert hooked_output.requires_grad is True
    assert metadata["runtime_checkpoint_input_grads_enabled"] is True
