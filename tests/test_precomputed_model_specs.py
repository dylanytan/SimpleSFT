"""Tests for precomputed model-spec snapshots."""

from __future__ import annotations

from pathlib import Path

from simplesft.models.precomputed_model_specs import (
    PrecomputedModelSpecSnapshot,
    load_precomputed_model_spec_snapshot,
    resolve_model_spec,
    save_precomputed_model_spec_snapshot,
)
from simplesft.types import ModelLinearLayerSpec, ModelSpec


def _toy_model_spec(*, model_name: str) -> ModelSpec:
    """Return a compact model spec for snapshot tests."""

    return ModelSpec(
        model_name=model_name,
        model_type="llama",
        num_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        intermediate_size=64,
        vocab_size=128,
        max_position_embeddings=128,
        total_params=10_000,
        trainable_linear_layers=(
            ModelLinearLayerSpec("layers.0.self_attn.q_proj", 32, 32, "attention"),
        ),
    )


def test_precomputed_model_spec_snapshot_round_trips(
    tmp_path: Path,
) -> None:
    """Snapshot save/load should preserve typed `ModelSpec` content."""

    snapshot = PrecomputedModelSpecSnapshot(
        schema_version=1,
        model_specs=(
            _toy_model_spec(model_name="toy-a"),
            _toy_model_spec(model_name="toy-b"),
        ),
        source_model_ids=("toy-a", "toy-b"),
    )
    output_path = save_precomputed_model_spec_snapshot(
        snapshot=snapshot,
        path=tmp_path / "model_specs.json",
    )
    loaded_snapshot = load_precomputed_model_spec_snapshot(path=output_path)
    assert loaded_snapshot.schema_version == 1
    assert loaded_snapshot.source_model_ids == ("toy-a", "toy-b")
    assert loaded_snapshot.spec_for_model_id(model_id="toy-b") is not None


def test_resolve_model_spec_uses_precomputed_snapshot(
    tmp_path: Path,
) -> None:
    """Model resolution should hit the snapshot before calling live inspect."""

    snapshot = PrecomputedModelSpecSnapshot(
        schema_version=1,
        model_specs=(_toy_model_spec(model_name="toy-hit"),),
        source_model_ids=("toy-hit",),
    )
    output_path = save_precomputed_model_spec_snapshot(
        snapshot=snapshot,
        path=tmp_path / "model_specs.json",
    )
    resolved_spec = resolve_model_spec(
        model_ref="toy-hit",
        inspect_fn=lambda model_ref: (_ for _ in ()).throw(
            AssertionError(f"Unexpected live inspect for {model_ref}")
        ),
        path=output_path,
    )
    assert resolved_spec.model_name == "toy-hit"


def test_resolve_model_spec_falls_back_when_snapshot_misses(
    tmp_path: Path,
) -> None:
    """Model resolution should use live inspection on snapshot misses."""

    snapshot = PrecomputedModelSpecSnapshot(
        schema_version=1,
        model_specs=(_toy_model_spec(model_name="toy-hit"),),
        source_model_ids=("toy-hit",),
    )
    output_path = save_precomputed_model_spec_snapshot(
        snapshot=snapshot,
        path=tmp_path / "model_specs.json",
    )
    resolved_spec = resolve_model_spec(
        model_ref="toy-miss",
        inspect_fn=lambda model_ref: _toy_model_spec(model_name=model_ref),
        path=output_path,
    )
    assert resolved_spec.model_name == "toy-miss"
