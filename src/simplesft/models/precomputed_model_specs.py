"""Checked-in precomputed model-spec snapshots for faster estimator startup."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from ..results.artifacts import load_model_spec_from_raw, model_spec_to_raw
from ..types import ModelSpec


@dataclass(frozen=True)
class PrecomputedModelSpecSnapshot:
    """Frozen snapshot of checked-in `ModelSpec` objects.

    Args:
        schema_version: Snapshot schema version.
        model_specs: Materialized precomputed model specs.
        source_model_ids: Model ids used to build the snapshot.

    Returns:
        Snapshot with lookup helpers for precomputed inspection data.
    """

    schema_version: int
    model_specs: tuple[ModelSpec, ...]
    source_model_ids: tuple[str, ...]

    def spec_for_model_id(self, *, model_id: str) -> ModelSpec | None:
        """Return one precomputed `ModelSpec` by model id when present."""

        for model_spec in self.model_specs:
            if model_spec.model_name == model_id:
                return model_spec
        return None


def _snapshot_path() -> Path:
    """Return the checked-in precomputed model-spec snapshot path."""

    return Path(__file__).with_name("precomputed_model_specs.json")


def snapshot_to_raw(
    *, snapshot: PrecomputedModelSpecSnapshot
) -> dict[str, Any]:
    """Serialize one snapshot into a JSON-friendly dictionary."""

    return {
        "schema_version": snapshot.schema_version,
        "source_model_ids": list(snapshot.source_model_ids),
        "model_specs": [
            model_spec_to_raw(model_spec=model_spec)
            for model_spec in snapshot.model_specs
        ],
    }


def save_precomputed_model_spec_snapshot(
    *,
    snapshot: PrecomputedModelSpecSnapshot,
    path: str | Path | None = None,
) -> Path:
    """Write one precomputed model-spec snapshot to JSON.

    Args:
        snapshot: Snapshot to persist.
        path: Optional output path.

    Returns:
        Resolved output path written to disk.
    """

    output_path = Path(path) if path is not None else _snapshot_path()
    output_path.write_text(
        json.dumps(snapshot_to_raw(snapshot=snapshot), indent=2),
        encoding="utf-8",
    )
    return output_path


@lru_cache(maxsize=None)
def load_precomputed_model_spec_snapshot(
    *, path: str | Path | None = None
) -> PrecomputedModelSpecSnapshot:
    """Load one checked-in precomputed model-spec snapshot.

    Args:
        path: Optional snapshot path.

    Returns:
        Frozen typed snapshot loaded from JSON.
    """

    snapshot_path = Path(path) if path is not None else _snapshot_path()
    raw = json.loads(snapshot_path.read_text(encoding="utf-8"))
    return PrecomputedModelSpecSnapshot(
        schema_version=int(raw["schema_version"]),
        model_specs=tuple(
            load_model_spec_from_raw(raw=item) for item in raw["model_specs"]
        ),
        source_model_ids=tuple(raw.get("source_model_ids", ())),
    )


def precomputed_model_spec_for_model_id(
    *, model_id: str, path: str | Path | None = None
) -> ModelSpec | None:
    """Return one precomputed `ModelSpec` when the snapshot contains it."""

    try:
        snapshot = load_precomputed_model_spec_snapshot(path=path)
    except FileNotFoundError:
        return None
    return snapshot.spec_for_model_id(model_id=model_id)


def resolve_model_spec(
    *,
    model_ref: str,
    inspect_fn: Callable[[str], ModelSpec],
    path: str | Path | None = None,
) -> ModelSpec:
    """Resolve a model id to a precomputed spec or a live inspected spec.

    Args:
        model_ref: Hugging Face model id or local model path.
        inspect_fn: Fallback inspection function for cache misses.
        path: Optional snapshot path override.

    Returns:
        Precomputed or live-inspected `ModelSpec`.

    Example:
        >>> from simplesft.types import ModelSpec
        >>> spec = resolve_model_spec(
        ...     model_ref="toy",
        ...     inspect_fn=lambda model_ref: ModelSpec(
        ...         model_name=model_ref,
        ...         model_type="llama",
        ...         num_layers=1,
        ...         hidden_size=8,
        ...         num_attention_heads=1,
        ...         intermediate_size=16,
        ...         vocab_size=32,
        ...         max_position_embeddings=64,
        ...         total_params=128,
        ...         trainable_linear_layers=(),
        ...     ),
        ... )
        >>> spec.model_name
        'toy'
    """

    precomputed_spec = precomputed_model_spec_for_model_id(
        model_id=model_ref,
        path=path,
    )
    if precomputed_spec is not None:
        return precomputed_spec
    return inspect_fn(model_ref)
