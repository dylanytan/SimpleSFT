"""Refresh an offline Hugging Face text-generation catalog audit snapshot."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi

from simplesft.models.architecture_registry import manifest_for_model_type
from simplesft.runtime import load_auto_config


SNAPSHOT_PATH = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "simplesft"
    / "model_catalog_seed_snapshot.json"
)


@dataclass(frozen=True)
class SeedModelStatus:
    """Classification for one top-seed model during catalog refresh."""

    model_id: str
    source: str
    rank: int
    status: str
    organization: str
    model_type: str = ""
    architecture_family: str = ""
    reason: str = ""


def _seed_models(*, api: HfApi, sort_key: str) -> list[tuple[int, str]]:
    """Return `(rank, model_id)` pairs for one seed list."""

    models = api.list_models(
        filter="text-generation",
        sort=sort_key,
        direction=-1,
        limit=100,
        full=False,
    )
    return [(index, model_info.id) for index, model_info in enumerate(models, start=1)]


def _classify_model(*, model_id: str, source: str, rank: int) -> SeedModelStatus:
    """Return support classification for one Hugging Face repo."""

    if model_id.lower().endswith("gguf") or "gguf" in model_id.lower():
        return SeedModelStatus(
            model_id=model_id,
            source=source,
            rank=rank,
            status="blocked_artifact",
            organization=model_id.split("/", 1)[0],
            reason="GGUF export is not a standard Transformers inspection target.",
        )
    try:
        config = load_auto_config(model_ref=model_id, trust_remote_code=False)
    except Exception as exc:  # noqa: BLE001
        reason = str(exc)
        status = "blocked_access" if "gated" in reason.lower() else "blocked_runtime"
        return SeedModelStatus(
            model_id=model_id,
            source=source,
            rank=rank,
            status=status,
            organization=model_id.split("/", 1)[0],
            reason=reason,
        )
    manifest = manifest_for_model_type(model_type=config.model_type)
    if manifest is None:
        return SeedModelStatus(
            model_id=model_id,
            source=source,
            rank=rank,
            status="unsupported_family",
            organization=model_id.split("/", 1)[0],
            model_type=config.model_type,
            reason="No explicit dense architecture manifest registered.",
        )
    return SeedModelStatus(
        model_id=model_id,
        source=source,
        rank=rank,
        status="supported",
        organization=model_id.split("/", 1)[0],
        model_type=config.model_type,
        architecture_family=manifest.family_spec.family_label,
    )


def main() -> None:
    """Refresh the offline catalog audit snapshot.

    Example:
        $ python scripts/refresh_model_catalog.py
    """

    api = HfApi()
    seed_sources = (
        ("downloads", _seed_models(api=api, sort_key="downloads")),
        ("trendingScore", _seed_models(api=api, sort_key="trendingScore")),
    )
    seen_model_ids: set[tuple[str, str]] = set()
    snapshot_rows: list[dict[str, Any]] = []
    for source, rows in seed_sources:
        for rank, model_id in rows:
            key = (source, model_id)
            if key in seen_model_ids:
                continue
            seen_model_ids.add(key)
            snapshot_rows.append(
                asdict(_classify_model(model_id=model_id, source=source, rank=rank))
            )
    SNAPSHOT_PATH.write_text(json.dumps(snapshot_rows, indent=2), encoding="utf-8")
    print(f"Wrote {len(snapshot_rows)} rows to {SNAPSHOT_PATH}")


if __name__ == "__main__":
    main()
