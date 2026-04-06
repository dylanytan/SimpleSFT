"""Precompute `ModelSpec` snapshots for the checked-in model catalog."""

from __future__ import annotations

import argparse

from simplesft.models.inspect import inspect_model
from simplesft.models.model_catalog import (
    load_model_catalog,
    public_model_catalog_entries,
)
from simplesft.models.precomputed_model_specs import (
    PrecomputedModelSpecSnapshot,
    save_precomputed_model_spec_snapshot,
)


def _parse_args() -> argparse.Namespace:
    """Return parsed CLI arguments for model-spec precomputation."""

    parser = argparse.ArgumentParser(
        description="Precompute inspected model specs into a checked-in JSON snapshot."
    )
    parser.add_argument(
        "--model",
        dest="model_ids",
        action="append",
        default=[],
        help="Specific model id to precompute. May be passed multiple times.",
    )
    parser.add_argument(
        "--include-nonpublic",
        action="store_true",
        help="Include checked-in catalog rows that are not in the public dropdown.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path. Defaults to the checked-in snapshot path.",
    )
    return parser.parse_args()


def _selected_model_ids(*, args: argparse.Namespace) -> tuple[str, ...]:
    """Return the ordered model ids selected for snapshot generation."""

    if args.model_ids:
        return tuple(dict.fromkeys(args.model_ids))
    if args.include_nonpublic:
        entries = load_model_catalog()
    else:
        entries = public_model_catalog_entries()
    return tuple(entry.model_id for entry in entries)


def _build_snapshot(*, model_ids: tuple[str, ...]) -> PrecomputedModelSpecSnapshot:
    """Inspect one ordered model list and return a frozen snapshot."""

    model_specs = tuple(inspect_model(model_ref=model_id) for model_id in model_ids)
    return PrecomputedModelSpecSnapshot(
        schema_version=1,
        model_specs=model_specs,
        source_model_ids=model_ids,
    )


def main() -> None:
    """Inspect catalog models and persist a precomputed snapshot.

    Example:
        $ python scripts/precompute_model_specs.py
    """

    args = _parse_args()
    model_ids = _selected_model_ids(args=args)
    assert model_ids, "No model ids selected for precomputation."
    snapshot = _build_snapshot(model_ids=model_ids)
    output_path = save_precomputed_model_spec_snapshot(
        snapshot=snapshot,
        path=args.output,
    )
    print(f"Wrote {len(snapshot.model_specs)} model specs to {output_path}")


if __name__ == "__main__":
    main()
