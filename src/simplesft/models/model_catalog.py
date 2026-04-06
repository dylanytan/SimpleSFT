"""Checked-in model catalog used by the local web UI."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelCatalogEntry:
    """One checked-in public model entry for the web catalog.

    Args:
        model_id: Full Hugging Face repo id.
        organization: Owning organization/user segment shown in the UI.
        display_name: Compact dropdown label.
        architecture_family: Normalized architecture-family label.
        model_type: Hugging Face config model type.
        is_trending_seed: Whether the entry came from the trending seed set.
        is_top_download_seed: Whether the entry came from the top-download seed set.
        catalog_priority: Lower values sort earlier within one organization.
        public_dropdown: Whether the entry appears in the default dropdown.

    Returns:
        Frozen catalog entry suitable for rendering and lookup.
    """

    model_id: str
    organization: str
    display_name: str
    architecture_family: str
    model_type: str
    is_trending_seed: bool
    is_top_download_seed: bool
    catalog_priority: int
    public_dropdown: bool


def _catalog_path() -> Path:
    """Return the checked-in model-catalog JSON path."""

    return Path(__file__).with_name("model_catalog.json")


def load_model_catalog() -> tuple[ModelCatalogEntry, ...]:
    """Load the checked-in model catalog.

    Returns:
        Immutable tuple of typed catalog entries.

    Example:
        >>> entries = load_model_catalog()
        >>> len(entries) > 0
        True
    """

    raw_entries = json.loads(_catalog_path().read_text(encoding="utf-8"))
    return tuple(ModelCatalogEntry(**raw_entry) for raw_entry in raw_entries)


def public_model_catalog_entries() -> tuple[ModelCatalogEntry, ...]:
    """Return public dropdown entries sorted by organization and priority."""

    entries = [entry for entry in load_model_catalog() if entry.public_dropdown]
    return tuple(
        sorted(
            entries,
            key=lambda entry: (
                entry.organization.lower(),
                entry.catalog_priority,
                entry.model_id.lower(),
            ),
        )
    )


def catalog_entry_for_model_id(*, model_id: str) -> ModelCatalogEntry | None:
    """Return one catalog entry by model id when present."""

    for entry in load_model_catalog():
        if entry.model_id == model_id:
            return entry
    return None


def model_select_options_html() -> str:
    """Render grouped model dropdown HTML for the web UI."""

    default_model_id = "Qwen/Qwen2.5-7B-Instruct"
    groups: dict[str, list[ModelCatalogEntry]] = {}
    for entry in public_model_catalog_entries():
        groups.setdefault(entry.organization, []).append(entry)
    option_groups = []
    for organization in sorted(groups):
        options = "".join(
            (
                f'<option value="{entry.model_id}"'
                f' data-architecture-family="{entry.architecture_family}"'
                f' data-model-type="{entry.model_type}"'
                f'{" selected" if entry.model_id == default_model_id else ""}>'
                f"{entry.model_id}</option>"
            )
            for entry in groups[organization]
        )
        option_groups.append(f'<optgroup label="{organization}">{options}</optgroup>')
    option_groups.append('<option value="__custom__">Custom model…</option>')
    return "".join(option_groups)
