"""Tests for the checked-in web model catalog."""

from __future__ import annotations

from simplesft.models.model_catalog import (
    model_select_options_html,
    public_model_catalog_entries,
)


def test_public_model_catalog_is_grouped_by_alphabetical_organization() -> None:
    """Public catalog entries should sort by organization then priority."""

    entries = public_model_catalog_entries()
    organizations = [entry.organization for entry in entries]
    assert organizations == sorted(organizations, key=str.lower)


def test_public_model_catalog_sorts_entries_within_organization() -> None:
    """Entries inside one organization should follow catalog priority."""

    allenai_entries = [
        entry
        for entry in public_model_catalog_entries()
        if entry.organization == "allenai"
    ]
    allenai_priorities = [entry.catalog_priority for entry in allenai_entries]
    assert allenai_priorities == sorted(allenai_priorities)
    assert allenai_entries[0].model_id == "allenai/Olmo-3.1-32B-Think"

    qwen_entries = [
        entry
        for entry in public_model_catalog_entries()
        if entry.organization == "Qwen"
    ]
    priorities = [entry.catalog_priority for entry in qwen_entries]
    assert priorities == sorted(priorities)
    assert qwen_entries[0].model_id == "Qwen/Qwen3-Coder-Next"


def test_model_select_options_html_contains_optgroups_and_custom_escape_hatch() -> None:
    """Dropdown HTML should render grouped organizations plus Custom model."""

    html = model_select_options_html()
    assert '<optgroup label="allenai">' in html
    assert '<optgroup label="Qwen">' in html
    assert '<optgroup label="microsoft">' in html
    assert 'value="allenai/Olmo-3-1025-7B"' in html
    assert 'value="facebook/opt-125m"' in html
    assert 'value="openai/gpt-oss-20b"' in html
    assert 'value="__custom__"' in html
