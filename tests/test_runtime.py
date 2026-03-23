"""Tests for runtime attention-backend helpers."""

import pytest

from simplesft.runtime import resolve_attention_implementation


def test_resolve_attention_implementation_for_standard() -> None:
    """Standard attention should keep the model-default implementation."""

    assert resolve_attention_implementation(attention_backend="standard") is None


def test_resolve_attention_implementation_rejects_unknown_backend() -> None:
    """Unknown backends should fail explicitly."""

    with pytest.raises(AssertionError):
        resolve_attention_implementation(attention_backend="unknown-backend")
