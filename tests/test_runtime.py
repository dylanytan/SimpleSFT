"""Tests for runtime attention-backend helpers."""

from dataclasses import dataclass
from typing import Optional

import pytest

import simplesft.runtime as runtime_module
from simplesft.runtime import resolve_attention_implementation


def test_resolve_attention_implementation_for_standard() -> None:
    """Standard attention should map to eager attention explicitly."""

    assert resolve_attention_implementation(attention_backend="standard") == "eager"


def test_resolve_attention_implementation_for_auto() -> None:
    """Auto attention should keep the model-default implementation."""

    assert resolve_attention_implementation(attention_backend="auto") is None


def test_resolve_attention_implementation_rejects_unknown_backend() -> None:
    """Unknown backends should fail explicitly."""

    with pytest.raises(AssertionError):
        resolve_attention_implementation(attention_backend="unknown-backend")


@dataclass
class _DummyConfig:
    """Typed config double used for runtime normalization tests."""

    model_type: str = "olmo3"
    rope_scaling: Optional[dict[str, object]] = None


def test_load_auto_config_normalizes_rope_scaling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Loaded configs should coerce rope-scaling beta values to floats."""

    monkeypatch.setattr(runtime_module, "prepare_transformers_runtime", lambda: False)
    monkeypatch.setattr(
        "transformers.AutoConfig.from_pretrained",
        lambda **_kwargs: _DummyConfig(
            rope_scaling={"beta_fast": 32, "beta_slow": 1, "rope_type": "yarn"}
        ),
    )
    config = runtime_module.load_auto_config(model_ref="toy")
    assert config.rope_scaling is not None
    assert isinstance(config.rope_scaling["beta_fast"], float)
    assert isinstance(config.rope_scaling["beta_slow"], float)
