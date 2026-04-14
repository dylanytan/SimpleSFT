"""Tests for TRL export / load helpers."""

from __future__ import annotations

from pathlib import Path

import yaml

from simplesft.results.export import load_trl_strategy_config


def test_load_trl_strategy_config_yaml_list(tmp_path: Path) -> None:
    """YAML list export should load as the first mapping."""

    path = tmp_path / "cfg.yaml"
    path.write_text(
        "- per_device_train_batch_size: 2\n"
        "  max_seq_length: 1024\n"
        "  use_peft: false\n",
        encoding="utf-8",
    )
    loaded = load_trl_strategy_config(path)
    assert loaded["per_device_train_batch_size"] == 2
    assert loaded["max_seq_length"] == 1024


def test_load_trl_strategy_config_plain_mapping(tmp_path: Path) -> None:
    """A single mapping file should load directly."""

    path = tmp_path / "cfg.yaml"
    path.write_text(
        "per_device_train_batch_size: 1\n"
        "max_seq_length: 2048\n"
        "use_peft: true\n",
        encoding="utf-8",
    )
    loaded = load_trl_strategy_config(path)
    assert loaded["use_peft"] is True


def test_load_trl_strategy_config_json_still_works(tmp_path: Path) -> None:
    """JSON syntax is valid YAML 1.2 and should parse."""

    path = tmp_path / "legacy.json"
    path.write_text(
        '[{"per_device_train_batch_size": 1, "max_seq_length": 128, "use_peft": false}]',
        encoding="utf-8",
    )
    loaded = load_trl_strategy_config(path)
    assert loaded["max_seq_length"] == 128


def test_exported_payload_can_include_model_name(tmp_path: Path) -> None:
    """Exported configs should be allowed to include the model name for convenience."""

    path = tmp_path / "cfg.yaml"
    path.write_text(
        "model: Qwen/Qwen2.5-0.5B-Instruct\n"
        "per_device_train_batch_size: 1\n"
        "max_seq_length: 128\n",
        encoding="utf-8",
    )
    loaded = load_trl_strategy_config(path)
    assert loaded["model"] == "Qwen/Qwen2.5-0.5B-Instruct"

