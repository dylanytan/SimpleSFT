"""Export heuristics to translate SimpleSFT candidates to Hugging Face TRL format."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml

from ..types import MemoryResult


def _map_to_trl_config(result: MemoryResult) -> dict:
    """Map a single candidate config and memory result to a TRL arguments dict."""

    config = result.config

    # Map optimizer name to typical Hugging Face optimizer choices.
    optim = config.optimizer_name.lower()
    if optim == "adamw":
        optim = "adamw_torch"

    trl_args = {
        # Convenience: keep the originating model alongside the strategy args.
        # (Ignored by TRL itself; consumed by downstream tooling / humans.)
        "model": result.model_name,

        # Model / Training loop defaults
        "per_device_train_batch_size": config.micro_batch_size_per_gpu,
        "max_seq_length": config.max_seq_len,
        "gradient_checkpointing": config.gradient_checkpointing,
        "optim": optim,
        
        # Precision
        "fp16": config.weight_dtype == "fp16",
        "bf16": config.weight_dtype == "bf16",
    }

    # Optional LoRA / PEFT mappings for trl sft
    if config.tuning_mode == "lora":
        trl_args["use_peft"] = True
        if config.lora is not None:
            trl_args["lora_r"] = config.lora.rank
            trl_args["lora_alpha"] = config.lora.alpha
            trl_args["lora_dropout"] = config.lora.dropout
            trl_args["lora_target_modules"] = list(config.lora.target_modules)
    else:
        trl_args["use_peft"] = False

    # Extract zero level for metadata
    zero_level = 0
    if config.distributed_mode == "zero2":
        zero_level = 2
    elif config.distributed_mode == "zero3":
        zero_level = 3

    # Optional reporting on memory so the runner script knows exactly why
    # this candidate was ranked this way. We tuck this under a metadata field.
    trl_args["_simplesft_metadata"] = {
        "global_peak_gb": result.global_peak_gb(),
        "headroom_gb": result.headroom_gb(),
        "distributed_mode": config.distributed_mode,
        "zero_level": zero_level,
        "dp_degree": config.data_parallel_degree(),
        "tp_degree": config.tensor_parallel_degree,
        "sp": config.sequence_parallel,
        "ckpt": config.gradient_checkpointing,
        "feasible": result.feasible,
    }

    return trl_args


def load_trl_strategy_config(path: str | Path) -> dict[str, Any]:
    """Load TRL strategy args written by SimpleSFT (YAML, or JSON as YAML 1.2).

    Args:
        path: Path to a YAML file containing one mapping, or a single-element list
            whose only element is that mapping (exported batch format).

    Returns:
        One TRL strategy dictionary (``run_sft`` / ``SFTTrainer`` kwargs subset).
    """

    raw_text = Path(path).read_text(encoding="utf-8")
    config_data = yaml.safe_load(raw_text)
    if isinstance(config_data, list):
        if len(config_data) == 0:
            raise ValueError("Config file contains an empty list.")
        config_data = config_data[0]
    if not isinstance(config_data, dict):
        raise TypeError("TRL strategy config must be a mapping or a one-element list of mappings.")
    return config_data


_YAML_DUMP_KWARGS = dict(
    default_flow_style=False,
    sort_keys=False,
    allow_unicode=True,
)


def trl_strategy_payload_to_yaml(config: dict[str, Any]) -> str:
    """Format one strategy dict like the web download (a single-element YAML list)."""

    return yaml.safe_dump([config], **_YAML_DUMP_KWARGS)


def trl_candidates_to_yaml_document(candidates: Iterable[MemoryResult]) -> str:
    """Serialize candidates to a YAML document (list of mappings)."""

    payload = [_map_to_trl_config(c) for c in candidates]
    return yaml.safe_dump(payload, **_YAML_DUMP_KWARGS)


def export_candidates_to_trl(
    candidates: Iterable[MemoryResult],
    export_path: str,
) -> None:
    """Export viable candidate strategies to a YAML file for Hub/TRL ingestion.

    Args:
        candidates: Candidate memory results to export.
        export_path: File path to write the YAML payload to.
    """

    text = trl_candidates_to_yaml_document(candidates)
    Path(export_path).write_text(text, encoding="utf-8")

