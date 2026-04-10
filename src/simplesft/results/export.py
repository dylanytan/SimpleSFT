"""Export heuristics to translate SimpleSFT candidates to Hugging Face TRL format."""

from __future__ import annotations

import json
from typing import Iterable

from ..types import MemoryResult


def _map_to_trl_config(result: MemoryResult) -> dict:
    """Map a single candidate config and memory result to a TRL arguments dict."""

    config = result.config

    # Map optimizer name to typical Hugging Face optimizer choices.
    optim = config.optimizer_name.lower()
    if optim == "adamw":
        optim = "adamw_torch"

    trl_args = {
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


def export_candidates_to_trl(
    candidates: Iterable[MemoryResult], 
    export_path: str,
) -> None:
    """Export viable candidate strategies to a JSON file for Hub/TRL ingestion.
    
    Args:
        candidates: A list of candidate memory results to export.
        export_path: The file path to write the JSON payload to.
    """
    
    payload = [_map_to_trl_config(c) for c in candidates]
    
    with open(export_path, "w") as f:
        json.dump(payload, f, indent=2)

