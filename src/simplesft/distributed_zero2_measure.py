"""Helpers for launching and aggregating ZeRO stage 2/3 measurement runs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist

from .artifacts import (
    load_memory_result,
    load_memory_result_from_raw,
    save_memory_result,
)
from .distributed_common import (
    build_torchrun_env,
    load_request,
    repo_root,
    serialize_model,
    torchrun_command,
)
from .measure import aggregate_rank_results
from .types import MemoryResult, ModelSpec, TrainingConfig
from .utils import maybe_get_deepspeed
from .zero2_measure import measure_zero2_local_peak_memory


def run_zero2_measurement(
    *, model: str | ModelSpec, config: TrainingConfig
) -> MemoryResult:
    """Launch a torchrun-based ZeRO measurement and return the aggregated result."""

    maybe_get_deepspeed()
    with tempfile.TemporaryDirectory(prefix="simplesft-zero2-") as temp_dir:
        request_path = Path(temp_dir) / "request.json"
        output_path = Path(temp_dir) / "result.json"
        launch_env, cross_numa_applied, cross_numa_reason = build_torchrun_env()
        request_path.write_text(
            json.dumps(
                {"model": serialize_model(model), "config": asdict(config)}, indent=2
            ),
            encoding="utf-8",
        )
        subprocess.run(
            torchrun_command(
                module_name="simplesft.distributed_zero2_measure",
                input_path=request_path,
                output_path=output_path,
                gpus_per_node=config.gpus_per_node,
            ),
            check=True,
            cwd=repo_root(),
            env=launch_env,
        )
        result = load_memory_result(path=output_path)
        result.metadata["cross_numa_nccl_env_applied"] = cross_numa_applied
        result.metadata["cross_numa_nccl_env_reason"] = cross_numa_reason
        return result


def _setup_zero2_runtime() -> int:
    """Initialize the local ZeRO worker runtime and return the local rank."""

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    deepspeed = maybe_get_deepspeed()
    deepspeed.init_distributed(dist_backend="nccl")
    return local_rank


def _gather_results(*, local_result: MemoryResult) -> list[dict[str, Any]]:
    """Gather serialized rank-local ZeRO measurement results across all ranks."""

    gathered_results: list[dict[str, Any] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_results, asdict(local_result))
    return [result for result in gathered_results if result is not None]


def main() -> None:
    """Run the torchrun worker entrypoint for ZeRO stage 2/3 measurement."""

    parser = argparse.ArgumentParser(
        prog="python -m simplesft.distributed_zero2_measure"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model, config = load_request(path=args.input)
    local_rank = _setup_zero2_runtime()
    local_result = measure_zero2_local_peak_memory(
        model=model,
        config=config,
        device_index=local_rank,
    )
    gathered_results = _gather_results(local_result=local_result)
    if dist.get_rank() == 0:
        aggregated_result = aggregate_rank_results(
            results=[
                load_memory_result_from_raw(raw=raw_result)
                for raw_result in gathered_results
            ],
            aggregation_tag=f"{config.distributed_mode}_aggregated",
            aggregation_assumption=(
                f"{config.distributed_mode} result aggregates the max across ranks."
            ),
        )
        save_memory_result(result=aggregated_result, path=args.output)
    dist.barrier(device_ids=[torch.cuda.current_device()])
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
