"""Command-line entrypoints for SimpleSFT."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .artifacts import (
    find_comparison_artifacts,
    load_benchmark_suite_result,
    load_comparison_result,
    save_comparison_result,
    save_memory_result,
    write_text_artifact,
)
from .benchmark import build_default_benchmark_cases, run_benchmark_suite
from .compare import compare_measurement_to_estimate
from .estimate import estimate_peak_memory
from .inspect import inspect_model
from .measure import measure_peak_memory
from .rebuild import rebuild_benchmark_suite_from_measurements
from .reporting import render_comparison_report, render_suite_report
from .search import search_configurations
from .types import LoRAConfig, TrainingConfig


def _print_json(*, payload: dict) -> None:
    """Print a JSON payload to stdout."""

    print(json.dumps(payload, indent=2))


def _build_training_config(args: argparse.Namespace) -> TrainingConfig:
    """Construct a `TrainingConfig` from CLI args."""

    lora_config = None
    if args.tuning_mode == "lora":
        lora_config = LoRAConfig(rank=args.lora_rank)
    return TrainingConfig(
        tuning_mode=args.tuning_mode,
        micro_batch_size_per_gpu=args.micro_batch_size_per_gpu,
        max_seq_len=args.max_seq_len,
        gradient_checkpointing=args.gradient_checkpointing,
        attention_backend=args.attention_backend,
        distributed_mode=args.distributed_mode,
        gpus_per_node=args.gpus_per_node,
        gpu_memory_gb=args.gpu_memory_gb,
        lora=lora_config,
    )


def _add_common_training_args(parser: argparse.ArgumentParser) -> None:
    """Register training config args on a parser."""

    parser.add_argument("--tuning-mode", default="full_ft", choices=["full_ft", "lora"])
    parser.add_argument("--micro-batch-size-per-gpu", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--attention-backend", default="standard")
    parser.add_argument(
        "--distributed-mode",
        default="single_gpu",
        choices=["single_gpu", "ddp", "zero2"],
    )
    parser.add_argument("--gpus-per-node", type=int, default=1)
    parser.add_argument("--gpu-memory-gb", type=float, default=24.0)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--output")


def _load_report_comparisons(*, input_dir: str):
    """Load all saved comparison artifacts under a directory."""

    return [
        load_comparison_result(path=path)
        for path in find_comparison_artifacts(root_dir=input_dir)
    ]


def _handle_inspect(args: argparse.Namespace) -> None:
    """Handle the `inspect` subcommand."""

    result = inspect_model(model_ref=args.model)
    _print_json(payload=asdict(result))


def _handle_estimate(args: argparse.Namespace) -> None:
    """Handle the `estimate` subcommand."""

    result = estimate_peak_memory(model=args.model, config=_build_training_config(args=args))
    if args.output:
        save_memory_result(result=result, path=args.output)
    _print_json(payload=asdict(result))


def _handle_measure(args: argparse.Namespace) -> None:
    """Handle the `measure` subcommand."""

    result = measure_peak_memory(model=args.model, config=_build_training_config(args=args))
    if args.output:
        save_memory_result(result=result, path=args.output)
    _print_json(payload=asdict(result))


def _handle_compare(args: argparse.Namespace) -> None:
    """Handle the `compare` subcommand."""

    config = _build_training_config(args=args)
    estimated = estimate_peak_memory(model=args.model, config=config)
    measured = measure_peak_memory(model=args.model, config=config)
    result = compare_measurement_to_estimate(measured=measured, estimated=estimated)
    if args.output:
        save_comparison_result(result=result, path=args.output)
    _print_json(payload=asdict(result))


def _handle_search(args: argparse.Namespace) -> None:
    """Handle the `search` subcommand."""

    lora_config = None
    if args.tuning_mode == "lora":
        lora_config = LoRAConfig(rank=args.lora_rank)
    configs = [
        TrainingConfig(
            tuning_mode=args.tuning_mode,
            micro_batch_size_per_gpu=micro_batch_size,
            max_seq_len=seq_len,
            distributed_mode=distributed_mode,
            gpu_memory_gb=args.gpu_memory_gb,
            gpus_per_node=args.gpus_per_node,
            lora=lora_config,
        )
        for seq_len in args.seq_lens
        for micro_batch_size in args.micro_batches
        for distributed_mode in args.distributed_modes
    ]
    result = search_configurations(model=args.model, configs=configs)
    _print_json(payload=asdict(result))


def _handle_benchmark(args: argparse.Namespace) -> None:
    """Handle the `benchmark` subcommand."""

    cases = build_default_benchmark_cases(
        model=args.model,
        seq_lens=args.seq_lens,
        micro_batches=args.micro_batches,
        tuning_modes=args.tuning_modes,
        distributed_modes=args.distributed_modes,
        attention_backends=args.attention_backends,
        gpu_memory_gb=args.gpu_memory_gb,
        gpus_per_node=args.gpus_per_node,
        lora_rank=args.lora_rank,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    suite_result, comparisons = run_benchmark_suite(
        cases=cases,
        output_dir=args.output_dir,
        include_measurement=args.measure,
        allow_measurement_failures=not args.strict_measure,
    )
    if args.report_path:
        report_text = render_suite_report(
            iteration_name="Benchmark Suite",
            suite_result=suite_result,
            comparisons=comparisons,
        )
        write_text_artifact(path=args.report_path, content=report_text)
    _print_json(payload=asdict(suite_result))


def _handle_report(args: argparse.Namespace) -> None:
    """Handle the `report` subcommand."""

    comparisons = _load_report_comparisons(input_dir=args.input_dir)
    if comparisons:
        report_text = render_comparison_report(
            iteration_name=args.iteration_name,
            comparisons=comparisons,
        )
    else:
        suite_index_path = f"{args.input_dir}/suite_index.json"
        suite_result = load_benchmark_suite_result(path=suite_index_path)
        report_text = render_suite_report(
            iteration_name=args.iteration_name,
            suite_result=suite_result,
            comparisons=[],
        )
    if args.output:
        write_text_artifact(path=args.output, content=report_text)
    print(report_text)


def _handle_rebuild_benchmark(args: argparse.Namespace) -> None:
    """Handle the `rebuild-benchmark` subcommand."""

    suite_result, comparisons = rebuild_benchmark_suite_from_measurements(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
    )
    if args.report_path:
        report_text = render_suite_report(
            iteration_name=args.iteration_name,
            suite_result=suite_result,
            comparisons=comparisons,
        )
        write_text_artifact(path=args.report_path, content=report_text)
    _print_json(payload=asdict(suite_result))


def main() -> None:
    """Run the SimpleSFT command-line interface."""

    parser = argparse.ArgumentParser(prog="simplesft")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("model")

    estimate_parser = subparsers.add_parser("estimate")
    estimate_parser.add_argument("model")
    _add_common_training_args(estimate_parser)

    measure_parser = subparsers.add_parser("measure")
    measure_parser.add_argument("model")
    _add_common_training_args(measure_parser)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("model")
    _add_common_training_args(compare_parser)

    search_parser = subparsers.add_parser("search")
    search_parser.add_argument("model")
    search_parser.add_argument("--seq-lens", nargs="+", type=int, required=True)
    search_parser.add_argument("--micro-batches", nargs="+", type=int, required=True)
    search_parser.add_argument(
        "--distributed-modes",
        nargs="+",
        default=["single_gpu", "ddp", "zero2"],
    )
    search_parser.add_argument("--tuning-mode", default="full_ft", choices=["full_ft", "lora"])
    search_parser.add_argument("--gpu-memory-gb", type=float, default=24.0)
    search_parser.add_argument("--gpus-per-node", type=int, default=1)
    search_parser.add_argument("--lora-rank", type=int, default=16)

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("model")
    benchmark_parser.add_argument("--output-dir", required=True)
    benchmark_parser.add_argument("--seq-lens", nargs="+", type=int, required=True)
    benchmark_parser.add_argument("--micro-batches", nargs="+", type=int, required=True)
    benchmark_parser.add_argument("--tuning-modes", nargs="+", default=["full_ft", "lora"])
    benchmark_parser.add_argument(
        "--distributed-modes",
        nargs="+",
        default=["single_gpu", "ddp", "zero2"],
    )
    benchmark_parser.add_argument("--attention-backends", nargs="+", default=["standard"])
    benchmark_parser.add_argument("--gpu-memory-gb", type=float, default=24.0)
    benchmark_parser.add_argument("--gpus-per-node", type=int, default=1)
    benchmark_parser.add_argument("--lora-rank", type=int, default=16)
    benchmark_parser.add_argument("--gradient-checkpointing", action="store_true")
    benchmark_parser.add_argument("--measure", action="store_true")
    benchmark_parser.add_argument("--strict-measure", action="store_true")
    benchmark_parser.add_argument("--report-path")

    report_parser = subparsers.add_parser("report")
    report_parser.add_argument("--input-dir", required=True)
    report_parser.add_argument("--iteration-name", default="Iteration Report")
    report_parser.add_argument("--output")

    rebuild_parser = subparsers.add_parser("rebuild-benchmark")
    rebuild_parser.add_argument("--source-dir", required=True)
    rebuild_parser.add_argument("--output-dir", required=True)
    rebuild_parser.add_argument("--iteration-name", default="Rebuilt Benchmark Suite")
    rebuild_parser.add_argument("--report-path")

    args = parser.parse_args()
    handlers = {
        "inspect": _handle_inspect,
        "estimate": _handle_estimate,
        "measure": _handle_measure,
        "compare": _handle_compare,
        "search": _handle_search,
        "benchmark": _handle_benchmark,
        "report": _handle_report,
        "rebuild-benchmark": _handle_rebuild_benchmark,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
