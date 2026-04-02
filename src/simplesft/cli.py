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
from .corpus_cleaning import clean_measurement_corpus
from .estimate import estimate_peak_memory
from .inspect import inspect_model
from .measure import measure_peak_memory
from .rebuild import rebuild_benchmark_suite_from_measurements
from .reporting import render_comparison_report, render_suite_report
from .search import search_configurations
from .types import EstimatorConfig, LoRAConfig, MeasurementConfig
from .web_server import serve_web_interface


def _print_json(*, payload: dict) -> None:
    """Print a JSON payload to stdout."""

    print(json.dumps(payload, indent=2))


def _build_measurement_config(args: argparse.Namespace) -> MeasurementConfig:
    """Construct a `MeasurementConfig` from CLI args."""

    lora_config = None
    if args.tuning_mode == "lora":
        lora_config = LoRAConfig(rank=args.lora_rank)
    return MeasurementConfig(
        tuning_mode=args.tuning_mode,
        optimizer_name=args.optimizer_name,
        optimizer_learning_rate=args.optimizer_learning_rate,
        optimizer_beta1=args.optimizer_beta1,
        optimizer_beta2=args.optimizer_beta2,
        optimizer_momentum=args.optimizer_momentum,
        optimizer_alpha=args.optimizer_alpha,
        optimizer_centered=args.optimizer_centered,
        micro_batch_size_per_gpu=args.micro_batch_size_per_gpu,
        max_seq_len=args.max_seq_len,
        gradient_checkpointing=args.gradient_checkpointing,
        attention_backend=args.attention_backend,
        distributed_mode=args.distributed_mode,
        gpus_per_node=args.gpus_per_node,
        gpu_memory_gb=args.gpu_memory_gb,
        lora=lora_config,
    )


def _build_estimator_config(args: argparse.Namespace) -> EstimatorConfig:
    """Construct an `EstimatorConfig` from CLI args."""

    lora_config = None
    if args.tuning_mode == "lora":
        lora_config = LoRAConfig(rank=args.lora_rank)
    return EstimatorConfig(
        tuning_mode=args.tuning_mode,
        optimizer_name=args.optimizer_name,
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
    parser.add_argument("--optimizer-name", default="adamw")
    parser.add_argument("--optimizer-learning-rate", type=float, default=1e-4)
    parser.add_argument("--optimizer-beta1", type=float, default=0.9)
    parser.add_argument("--optimizer-beta2", type=float, default=0.999)
    parser.add_argument("--optimizer-momentum", type=float, default=0.0)
    parser.add_argument("--optimizer-alpha", type=float, default=0.99)
    parser.add_argument("--optimizer-centered", action="store_true")
    parser.add_argument("--micro-batch-size-per-gpu", type=int, default=1)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--attention-backend", default="standard")
    parser.add_argument(
        "--distributed-mode",
        default="single_gpu",
        choices=["single_gpu", "ddp", "zero2", "zero3"],
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

    result = estimate_peak_memory(
        model=args.model, config=_build_estimator_config(args=args)
    )
    if args.output:
        save_memory_result(result=result, path=args.output)
    _print_json(payload=asdict(result))


def _handle_measure(args: argparse.Namespace) -> None:
    """Handle the `measure` subcommand."""

    result = measure_peak_memory(
        model=args.model, config=_build_measurement_config(args=args)
    )
    if args.output:
        save_memory_result(result=result, path=args.output)
    _print_json(payload=asdict(result))


def _handle_compare(args: argparse.Namespace) -> None:
    """Handle the `compare` subcommand."""

    measurement_config = _build_measurement_config(args=args)
    estimated = estimate_peak_memory(
        model=args.model,
        config=measurement_config.to_estimator_config(),
    )
    measured = measure_peak_memory(model=args.model, config=measurement_config)
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
        EstimatorConfig(
            tuning_mode=args.tuning_mode,
            optimizer_name=args.optimizer_name,
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

    config_overrides = {
        "optimizer_learning_rate": args.optimizer_learning_rate,
        "optimizer_beta1": args.optimizer_beta1,
        "optimizer_beta2": args.optimizer_beta2,
        "optimizer_momentum": args.optimizer_momentum,
        "optimizer_alpha": args.optimizer_alpha,
        "optimizer_centered": args.optimizer_centered,
    }
    cases = build_default_benchmark_cases(
        model=args.model,
        seq_lens=args.seq_lens,
        micro_batches=args.micro_batches,
        tuning_modes=args.tuning_modes,
        distributed_modes=args.distributed_modes,
        attention_backends=args.attention_backends,
        optimizer_names=args.optimizer_names,
        gpu_memory_gb=args.gpu_memory_gb,
        gpus_per_node=args.gpus_per_node,
        lora_rank=args.lora_rank,
        gradient_checkpointing=args.gradient_checkpointing,
        config_overrides=config_overrides,
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


def _handle_clean_corpus(args: argparse.Namespace) -> None:
    """Handle the `clean-corpus` subcommand."""

    result = clean_measurement_corpus(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
    )
    _print_json(payload=asdict(result))


def _handle_web(args: argparse.Namespace) -> None:
    """Handle the `web` subcommand."""

    serve_web_interface(
        host=args.host,
        port=args.port,
        open_browser=not args.no_browser,
    )


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
        default=["single_gpu", "ddp", "zero2", "zero3"],
    )
    search_parser.add_argument(
        "--tuning-mode", default="full_ft", choices=["full_ft", "lora"]
    )
    search_parser.add_argument("--optimizer-name", default="adamw")
    search_parser.add_argument("--gpu-memory-gb", type=float, default=24.0)
    search_parser.add_argument("--gpus-per-node", type=int, default=1)
    search_parser.add_argument("--lora-rank", type=int, default=16)

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("model")
    benchmark_parser.add_argument("--output-dir", required=True)
    benchmark_parser.add_argument("--seq-lens", nargs="+", type=int, required=True)
    benchmark_parser.add_argument("--micro-batches", nargs="+", type=int, required=True)
    benchmark_parser.add_argument(
        "--tuning-modes", nargs="+", default=["full_ft", "lora"]
    )
    benchmark_parser.add_argument(
        "--distributed-modes",
        nargs="+",
        default=["single_gpu", "ddp", "zero2", "zero3"],
    )
    benchmark_parser.add_argument(
        "--attention-backends", nargs="+", default=["standard"]
    )
    benchmark_parser.add_argument("--optimizer-names", nargs="+", default=["adamw"])
    benchmark_parser.add_argument("--optimizer-learning-rate", type=float, default=1e-4)
    benchmark_parser.add_argument("--optimizer-beta1", type=float, default=0.9)
    benchmark_parser.add_argument("--optimizer-beta2", type=float, default=0.999)
    benchmark_parser.add_argument("--optimizer-momentum", type=float, default=0.0)
    benchmark_parser.add_argument("--optimizer-alpha", type=float, default=0.99)
    benchmark_parser.add_argument("--optimizer-centered", action="store_true")
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

    clean_parser = subparsers.add_parser("clean-corpus")
    clean_parser.add_argument("--root-dir", required=True)
    clean_parser.add_argument("--output-dir", required=True)

    web_parser = subparsers.add_parser("web")
    web_parser.add_argument("--host", default="127.0.0.1")
    web_parser.add_argument("--port", type=int, default=8765)
    web_parser.add_argument("--no-browser", action="store_true")

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
        "clean-corpus": _handle_clean_corpus,
        "web": _handle_web,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
