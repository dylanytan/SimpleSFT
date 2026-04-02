"""Local web server for interactive SimpleSFT estimates."""

from __future__ import annotations

import errno
import json
import math
import webbrowser
from dataclasses import asdict, dataclass, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Optional, Union
from urllib.error import URLError
from urllib.request import urlopen

from .estimate import estimate_peak_memory
from .inspect import inspect_model
from .types import EstimatorConfig, LoRAConfig, MemoryResult, ModelSpec
from .web_assets import APP_HTML


EstimateFn = Callable[[Union[str, ModelSpec], EstimatorConfig], MemoryResult]
InspectFn = Callable[[str], ModelSpec]


@dataclass(frozen=True)
class WebRecommendationCandidate:
    """One candidate training strategy considered by the web UI.

    Args:
        distributed_mode: Candidate distributed backend.
        gradient_checkpointing: Whether checkpointing is enabled.
        tensor_parallel_degree: Megatron-style tensor-parallel degree.
        sequence_parallel: Whether sequence parallelism is enabled.
        data_parallel_degree: Remaining data-parallel degree after TP.
        feasible: Whether the estimate fits the requested GPU budget.
        global_peak_gb: Predicted peak memory per rank in GiB.
        headroom_gb: Predicted VRAM headroom per rank in GiB.
        estimated_slowdown_percent: Estimated throughput slowdown versus the
            least sharded baseline.
        ranking_score: Internal ranking score where lower is better.
    """

    distributed_mode: str
    gradient_checkpointing: bool
    tensor_parallel_degree: int
    sequence_parallel: bool
    data_parallel_degree: int
    feasible: bool
    global_peak_gb: float
    headroom_gb: float
    estimated_slowdown_percent: float
    ranking_score: float

    def parallelism_label(self) -> str:
        """Return a compact `dp × tp = world` label for the candidate.

        Returns:
            Human-readable decomposition of the candidate world size.
        """

        world_size = self.data_parallel_degree * self.tensor_parallel_degree
        return (
            f"dp={self.data_parallel_degree} · "
            f"tp={self.tensor_parallel_degree} · "
            f"world={world_size}"
        )


@dataclass(frozen=True)
class WebRecommendation:
    """Recommended training strategy for one estimate request.

    Args:
        strategy_source: `recommended` or `request_override`.
        config: Final estimator config used for the estimate.
        rationale: Short explanation of the recommendation.
        candidates: Candidate strategies considered during selection.
    """

    strategy_source: str
    config: EstimatorConfig
    rationale: tuple[str, ...]
    candidates: tuple[WebRecommendationCandidate, ...]


def _json_response(
    *, handler: BaseHTTPRequestHandler, status: HTTPStatus, payload: dict[str, Any]
) -> None:
    """Write a JSON response to the client.

    Args:
        handler: Active HTTP request handler.
        status: HTTP status code.
        payload: JSON-serializable response object.
    """

    content = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(content)))
    handler.end_headers()
    handler.wfile.write(content)


def _html_response(*, handler: BaseHTTPRequestHandler, content: str) -> None:
    """Write the main HTML application to the client.

    Args:
        handler: Active HTTP request handler.
        content: HTML content to return.
    """

    body = content.encode("utf-8")
    handler.send_response(HTTPStatus.OK)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_json_payload(*, handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    """Read and decode a JSON request body.

    Args:
        handler: Active HTTP request handler.

    Returns:
        Parsed JSON payload.
    """

    content_length = int(handler.headers.get("Content-Length", "0"))
    raw_body = handler.rfile.read(content_length)
    return json.loads(raw_body.decode("utf-8"))


def _build_lora_config(*, payload: dict[str, Any]) -> Optional[LoRAConfig]:
    """Build a `LoRAConfig` from a request payload.

    Args:
        payload: JSON request body.

    Returns:
        Parsed LoRA config or `None`.
    """

    raw_lora = payload.get("lora")
    if raw_lora is None:
        return None
    return LoRAConfig(
        rank=int(raw_lora["rank"]),
        alpha=float(raw_lora.get("alpha", raw_lora["rank"])),
        dropout=float(raw_lora.get("dropout", 0.0)),
        target_modules=tuple(raw_lora["target_modules"]),
        bias=str(raw_lora.get("bias", "none")),
    )


def _recommended_modes(*, config: EstimatorConfig) -> tuple[str, ...]:
    """Return the recommended distributed-mode search order for one request."""

    if config.gpus_per_node * config.num_nodes == 1:
        return ("single_gpu",)
    return ("ddp", "zero2", "zero3")


def _divisors(*, value: int) -> tuple[int, ...]:
    """Return the positive divisors for one integer in ascending order."""

    divisors = {divisor for divisor in range(1, value + 1) if value % divisor == 0}
    return tuple(sorted(divisors))


def _candidate_tensor_parallel_degrees(
    *, payload: dict[str, Any], available_gpu_count: int
) -> tuple[int, ...]:
    """Return tensor-parallel degrees to evaluate for one request."""

    explicit_degree = payload.get("tensor_parallel_degree")
    if explicit_degree is not None:
        return (int(explicit_degree),)
    return _divisors(value=max(1, available_gpu_count))


def _candidate_sequence_parallel_values(
    *, payload: dict[str, Any], tensor_parallel_degree: int
) -> tuple[bool, ...]:
    """Return sequence-parallel settings to evaluate for one TP degree."""

    explicit_value = payload.get("sequence_parallel")
    if explicit_value is not None:
        return (bool(explicit_value),)
    if tensor_parallel_degree == 1:
        return (False,)
    return (False, True)


def _is_valid_strategy(
    *,
    available_gpu_count: int,
    distributed_mode: str,
    tensor_parallel_degree: int,
    sequence_parallel: bool,
) -> bool:
    """Return whether one optimization combination is structurally valid.

    Args:
        available_gpu_count: Total number of visible GPUs.
        distributed_mode: Candidate distributed backend.
        tensor_parallel_degree: Candidate tensor-parallel degree.
        sequence_parallel: Whether sequence parallelism is enabled.

    Returns:
        `True` when the combination is valid for the available world size.
    """

    if tensor_parallel_degree < 1:
        return False
    if available_gpu_count % tensor_parallel_degree != 0:
        return False
    if sequence_parallel and tensor_parallel_degree == 1:
        return False
    data_parallel_degree = available_gpu_count // tensor_parallel_degree
    if distributed_mode == "single_gpu":
        return available_gpu_count == 1 and tensor_parallel_degree == 1
    if distributed_mode == "ddp":
        return data_parallel_degree >= 2
    if distributed_mode in {"zero2", "zero3"}:
        return data_parallel_degree >= 2
    return False


def _strategy_candidates_from_payload(
    *, payload: dict[str, Any], config: EstimatorConfig
) -> tuple[tuple[str, bool, int, bool], ...]:
    """Build candidate strategy tuples for one request."""

    explicit_mode = payload.get("distributed_mode")
    explicit_checkpointing = payload.get("gradient_checkpointing")
    modes = (
        (str(explicit_mode),)
        if explicit_mode is not None
        else _recommended_modes(config=config)
    )
    checkpoint_options = (
        (bool(explicit_checkpointing),)
        if explicit_checkpointing is not None
        else (False, True)
    )
    candidates: list[tuple[str, bool, int, bool]] = []
    for distributed_mode in modes:
        available_gpu_count = (
            1 if distributed_mode == "single_gpu" else config.available_gpu_count()
        )
        for checkpointing in checkpoint_options:
            for tensor_parallel_degree in _candidate_tensor_parallel_degrees(
                payload=payload,
                available_gpu_count=available_gpu_count,
            ):
                for sequence_parallel in _candidate_sequence_parallel_values(
                    payload=payload,
                    tensor_parallel_degree=tensor_parallel_degree,
                ):
                    if not _is_valid_strategy(
                        available_gpu_count=available_gpu_count,
                        distributed_mode=distributed_mode,
                        tensor_parallel_degree=tensor_parallel_degree,
                        sequence_parallel=sequence_parallel,
                    ):
                        continue
                    candidates.append(
                        (
                            distributed_mode,
                            checkpointing,
                            tensor_parallel_degree,
                            sequence_parallel,
                        )
                    )
    return tuple(candidates)


def _safety_headroom_gb(*, gpu_memory_gb: float) -> float:
    """Return a modest safety headroom target for one GPU size."""

    return max(2.0, min(gpu_memory_gb * 0.10, 8.0))


def _preferred_headroom_gb(*, gpu_memory_gb: float) -> float:
    """Return the preferred headroom target used for candidate ranking.

    Args:
        gpu_memory_gb: Per-GPU memory capacity in GiB.

    Returns:
        Preferred headroom target in GiB with a modest cap.
    """

    return max(
        _safety_headroom_gb(gpu_memory_gb=gpu_memory_gb),
        min(gpu_memory_gb * 0.20, 16.0),
    )


def _distributed_slowdown_fraction(*, distributed_mode: str) -> float:
    """Return a rough throughput slowdown prior for one backend.

    Args:
        distributed_mode: Candidate distributed backend.

    Returns:
        Estimated slowdown fraction versus a least-sharded baseline.
    """

    return {
        "single_gpu": 0.00,
        "ddp": 0.03,
        "zero2": 0.08,
        "zero3": 0.18,
    }[distributed_mode]


def _tensor_parallel_slowdown_fraction(*, tensor_parallel_degree: int) -> float:
    """Return a rough slowdown prior for one tensor-parallel degree.

    Args:
        tensor_parallel_degree: Megatron-style tensor-parallel degree.

    Returns:
        Estimated slowdown fraction from TP communication overhead.
    """

    if tensor_parallel_degree <= 1:
        return 0.0
    return 0.08 * math.log2(float(tensor_parallel_degree))


def _estimated_training_slowdown_fraction(*, result: MemoryResult) -> float:
    """Estimate throughput slowdown from the selected optimization stack.

    Args:
        result: Memory estimate result for one candidate config.

    Returns:
        Estimated slowdown fraction relative to a least-sharded baseline.

    Example:
        >>> slowdown = _estimated_training_slowdown_fraction(result=result)
        >>> slowdown >= 0.0
        True
    """

    multiplier = 1.0
    multiplier *= 1.0 + _distributed_slowdown_fraction(
        distributed_mode=result.config.distributed_mode
    )
    if result.config.gradient_checkpointing:
        multiplier *= 1.30
    multiplier *= 1.0 + _tensor_parallel_slowdown_fraction(
        tensor_parallel_degree=result.config.tensor_parallel_degree
    )
    if result.config.sequence_parallel:
        multiplier *= 1.04
    return multiplier - 1.0


def _headroom_ranking_adjustment(*, result: MemoryResult) -> float:
    """Return the ranking adjustment contributed by memory headroom.

    Args:
        result: Memory estimate result for one candidate.

    Returns:
        Positive values penalize tight headroom. Negative values reward
        additional headroom up to a capped preferred target.
    """

    target_headroom = _safety_headroom_gb(gpu_memory_gb=result.config.gpu_memory_gb)
    preferred_headroom = _preferred_headroom_gb(
        gpu_memory_gb=result.config.gpu_memory_gb
    )
    actual_headroom = result.headroom_gb()
    shortfall = max(0.0, target_headroom - actual_headroom)
    capped_headroom = min(max(actual_headroom, 0.0), preferred_headroom)
    headroom_bonus = 5.0 * (capped_headroom / preferred_headroom)
    return shortfall * 5.0 - headroom_bonus


def _strategy_score(*, result: MemoryResult) -> float:
    """Score one candidate strategy so lower is better.

    Args:
        result: Memory estimate result for one candidate.

    Returns:
        Ranking score combining infeasibility, headroom shortfall, and
        estimated training slowdown.
    """

    slowdown_percent = _estimated_training_slowdown_fraction(result=result) * 100.0
    if not result.feasible:
        return 1_000_000.0 + abs(result.headroom_gb()) * 1_000.0 + slowdown_percent
    return slowdown_percent + _headroom_ranking_adjustment(result=result)


def _candidate_from_result(
    *,
    distributed_mode: str,
    gradient_checkpointing: bool,
    tensor_parallel_degree: int,
    sequence_parallel: bool,
    result: MemoryResult,
) -> WebRecommendationCandidate:
    """Convert one estimate result into a serializable candidate summary."""

    estimated_slowdown_percent = (
        _estimated_training_slowdown_fraction(result=result) * 100.0
    )
    data_parallel_degree = result.config.data_parallel_degree()
    return WebRecommendationCandidate(
        distributed_mode=distributed_mode,
        gradient_checkpointing=gradient_checkpointing,
        tensor_parallel_degree=tensor_parallel_degree,
        sequence_parallel=sequence_parallel,
        data_parallel_degree=data_parallel_degree,
        feasible=result.feasible,
        global_peak_gb=result.global_peak_gb(),
        headroom_gb=result.headroom_gb(),
        estimated_slowdown_percent=estimated_slowdown_percent,
        ranking_score=_strategy_score(result=result),
    )


def _recommendation_rationale(
    *,
    recommendation: WebRecommendationCandidate,
    all_candidates: tuple[WebRecommendationCandidate, ...],
    gpu_memory_gb: float,
) -> tuple[str, ...]:
    """Build a short explanation for the chosen strategy."""

    faster_rejected = [
        candidate
        for candidate in all_candidates
        if candidate.ranking_score < recommendation.ranking_score
    ]
    rationale = [
        (
            "Recommended "
            f"{recommendation.distributed_mode}"
            f" with dp={recommendation.data_parallel_degree}"
            f"{' + tp=' + str(recommendation.tensor_parallel_degree) if recommendation.tensor_parallel_degree > 1 else ''}"
            f"{' + sp' if recommendation.sequence_parallel else ''}"
            f" with checkpointing {'on' if recommendation.gradient_checkpointing else 'off'}."
        )
    ]
    rationale.append(
        f"Estimated training slowdown is {recommendation.estimated_slowdown_percent:.1f}% versus a least-sharded baseline."
    )
    rationale.append("Ranking balances estimated slowdown against memory headroom.")
    if faster_rejected and all(not candidate.feasible for candidate in faster_rejected):
        rationale.append(
            "All lower-slowdown candidates exceeded the requested GPU memory budget."
        )
    target_headroom = _safety_headroom_gb(gpu_memory_gb=gpu_memory_gb)
    if recommendation.feasible and recommendation.headroom_gb >= target_headroom:
        rationale.append(
            f"The estimate keeps at least {target_headroom:.1f} GiB of headroom per GPU."
        )
    elif recommendation.feasible:
        rationale.append(
            "This is the fastest feasible option, but it runs close to the memory limit."
        )
    return tuple(rationale)


def _recommend_estimator_config(
    *,
    payload: dict[str, Any],
    base_config: EstimatorConfig,
    model_spec: ModelSpec,
    estimate_fn: EstimateFn,
) -> tuple[EstimatorConfig, WebRecommendation]:
    """Choose a training strategy for one web estimate request.

    Args:
        payload: Raw request payload, optionally containing explicit overrides.
        base_config: Base structural config built from the request.
        model_spec: Inspected model specification.
        estimate_fn: Estimate function dependency.

    Returns:
        Final config plus a typed recommendation summary.
    """

    candidate_pairs = _strategy_candidates_from_payload(
        payload=payload, config=base_config
    )
    assert candidate_pairs, (
        "No valid optimization combinations for the requested GPU count. "
        "After tensor parallelism, DDP and ZeRO need at least 2-way data parallelism. "
        "Sequence parallel also requires tensor_parallel_degree > 1."
    )
    results = []
    for (
        distributed_mode,
        checkpointing,
        tensor_parallel_degree,
        sequence_parallel,
    ) in candidate_pairs:
        candidate_config = replace(
            base_config,
            distributed_mode=distributed_mode,
            gradient_checkpointing=checkpointing,
            tensor_parallel_degree=tensor_parallel_degree,
            sequence_parallel=sequence_parallel,
        )
        result = estimate_fn(model_spec, candidate_config)
        candidate = _candidate_from_result(
            distributed_mode=distributed_mode,
            gradient_checkpointing=checkpointing,
            tensor_parallel_degree=tensor_parallel_degree,
            sequence_parallel=sequence_parallel,
            result=result,
        )
        results.append((candidate, candidate_config, result))
    best_candidate, best_config, _ = min(
        results, key=lambda item: item[0].ranking_score
    )
    explicit_strategy = (
        "distributed_mode" in payload
        or "gradient_checkpointing" in payload
        or "tensor_parallel_degree" in payload
        or "sequence_parallel" in payload
    )
    rationale = (
        ("Using the explicit training strategy supplied in the request.",)
        if explicit_strategy
        else _recommendation_rationale(
            recommendation=best_candidate,
            all_candidates=tuple(item[0] for item in results),
            gpu_memory_gb=base_config.gpu_memory_gb,
        )
    )
    recommendation = WebRecommendation(
        strategy_source="request_override" if explicit_strategy else "recommended",
        config=best_config,
        rationale=rationale,
        candidates=tuple(
            sorted(
                (item[0] for item in results),
                key=lambda candidate: candidate.ranking_score,
            )
        ),
    )
    return best_config, recommendation


def build_estimator_config_from_payload(*, payload: dict[str, Any]) -> EstimatorConfig:
    """Build an estimator config from one API request payload.

    Args:
        payload: JSON request body.

    Returns:
        Parsed `EstimatorConfig`.

    Example:
        >>> config = build_estimator_config_from_payload(
        ...     payload={"tuning_mode": "full_ft", "max_seq_len": 128}
        ... )
        >>> config.max_seq_len
        128
    """

    return EstimatorConfig(
        tuning_mode=str(payload["tuning_mode"]),
        distributed_mode=str(payload.get("distributed_mode", "single_gpu")),
        optimizer_name=str(payload.get("optimizer_name", "adamw")),
        attention_backend=str(payload.get("attention_backend", "flash2")),
        gradient_checkpointing=bool(payload.get("gradient_checkpointing", False)),
        tensor_parallel_degree=int(payload.get("tensor_parallel_degree", 1)),
        sequence_parallel=bool(payload.get("sequence_parallel", False)),
        vocab_parallel_logits=bool(payload.get("vocab_parallel_logits", True)),
        max_seq_len=int(payload.get("max_seq_len", 512)),
        micro_batch_size_per_gpu=int(payload.get("micro_batch_size_per_gpu", 1)),
        gpus_per_node=int(payload.get("gpus_per_node", 1)),
        num_nodes=int(payload.get("num_nodes", 1)),
        gpu_memory_gb=float(payload.get("gpu_memory_gb", 24.0)),
        weight_dtype=str(payload.get("weight_dtype", "bf16")),
        grad_dtype=str(payload.get("grad_dtype", "bf16")),
        optimizer_state_dtype=str(payload.get("optimizer_state_dtype", "auto")),
        optimizer_update_dtype=str(payload.get("optimizer_update_dtype", "auto")),
        master_weight_dtype=str(payload.get("master_weight_dtype", "fp32")),
        adapter_weight_dtype=str(payload.get("adapter_weight_dtype", "fp32")),
        adapter_grad_dtype=str(payload.get("adapter_grad_dtype", "fp32")),
        adapter_optimizer_state_dtype=str(
            payload.get("adapter_optimizer_state_dtype", "fp32")
        ),
        use_master_weights=bool(payload.get("use_master_weights", False)),
        lora=_build_lora_config(payload=payload),
        runtime_floor_gb_override=payload.get("runtime_floor_gb_override"),
    )


def _build_estimate_payload(
    *,
    request_payload: dict[str, Any],
    estimate_fn: EstimateFn,
    inspect_fn: InspectFn,
) -> dict[str, Any]:
    """Build the JSON response payload for one estimate request.

    Args:
        request_payload: Parsed request body.
        estimate_fn: Estimate function dependency.
        inspect_fn: Model inspection dependency.

    Returns:
        JSON-serializable estimate response.
    """

    model_ref = str(request_payload["model"]).strip()
    assert model_ref, "Model is required."
    model_spec = inspect_fn(model_ref)
    base_config = build_estimator_config_from_payload(payload=request_payload)
    config, recommendation = _recommend_estimator_config(
        payload=request_payload,
        base_config=base_config,
        model_spec=model_spec,
        estimate_fn=estimate_fn,
    )
    result = estimate_fn(model_spec, config)
    estimate_payload = asdict(result)
    estimate_payload["global_peak_gb"] = result.global_peak_gb()
    estimate_payload["headroom_gb"] = result.headroom_gb()
    estimate_payload["comparable_metadata"] = result.comparable_metadata()
    return {
        "model_spec": asdict(model_spec),
        "recommendation": asdict(recommendation),
        "estimate": estimate_payload,
    }


def build_request_handler(
    *,
    estimate_fn: EstimateFn = estimate_peak_memory,
    inspect_fn: InspectFn = inspect_model,
) -> type[BaseHTTPRequestHandler]:
    """Build a request-handler class bound to estimator dependencies.

    Args:
        estimate_fn: Memory-estimation dependency.
        inspect_fn: Model-inspection dependency.

    Returns:
        `BaseHTTPRequestHandler` subclass serving the web UI.
    """

    class RequestHandler(BaseHTTPRequestHandler):
        """HTTP handler for the SimpleSFT estimator web app."""

        def do_GET(self) -> None:  # noqa: N802
            """Serve the HTML app or a simple health endpoint."""

            if self.path == "/":
                _html_response(handler=self, content=APP_HTML)
                return
            if self.path == "/api/health":
                _json_response(
                    handler=self,
                    status=HTTPStatus.OK,
                    payload={"ok": True},
                )
                return
            _json_response(
                handler=self,
                status=HTTPStatus.NOT_FOUND,
                payload={"error": f"Unknown path: {self.path}"},
            )

        def do_POST(self) -> None:  # noqa: N802
            """Handle estimate requests."""

            if self.path != "/api/estimate":
                _json_response(
                    handler=self,
                    status=HTTPStatus.NOT_FOUND,
                    payload={"error": f"Unknown path: {self.path}"},
                )
                return
            try:
                payload = _read_json_payload(handler=self)
                response_payload = _build_estimate_payload(
                    request_payload=payload,
                    estimate_fn=estimate_fn,
                    inspect_fn=inspect_fn,
                )
                _json_response(
                    handler=self,
                    status=HTTPStatus.OK,
                    payload=response_payload,
                )
            except Exception as exc:
                _json_response(
                    handler=self,
                    status=HTTPStatus.BAD_REQUEST,
                    payload={"error": str(exc)},
                )

        def log_message(self, format: str, *args: Any) -> None:
            """Suppress default noisy request logging."""

            return

    return RequestHandler


def create_web_server(
    *,
    host: str,
    port: int,
    estimate_fn: EstimateFn = estimate_peak_memory,
    inspect_fn: InspectFn = inspect_model,
) -> ThreadingHTTPServer:
    """Create the threaded SimpleSFT web server.

    Args:
        host: Bind host.
        port: Bind port. Use `0` for an ephemeral port.
        estimate_fn: Memory-estimation dependency.
        inspect_fn: Model-inspection dependency.

    Returns:
        Configured `ThreadingHTTPServer`.
    """

    handler = build_request_handler(estimate_fn=estimate_fn, inspect_fn=inspect_fn)
    return ThreadingHTTPServer((host, port), handler)


def _probe_existing_web_server(*, host: str, port: int) -> bool:
    """Check whether a SimpleSFT web server is already running on one port.

    Args:
        host: Bind host to probe.
        port: Port to probe.

    Returns:
        `True` when the health endpoint responds with `{"ok": true}`.
    """

    url = f"http://{host}:{port}/api/health"
    try:
        with urlopen(url, timeout=0.5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, json.JSONDecodeError):
        return False
    return payload == {"ok": True}


def _create_or_reuse_server(
    *,
    host: str,
    port: int,
    estimate_fn: EstimateFn,
    inspect_fn: InspectFn,
) -> tuple[Optional[ThreadingHTTPServer], str, Optional[str]]:
    """Create a web server or reuse an existing healthy instance.

    Args:
        host: Requested bind host.
        port: Requested bind port.
        estimate_fn: Memory-estimation dependency.
        inspect_fn: Model-inspection dependency.

    Returns:
        Tuple of `(server, url, notice)`. `server` is `None` when a healthy
        existing server is already serving the requested port.

    Example:
        >>> server, url, notice = _create_or_reuse_server(
        ...     host="127.0.0.1",
        ...     port=0,
        ...     estimate_fn=estimate_peak_memory,
        ...     inspect_fn=inspect_model,
        ... )
        >>> server is not None
        True
    """

    try:
        server = create_web_server(
            host=host,
            port=port,
            estimate_fn=estimate_fn,
            inspect_fn=inspect_fn,
        )
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise
        if _probe_existing_web_server(host=host, port=port):
            return None, f"http://{host}:{port}", "Reusing existing SimpleSFT web UI."
        fallback_server = create_web_server(
            host=host,
            port=0,
            estimate_fn=estimate_fn,
            inspect_fn=inspect_fn,
        )
        fallback_url = f"http://{host}:{fallback_server.server_port}"
        return (
            fallback_server,
            fallback_url,
            f"Port {port} was busy; bound to {fallback_server.server_port} instead.",
        )
    server_url = f"http://{host}:{server.server_port}"
    return server, server_url, None


def serve_web_interface(
    *, host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True
) -> None:
    """Serve the local web UI until interrupted.

    Args:
        host: Bind host.
        port: Bind port.
        open_browser: Whether to open the browser automatically.
    """

    server, url, notice = _create_or_reuse_server(
        host=host,
        port=port,
        estimate_fn=estimate_peak_memory,
        inspect_fn=inspect_model,
    )
    if notice is not None:
        print(notice)
    if open_browser:
        webbrowser.open(url)
    print(f"SimpleSFT web UI listening on {url}")
    if server is None:
        return
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
