"""Tests for the local SimpleSFT web interface."""

from __future__ import annotations

import json
import socket
import threading
from contextlib import contextmanager
from typing import Iterator
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from simplesft.estimator.estimate import estimate_peak_memory
from simplesft.types import (
    EstimatorConfig,
    MemoryComponentBreakdown,
    MemoryResult,
    ModelLinearLayerSpec,
    ModelSpec,
)
from simplesft.web.server import (
    _estimated_training_slowdown_fraction,
    _strategy_score,
    _create_or_reuse_server,
    build_estimator_config_from_payload,
    create_web_server,
)


def _toy_model_spec() -> ModelSpec:
    """Return a compact model spec for web-server tests."""

    return ModelSpec(
        model_name="toy",
        model_type="llama",
        num_layers=2,
        hidden_size=32,
        num_attention_heads=4,
        intermediate_size=64,
        vocab_size=128,
        max_position_embeddings=128,
        total_params=10_000,
        trainable_linear_layers=(
            ModelLinearLayerSpec("layers.0.self_attn.q_proj", 32, 32, "attention"),
            ModelLinearLayerSpec("layers.0.self_attn.k_proj", 32, 32, "attention"),
            ModelLinearLayerSpec("layers.0.mlp.up_proj", 32, 64, "mlp"),
        ),
    )


def _scored_result(*, global_peak_gb: float) -> MemoryResult:
    """Build a minimal feasible result with controllable headroom.

    Args:
        global_peak_gb: Peak memory per GPU in GiB.

    Returns:
        Minimal estimate result suitable for ranking tests.
    """

    gib_bytes = 1024**3
    return MemoryResult(
        mode="estimate",
        model_name="toy",
        config=EstimatorConfig(
            tuning_mode="full_ft",
            distributed_mode="zero2",
            gpus_per_node=4,
            gpu_memory_gb=80.0,
        ),
        breakdown=MemoryComponentBreakdown(),
        phase_records=(),
        peak_phase="backward",
        global_peak_bytes=int(global_peak_gb * gib_bytes),
        feasible=global_peak_gb <= 80.0,
    )


@contextmanager
def _running_server() -> Iterator[str]:
    """Run the web server on an ephemeral local port for one test."""

    server = create_web_server(
        host="127.0.0.1",
        port=0,
        inspect_fn=lambda model_ref: _toy_model_spec(),
        estimate_fn=lambda model, config: estimate_peak_memory(model, config),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_build_estimator_config_from_payload_parses_lora() -> None:
    """Web payload parsing should build the structural estimator config."""

    config = build_estimator_config_from_payload(
        payload={
            "tuning_mode": "lora",
            "distributed_mode": "zero3",
            "max_seq_len": 2048,
            "lora": {
                "rank": 8,
                "alpha": 16,
                "dropout": 0.05,
                "target_modules": ["q_proj", "k_proj"],
                "bias": "none",
            },
        }
    )
    assert isinstance(config, EstimatorConfig)
    assert config.distributed_mode == "zero3"
    assert config.lora is not None
    assert config.lora.rank == 8


def test_build_estimator_config_from_payload_defaults_to_flash2() -> None:
    """Web payload parsing should default the attention backend to flash2."""

    config = build_estimator_config_from_payload(
        payload={
            "tuning_mode": "full_ft",
            "max_seq_len": 512,
        }
    )
    assert config.attention_backend == "flash2"


def test_build_estimator_config_from_payload_parses_tp_and_sp() -> None:
    """Web payload parsing should accept TP/SP overrides."""

    config = build_estimator_config_from_payload(
        payload={
            "tuning_mode": "full_ft",
            "distributed_mode": "zero2",
            "gpus_per_node": 4,
            "tensor_parallel_degree": 2,
            "sequence_parallel": True,
        }
    )
    assert config.tensor_parallel_degree == 2
    assert config.sequence_parallel is True


def test_web_server_serves_html_and_estimate_json() -> None:
    """The local web server should serve the UI and JSON estimate endpoint."""

    with _running_server() as base_url:
        html = urlopen(f"{base_url}/").read().decode("utf-8")
        assert "Memory estimate workbench" in html
        assert "Memory Composition" in html
        assert "Candidate Strategies" in html
        assert "Est. slowdown" in html
        assert "Fixed / Peak / Free" in html
        assert "Itemized view" in html
        assert "bar-marker peak" in html
        assert "bar-marker gpu" in html
        assert "Backend buffers" in html
        assert "Peak overhead" in html
        assert "markerLabelStyle" in html
        assert 'return "optim";' in html
        assert "memory-tooltip" in html
        assert "showMemoryTooltip" in html
        assert 'data-tooltip="${escapeAttribute(tooltip)}"' in html
        assert "submitEstimate();" in html
        assert "Data parallel" in html
        assert "World size" in html
        assert "Overhead" in html
        assert 'id="model_select"' in html
        assert '<optgroup label="Qwen">' in html
        assert '<optgroup label="microsoft">' in html
        assert 'id="lora_target_modules"' in html
        assert 'id="tensor_parallel_degree"' in html
        assert 'id="sequence_parallel"' in html
        assert 'id="vocab_parallel_logits"' in html
        assert "multiple" in html
        assert 'name="use_master_weights"' not in html
        assert 'name="distributed_mode"' not in html
        assert 'name="gradient_checkpointing"' not in html
        assert "Per-GPU peak snapshot" not in html
        assert "Checkpoint recompute is folded into activations" not in html
        assert "Raw JSON" not in html
        assert "Model Architecture" not in html
        assert "Recommended Strategy" not in html
        assert "Retained Activations" not in html
        assert "Workspace Windows" not in html
        assert "Phase Peaks" not in html
        assert html.index("renderBreakdownBar(displayBreakdown),") < html.index(
            'renderTable(\n          "Candidate Strategies",'
        )
        request = Request(
            url=f"{base_url}/api/estimate",
            data=json.dumps(
                {
                    "model": "toy",
                    "tuning_mode": "full_ft",
                    "max_seq_len": 128,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        payload = json.loads(urlopen(request).read().decode("utf-8"))
        assert payload["model_spec"]["model_name"] == "toy"
        assert payload["estimate"]["mode"] == "estimate"
        assert payload["support_status"] == "officially supported"
        assert payload["catalog_status"] == "custom model path"
        assert payload["recommendation"]["strategy_source"] == "recommended"
        assert payload["estimate"]["config"]["distributed_mode"] == "single_gpu"
        assert payload["estimate"]["config"]["tensor_parallel_degree"] >= 1
        assert payload["estimate"]["metadata"]["data_parallel_degree"] >= 1
        assert (
            payload["model_spec"]["architecture_family"]["family_label"]
            == "llama_dense"
        )
        assert payload["model_spec"]["tensor_layout_summary"] == (
            "col(qkv/up) · row(o/down) · vocab(embed/lm_head)"
        )
        assert (
            payload["recommendation"]["candidates"][0]["estimated_slowdown_percent"]
            >= 0.0
        )
        assert "dp=" in payload["recommendation"]["rationale"][0]
        assert (
            payload["estimate"]["debug"]["activations"]["hook_visible_activation_bytes"]
            > 0
        )


def test_web_server_recommends_distributed_mode_for_multi_gpu_requests() -> None:
    """Multi-GPU web requests should recommend a distributed backend."""

    with _running_server() as base_url:
        request = Request(
            url=f"{base_url}/api/estimate",
            data=json.dumps(
                {
                    "model": "toy",
                    "tuning_mode": "full_ft",
                    "max_seq_len": 128,
                    "gpus_per_node": 4,
                    "gpu_memory_gb": 80,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        payload = json.loads(urlopen(request).read().decode("utf-8"))
        assert payload["recommendation"]["config"]["distributed_mode"] != "single_gpu"


def test_web_server_allows_tp_only_on_two_gpus() -> None:
    """Two-GPU requests should admit a pure tensor-parallel candidate."""

    with _running_server() as base_url:
        request = Request(
            url=f"{base_url}/api/estimate",
            data=json.dumps(
                {
                    "model": "toy",
                    "tuning_mode": "full_ft",
                    "max_seq_len": 128,
                    "gpus_per_node": 2,
                    "gpu_memory_gb": 80,
                    "tensor_parallel_degree": 2,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        payload = json.loads(urlopen(request).read().decode("utf-8"))
        assert payload["recommendation"]["config"]["distributed_mode"] == "tp_only"
        assert payload["recommendation"]["config"]["tensor_parallel_degree"] == 2
        assert payload["estimate"]["metadata"]["data_parallel_degree"] == 1


def test_slowdown_model_prefers_zero2_over_ddp_tp2_when_both_fit() -> None:
    """Slowdown scoring should prefer lower-overhead ZeRO over extra TP."""

    model = _toy_model_spec()
    zero2_result = estimate_peak_memory(
        model,
        EstimatorConfig(
            tuning_mode="full_ft",
            distributed_mode="zero2",
            max_seq_len=128,
            micro_batch_size_per_gpu=1,
            gpus_per_node=4,
            gpu_memory_gb=80.0,
        ),
    )
    ddp_tp2_result = estimate_peak_memory(
        model,
        EstimatorConfig(
            tuning_mode="full_ft",
            distributed_mode="ddp",
            tensor_parallel_degree=2,
            max_seq_len=128,
            micro_batch_size_per_gpu=1,
            gpus_per_node=4,
            gpu_memory_gb=80.0,
        ),
    )
    assert zero2_result.feasible
    assert ddp_tp2_result.feasible
    assert _estimated_training_slowdown_fraction(result=zero2_result) < (
        _estimated_training_slowdown_fraction(result=ddp_tp2_result)
    )


def test_ranking_prefers_more_headroom_for_same_strategy() -> None:
    """Ranking should prefer additional headroom when slowdown is unchanged."""

    roomier_result = _scored_result(global_peak_gb=68.0)
    tighter_result = _scored_result(global_peak_gb=76.0)
    assert _estimated_training_slowdown_fraction(result=roomier_result) == (
        _estimated_training_slowdown_fraction(result=tighter_result)
    )
    assert roomier_result.headroom_gb() > tighter_result.headroom_gb()
    assert _strategy_score(result=roomier_result) < _strategy_score(
        result=tighter_result
    )


def test_web_server_rejects_invalid_zero_tp_configuration() -> None:
    """Invalid ZeRO+TP combinations should be rejected by the API."""

    with _running_server() as base_url:
        request = Request(
            url=f"{base_url}/api/estimate",
            data=json.dumps(
                {
                    "model": "toy",
                    "tuning_mode": "full_ft",
                    "max_seq_len": 128,
                    "gpus_per_node": 4,
                    "distributed_mode": "zero2",
                    "tensor_parallel_degree": 4,
                    "sequence_parallel": True,
                }
            ).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            urlopen(request)
        except HTTPError as exc:
            payload = json.loads(exc.read().decode("utf-8"))
            assert exc.code == 400
            assert "No valid optimization combinations" in payload["error"]
        else:
            raise AssertionError("Expected invalid TP/ZeRO combination to fail.")


def test_create_or_reuse_server_reuses_existing_health_endpoint() -> None:
    """Existing healthy SimpleSFT servers should be reused instead of failing."""

    with _running_server() as base_url:
        port = int(base_url.rsplit(":", maxsplit=1)[1])
        server, url, notice = _create_or_reuse_server(
            host="127.0.0.1",
            port=port,
            inspect_fn=lambda model_ref: _toy_model_spec(),
            estimate_fn=lambda model, config: estimate_peak_memory(model, config),
        )
        assert server is None
        assert url == base_url
        assert notice is not None


def test_create_or_reuse_server_falls_back_when_port_is_busy() -> None:
    """Busy non-SimpleSFT ports should fall back to an ephemeral local port."""

    blocker = socket.socket()
    blocker.bind(("127.0.0.1", 0))
    blocker.listen(1)
    port = blocker.getsockname()[1]
    try:
        server, url, notice = _create_or_reuse_server(
            host="127.0.0.1",
            port=port,
            inspect_fn=lambda model_ref: _toy_model_spec(),
            estimate_fn=lambda model, config: estimate_peak_memory(model, config),
        )
        assert server is not None
        assert f":{port}" not in url
        assert notice is not None
        server.server_close()
    finally:
        blocker.close()
