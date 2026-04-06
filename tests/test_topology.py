"""Tests for topology-aware NCCL environment handling."""

import simplesft.measurement.topology as topology_module


def test_maybe_apply_cross_numa_nccl_env_sets_nccl_p2p_disable(monkeypatch) -> None:
    """Cross-NUMA 2xA100 hosts should disable NCCL P2P."""

    monkeypatch.setattr(
        topology_module,
        "read_gpu_product_names",
        lambda: ["NVIDIA A100-PCIE-40GB", "NVIDIA A100-PCIE-40GB"],
    )
    monkeypatch.setattr(
        topology_module,
        "read_topology_output",
        lambda: (
            "GPU0 GPU1 CPU Affinity NUMA Affinity\n"
            "GPU0 X SYS 0-31 0\n"
            "GPU1 SYS X 32-63 1\n"
        ),
    )
    fake_env: dict[str, str] = {}
    applied, reason = topology_module.maybe_apply_cross_numa_nccl_env(env=fake_env)
    assert applied is True
    assert reason == "cross-NUMA topology detected"
    assert fake_env["NCCL_P2P_DISABLE"] == "1"


def test_maybe_apply_cross_numa_nccl_env_skips_non_matching_shape(monkeypatch) -> None:
    """Non-2xA100 hosts should skip the cross-NUMA NCCL override."""

    monkeypatch.setattr(
        topology_module, "read_gpu_product_names", lambda: ["NVIDIA L4"]
    )
    fake_env: dict[str, str] = {}
    applied, reason = topology_module.maybe_apply_cross_numa_nccl_env(env=fake_env)
    assert applied is False
    assert reason == "requires exactly two A100 GPUs"
    assert "NCCL_P2P_DISABLE" not in fake_env
