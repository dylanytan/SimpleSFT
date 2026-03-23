"""Tests for phase-local workspace attribution helpers."""

from simplesft.attribution import build_workspace_proxy_metadata
from simplesft.types import MemoryComponentBreakdown, PhaseMemoryRecord


def _phase_record(*, phase_name: str, reserved_bytes: int, peak_reserved_bytes: int) -> PhaseMemoryRecord:
    """Build a compact phase record for workspace attribution tests."""

    return PhaseMemoryRecord(
        phase_name=phase_name,
        allocated_bytes=reserved_bytes,
        reserved_bytes=reserved_bytes,
        peak_allocated_bytes=peak_reserved_bytes,
        peak_reserved_bytes=peak_reserved_bytes,
        delta_allocated_bytes=0,
        delta_reserved_bytes=0,
    )


def test_build_workspace_proxy_metadata_returns_phase_local_proxies() -> None:
    """Workspace proxies should capture forward, backward, and optimizer spikes."""

    metadata = build_workspace_proxy_metadata(
        phase_records=[
            _phase_record(
                phase_name="batch_materialization",
                reserved_bytes=10,
                peak_reserved_bytes=10,
            ),
            _phase_record(phase_name="forward", reserved_bytes=20, peak_reserved_bytes=25),
            _phase_record(phase_name="backward", reserved_bytes=30, peak_reserved_bytes=38),
            _phase_record(
                phase_name="optimizer_step",
                reserved_bytes=40,
                peak_reserved_bytes=45,
            ),
        ],
        breakdown=MemoryComponentBreakdown(activation_bytes=8, gradient_bytes=6),
    )
    assert metadata["forward_workspace_proxy_bytes"] == 7
    assert metadata["backward_workspace_proxy_bytes"] == 12
    assert metadata["optimizer_workspace_proxy_bytes"] == 15
