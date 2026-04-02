"""Tests for phase-local workspace attribution helpers."""

from __future__ import annotations

from typing import Optional

from simplesft.attribution import (
    build_reserved_carryover_metadata,
    build_workspace_proxy_metadata,
)
from simplesft.types import MemoryComponentBreakdown, PhaseMemoryRecord


def _phase_record(
    *,
    phase_name: str,
    reserved_bytes: int,
    peak_reserved_bytes: int,
    allocated_bytes: Optional[int] = None,
) -> PhaseMemoryRecord:
    """Build a compact phase record for workspace attribution tests."""

    return PhaseMemoryRecord(
        phase_name=phase_name,
        allocated_bytes=reserved_bytes if allocated_bytes is None else allocated_bytes,
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
            _phase_record(
                phase_name="forward", reserved_bytes=20, peak_reserved_bytes=25
            ),
            _phase_record(
                phase_name="backward", reserved_bytes=30, peak_reserved_bytes=38
            ),
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


def test_build_reserved_carryover_metadata_returns_reserved_minus_allocated() -> None:
    """Carry-over metadata should reflect reserved-over-allocated bytes."""

    metadata = build_reserved_carryover_metadata(
        phase_records=[
            _phase_record(
                phase_name="forward", reserved_bytes=20, peak_reserved_bytes=25
            ),
            _phase_record(
                phase_name="loss_materialization",
                reserved_bytes=19,
                peak_reserved_bytes=22,
                allocated_bytes=17,
            ),
            _phase_record(
                phase_name="backward",
                reserved_bytes=30,
                peak_reserved_bytes=38,
                allocated_bytes=26,
            ),
            _phase_record(
                phase_name="optimizer_step",
                reserved_bytes=35,
                peak_reserved_bytes=45,
                allocated_bytes=25,
            ),
            _phase_record(
                phase_name="zero_grad",
                reserved_bytes=33,
                peak_reserved_bytes=35,
                allocated_bytes=30,
            ),
            _phase_record(
                phase_name="step_end",
                reserved_bytes=31,
                peak_reserved_bytes=31,
                allocated_bytes=31,
            ),
        ]
    )
    assert metadata["reserved_carryover_to_loss_materialization_bytes"] == 2
    assert metadata["reserved_carryover_to_backward_bytes"] == 4
    assert metadata["reserved_carryover_to_optimizer_step_bytes"] == 10
