"""Tests for allocator-level soft and stressed peak models."""

from __future__ import annotations

from simplesft.allocator_model import (
    build_allocator_metadata,
    build_allocator_phase_states,
    selected_allocator_peak_state,
)
from simplesft.types import PhaseMemoryRecord


def _phase_record(
    *,
    phase_name: str,
    peak_allocated_bytes: int,
    peak_reserved_bytes: int,
) -> PhaseMemoryRecord:
    """Return a compact phase record for allocator-model tests."""

    return PhaseMemoryRecord(
        phase_name=phase_name,
        allocated_bytes=peak_allocated_bytes,
        reserved_bytes=peak_reserved_bytes,
        peak_allocated_bytes=peak_allocated_bytes,
        peak_reserved_bytes=peak_reserved_bytes,
        delta_allocated_bytes=0,
        delta_reserved_bytes=0,
    )


def test_allocator_stressed_peak_reclaims_reclaimable_idle_cache() -> None:
    """Stressed peaks should drop reclaimable idle cache above capacity."""

    states = build_allocator_phase_states(
        phase_records=(
            _phase_record(
                phase_name="forward",
                peak_allocated_bytes=80,
                peak_reserved_bytes=120,
            ),
        ),
        runtime_reserved_only_bytes=10,
        gpu_capacity_bytes=100,
        stress_trigger_fraction=1.0,
    )
    assert states[0].idle_cache_peak_bytes == 40
    assert states[0].pinned_reserved_peak_bytes == 10
    assert states[0].reclaimable_idle_cache_bytes == 30
    assert states[0].stressed_reserved_peak_bytes == 90


def test_allocator_stressed_peak_keeps_soft_peak_below_trigger() -> None:
    """Stressed peaks should match soft peaks when capacity is not exceeded."""

    states = build_allocator_phase_states(
        phase_records=(
            _phase_record(
                phase_name="forward",
                peak_allocated_bytes=80,
                peak_reserved_bytes=95,
            ),
        ),
        runtime_reserved_only_bytes=10,
        gpu_capacity_bytes=100,
        stress_trigger_fraction=1.0,
    )
    assert states[0].stressed_reserved_peak_bytes == 95


def test_allocator_metadata_tracks_soft_and_stressed_peaks() -> None:
    """Allocator metadata should expose all three top-level peak modes."""

    states = build_allocator_phase_states(
        phase_records=(
            _phase_record(
                phase_name="forward",
                peak_allocated_bytes=80,
                peak_reserved_bytes=120,
            ),
            _phase_record(
                phase_name="backward",
                peak_allocated_bytes=100,
                peak_reserved_bytes=115,
            ),
        ),
        runtime_reserved_only_bytes=10,
        gpu_capacity_bytes=100,
        stress_trigger_fraction=1.0,
    )
    metadata = build_allocator_metadata(allocator_phase_states=states)
    assert metadata["soft_peak_phase"] == "forward"
    assert metadata["stressed_peak_phase"] == "backward"
    assert metadata["allocated_peak_phase"] == "backward"


def test_selected_allocator_peak_state_uses_requested_mode() -> None:
    """Allocator peak selection should honor allocated vs reserved modes."""

    states = build_allocator_phase_states(
        phase_records=(
            _phase_record(
                phase_name="forward",
                peak_allocated_bytes=90,
                peak_reserved_bytes=120,
            ),
            _phase_record(
                phase_name="backward",
                peak_allocated_bytes=100,
                peak_reserved_bytes=110,
            ),
        ),
        runtime_reserved_only_bytes=10,
        gpu_capacity_bytes=100,
        stress_trigger_fraction=1.0,
    )
    assert (
        selected_allocator_peak_state(
            allocator_phase_states=states,
            allocator_peak_mode="soft_reserved",
        ).phase_name
        == "forward"
    )
    assert (
        selected_allocator_peak_state(
            allocator_phase_states=states,
            allocator_peak_mode="stressed_reserved",
        ).phase_name
        == "backward"
    )
    assert (
        selected_allocator_peak_state(
            allocator_phase_states=states,
            allocator_peak_mode="allocated",
        ).phase_name
        == "backward"
    )
