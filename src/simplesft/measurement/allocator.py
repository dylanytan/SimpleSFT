"""Allocator-level peak models derived from phase-local allocated/reserved peaks."""

from __future__ import annotations

from dataclasses import dataclass

from ..types import PhaseMemoryRecord


@dataclass(frozen=True)
class AllocatorPhaseState:
    """Allocator state derived from one phase peak.

    Args:
        phase_name: Phase label for the state.
        allocated_peak_bytes: Instantaneous allocated peak for the phase.
        soft_reserved_peak_bytes: Reserved peak before stress-triggered reclaim.
        pinned_reserved_peak_bytes: Idle reserved bytes treated as non-reclaimable.
        idle_cache_peak_bytes: Reserved-but-unallocated bytes at the phase peak.
        reclaimable_idle_cache_bytes: Idle cache eligible for reclaim under stress.
        stressed_reserved_peak_bytes: Reserved peak after reclaiming idle cache.

    Returns:
        Frozen dataclass describing allocator-level phase state.
    """

    phase_name: str
    allocated_peak_bytes: int
    soft_reserved_peak_bytes: int
    pinned_reserved_peak_bytes: int
    idle_cache_peak_bytes: int
    reclaimable_idle_cache_bytes: int
    stressed_reserved_peak_bytes: int


def _idle_cache_peak_bytes(*, record: PhaseMemoryRecord) -> int:
    """Return reserved-but-unallocated bytes at the phase peak."""

    return max(0, record.peak_reserved_bytes - record.peak_allocated_bytes)


def _pinned_reserved_peak_bytes(
    *,
    idle_cache_peak_bytes: int,
    runtime_reserved_only_bytes: int,
) -> int:
    """Return non-reclaimable idle reserved bytes at the phase peak."""

    return min(idle_cache_peak_bytes, runtime_reserved_only_bytes)


def _stressed_reserved_peak_bytes(
    *,
    soft_reserved_peak_bytes: int,
    reclaimable_idle_cache_bytes: int,
    stress_trigger_bytes: int,
) -> int:
    """Return reserved peak after reclaiming idle cache under pressure."""

    if soft_reserved_peak_bytes <= stress_trigger_bytes:
        return soft_reserved_peak_bytes
    return soft_reserved_peak_bytes - reclaimable_idle_cache_bytes


def build_allocator_phase_states(
    *,
    phase_records: tuple[PhaseMemoryRecord, ...] | list[PhaseMemoryRecord],
    runtime_reserved_only_bytes: int,
    gpu_capacity_bytes: int,
    stress_trigger_fraction: float,
) -> tuple[AllocatorPhaseState, ...]:
    """Return allocator phase states for soft and stressed reserve models.

    Args:
        phase_records: Ordered phase timeline for one estimate.
        runtime_reserved_only_bytes: Runtime pool bytes treated as pinned reserve.
        gpu_capacity_bytes: Per-rank GPU capacity in bytes.
        stress_trigger_fraction: Fraction of capacity that triggers reclaim.

    Returns:
        Per-phase allocator states with allocated, soft-reserved, and
        stressed-reserved peaks.
    """

    stress_trigger_bytes = int(round(gpu_capacity_bytes * stress_trigger_fraction))
    states: list[AllocatorPhaseState] = []
    for record in phase_records:
        idle_cache_bytes = _idle_cache_peak_bytes(record=record)
        pinned_reserved_bytes = _pinned_reserved_peak_bytes(
            idle_cache_peak_bytes=idle_cache_bytes,
            runtime_reserved_only_bytes=runtime_reserved_only_bytes,
        )
        reclaimable_idle_bytes = max(0, idle_cache_bytes - pinned_reserved_bytes)
        stressed_reserved_bytes = _stressed_reserved_peak_bytes(
            soft_reserved_peak_bytes=record.peak_reserved_bytes,
            reclaimable_idle_cache_bytes=reclaimable_idle_bytes,
            stress_trigger_bytes=stress_trigger_bytes,
        )
        states.append(
            AllocatorPhaseState(
                phase_name=record.phase_name,
                allocated_peak_bytes=record.peak_allocated_bytes,
                soft_reserved_peak_bytes=record.peak_reserved_bytes,
                pinned_reserved_peak_bytes=pinned_reserved_bytes,
                idle_cache_peak_bytes=idle_cache_bytes,
                reclaimable_idle_cache_bytes=reclaimable_idle_bytes,
                stressed_reserved_peak_bytes=stressed_reserved_bytes,
            )
        )
    return tuple(states)


def selected_allocator_peak_state(
    *,
    allocator_phase_states: tuple[AllocatorPhaseState, ...],
    allocator_peak_mode: str,
) -> AllocatorPhaseState:
    """Return the phase selected by the configured allocator peak mode.

    Args:
        allocator_phase_states: Per-phase allocator states.
        allocator_peak_mode: One of `soft_reserved`, `stressed_reserved`,
            or `allocated`.

    Returns:
        The allocator phase state with the highest peak under the selected mode.
    """

    if allocator_peak_mode == "allocated":
        return max(
            allocator_phase_states,
            key=lambda state: state.allocated_peak_bytes,
        )
    if allocator_peak_mode == "stressed_reserved":
        return max(
            allocator_phase_states,
            key=lambda state: state.stressed_reserved_peak_bytes,
        )
    return max(
        allocator_phase_states,
        key=lambda state: state.soft_reserved_peak_bytes,
    )


def build_allocator_metadata(
    *,
    allocator_phase_states: tuple[AllocatorPhaseState, ...],
) -> dict[str, int | str]:
    """Return allocator-level metadata for estimate reporting.

    Args:
        allocator_phase_states: Per-phase allocator states.

    Returns:
        Mapping with global peaks and the selected reclaimable idle-cache state.
    """

    soft_peak_state = selected_allocator_peak_state(
        allocator_phase_states=allocator_phase_states,
        allocator_peak_mode="soft_reserved",
    )
    stressed_peak_state = selected_allocator_peak_state(
        allocator_phase_states=allocator_phase_states,
        allocator_peak_mode="stressed_reserved",
    )
    allocated_peak_state = selected_allocator_peak_state(
        allocator_phase_states=allocator_phase_states,
        allocator_peak_mode="allocated",
    )
    return {
        "soft_global_peak_bytes": soft_peak_state.soft_reserved_peak_bytes,
        "soft_peak_phase": soft_peak_state.phase_name,
        "stressed_global_peak_bytes": stressed_peak_state.stressed_reserved_peak_bytes,
        "stressed_peak_phase": stressed_peak_state.phase_name,
        "allocated_global_peak_bytes": allocated_peak_state.allocated_peak_bytes,
        "allocated_peak_phase": allocated_peak_state.phase_name,
        "stressed_peak_idle_cache_bytes": stressed_peak_state.idle_cache_peak_bytes,
        "stressed_peak_reclaimable_idle_cache_bytes": (
            stressed_peak_state.reclaimable_idle_cache_bytes
        ),
        "stressed_peak_pinned_reserved_bytes": (
            stressed_peak_state.pinned_reserved_peak_bytes
        ),
    }
