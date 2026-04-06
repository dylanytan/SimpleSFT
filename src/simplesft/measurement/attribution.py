"""Helpers for attributing phase-local workspace from memory traces."""

from __future__ import annotations

from ..types import MemoryComponentBreakdown, PhaseMemoryRecord


def build_workspace_proxy_metadata(
    *,
    phase_records: tuple[PhaseMemoryRecord, ...] | list[PhaseMemoryRecord],
    breakdown: MemoryComponentBreakdown,
) -> dict[str, int]:
    """Build phase-local workspace proxy metrics from phase peaks.

    Args:
        phase_records: Ordered phase records for a measured or estimated step.
        breakdown: Modular memory breakdown for the same step.

    Returns:
        Mapping with forward, backward, and optimizer workspace proxy bytes.

    Example:
        >>> records = [
        ...     PhaseMemoryRecord(phase_name="batch_materialization", reserved_bytes=10),
        ...     PhaseMemoryRecord(phase_name="forward", peak_reserved_bytes=25, reserved_bytes=20),
        ...     PhaseMemoryRecord(phase_name="backward", peak_reserved_bytes=35, reserved_bytes=30),
        ...     PhaseMemoryRecord(phase_name="optimizer_step", peak_reserved_bytes=40, reserved_bytes=40),
        ... ]
        >>> build_workspace_proxy_metadata(
        ...     phase_records=records,
        ...     breakdown=MemoryComponentBreakdown(activation_bytes=8, gradient_bytes=6),
        ... )["forward_workspace_proxy_bytes"]
        7
    """

    phase_by_name = {record.phase_name: record for record in phase_records}
    batch_record = phase_by_name["batch_materialization"]
    forward_record = phase_by_name["forward"]
    backward_record = phase_by_name["backward"]
    optimizer_record = phase_by_name["optimizer_step"]
    forward_workspace_bytes = max(
        0,
        forward_record.peak_reserved_bytes
        - batch_record.reserved_bytes
        - breakdown.activation_bytes,
    )
    backward_workspace_bytes = max(
        0,
        backward_record.peak_reserved_bytes
        - forward_record.reserved_bytes
        - breakdown.gradient_bytes,
    )
    optimizer_workspace_bytes = max(
        0,
        optimizer_record.peak_reserved_bytes - backward_record.reserved_bytes,
    )
    return {
        "forward_workspace_proxy_bytes": forward_workspace_bytes,
        "backward_workspace_proxy_bytes": backward_workspace_bytes,
        "optimizer_workspace_proxy_bytes": optimizer_workspace_bytes,
    }


def build_reserved_carryover_metadata(
    *,
    phase_records: tuple[PhaseMemoryRecord, ...] | list[PhaseMemoryRecord],
) -> dict[str, int]:
    """Build reserved-memory carry-over metrics between phase boundaries.

    Args:
        phase_records: Ordered phase records for a measured or estimated step.

    Returns:
        Mapping with reserved-over-allocated carry-over bytes for key transitions.

    Example:
        >>> records = [
        ...     PhaseMemoryRecord(phase_name="forward", allocated_bytes=10, reserved_bytes=12),
        ...     PhaseMemoryRecord(phase_name="loss_materialization", allocated_bytes=9, reserved_bytes=11),
        ... ]
        >>> build_reserved_carryover_metadata(phase_records=records)["reserved_carryover_to_loss_materialization_bytes"]
        2
    """

    phase_by_name = {record.phase_name: record for record in phase_records}
    target_phases = (
        "loss_materialization",
        "backward",
        "optimizer_step",
        "zero_grad",
        "step_end",
    )
    return {
        f"reserved_carryover_to_{phase_name}_bytes": max(
            0,
            phase_by_name[phase_name].reserved_bytes
            - phase_by_name[phase_name].allocated_bytes,
        )
        for phase_name in target_phases
        if phase_name in phase_by_name
    }
