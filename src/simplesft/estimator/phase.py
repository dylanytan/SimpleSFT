"""Phase assembly and peak selection for the simplified estimator."""

from __future__ import annotations

from dataclasses import dataclass

from ..types import PhaseMemoryRecord, PhasePeakDebug


@dataclass(frozen=True)
class PhaseInputs:
    """Named inputs used to assemble the phase timeline."""

    parameter_bytes: int
    gradient_bytes: int
    optimizer_state_bytes: int
    master_weight_bytes: int
    runtime_support_bytes: int
    persistent_backend_buffer_bytes: int
    forward_activation_bytes: int
    backward_activation_bytes: int
    forward_workspace_bytes: int
    backward_workspace_bytes: int
    optimizer_workspace_bytes: int
    backward_end_state_bytes: int
    optimizer_reserved_carryover_bytes: int


def _record(*, phase_name: str, peak_bytes: int) -> PhaseMemoryRecord:
    """Return a synthetic phase record with allocated and reserved peaks aligned."""

    return PhaseMemoryRecord(
        phase_name=phase_name,
        allocated_bytes=peak_bytes,
        reserved_bytes=peak_bytes,
        peak_allocated_bytes=peak_bytes,
        peak_reserved_bytes=peak_bytes,
        delta_allocated_bytes=0,
        delta_reserved_bytes=0,
    )


def _resident_floor(*, inputs: PhaseInputs, include_gradients: bool) -> int:
    """Return the resident floor for one phase."""

    resident_bytes = (
        inputs.parameter_bytes
        + inputs.optimizer_state_bytes
        + inputs.master_weight_bytes
        + inputs.runtime_support_bytes
        + inputs.persistent_backend_buffer_bytes
    )
    if include_gradients:
        resident_bytes += inputs.gradient_bytes
    return resident_bytes


def build_phase_records(*, inputs: PhaseInputs) -> tuple[PhaseMemoryRecord, ...]:
    """Build the simplified training-phase timeline."""

    baseline_bytes = _resident_floor(inputs=inputs, include_gradients=False)
    forward_peak_bytes = (
        baseline_bytes
        + inputs.forward_activation_bytes
        + inputs.forward_workspace_bytes
    )
    backward_peak_bytes = (
        _resident_floor(inputs=inputs, include_gradients=True)
        + inputs.backward_activation_bytes
        + inputs.backward_workspace_bytes
    )
    optimizer_peak_bytes = (
        _resident_floor(inputs=inputs, include_gradients=True)
        + inputs.backward_end_state_bytes
        + inputs.optimizer_workspace_bytes
        + inputs.optimizer_reserved_carryover_bytes
    )
    return (
        _record(phase_name="post_init_baseline", peak_bytes=baseline_bytes),
        _record(phase_name="forward", peak_bytes=forward_peak_bytes),
        _record(phase_name="loss_materialization", peak_bytes=forward_peak_bytes),
        _record(phase_name="backward", peak_bytes=backward_peak_bytes),
        _record(phase_name="optimizer_step", peak_bytes=optimizer_peak_bytes),
        _record(phase_name="step_end", peak_bytes=baseline_bytes),
    )


def build_phase_peak_debug(
    *, phase_records: tuple[PhaseMemoryRecord, ...], backward_end_state_bytes: int
) -> PhasePeakDebug:
    """Build phase-peak debug metadata from the synthetic timeline."""

    peak_record = max(
        (
            record
            for record in phase_records
            if record.phase_name in {"forward", "backward", "optimizer_step"}
        ),
        key=lambda record: record.peak_allocated_bytes,
    )
    peak_by_name = {
        record.phase_name: record.peak_allocated_bytes for record in phase_records
    }
    global_peak_bytes = peak_record.peak_allocated_bytes
    return PhasePeakDebug(
        forward_peak_bytes=peak_by_name.get("forward", 0),
        backward_peak_bytes=peak_by_name.get("backward", 0),
        optimizer_peak_bytes=peak_by_name.get("optimizer_step", 0),
        global_peak_bytes=global_peak_bytes,
        global_peak_phase=peak_record.phase_name,
        soft_reserved_global_peak_bytes=global_peak_bytes,
        stressed_reserved_global_peak_bytes=global_peak_bytes,
        backward_end_state_bytes=backward_end_state_bytes,
    )
