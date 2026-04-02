"""Configuration search helpers for SimpleSFT estimates."""

from __future__ import annotations

from typing import Iterable, Mapping

from .estimate import estimate_peak_memory
from .types import EstimatorConfig, MemoryResult, ModelSpec, SearchResult


SIMPLICITY_ORDER = {
    "single_gpu": 0,
    "ddp": 1,
    "zero2": 2,
    "zero3": 3,
}


def _candidate_sort_key(
    result: MemoryResult,
    *,
    simplicity_order: Mapping[str, int],
) -> tuple[int, int, float]:
    """Return the ranking key for one candidate estimate."""

    complexity = simplicity_order[result.config.distributed_mode]
    feasible_rank = 0 if result.feasible else 1
    return feasible_rank, complexity, -result.headroom_gb()


def search_configurations(
    *,
    model: str | ModelSpec,
    configs: Iterable[EstimatorConfig],
    simplicity_order: Mapping[str, int] | None = None,
) -> SearchResult:
    """Estimate and rank a list of candidate training configurations.

    Args:
        model: Hugging Face model id, local model path, or precomputed spec.
        configs: Candidate training configurations to evaluate.

    Returns:
        SearchResult containing ranked feasible and infeasible candidates.

    Example:
        >>> from simplesft.types import EstimatorConfig
        >>> result = search_configurations(
        ...     model="sshleifer/tiny-gpt2",
        ...     configs=[EstimatorConfig(tuning_mode="full_ft", max_seq_len=16)],
        ... )
        >>> len(result.candidates) + len(result.infeasible_candidates)
        1
    """

    resolved_simplicity_order = simplicity_order or SIMPLICITY_ORDER
    estimates = [estimate_peak_memory(model=model, config=config) for config in configs]
    estimates.sort(
        key=lambda result: _candidate_sort_key(
            result,
            simplicity_order=resolved_simplicity_order,
        )
    )
    feasible = tuple(result for result in estimates if result.feasible)
    infeasible = tuple(result for result in estimates if not result.feasible)
    return SearchResult(candidates=feasible, infeasible_candidates=infeasible)
