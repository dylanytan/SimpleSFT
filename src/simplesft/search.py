"""Configuration search helpers for SimpleSFT estimates."""

from __future__ import annotations

from typing import Iterable

from .estimate import estimate_peak_memory
from .types import MemoryResult, ModelSpec, SearchResult, TrainingConfig


SIMPLICITY_ORDER = {
    "single_gpu": 0,
    "ddp": 1,
    "zero2": 2,
}


def _candidate_sort_key(result: MemoryResult) -> tuple[int, int, float]:
    """Return the ranking key for one candidate estimate."""

    complexity = SIMPLICITY_ORDER[result.config.distributed_mode]
    feasible_rank = 0 if result.feasible else 1
    return feasible_rank, complexity, -result.headroom_gb()


def search_configurations(
    *,
    model: str | ModelSpec,
    configs: Iterable[TrainingConfig],
) -> SearchResult:
    """Estimate and rank a list of candidate training configurations.

    Args:
        model: Hugging Face model id, local model path, or precomputed spec.
        configs: Candidate training configurations to evaluate.

    Returns:
        SearchResult containing ranked feasible and infeasible candidates.

    Example:
        >>> from simplesft.types import TrainingConfig
        >>> result = search_configurations(
        ...     model="sshleifer/tiny-gpt2",
        ...     configs=[TrainingConfig(tuning_mode="full_ft", max_seq_len=16)],
        ... )
        >>> len(result.candidates) + len(result.infeasible_candidates)
        1
    """

    estimates = [estimate_peak_memory(model=model, config=config) for config in configs]
    estimates.sort(key=_candidate_sort_key)
    feasible = tuple(result for result in estimates if result.feasible)
    infeasible = tuple(result for result in estimates if not result.feasible)
    return SearchResult(candidates=feasible, infeasible_candidates=infeasible)
