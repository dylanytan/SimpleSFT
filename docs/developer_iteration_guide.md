# Developer Iteration Guide

`SimpleSFT` is intentionally built for manual estimator refinement.

## Loop

1. Run a benchmark corpus with `simplesft benchmark --measure`.
2. Rewrite the estimator or measurement attribution based on the mismatch pattern.
3. Rebuild estimates and comparisons from the saved measurements with
   `simplesft rebuild-benchmark`.
4. Review the regenerated suite report with `simplesft report`.
5. Repeat for four planned developer iterations.

## Iteration Checklist

Each iteration should answer:

- Which benchmark cases were run.
- Which components or phases had the largest error.
- What code or formula changes were made.
- Whether total, phase, and component error improved.
- What remains unexplained.

## Recommended Four Iterations

1. Baseline instrumentation and first end-to-end estimate.
2. Structural fixes for LoRA, activations, optimizer state, and phase peaks.
3. Transient-memory and backend refinement for DDP and ZeRO-2.
4. Hardening, safety margins, and search-readiness.
