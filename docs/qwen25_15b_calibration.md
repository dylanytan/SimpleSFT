# Qwen 2.5 1.5B Calibration Notes

This document records the current calibration status for
`Qwen/Qwen2.5-1.5B` on single `NVIDIA A100-PCIE-40GB` GPUs.

## Benchmark Corpus

- Model: `Qwen/Qwen2.5-1.5B`
- GPUs: `NVIDIA A100-PCIE-40GB`
- Weight dtype: `bf16`
- Optimizer: `AdamW`
- Tuning modes: `full_ft`, `lora`
- Attention backend: `standard`
- Distributed mode: `single_gpu`
- Warmup steps: `1`

## Artifact Directories

- Initial 8-case suite:
  `benchmark_artifacts/qwen25_15b_seq128_512_mb1_mb2_iter1`
- Size-aware calibrated 8-case suite:
  `benchmark_artifacts/qwen25_15b_seq128_512_mb1_mb2_iter2`

## Error Progression

- Iteration 1, 8 cases, `seq={128,512}`, `mb={1,2}`
  - Mean global peak error: `0.741 GiB`
  - Median global relative error: `5.01%`
- Iteration 2, same 8 cases
  - Mean global peak error: `0.152 GiB`
  - Median global relative error: `1.33%`

## Current 8-Case Results

- `full_ft / seq=128 / mb=1`
  - measured: `15.400 GiB`
  - estimated: `15.172 GiB`
  - relative error: `1.48%`
- `full_ft / seq=128 / mb=2`
  - measured: `15.523 GiB`
  - estimated: `15.349 GiB`
  - relative error: `1.12%`
- `full_ft / seq=512 / mb=1`
  - measured: `16.121 GiB`
  - estimated: `16.309 GiB`
  - relative error: `1.17%`
- `full_ft / seq=512 / mb=2`
  - measured: `17.910 GiB`
  - estimated: `18.228 GiB`
  - relative error: `1.78%`
- `lora / seq=128 / mb=1`
  - measured: `3.717 GiB`
  - estimated: `3.605 GiB`
  - relative error: `3.00%`
- `lora / seq=128 / mb=2`
  - measured: `4.188 GiB`
  - estimated: `4.037 GiB`
  - relative error: `3.60%`
- `lora / seq=512 / mb=1`
  - measured: `5.539 GiB`
  - estimated: `5.538 GiB`
  - relative error: `0.02%`
- `lora / seq=512 / mb=2`
  - measured: `7.928 GiB`
  - estimated: `7.886 GiB`
  - relative error: `0.53%`

## Out-Of-Sample Check: Gradient Checkpointing

These runs were measured ad hoc after iteration 2 and are not yet persisted as a
benchmark suite artifact directory.

- `full_ft / seq=512 / mb=1 / gradient_checkpointing=True`
  - measured: `15.947 GiB`
  - estimated: `16.309 GiB`
  - relative error: `2.27%`
- `full_ft / seq=512 / mb=2 / gradient_checkpointing=True`
  - measured: `17.719 GiB`
  - estimated: `18.228 GiB`
  - relative error: `2.88%`
- `lora / seq=512 / mb=1 / gradient_checkpointing=True`
  - measured: `5.502 GiB`
  - estimated: `5.538 GiB`
  - relative error: `0.65%`
- `lora / seq=512 / mb=2 / gradient_checkpointing=True`
  - measured: `7.910 GiB`
  - estimated: `7.886 GiB`
  - relative error: `0.31%`

## Key Findings

- The 0.5B-tuned estimator did not transfer cleanly to 1.5B without a second
  size-aware pass.
- For large single-GPU `full_ft`, optimizer-step transient memory is much larger
  than the 0.5B branch predicted and must be modeled explicitly.
- For large single-GPU `lora`, long-sequence peaks are optimizer-step dominated.
  Summing activation and transient bytes directly at backward over-predicts peak
  memory.
- A size-gated single-GPU branch improved transfer without degrading the existing
  0.5B fit.

## Remaining Gaps

- Phase labeling for large-model LoRA still reports `backward` in some estimated
  cases when the measured peak is `optimizer_step`, even though the peak value is
  now close.
- Component alignment for `transient_bytes` and `runtime_reserve_bytes` is still
  rougher than global peak alignment.
- Runtime DDP and ZeRO-2 measurement remain unimplemented.
