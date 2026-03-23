# Qwen 2.5 0.5B Calibration Notes

This document records the current calibration status for
`Qwen/Qwen2.5-0.5B` on a single `NVIDIA A100-PCIE-40GB` using
`CUDA_VISIBLE_DEVICES=0`.

## Benchmark Corpus

- Model: `Qwen/Qwen2.5-0.5B`
- GPU: `NVIDIA A100-PCIE-40GB`
- Weight dtype: `bf16`
- Optimizer: `AdamW`
- Tuning modes: `full_ft`, `lora`
- Attention backend: `standard`
- Distributed mode: `single_gpu`
- Warmup steps: `1`

## Artifact Directories

- Iteration 1:
  `benchmark_artifacts/qwen25_05b_seq128_512_mb1_v1`
- Iteration 4, original 4-case corpus:
  `benchmark_artifacts/qwen25_05b_seq128_512_mb1_iter4`
- Iteration 5, expanded 8-case corpus:
  `benchmark_artifacts/qwen25_05b_seq128_512_mb1_mb2_iter5`

## Error Progression

- Iteration 1, 4 cases, `seq={128,512}`, `mb=1`
  - Mean global peak error: `3.708 GiB`
  - Median global relative error: `108.93%`
- Iteration 4, same 4 cases
  - Mean global peak error: `0.060 GiB`
  - Median global relative error: `1.99%`
- Iteration 5, expanded 8 cases, `seq={128,512}`, `mb={1,2}`
  - Mean global peak error: `0.070 GiB`
  - Median global relative error: `2.09%`

## Current 8-Case Results

- `full_ft / seq=128 / mb=1`
  - measured: `4.857 GiB`
  - estimated: `4.969 GiB`
  - relative error: `2.29%`
- `full_ft / seq=128 / mb=2`
  - measured: `4.967 GiB`
  - estimated: `5.036 GiB`
  - relative error: `1.40%`
- `full_ft / seq=512 / mb=1`
  - measured: `5.088 GiB`
  - estimated: `5.172 GiB`
  - relative error: `1.65%`
- `full_ft / seq=512 / mb=2`
  - measured: `6.309 GiB`
  - estimated: `6.294 GiB`
  - relative error: `0.23%`
- `lora / seq=128 / mb=1`
  - measured: `1.424 GiB`
  - estimated: `1.498 GiB`
  - relative error: `5.21%`
- `lora / seq=128 / mb=2`
  - measured: `1.854 GiB`
  - estimated: `1.793 GiB`
  - relative error: `3.24%`
- `lora / seq=512 / mb=1`
  - measured: `2.674 GiB`
  - estimated: `2.739 GiB`
  - relative error: `2.43%`
- `lora / seq=512 / mb=2`
  - measured: `4.357 GiB`
  - estimated: `4.275 GiB`
  - relative error: `1.88%`

## Out-Of-Sample Check: Gradient Checkpointing

These runs were measured ad hoc after iteration 5 and are not yet persisted as a
benchmark suite artifact directory.

- `full_ft / seq=512 / mb=1 / gradient_checkpointing=True`
  - measured: `5.043 GiB`
  - estimated: `5.172 GiB`
  - relative error: `2.55%`
- `full_ft / seq=512 / mb=2 / gradient_checkpointing=True`
  - measured: `6.301 GiB`
  - estimated: `6.294 GiB`
  - relative error: `0.11%`
- `lora / seq=512 / mb=1 / gradient_checkpointing=True`
  - measured: `2.680 GiB`
  - estimated: `2.739 GiB`
  - relative error: `2.21%`
- `lora / seq=512 / mb=2 / gradient_checkpointing=True`
  - measured: `4.361 GiB`
  - estimated: `4.275 GiB`
  - relative error: `1.97%`

## Key Findings

- AdamW state already exists at `post_init_baseline` when `warmup_steps > 0`.
  Modeling optimizer state as appearing only at `optimizer_step` over-predicts
  full-FT peaks by about `1 GiB`.
- In the current PEFT stack, LoRA adapter weights, gradients, and AdamW state are
  materialized as `fp32`.
- For single-GPU `full_ft`, the dominant transient at `optimizer_step` scales well
  as:
  - `max(0.94 * parameter_bytes, 1.8 * activation_bytes)`
- For the current measurement schema, `activation_bytes` is a module-output
  summary. Gradient checkpointing does not directly reduce this component, so it
  should not be halved inside the estimator.

## Remaining Gaps

- Component alignment for `transient_bytes` and `runtime_reserve_bytes` is still
  rougher than global peak alignment.
- Runtime DDP and ZeRO-2 measurement remain unimplemented.
- FlashAttention-specific calibration is still missing.
