# Qwen 2.5 ZeRO-2 Calibration Notes

This document records the current 2-GPU ZeRO-2 calibration status for
`Qwen/Qwen2.5-0.5B` and `Qwen/Qwen2.5-1.5B` on this host.

## Benchmark Corpus

- GPUs: `2 x NVIDIA A100-PCIE-40GB`
- Topology: cross-NUMA; spawned jobs set `NCCL_P2P_DISABLE=1`
- Weight dtype: `bf16`
- Optimizer: `AdamW`
- Distributed mode: `zero2`
- Tuning modes: `full_ft`, `lora`
- Attention backend: `standard`
- Warmup steps: `1`
- Cases per suite: `8`
- Sweep: `seq={128,512}`, `mb={1,2}`

## Artifact Directories

- `0.5B` measured suite, iteration 1:
  `benchmark_artifacts/qwen25_05b_zero2_seq128_512_mb1_mb2_iter1`
- `0.5B` rebuilt calibrated suite, iteration 2:
  `benchmark_artifacts/qwen25_05b_zero2_seq128_512_mb1_mb2_iter2`
- `1.5B` measured suite, iteration 1:
  `benchmark_artifacts/qwen25_15b_zero2_seq128_512_mb1_mb2_iter1`
- `1.5B` rebuilt calibrated suite, iteration 2:
  `benchmark_artifacts/qwen25_15b_zero2_seq128_512_mb1_mb2_iter2`
- `0.5B` flash ZeRO-2 measured suite, iteration 1:
  `benchmark_artifacts/qwen25_05b_flash_zero2_seq128_512_mb1_mb2_iter1`
- `0.5B` flash ZeRO-2 rebuilt suite, iteration 2:
  `benchmark_artifacts/qwen25_05b_flash_zero2_seq128_512_mb1_mb2_iter2`
- `1.5B` flash ZeRO-2 measured suite, iteration 1:
  `benchmark_artifacts/qwen25_15b_flash_zero2_seq128_512_mb1_mb2_iter1`
- `1.5B` flash ZeRO-2 rebuilt suite, iteration 2:
  `benchmark_artifacts/qwen25_15b_flash_zero2_seq128_512_mb1_mb2_iter2`
- `0.5B` ZeRO-2 checkpointing measured suite, iteration 1:
  `benchmark_artifacts/qwen25_05b_zero2_ckpt_seq128_512_mb1_mb2_iter1`
- `0.5B` ZeRO-2 checkpointing rebuilt suite, iteration 2:
  `benchmark_artifacts/qwen25_05b_zero2_ckpt_seq128_512_mb1_mb2_iter2`
- `1.5B` ZeRO-2 checkpointing measured suite, iteration 1:
  `benchmark_artifacts/qwen25_15b_zero2_ckpt_seq128_512_mb1_mb2_iter1`
- `1.5B` ZeRO-2 checkpointing rebuilt suite, iteration 2:
  `benchmark_artifacts/qwen25_15b_zero2_ckpt_seq128_512_mb1_mb2_iter2`

## Error Progression

- `Qwen/Qwen2.5-0.5B`
  - iteration 1 mean global peak error: `1.639 GiB`
  - iteration 1 median global relative error: `23.21%`
  - iteration 2 mean global peak error: `0.218 GiB`
  - iteration 2 median global relative error: `3.30%`
  - iteration 2 max global relative error: `7.86%`
- `Qwen/Qwen2.5-1.5B`
  - iteration 1 mean global peak error: `6.243 GiB`
  - iteration 1 median global relative error: `27.39%`
  - iteration 2 mean global peak error: `0.241 GiB`
  - iteration 2 median global relative error: `1.50%`
  - iteration 2 max global relative error: `2.76%`

## Iteration 2 Results

- `0.5B / full_ft / seq=128 / mb=1`
  - measured: `8.703 GiB`
  - estimated: `8.963 GiB`
  - relative error: `2.99%`
- `0.5B / full_ft / seq=128 / mb=2`
  - measured: `9.143 GiB`
  - estimated: `9.160 GiB`
  - relative error: `0.19%`
- `0.5B / full_ft / seq=512 / mb=1`
  - measured: `9.221 GiB`
  - estimated: `9.554 GiB`
  - relative error: `3.61%`
- `0.5B / full_ft / seq=512 / mb=2`
  - measured: `10.902 GiB`
  - estimated: `10.341 GiB`
  - relative error: `5.15%`
- `0.5B / lora / seq=128 / mb=1`
  - measured: `3.266 GiB`
  - estimated: `3.522 GiB`
  - relative error: `7.86%`
- `0.5B / lora / seq=128 / mb=2`
  - measured: `3.650 GiB`
  - estimated: `3.897 GiB`
  - relative error: `6.75%`
- `0.5B / lora / seq=512 / mb=1`
  - measured: `4.393 GiB`
  - estimated: `4.439 GiB`
  - relative error: `1.05%`
- `0.5B / lora / seq=512 / mb=2`
  - measured: `5.916 GiB`
  - estimated: `5.937 GiB`
  - relative error: `0.35%`
- `1.5B / full_ft / seq=128 / mb=1`
  - measured: `25.186 GiB`
  - estimated: `25.117 GiB`
  - relative error: `0.27%`
- `1.5B / full_ft / seq=128 / mb=2`
  - measured: `25.832 GiB`
  - estimated: `25.511 GiB`
  - relative error: `1.24%`
- `1.5B / full_ft / seq=512 / mb=1`
  - measured: `27.045 GiB`
  - estimated: `26.299 GiB`
  - relative error: `2.76%`
- `1.5B / full_ft / seq=512 / mb=2`
  - measured: `28.213 GiB`
  - estimated: `27.874 GiB`
  - relative error: `1.20%`
- `1.5B / lora / seq=128 / mb=1`
  - measured: `5.525 GiB`
  - estimated: `5.428 GiB`
  - relative error: `1.76%`
- `1.5B / lora / seq=128 / mb=2`
  - measured: `6.033 GiB`
  - estimated: `6.044 GiB`
  - relative error: `0.17%`
- `1.5B / lora / seq=512 / mb=1`
  - measured: `7.043 GiB`
  - estimated: `6.890 GiB`
  - relative error: `2.17%`
- `1.5B / lora / seq=512 / mb=2`
  - measured: `9.158 GiB`
  - estimated: `9.351 GiB`
  - relative error: `2.11%`

## Structural Changes In The Estimator

- ZeRO-2 parameters are now modeled as replicated per rank.
- ZeRO-2 `full_ft` optimizer states are modeled as fp32 and sharded.
- ZeRO-2 gradients are treated as transient pressure instead of persistent
  rank-local residency, which matches the current measurement path much better.
- ZeRO-2 `full_ft` reserve is parameter-scaled.
- ZeRO-2 `full_ft` transient memory is modeled as a parameter-scaled base plus
  an activation-scaled term.
- ZeRO-2 `lora` uses a separate transient and reserve branch.
- ZeRO-2 `full_ft` peaks are placed at `optimizer_step`; ZeRO-2 `lora` peaks are
  placed at `backward`.
- Comparison artifacts now include workspace-proxy errors so phase-local
  workspace pressure is visible during calibration.

## Key Findings

- On this host, ZeRO-2 per-rank peak memory is not lower than single-GPU memory
  for these models. The sharded optimizer-state savings are outweighed by large
  transient communication and workspace pressure.
- `full_ft` and `lora` need different ZeRO-2 phase models. `full_ft` is
  optimizer-step dominated. `lora` is backward dominated on this corpus.
- The largest initial error came from a structural mistake: sharding parameter
  residency under ZeRO-2. Fixing that was necessary before any useful transient
  calibration could happen.
- Workspace-proxy errors remain much rougher than global peak errors, especially
  for forward-phase workspace attribution on the 1.5B runs.

## FlashAttention Under ZeRO-2

Measured flash suites were rebuilt against the current estimator:

- `Qwen/Qwen2.5-0.5B`
  - mean global peak error: `0.343 GiB`
  - median global relative error: `3.68%`
  - max global relative error: `13.24%`
- `Qwen/Qwen2.5-1.5B`
  - mean global peak error: `0.708 GiB`
  - median global relative error: `4.53%`
  - max global relative error: `12.21%`

For this ZeRO-2 corpus, the measured global peak bytes were identical to the
standard-attention ZeRO-2 suites in every matched case.

Interpretation:

- `flash2` does not change the measured global peak in the current ZeRO-2
  harness for either Qwen size.
- The likely reason is that ZeRO-2 `full_ft` remains optimizer-step dominated
  and ZeRO-2 LoRA remains backward dominated, so any forward/backward attention
  savings do not move the overall maximum.
- Workspace-proxy instrumentation still matters because attention-kernel
  differences could show up in phase-local or op-local analysis even when the
  global maximum is unchanged.

## Gradient Checkpointing Under ZeRO-2

Measured checkpointed suites were rebuilt against the current estimator:

- `Qwen/Qwen2.5-0.5B`
  - mean global peak error: `0.218 GiB`
  - median global relative error: `3.30%`
  - max global relative error: `7.86%`
- `Qwen/Qwen2.5-1.5B`
  - mean global peak error: `0.241 GiB`
  - median global relative error: `1.50%`
  - max global relative error: `2.76%`

For this ZeRO-2 corpus, the measured global peak bytes were also identical to
the non-checkpointed ZeRO-2 suites in every matched case.

Interpretation:

- Gradient checkpointing does not change the measured global peak in the current
  ZeRO-2 harness.
- That result is consistent with the measured phase structure: the peak stays at
  `optimizer_step` for `full_ft` and at `backward` for LoRA, so reducing
  retained activations is not the limiting lever here.

## Remaining Gaps

- `runtime_reserve_bytes` is still the roughest component term in the ZeRO-2
  branch even though global peak alignment is now good.
- Forward-phase workspace proxies are still under-modeled, especially on the
  larger model.
- FlashAttention has not yet been measured together with ZeRO-2 in the current
  corpus.
