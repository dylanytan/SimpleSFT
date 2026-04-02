# Qwen 2.5 ZeRO-2 Calibration Notes

This document records the current ZeRO-2 calibration status for
`Qwen/Qwen2.5-0.5B`, `Qwen/Qwen2.5-1.5B`, and `Qwen/Qwen2.5-7B` on this host.

## Benchmark Corpus

- GPUs:
  - `2 x NVIDIA A100-PCIE-40GB` for the 0.5B and 1.5B corpora
  - `3 x NVIDIA A100 80GB` for the 7B corpus
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
- `0.5B` rebuilt standard suite, iteration 4:
  `benchmark_artifacts/qwen25_05b_zero2_seq128_512_mb1_mb2_iter4`
- `1.5B` rebuilt standard suite, iteration 4:
  `benchmark_artifacts/qwen25_15b_zero2_seq128_512_mb1_mb2_iter4`
- `0.5B` rebuilt long-context suite, iteration 3:
  `benchmark_artifacts/qwen25_05b_zero2_seq1024_2048_mb1_iter3`
- `1.5B` rebuilt long-context suite, iteration 3:
  `benchmark_artifacts/qwen25_15b_zero2_seq1024_2048_mb1_iter3`
- `7B` measured suite, iteration 1:
  `benchmark_artifacts/qwen25_7b_zero2_seq512_1024_mb1_iter1`
- `7B` rebuilt calibrated suite, iteration 3:
  `benchmark_artifacts/qwen25_7b_zero2_seq512_1024_mb1_iter3`
- `0.5B` rebuilt flash suite, iteration 4:
  `benchmark_artifacts/qwen25_05b_flash_zero2_seq128_512_mb1_mb2_iter4`
- `1.5B` rebuilt flash suite, iteration 4:
  `benchmark_artifacts/qwen25_15b_flash_zero2_seq128_512_mb1_mb2_iter4`
- `0.5B` rebuilt flash long-context suite, iteration 4:
  `benchmark_artifacts/qwen25_05b_flash_zero2_seq1024_2048_mb1_iter4`
- `1.5B` rebuilt flash long-context suite, iteration 4:
  `benchmark_artifacts/qwen25_15b_flash_zero2_seq1024_2048_mb1_iter4`
- `0.5B` rebuilt checkpointed suite, iteration 3:
  `benchmark_artifacts/qwen25_05b_zero2_ckpt_seq128_512_mb1_mb2_iter3`
- `1.5B` rebuilt checkpointed suite, iteration 3:
  `benchmark_artifacts/qwen25_15b_zero2_ckpt_seq128_512_mb1_mb2_iter3`

## Error Progression

- `Qwen/Qwen2.5-0.5B`
  - iteration 1 mean global peak error: `1.639 GiB`
  - iteration 1 median global relative error: `23.21%`
  - iteration 2 mean global peak error: `0.218 GiB`
  - iteration 2 median global relative error: `3.30%`
  - iteration 2 max global relative error: `7.86%`
  - iteration 4 mean global peak error: `0.183 GiB`
  - iteration 4 median global relative error: `1.17%`
  - iteration 4 max global relative error: `6.08%`
- `Qwen/Qwen2.5-1.5B`
  - iteration 1 mean global peak error: `6.243 GiB`
  - iteration 1 median global relative error: `27.39%`
  - iteration 2 mean global peak error: `0.241 GiB`
  - iteration 2 median global relative error: `1.50%`
  - iteration 2 max global relative error: `2.76%`
  - iteration 4 mean global peak error: `0.117 GiB`
  - iteration 4 median global relative error: `0.33%`
  - iteration 4 max global relative error: `1.51%`
- `Qwen/Qwen2.5-7B`
  - iteration 1 mean global peak error: `23.407 GiB`
  - iteration 1 median global relative error: `42.15%`
  - iteration 3 mean global peak error: `0.232 GiB`
  - iteration 3 median global relative error: `0.22%`
  - iteration 3 max global relative error: `0.95%`

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
- LoRA ZeRO-2 reserve is now independent of activation bytes; it is modeled as a
  runtime baseline term tied to model residency and world size.
- LoRA ZeRO-2 backward workspace is now driven by effective microstep tokens
  with an explicit long-context correction, rather than frozen-model scale.
- LoRA ZeRO-2 forward workspace now uses full-model residency plus activation
  pressure, instead of adapter-state scale.
- FlashAttention no longer changes `activation_bytes` in the estimator because
  the measurement path defines that component as retained tensors, not attention
  kernel workspace.

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
  - short-context rebuilt suite mean global peak error: `0.183 GiB`
  - short-context rebuilt suite median global relative error: `1.17%`
  - long-context rebuilt suite mean global peak error: `0.201 GiB`
  - long-context rebuilt suite median global relative error: `1.57%`
- `Qwen/Qwen2.5-1.5B`
  - short-context rebuilt suite mean global peak error: `0.117 GiB`
  - short-context rebuilt suite median global relative error: `0.33%`
  - long-context rebuilt suite mean global peak error: `0.141 GiB`
  - long-context rebuilt suite median global relative error: `0.85%`

For this ZeRO-2 corpus, the measured global peak bytes were identical to the
standard-attention ZeRO-2 suites in every matched case.

## Iteration 5: Phase-Decomposed LoRA ZeRO-2

This pass addressed a structural weakness in the LoRA ZeRO-2 estimator rather
than only retuning constants.

### What Changed

- LoRA ZeRO-2 no longer uses one monolithic transient heuristic.
- The estimator now separates LoRA ZeRO-2 peak pressure into:
  - a baseline runtime reserve,
  - a forward workspace term driven by microstep token pressure, retained
    activation pressure, and persistent parameter residency,
  - a backward workspace term driven by effective microstep tokens with an
    explicit long-sequence correction.
- The LoRA ZeRO-2 reserve no longer shrinks aggressively with world size. New
  `3 x 80GB` LoRA measurements on the `0.5B` and `1.5B` models showed that the
  previous world-size dependence was incorrect.

### Additional Artifact Directories

- `0.5B` rebuilt standard suite, iteration 5:
  `benchmark_artifacts/qwen25_05b_zero2_seq128_512_mb1_mb2_iter4`
- `1.5B` rebuilt standard suite, iteration 5:
  `benchmark_artifacts/qwen25_15b_zero2_seq128_512_mb1_mb2_iter4`
- `0.5B` rebuilt long-context suite, iteration 5:
  `benchmark_artifacts/qwen25_05b_zero2_seq1024_2048_mb1_iter3`
- `1.5B` rebuilt long-context suite, iteration 5:
  `benchmark_artifacts/qwen25_15b_zero2_seq1024_2048_mb1_iter3`
- `7B` rebuilt calibrated suite, iteration 5:
  `benchmark_artifacts/qwen25_7b_zero2_seq512_1024_mb1_iter3`
- `0.5B` world-size-disambiguation suite, measured on `3 x 80GB`:
  `benchmark_artifacts/qwen25_05b_zero2_world3_lora_seq512_1024_mb1_iter1`
- `0.5B` world-size-disambiguation suite, rebuilt:
  `benchmark_artifacts/qwen25_05b_zero2_world3_lora_seq512_1024_mb1_iter2`
- `1.5B` world-size-disambiguation suite, measured on `3 x 80GB`:
  `benchmark_artifacts/qwen25_15b_zero2_world3_lora_seq512_1024_mb1_iter1`
- `1.5B` world-size-disambiguation suite, rebuilt:
  `benchmark_artifacts/qwen25_15b_zero2_world3_lora_seq512_1024_mb1_iter2`

### Standard-Corpus Improvement

Across the 32-case standard ZeRO-2 corpus spanning `0.5B`, `1.5B`, and `7B`:

- before this pass:
  - mean global peak error: `0.312 GiB`
  - median global relative error: `2.64%`
  - max global relative error: `15.37%`
  - LoRA mean global peak error: `0.341 GiB`
  - LoRA median global relative error: `3.57%`
  - LoRA max global relative error: `15.37%`
- after this pass:
  - mean global peak error: `0.165 GiB`
  - median global relative error: `0.87%`
  - max global relative error: `6.08%`
  - LoRA mean global peak error: `0.080 GiB`
  - LoRA median global relative error: `0.70%`
  - LoRA max global relative error: `2.40%`

### Suite-Level Before/After

- `0.5B`, short-context standard:
  - before mean error: `0.327 GiB`
  - after mean error: `0.186 GiB`
- `1.5B`, short-context standard:
  - before mean error: `0.281 GiB`
  - after mean error: `0.151 GiB`
- `0.5B`, long-context standard:
  - before mean error: `0.439 GiB`
  - after mean error: `0.195 GiB`
- `1.5B`, long-context standard:
  - before mean error: `0.303 GiB`
  - after mean error: `0.149 GiB`
- `7B`, standard:
  - before mean error: `0.346 GiB`
  - after mean error: `0.239 GiB`
- `0.5B`, `3 x 80GB` LoRA validation:
  - before mean error: `0.119 GiB`
  - after mean error: `0.030 GiB`
- `1.5B`, `3 x 80GB` LoRA validation:
  - before mean error: `0.259 GiB`
  - after mean error: `0.089 GiB`

### Flash And Checkpointing Validation

The new phase-decomposed LoRA branch generalized cleanly on long-context flash
ZeRO-2 suites:

- `0.5B` flash long-context:
  - before mean error: `0.201 GiB`
  - after mean error: `0.195 GiB`
  - before LoRA mean error: `0.067 GiB`
  - after LoRA mean error: `0.055 GiB`
- `1.5B` flash long-context:
  - before mean error: `0.141 GiB`
  - after mean error: `0.149 GiB`
  - before LoRA mean error: `0.198 GiB`
  - after LoRA mean error: `0.213 GiB`

Short-context flash and checkpointed suites moved only slightly. Those cases are
already low-error and remain dominated by the same global peak phase, so this
iteration prioritized cross-size structural correctness on the standard corpus.

Interpretation:

- `flash2` does not change the measured global peak in the current ZeRO-2
  harness for either Qwen size.
- The likely reason is that ZeRO-2 `full_ft` remains optimizer-step dominated
  and ZeRO-2 LoRA remains backward dominated, so any forward/backward attention
  savings do not move the overall maximum.
- In this measurement contract, `activation_bytes` represents retained tensors.
  Flash-related differences must therefore be modeled in workspace terms or
  phase placement, not as retained-activation savings.

## Gradient Checkpointing Under ZeRO-2

Measured checkpointed suites were rebuilt against the current estimator:

- `Qwen/Qwen2.5-0.5B`
  - rebuilt suite mean global peak error: `0.183 GiB`
  - rebuilt suite median global relative error: `1.17%`
  - rebuilt suite max global relative error: `6.08%`
- `Qwen/Qwen2.5-1.5B`
  - rebuilt suite mean global peak error: `0.117 GiB`
  - rebuilt suite median global relative error: `0.33%`
  - rebuilt suite max global relative error: `1.51%`

For this ZeRO-2 corpus, the measured global peak bytes were also identical to
the non-checkpointed ZeRO-2 suites in every matched case.

Interpretation:

- Gradient checkpointing does not change the measured global peak in the current
  ZeRO-2 harness.
- That result is consistent with the measured phase structure: the peak stays at
  `optimizer_step` for `full_ft` and at `backward` for LoRA, so reducing
  retained activations is not the limiting lever here.

## Remaining Gaps

- `runtime_reserve_bytes` and phase-local workspace proxies remain rougher than
  global peak bytes even after the structural fixes.
- The current high-quality fit is for the Qwen dense-decoder ZeRO-2 corpus.
  Other architectures still need their own measurement passes.
- If FlashAttention or checkpointing ever moves the global peak on a future
  corpus, the estimator will need a dedicated workspace model for that regime.
