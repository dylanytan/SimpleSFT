# SimpleSFT

SimpleSFT measures and estimates peak memory for supervised fine-tuning runs.

## Goals

- Measure real training-step memory with granular phase and component breakdowns.
- Estimate peak memory analytically with the same modular breakdown.
- Make estimator mistakes obvious enough that a developer can refine the model across repeated calibration iterations.

## Current Scope

- Hugging Face language models across the current GPT/Qwen/OLMo support surface.
- Qwen vision-language models with synthetic image-token estimation and measurement inputs.
- `full_ft` and `lora` tuning modes.
- AdamW memory modeling.
- Estimation support for `single_gpu`, `ddp`, `zero2`, and `zero3`.
- Runtime measurement support for CUDA environments, with explicit optional-runtime hooks for LoRA and DeepSpeed ZeRO stage 2/3.

## Install

```bash
pip install -e .
```

Optional dependencies:

```bash
pip install -e .[dev]
pip install -e .[lora]
pip install -e .[zero2]
```

The `zero2` extra installs the DeepSpeed dependency used by both ZeRO-2 and ZeRO-3 runtime measurement.

## CLI

```bash
simplesft inspect sshleifer/tiny-gpt2
simplesft estimate sshleifer/tiny-gpt2 --max-seq-len 128
simplesft measure sshleifer/tiny-gpt2 --max-seq-len 128
simplesft compare sshleifer/tiny-gpt2 --max-seq-len 128
simplesft search sshleifer/tiny-gpt2 --seq-lens 128 256 --micro-batches 1 2
simplesft estimate Qwen/Qwen2.5-VL-3B-Instruct --max-seq-len 2048 --distributed-mode zero3 --gpus-per-node 4
simplesft web --host 127.0.0.1 --port 8765
simplesft benchmark sshleifer/tiny-gpt2 --output-dir benchmark_artifacts/iter1 --seq-lens 64 128 --micro-batches 1 2
simplesft rebuild-benchmark --source-dir benchmark_artifacts/iter1 --output-dir benchmark_artifacts/iter2
simplesft report --input-dir benchmark_artifacts/iter1 --iteration-name "Iteration 1"
```

The web UI is a local stdlib server. It exposes a form-driven estimator
workbench on `/` and a JSON API on `/api/estimate`.

## Developer Iteration

See [docs/developer_iteration_guide.md](docs/developer_iteration_guide.md) for the manual four-iteration calibration loop.
See [docs/qwen25_05b_calibration.md](docs/qwen25_05b_calibration.md) for the
current Qwen 0.5B calibration results and measured artifact directories.
See [docs/qwen25_15b_calibration.md](docs/qwen25_15b_calibration.md) for the
current Qwen 1.5B calibration results and transfer notes.
See [docs/qwen25_zero2_calibration.md](docs/qwen25_zero2_calibration.md) for the
current ZeRO-2 measurement and calibration notes, including the 7B pass and the
latest rebuilt flash/checkpointed suites.

Recent persisted benchmark suites:

- `benchmark_artifacts/qwen25_05b_ddp_seq128_512_mb1_mb2_iter1`
- `benchmark_artifacts/qwen25_15b_ddp_seq128_512_mb1_mb2_iter1`
- `benchmark_artifacts/qwen25_05b_ckpt_seq128_512_mb1_mb2_iter1`
- `benchmark_artifacts/qwen25_15b_ckpt_seq128_512_mb1_mb2_iter1`
- `benchmark_artifacts/qwen25_05b_flash_seq512_1024_2048_mb1_iter1`
- `benchmark_artifacts/qwen25_15b_flash_seq512_1024_2048_mb1_iter1`
- `benchmark_artifacts/qwen25_05b_zero2_seq128_512_mb1_mb2_iter1`
- `benchmark_artifacts/qwen25_05b_zero2_seq128_512_mb1_mb2_iter2`
- `benchmark_artifacts/qwen25_15b_zero2_seq128_512_mb1_mb2_iter1`
- `benchmark_artifacts/qwen25_15b_zero2_seq128_512_mb1_mb2_iter2`
- `benchmark_artifacts/qwen25_05b_flash_zero2_seq128_512_mb1_mb2_iter1`
- `benchmark_artifacts/qwen25_05b_flash_zero2_seq128_512_mb1_mb2_iter2`
- `benchmark_artifacts/qwen25_15b_flash_zero2_seq128_512_mb1_mb2_iter1`
- `benchmark_artifacts/qwen25_15b_flash_zero2_seq128_512_mb1_mb2_iter2`
- `benchmark_artifacts/qwen25_05b_zero2_ckpt_seq128_512_mb1_mb2_iter1`
- `benchmark_artifacts/qwen25_05b_zero2_ckpt_seq128_512_mb1_mb2_iter2`
- `benchmark_artifacts/qwen25_15b_zero2_ckpt_seq128_512_mb1_mb2_iter1`
- `benchmark_artifacts/qwen25_15b_zero2_ckpt_seq128_512_mb1_mb2_iter2`

## Notes

- The package patches a broken optional `torchvision` path in this environment so Hugging Face language-model and Qwen-VL inspection can run.
- Real measurement still requires CUDA.
- On 2xA100 cross-NUMA hosts, DDP measurement now automatically sets `NCCL_P2P_DISABLE=1` for the spawned worker job.
- ZeRO-2 and ZeRO-3 measurement are implemented with DeepSpeed and use the same cross-NUMA launch policy as DDP on this host.
- Comparison artifacts now include workspace-proxy errors so FlashAttention and ZeRO-2 workspace effects are visible separately from module-output activations.
- FlashAttention 2 measurement works when `flash-attn` is installed. In this environment the direct wheel install succeeds even when `pip install flash-attn` hits a cross-filesystem wheel-staging error.
- On the current ZeRO-2 Qwen corpus, `flash2` and gradient checkpointing do not change the measured global peak; optimizer-step or backward pressure still dominates.
- In the current measurement contract, `activation_bytes` means retained tensors. FlashAttention should therefore affect workspace terms or phase placement, not retained-activation accounting.
