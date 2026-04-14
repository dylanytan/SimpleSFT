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

## Training (SFT)

SimpleSFT can export a feasible training strategy (batch size, seq len, precision, LoRA vs full fine-tune, etc.) into a Hugging Face TRL-compatible YAML file, and then run an actual supervised fine-tuning job using TRL's `SFTTrainer`.

Export a candidate strategy config from `search`:

```bash
simplesft search sshleifer/tiny-gpt2 \
  --seq-lens 128 256 \
  --micro-batches 1 2 \
  --export-file ./trl_strategy.yaml
```

Then run training with that exported strategy:

```bash
simplesft train \
  --config ./trl_strategy.yaml \
  --dataset timdettmers/openassistant-guanaco \
  --output-dir ./sft_output \
  --test-run
```

Notes:

- The exported YAML may contain a list of candidate strategies; the training runner will use the first entry.
- Override the model (otherwise it is read from the exported config's `model` field):

```bash
simplesft train --config ./trl_strategy.yaml --model sshleifer/tiny-gpt2 --dataset timdettmers/openassistant-guanaco
```

- For datasets that need a subset/config (e.g. language splits), pass `--dataset-config`.
- For non-plain-text datasets, SimpleSFT will infer a formatting function automatically, or you can provide `--format-template` with `{Column}` placeholders.
- The same runner is available as a standalone script if you prefer: `python scripts/run_sft.py --config ./trl_strategy.yaml --dataset ...`.

## Developer Iteration

See [docs/developer_iteration_guide.md](docs/developer_iteration_guide.md) for the manual four-iteration calibration loop.
See [docs/qwen25_05b_calibration.md](docs/qwen25_05b_calibration.md) for the
current Qwen 0.5B calibration results and measured artifact directories.
See [docs/qwen25_15b_calibration.md](docs/qwen25_15b_calibration.md) for the
current Qwen 1.5B calibration results and transfer notes.
See [docs/qwen25_zero2_calibration.md](docs/qwen25_zero2_calibration.md) for the
current ZeRO-2 measurement and calibration notes, including the 7B pass and the
latest rebuilt flash/checkpointed suites.

Rebuild the cleaned-corpus phase-aligned calibration report with:

```bash
python scripts/report_phase_aligned_calibration.py \
  --canonical-csv benchmark_artifacts/_cleaned_corpus/canonical_measurements.csv \
  --output benchmark_artifacts/_cleaned_corpus/phase_aligned_report_latest.md \
  --tex-output benchmark_artifacts/_cleaned_corpus/phase_aligned_report_latest.tex
```

The canonical cleaned corpus now includes curated checkpointed long-context OLMo
3 7B A100 suites alongside the earlier Qwen calibration artifacts.

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
