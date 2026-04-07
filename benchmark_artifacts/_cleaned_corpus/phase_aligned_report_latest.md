# Phase-Aligned Calibration Replay

- Replayed rows: `179`
- Failed rows: `151`
- Global MAE: `1.099 GiB`
- Mean absolute relative error: `10.5%`
- Median absolute relative error: `6.2%`
- Phase match rate: `64.8%`
- Retained-forward proxy MAE: `0.577 GiB`

## Component MAE

| Metric | Raw | Phase-Aligned |
| --- | ---: | ---: |
| parameter_bytes | 0.002 GiB | 0.002 GiB |
| gradient_bytes | 0.222 GiB | 0.222 GiB |
| optimizer_state_bytes | 0.064 GiB | 0.064 GiB |
| activation_bytes | 1.273 GiB | `n/a` |
| transient_bytes | 3.104 GiB | 2.982 GiB |
| runtime_reserve_bytes | 0.490 GiB | 0.490 GiB |


## Phase Confusion

| Measured | Estimated | Count |
| --- | --- | ---: |
| backward | backward | 58 |
| backward | optimizer_step | 3 |
| optimizer_step | backward | 60 |
| optimizer_step | optimizer_step | 58 |


## Targeted Slices

| Slice | Count | Global MAE | Proxy MAE | Phase Match |
| --- | ---: | ---: | ---: | ---: |
| Non-Eager Long-Seq Full FT | 1 | 13.451 GiB | 2.879 GiB | 100.0% |
| Non-Eager Long-Seq LoRA | 12 | 1.691 GiB | 0.949 GiB | 66.7% |
| ZeRO-2 | 68 | 1.365 GiB | 0.561 GiB | 97.1% |
| ZeRO-3 | 0 | 0.000 GiB | 0.000 GiB | 0.0% |


### By measured phase

| Group | Count | Global MAE | Proxy MAE | Phase Match |
| --- | ---: | ---: | ---: | ---: |
| backward | 61 | 1.333 GiB | 0.643 GiB | 95.1% |
| optimizer_step | 118 | 0.978 GiB | 0.543 GiB | 49.2% |


### By attention backend

| Group | Count | Global MAE | Proxy MAE | Phase Match |
| --- | ---: | ---: | ---: | ---: |
| flash2 | 68 | 1.266 GiB | 0.601 GiB | 77.9% |
| sdpa | 111 | 0.997 GiB | 0.562 GiB | 56.8% |


### By sequence length

| Group | Count | Global MAE | Proxy MAE | Phase Match |
| --- | ---: | ---: | ---: | ---: |
| 1024 | 19 | 0.901 GiB | 0.587 GiB | 47.4% |
| 128 | 16 | 1.495 GiB | 0.107 GiB | 100.0% |
| 2048 | 57 | 1.158 GiB | 1.009 GiB | 52.6% |
| 256 | 6 | 0.300 GiB | 0.087 GiB | 100.0% |
| 4096 | 9 | 1.563 GiB | 0.784 GiB | 55.6% |
| 512 | 68 | 0.797 GiB | 0.266 GiB | 67.6% |
| 8192 | 4 | 4.921 GiB | 1.804 GiB | 100.0% |


### By distributed mode

| Group | Count | Global MAE | Proxy MAE | Phase Match |
| --- | ---: | ---: | ---: | ---: |
| ddp | 10 | 1.101 GiB | 0.357 GiB | 70.0% |
| single_gpu | 101 | 0.920 GiB | 0.609 GiB | 42.6% |
| zero2 | 68 | 1.365 GiB | 0.561 GiB | 97.1% |


### By tuning mode

| Group | Count | Global MAE | Proxy MAE | Phase Match |
| --- | ---: | ---: | ---: | ---: |
| full_ft | 100 | 1.101 GiB | 0.762 GiB | 69.0% |
| lora | 79 | 1.097 GiB | 0.343 GiB | 59.5% |


## Top 20 Overestimates

| Model | Backend | Seq | Dist | Tune | Measured | Estimated | Signed Error |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: |
| HuggingFaceTB/SmolLM2-1.7B | sdpa | 8192 | single_gpu | full_ft | 36.568 GiB | 50.019 GiB | 13.451 GiB |
| Qwen/Qwen2.5-14B | sdpa | 2048 | zero2 | lora | 46.922 GiB | 53.770 GiB | 6.848 GiB |
| HuggingFaceTB/SmolLM2-1.7B | sdpa | 2048 | single_gpu | full_ft | 17.672 GiB | 22.293 GiB | 4.621 GiB |
| allenai/OLMo-1B-hf | sdpa | 2048 | single_gpu | full_ft | 11.762 GiB | 15.314 GiB | 3.552 GiB |
| EleutherAI/pythia-1b-deduped | sdpa | 2048 | single_gpu | full_ft | 10.566 GiB | 13.944 GiB | 3.378 GiB |
| Qwen/Qwen2.5-1.5B | flash2 | 2048 | single_gpu | full_ft | 19.281 GiB | 22.583 GiB | 3.302 GiB |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | sdpa | 2048 | single_gpu | full_ft | 12.049 GiB | 14.958 GiB | 2.909 GiB |
| Qwen/Qwen2.5-7B | sdpa | 8192 | zero2 | lora | 66.758 GiB | 69.482 GiB | 2.724 GiB |
| Qwen/Qwen2.5-1.5B | sdpa | 512 | zero2 | full_ft | 29.936 GiB | 32.415 GiB | 2.480 GiB |
| HuggingFaceTB/SmolLM2-1.7B | sdpa | 512 | zero2 | full_ft | 27.385 GiB | 29.805 GiB | 2.420 GiB |
| Qwen/Qwen2.5-0.5B | sdpa | 512 | zero2 | full_ft | 9.268 GiB | 11.601 GiB | 2.333 GiB |
| HuggingFaceTB/SmolLM2-1.7B | sdpa | 8192 | single_gpu | lora | 29.916 GiB | 32.042 GiB | 2.126 GiB |
| Qwen/Qwen2.5-1.5B | flash2 | 128 | zero2 | full_ft | 25.186 GiB | 27.150 GiB | 1.964 GiB |
| Qwen/Qwen2.5-1.5B | flash2 | 128 | zero2 | full_ft | 25.186 GiB | 27.150 GiB | 1.964 GiB |
| Qwen/Qwen2.5-7B-Instruct | flash2 | 4096 | single_gpu | lora | 46.051 GiB | 47.963 GiB | 1.912 GiB |
| HuggingFaceTB/SmolLM2-1.7B | sdpa | 2048 | ddp | full_ft | 20.713 GiB | 22.593 GiB | 1.880 GiB |
| Qwen/Qwen2.5-0.5B | flash2 | 128 | zero2 | full_ft | 8.703 GiB | 10.514 GiB | 1.811 GiB |
| Qwen/Qwen2.5-0.5B | flash2 | 128 | zero2 | full_ft | 8.703 GiB | 10.514 GiB | 1.811 GiB |
| Qwen/Qwen2.5-14B | sdpa | 1024 | zero2 | lora | 39.428 GiB | 41.135 GiB | 1.707 GiB |
| Qwen/Qwen2.5-7B | sdpa | 4096 | zero2 | lora | 40.611 GiB | 42.271 GiB | 1.660 GiB |


## Top 20 Underestimates

| Model | Backend | Seq | Dist | Tune | Measured | Estimated | Signed Error |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: |
| allenai/Olmo-3-1025-7B | sdpa | 2048 | single_gpu | full_ft | 78.561 GiB | 74.935 GiB | -3.626 GiB |
| Qwen/Qwen2.5-7B-Instruct | sdpa | 4096 | single_gpu | lora | 24.033 GiB | 21.217 GiB | -2.816 GiB |
| Qwen/Qwen2.5-7B-Instruct | flash2 | 4096 | single_gpu | lora | 24.033 GiB | 21.217 GiB | -2.816 GiB |
| allenai/Olmo-3-1025-7B | flash2 | 2048 | single_gpu | full_ft | 77.221 GiB | 74.935 GiB | -2.286 GiB |
| Qwen/Qwen2.5-1.5B | flash2 | 2048 | zero2 | full_ft | 31.688 GiB | 29.492 GiB | -2.196 GiB |
| Qwen/Qwen2.5-1.5B | flash2 | 2048 | zero2 | full_ft | 31.688 GiB | 29.492 GiB | -2.196 GiB |
| Qwen/Qwen2.5-0.5B | flash2 | 1024 | zero2 | lora | 5.916 GiB | 3.740 GiB | -2.176 GiB |
| Qwen/Qwen2.5-0.5B | flash2 | 512 | zero2 | lora | 5.916 GiB | 3.740 GiB | -2.176 GiB |
| Qwen/Qwen2.5-0.5B | flash2 | 1024 | zero2 | lora | 5.916 GiB | 3.740 GiB | -2.176 GiB |
| Qwen/Qwen2.5-0.5B | flash2 | 512 | zero2 | lora | 5.916 GiB | 3.740 GiB | -2.176 GiB |
| Qwen/Qwen2.5-1.5B | flash2 | 1024 | zero2 | lora | 9.158 GiB | 7.235 GiB | -1.923 GiB |
| Qwen/Qwen2.5-1.5B | flash2 | 512 | zero2 | lora | 9.158 GiB | 7.235 GiB | -1.923 GiB |
| Qwen/Qwen2.5-1.5B | flash2 | 1024 | zero2 | lora | 9.158 GiB | 7.235 GiB | -1.923 GiB |
| Qwen/Qwen2.5-1.5B | flash2 | 512 | zero2 | lora | 9.158 GiB | 7.235 GiB | -1.923 GiB |
| Qwen/Qwen2.5-0.5B | flash2 | 2048 | single_gpu | lora | 7.947 GiB | 6.239 GiB | -1.709 GiB |
| Qwen/Qwen3-0.6B | sdpa | 512 | single_gpu | full_ft | 5.744 GiB | 4.068 GiB | -1.676 GiB |
| Qwen/Qwen2.5-0.5B | flash2 | 512 | zero2 | lora | 4.393 GiB | 2.740 GiB | -1.653 GiB |
| Qwen/Qwen2.5-0.5B | flash2 | 512 | zero2 | lora | 4.393 GiB | 2.740 GiB | -1.653 GiB |
| Qwen/Qwen2.5-1.5B | flash2 | 512 | zero2 | lora | 7.043 GiB | 5.476 GiB | -1.567 GiB |
| Qwen/Qwen2.5-1.5B | flash2 | 512 | zero2 | lora | 7.043 GiB | 5.476 GiB | -1.567 GiB |


## Top 20 Phase Mismatches

| Model | Backend | Seq | Dist | Tune | Measured Phase | Estimated Phase | Global Error | Proxy Error |
| --- | --- | ---: | --- | --- | --- | --- | ---: | ---: |
| HuggingFaceTB/SmolLM2-1.7B | sdpa | 2048 | single_gpu | full_ft | optimizer_step | backward | 4.621 GiB | 0.720 GiB |
| allenai/Olmo-3-1025-7B | sdpa | 2048 | single_gpu | full_ft | optimizer_step | backward | 3.626 GiB | 3.505 GiB |
| allenai/OLMo-1B-hf | sdpa | 2048 | single_gpu | full_ft | optimizer_step | backward | 3.552 GiB | 0.528 GiB |
| EleutherAI/pythia-1b-deduped | sdpa | 2048 | single_gpu | full_ft | optimizer_step | backward | 3.378 GiB | 1.531 GiB |
| Qwen/Qwen2.5-1.5B | flash2 | 2048 | single_gpu | full_ft | optimizer_step | backward | 3.302 GiB | 2.604 GiB |
| TinyLlama/TinyLlama-1.1B-Chat-v1.0 | sdpa | 2048 | single_gpu | full_ft | optimizer_step | backward | 2.909 GiB | 0.441 GiB |
| allenai/Olmo-3-1025-7B | flash2 | 2048 | single_gpu | full_ft | optimizer_step | backward | 2.286 GiB | 3.505 GiB |
| Qwen/Qwen2.5-7B-Instruct | flash2 | 4096 | single_gpu | lora | optimizer_step | backward | 1.912 GiB | 0.017 GiB |
| HuggingFaceTB/SmolLM2-1.7B | sdpa | 2048 | ddp | full_ft | optimizer_step | backward | 1.880 GiB | 0.720 GiB |
| Qwen/Qwen3-0.6B | sdpa | 512 | single_gpu | full_ft | optimizer_step | backward | 1.676 GiB | 0.333 GiB |
| allenai/OLMo-1B-hf | sdpa | 2048 | ddp | full_ft | optimizer_step | backward | 1.642 GiB | 0.528 GiB |
| allenai/Olmo-3-1025-7B | flash2 | 4096 | single_gpu | lora | optimizer_step | backward | 1.591 GiB | 1.743 GiB |
| Qwen/Qwen3-0.6B | sdpa | 512 | single_gpu | full_ft | optimizer_step | backward | 1.531 GiB | 0.333 GiB |
| Qwen/Qwen3-0.6B | sdpa | 512 | ddp | full_ft | backward | optimizer_step | 1.510 GiB | 0.335 GiB |
| EleutherAI/pythia-1b-deduped | sdpa | 2048 | single_gpu | lora | optimizer_step | backward | 1.477 GiB | 1.366 GiB |
| Qwen/Qwen3-0.6B | sdpa | 2048 | single_gpu | lora | optimizer_step | backward | 1.119 GiB | 0.007 GiB |
| Qwen/Qwen3-0.6B | sdpa | 2048 | single_gpu | lora | optimizer_step | backward | 1.050 GiB | 0.007 GiB |
| Qwen/Qwen3-0.6B | sdpa | 2048 | single_gpu | lora | optimizer_step | backward | 1.034 GiB | 0.007 GiB |
| Qwen/Qwen3-0.6B | sdpa | 2048 | single_gpu | lora | optimizer_step | backward | 1.033 GiB | 0.007 GiB |
| Qwen/Qwen2.5-1.5B | flash2 | 2048 | single_gpu | lora | optimizer_step | backward | 1.017 GiB | 0.721 GiB |


## Failures

- `tiiuae/falcon-rw-1b::full_ft-single_gpu-adamw-standard-seq256-mb1::Unsupported model_type `falcon`. Missing explicit dense architecture manifest for this family.`
- `tiiuae/falcon-rw-1b::full_ft-single_gpu-adamw-standard-seq1024-mb1::Unsupported model_type `falcon`. Missing explicit dense architecture manifest for this family.`
- `tiiuae/falcon-rw-1b::full_ft-single_gpu-adamw-standard-seq2048-mb1::Unsupported model_type `falcon`. Missing explicit dense architecture manifest for this family.`
- `tiiuae/falcon-rw-1b::lora-single_gpu-adamw-standard-seq256-mb1::Unsupported model_type `falcon`. Missing explicit dense architecture manifest for this family.`
- `tiiuae/falcon-rw-1b::lora-single_gpu-adamw-standard-seq1024-mb1::Unsupported model_type `falcon`. Missing explicit dense architecture manifest for this family.`
- `tiiuae/falcon-rw-1b::lora-single_gpu-adamw-standard-seq2048-mb1::Unsupported model_type `falcon`. Missing explicit dense architecture manifest for this family.`
- `Qwen/Qwen2.5-0.5B::full_ft-zero2-adamw-standard-seq512-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen25_05b_optimizer_zero2_fullft_iter7/fullft-zero2-adamw-standard-seq512-mb1/measurement.json'`
- `Qwen/Qwen2.5-0.5B::full_ft-zero2-adam-standard-seq512-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen25_05b_optimizer_zero2_fullft_iter7/fullft-zero2-adam-standard-seq512-mb1/measurement.json'`
- `Qwen/Qwen2.5-0.5B::full_ft-zero2-sgd-standard-seq512-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen25_05b_optimizer_zero2_fullft_iter7/fullft-zero2-sgd-standard-seq512-mb1/measurement.json'`
- `Qwen/Qwen2.5-0.5B::full_ft-zero2-rmsprop-standard-seq512-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen25_05b_optimizer_zero2_fullft_iter7/fullft-zero2-rmsprop-standard-seq512-mb1/measurement.json'`
- `Qwen/Qwen2.5-0.5B::full_ft-zero2-adagrad-standard-seq512-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen25_05b_optimizer_zero2_fullft_iter7/fullft-zero2-adagrad-standard-seq512-mb1/measurement.json'`
- `Qwen/Qwen2.5-0.5B::full_ft-zero2-adafactor-standard-seq512-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen25_05b_optimizer_zero2_fullft_iter7/fullft-zero2-adafactor-standard-seq512-mb1/measurement.json'`
- `Qwen/Qwen2.5-1.5B::full_ft-zero2-adamw-standard-seq512-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen25_15b_optimizer_zero2_fullft_iter6/fullft-zero2-adamw-standard-seq512-mb1/measurement.json'`
- `Qwen/Qwen2.5-1.5B::full_ft-zero2-adam-standard-seq512-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen25_15b_optimizer_zero2_fullft_iter6/fullft-zero2-adam-standard-seq512-mb1/measurement.json'`
- `Qwen/Qwen2.5-1.5B::full_ft-zero2-sgd-standard-seq512-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen25_15b_optimizer_zero2_fullft_iter6/fullft-zero2-sgd-standard-seq512-mb1/measurement.json'`
- `Qwen/Qwen2.5-1.5B::full_ft-zero2-rmsprop-standard-seq512-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen25_15b_optimizer_zero2_fullft_iter6/fullft-zero2-rmsprop-standard-seq512-mb1/measurement.json'`
- `Qwen/Qwen2.5-1.5B::full_ft-zero2-adagrad-standard-seq512-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen25_15b_optimizer_zero2_fullft_iter6/fullft-zero2-adagrad-standard-seq512-mb1/measurement.json'`
- `Qwen/Qwen2.5-1.5B::full_ft-zero2-adafactor-standard-seq512-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen25_15b_optimizer_zero2_fullft_iter6/fullft-zero2-adafactor-standard-seq512-mb1/measurement.json'`
- `Qwen/Qwen2.5-1.5B::full_ft-zero2-adamw-standard-seq512-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen25_15b_optimizer_zero2_fullft_iter9/fullft-zero2-adamw-standard-seq512-mb1/measurement.json'`
- `Qwen/Qwen2.5-1.5B::full_ft-zero2-adamw-standard-seq1024-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen25_15b_optimizer_zero2_fullft_iter9/fullft-zero2-adamw-standard-seq1024-mb1/measurement.json'`
- `Qwen/Qwen2.5-1.5B::full_ft-zero2-adam-standard-seq512-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen25_15b_optimizer_zero2_fullft_iter9/fullft-zero2-adam-standard-seq512-mb1/measurement.json'`
- `Qwen/Qwen2.5-1.5B::full_ft-zero2-adam-standard-seq1024-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen25_15b_optimizer_zero2_fullft_iter9/fullft-zero2-adam-standard-seq1024-mb1/measurement.json'`
- `Qwen/Qwen2.5-1.5B::full_ft-zero2-sgd-standard-seq512-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen25_15b_optimizer_zero2_fullft_iter9/fullft-zero2-sgd-standard-seq512-mb1/measurement.json'`
- `Qwen/Qwen2.5-1.5B::full_ft-zero2-sgd-standard-seq1024-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen25_15b_optimizer_zero2_fullft_iter9/fullft-zero2-sgd-standard-seq1024-mb1/measurement.json'`
- `Qwen/Qwen2.5-1.5B::full_ft-zero2-rmsprop-standard-seq512-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen25_15b_optimizer_zero2_fullft_iter9/fullft-zero2-rmsprop-standard-seq512-mb1/measurement.json'`
- `Qwen/Qwen2.5-1.5B::full_ft-zero2-rmsprop-standard-seq1024-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen25_15b_optimizer_zero2_fullft_iter9/fullft-zero2-rmsprop-standard-seq1024-mb1/measurement.json'`
- `Qwen/Qwen2.5-1.5B::full_ft-zero2-adagrad-standard-seq512-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen25_15b_optimizer_zero2_fullft_iter9/fullft-zero2-adagrad-standard-seq512-mb1/measurement.json'`
- `Qwen/Qwen2.5-1.5B::full_ft-zero2-adagrad-standard-seq1024-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen25_15b_optimizer_zero2_fullft_iter9/fullft-zero2-adagrad-standard-seq1024-mb1/measurement.json'`
- `Qwen/Qwen2.5-1.5B::full_ft-zero2-adafactor-standard-seq512-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen25_15b_optimizer_zero2_fullft_iter9/fullft-zero2-adafactor-standard-seq512-mb1/measurement.json'`
- `Qwen/Qwen2.5-1.5B::full_ft-zero2-adafactor-standard-seq1024-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen25_15b_optimizer_zero2_fullft_iter9/fullft-zero2-adafactor-standard-seq1024-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::full_ft-single_gpu-adamw-flash2-seq2048-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_flash_modes_iter3/fullft-singlegpu-adamw-flash2-seq2048-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::full_ft-single_gpu-adamw-flash2-seq8192-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_flash_modes_iter3/fullft-singlegpu-adamw-flash2-seq8192-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::full_ft-single_gpu-adafactor-flash2-seq2048-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_flash_modes_iter3/fullft-singlegpu-adafactor-flash2-seq2048-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::full_ft-single_gpu-adafactor-flash2-seq8192-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_flash_modes_iter3/fullft-singlegpu-adafactor-flash2-seq8192-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::full_ft-zero2-adamw-flash2-seq2048-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_flash_modes_iter3/fullft-zero2-adamw-flash2-seq2048-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::full_ft-zero2-adamw-flash2-seq8192-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_flash_modes_iter3/fullft-zero2-adamw-flash2-seq8192-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::full_ft-zero2-adafactor-flash2-seq2048-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_flash_modes_iter3/fullft-zero2-adafactor-flash2-seq2048-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::full_ft-zero2-adafactor-flash2-seq8192-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_flash_modes_iter3/fullft-zero2-adafactor-flash2-seq8192-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::lora-single_gpu-adamw-flash2-seq2048-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_flash_modes_iter3/lora-singlegpu-adamw-flash2-seq2048-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::lora-single_gpu-adamw-flash2-seq8192-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_flash_modes_iter3/lora-singlegpu-adamw-flash2-seq8192-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::lora-single_gpu-adafactor-flash2-seq2048-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_flash_modes_iter3/lora-singlegpu-adafactor-flash2-seq2048-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::lora-single_gpu-adafactor-flash2-seq8192-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_flash_modes_iter3/lora-singlegpu-adafactor-flash2-seq8192-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::lora-zero2-adamw-flash2-seq2048-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_flash_modes_iter3/lora-zero2-adamw-flash2-seq2048-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::lora-zero2-adamw-flash2-seq8192-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_flash_modes_iter3/lora-zero2-adamw-flash2-seq8192-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::lora-zero2-adafactor-flash2-seq2048-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_flash_modes_iter3/lora-zero2-adafactor-flash2-seq2048-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::lora-zero2-adafactor-flash2-seq8192-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_flash_modes_iter3/lora-zero2-adafactor-flash2-seq8192-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::full_ft-ddp-adamw-standard-seq512-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_optimizer_ddp_fullft_iter6/fullft-ddp-adamw-standard-seq512-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::full_ft-ddp-adam-standard-seq512-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_optimizer_ddp_fullft_iter6/fullft-ddp-adam-standard-seq512-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::full_ft-ddp-sgd-standard-seq512-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_optimizer_ddp_fullft_iter6/fullft-ddp-sgd-standard-seq512-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::full_ft-ddp-rmsprop-standard-seq512-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_optimizer_ddp_fullft_iter6/fullft-ddp-rmsprop-standard-seq512-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::full_ft-ddp-adagrad-standard-seq512-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_optimizer_ddp_fullft_iter6/fullft-ddp-adagrad-standard-seq512-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::full_ft-ddp-adafactor-standard-seq512-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_optimizer_ddp_fullft_iter6/fullft-ddp-adafactor-standard-seq512-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::full_ft-ddp-adamw-standard-seq512-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_optimizer_ddp_fullft_iter9/fullft-ddp-adamw-standard-seq512-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::full_ft-ddp-adamw-standard-seq2048-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_optimizer_ddp_fullft_iter9/fullft-ddp-adamw-standard-seq2048-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::full_ft-ddp-adam-standard-seq512-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_optimizer_ddp_fullft_iter9/fullft-ddp-adam-standard-seq512-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::full_ft-ddp-adam-standard-seq2048-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_optimizer_ddp_fullft_iter9/fullft-ddp-adam-standard-seq2048-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::full_ft-ddp-sgd-standard-seq512-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_optimizer_ddp_fullft_iter9/fullft-ddp-sgd-standard-seq512-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::full_ft-ddp-sgd-standard-seq2048-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_optimizer_ddp_fullft_iter9/fullft-ddp-sgd-standard-seq2048-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::full_ft-ddp-rmsprop-standard-seq512-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_optimizer_ddp_fullft_iter9/fullft-ddp-rmsprop-standard-seq512-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::full_ft-ddp-rmsprop-standard-seq2048-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_optimizer_ddp_fullft_iter9/fullft-ddp-rmsprop-standard-seq2048-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::full_ft-ddp-adagrad-standard-seq512-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_optimizer_ddp_fullft_iter9/fullft-ddp-adagrad-standard-seq512-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::full_ft-ddp-adagrad-standard-seq2048-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_optimizer_ddp_fullft_iter9/fullft-ddp-adagrad-standard-seq2048-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::full_ft-ddp-adafactor-standard-seq512-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_optimizer_ddp_fullft_iter9/fullft-ddp-adafactor-standard-seq512-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::full_ft-ddp-adafactor-standard-seq2048-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_optimizer_ddp_fullft_iter9/fullft-ddp-adafactor-standard-seq2048-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::full_ft-single_gpu-adamw-standard-seq512-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_optimizer_single_fullft_iter6/fullft-singlegpu-adamw-standard-seq512-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::full_ft-single_gpu-adamw-standard-seq2048-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_optimizer_single_fullft_iter6/fullft-singlegpu-adamw-standard-seq2048-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::full_ft-single_gpu-adam-standard-seq512-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_optimizer_single_fullft_iter6/fullft-singlegpu-adam-standard-seq512-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::full_ft-single_gpu-adam-standard-seq2048-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_optimizer_single_fullft_iter6/fullft-singlegpu-adam-standard-seq2048-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::full_ft-single_gpu-sgd-standard-seq512-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_optimizer_single_fullft_iter6/fullft-singlegpu-sgd-standard-seq512-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::full_ft-single_gpu-sgd-standard-seq2048-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_optimizer_single_fullft_iter6/fullft-singlegpu-sgd-standard-seq2048-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::full_ft-single_gpu-rmsprop-standard-seq512-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_optimizer_single_fullft_iter6/fullft-singlegpu-rmsprop-standard-seq512-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::full_ft-single_gpu-rmsprop-standard-seq2048-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_optimizer_single_fullft_iter6/fullft-singlegpu-rmsprop-standard-seq2048-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::full_ft-single_gpu-adagrad-standard-seq512-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_optimizer_single_fullft_iter6/fullft-singlegpu-adagrad-standard-seq512-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::full_ft-single_gpu-adagrad-standard-seq2048-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_optimizer_single_fullft_iter6/fullft-singlegpu-adagrad-standard-seq2048-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::full_ft-single_gpu-adafactor-standard-seq512-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_optimizer_single_fullft_iter6/fullft-singlegpu-adafactor-standard-seq512-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::full_ft-single_gpu-adafactor-standard-seq2048-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_optimizer_single_fullft_iter6/fullft-singlegpu-adafactor-standard-seq2048-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::lora-single_gpu-adamw-standard-seq512-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_optimizer_single_lora_iter6/lora-singlegpu-adamw-standard-seq512-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::lora-single_gpu-adamw-standard-seq2048-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_optimizer_single_lora_iter6/lora-singlegpu-adamw-standard-seq2048-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::lora-single_gpu-adam-standard-seq512-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_optimizer_single_lora_iter6/lora-singlegpu-adam-standard-seq512-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::lora-single_gpu-adam-standard-seq2048-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_optimizer_single_lora_iter6/lora-singlegpu-adam-standard-seq2048-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::lora-single_gpu-sgd-standard-seq512-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_optimizer_single_lora_iter6/lora-singlegpu-sgd-standard-seq512-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::lora-single_gpu-sgd-standard-seq2048-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_optimizer_single_lora_iter6/lora-singlegpu-sgd-standard-seq2048-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::lora-single_gpu-rmsprop-standard-seq512-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_optimizer_single_lora_iter6/lora-singlegpu-rmsprop-standard-seq512-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::lora-single_gpu-rmsprop-standard-seq2048-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_optimizer_single_lora_iter6/lora-singlegpu-rmsprop-standard-seq2048-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::lora-single_gpu-adagrad-standard-seq512-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_optimizer_single_lora_iter6/lora-singlegpu-adagrad-standard-seq512-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::lora-single_gpu-adagrad-standard-seq2048-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_optimizer_single_lora_iter6/lora-singlegpu-adagrad-standard-seq2048-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::lora-single_gpu-adafactor-standard-seq512-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_optimizer_single_lora_iter6/lora-singlegpu-adafactor-standard-seq512-mb1/measurement.json'`
- `Qwen/Qwen3-0.6B::lora-single_gpu-adafactor-standard-seq2048-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen3_06b_optimizer_single_lora_iter6/lora-singlegpu-adafactor-standard-seq2048-mb1/measurement.json'`
- `allenai/OLMo-2-1124-13B::lora-single_gpu-adamw-sdpa-seq5120-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/border2_olmo2_13b_lora_sg_sdpa_seq5120_20260406/lora-singlegpu-adamw-sdpa-seq5120-mb1/measurement.json'`
- `Qwen/Qwen2.5-14B::lora-single_gpu-adamw-sdpa-seq4608-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/border2_qwen25_14b_lora_sg_sdpa_seq4608_20260406/lora-singlegpu-adamw-sdpa-seq4608-mb1/measurement.json'`
- `Qwen/Qwen2.5-7B::lora-single_gpu-adamw-sdpa-seq9728-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/border2_qwen25_7b_lora_sg_sdpa_seq9728_20260406/lora-singlegpu-adamw-sdpa-seq9728-mb1/measurement.json'`
- `allenai/OLMo-2-1124-13B::lora-single_gpu-adamw-sdpa-seq4096-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/high_vram_olmo2_13b_lora_sg_seq4096/lora-singlegpu-adamw-sdpa-seq4096-mb1/measurement.json'`
- `allenai/OLMo-2-1124-13B::lora-single_gpu-adamw-flash2-seq4096-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/high_vram_olmo2_13b_lora_sg_seq4096/lora-singlegpu-adamw-flash2-seq4096-mb1/measurement.json'`
- `allenai/OLMo-2-1124-13B::lora-single_gpu-adamw-sdpa-seq4608-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/high_vram_olmo2_13b_lora_sg_seq4608_20260406/lora-singlegpu-adamw-sdpa-seq4608-mb1/measurement.json'`
- `allenai/OLMo-2-1124-13B::lora-single_gpu-adamw-flash2-seq4608-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/high_vram_olmo2_13b_lora_sg_seq4608_20260406/lora-singlegpu-adamw-flash2-seq4608-mb1/measurement.json'`
- `allenai/Olmo-3-1025-7B::lora-single_gpu-adamw-sdpa-seq7808-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/high_vram_olmo3_7b_lora_sg_seq7808/lora-singlegpu-adamw-sdpa-seq7808-mb1/measurement.json'`
- `allenai/Olmo-3-1025-7B::lora-single_gpu-adamw-flash2-seq7808-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/high_vram_olmo3_7b_lora_sg_seq7808/lora-singlegpu-adamw-flash2-seq7808-mb1/measurement.json'`
- `allenai/Olmo-3-1025-7B::lora-single_gpu-adamw-sdpa-seq8704-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/high_vram_olmo3_7b_lora_sg_seq8704_20260406/lora-singlegpu-adamw-sdpa-seq8704-mb1/measurement.json'`
- `allenai/Olmo-3-1025-7B::lora-single_gpu-adamw-flash2-seq8704-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/high_vram_olmo3_7b_lora_sg_seq8704_20260406/lora-singlegpu-adamw-flash2-seq8704-mb1/measurement.json'`
- `Qwen/Qwen2.5-14B::lora-single_gpu-adamw-sdpa-seq3328-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/high_vram_qwen25_14b_lora_sg_seq3328/lora-singlegpu-adamw-sdpa-seq3328-mb1/measurement.json'`
- `Qwen/Qwen2.5-14B::lora-single_gpu-adamw-flash2-seq3328-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/high_vram_qwen25_14b_lora_sg_seq3328/lora-singlegpu-adamw-flash2-seq3328-mb1/measurement.json'`
- `Qwen/Qwen2.5-14B::lora-single_gpu-adamw-sdpa-seq4096-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/high_vram_qwen25_14b_lora_sg_seq4096_20260406/lora-singlegpu-adamw-sdpa-seq4096-mb1/measurement.json'`
- `Qwen/Qwen2.5-14B::lora-single_gpu-adamw-flash2-seq4096-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/high_vram_qwen25_14b_lora_sg_seq4096_20260406/lora-singlegpu-adamw-flash2-seq4096-mb1/measurement.json'`
- `Qwen/Qwen2.5-7B::lora-single_gpu-adamw-sdpa-seq8192-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/high_vram_qwen25_7b_lora_sg/lora-singlegpu-adamw-sdpa-seq8192-mb1/measurement.json'`
- `Qwen/Qwen2.5-7B::lora-single_gpu-adamw-flash2-seq8192-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/high_vram_qwen25_7b_lora_sg/lora-singlegpu-adamw-flash2-seq8192-mb1/measurement.json'`
- `Qwen/Qwen2.5-7B::lora-single_gpu-adamw-sdpa-seq9216-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/high_vram_qwen25_7b_lora_sg_seq9216_20260406/lora-singlegpu-adamw-sdpa-seq9216-mb1/measurement.json'`
- `Qwen/Qwen2.5-7B::lora-single_gpu-adamw-flash2-seq9216-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/high_vram_qwen25_7b_lora_sg_seq9216_20260406/lora-singlegpu-adamw-flash2-seq9216-mb1/measurement.json'`
- `Qwen/Qwen2.5-0.5B::full_ft-single_gpu-flash2-seq512-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen25_05b_flash_seq512_1024_2048_mb1_iter1/fullft-singlegpu-flash2-seq512-mb1/measurement.json'`
- `Qwen/Qwen2.5-0.5B::full_ft-single_gpu-flash2-seq1024-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen25_05b_flash_seq512_1024_2048_mb1_iter1/fullft-singlegpu-flash2-seq1024-mb1/measurement.json'`
- `Qwen/Qwen2.5-0.5B::full_ft-single_gpu-flash2-seq2048-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen25_05b_flash_seq512_1024_2048_mb1_iter1/fullft-singlegpu-flash2-seq2048-mb1/measurement.json'`
- `Qwen/Qwen2.5-0.5B::lora-single_gpu-flash2-seq512-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen25_05b_flash_seq512_1024_2048_mb1_iter1/lora-singlegpu-flash2-seq512-mb1/measurement.json'`
- `Qwen/Qwen2.5-0.5B::lora-single_gpu-flash2-seq1024-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen25_05b_flash_seq512_1024_2048_mb1_iter1/lora-singlegpu-flash2-seq1024-mb1/measurement.json'`
- `Qwen/Qwen2.5-0.5B::lora-single_gpu-flash2-seq2048-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen25_05b_flash_seq512_1024_2048_mb1_iter1/lora-singlegpu-flash2-seq2048-mb1/measurement.json'`
- `Qwen/Qwen2.5-1.5B::full_ft-single_gpu-flash2-seq512-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen25_15b_flash_seq512_1024_2048_mb1_iter1/fullft-singlegpu-flash2-seq512-mb1/measurement.json'`
- `Qwen/Qwen2.5-1.5B::full_ft-single_gpu-flash2-seq1024-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen25_15b_flash_seq512_1024_2048_mb1_iter1/fullft-singlegpu-flash2-seq1024-mb1/measurement.json'`
- `Qwen/Qwen2.5-1.5B::full_ft-single_gpu-flash2-seq2048-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen25_15b_flash_seq512_1024_2048_mb1_iter1/fullft-singlegpu-flash2-seq2048-mb1/measurement.json'`
- `Qwen/Qwen2.5-1.5B::lora-single_gpu-flash2-seq512-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen25_15b_flash_seq512_1024_2048_mb1_iter1/lora-singlegpu-flash2-seq512-mb1/measurement.json'`
- `Qwen/Qwen2.5-1.5B::lora-single_gpu-flash2-seq1024-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen25_15b_flash_seq512_1024_2048_mb1_iter1/lora-singlegpu-flash2-seq1024-mb1/measurement.json'`
- `Qwen/Qwen2.5-1.5B::lora-single_gpu-flash2-seq2048-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/qwen25_15b_flash_seq512_1024_2048_mb1_iter1/lora-singlegpu-flash2-seq2048-mb1/measurement.json'`
- `Qwen/Qwen2.5-0.5B::lora-single_gpu-adamw-flash2-seq2048-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/refresh_lora_flash2_qwen25_05b/lora-singlegpu-adamw-flash2-seq2048-mb1/measurement.json'`
- `Qwen/Qwen2.5-1.5B::lora-single_gpu-adamw-flash2-seq2048-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/refresh_lora_flash2_qwen25_15b/lora-singlegpu-adamw-flash2-seq2048-mb1/measurement.json'`
- `allenai/OLMo-1B-hf::lora-single_gpu-adamw-sdpa-ckpt-seq4096-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/single_gpu_80gb_ckpt_scaled_20260406/olmo1b_lora_seq4096_ckpt/lora-singlegpu-adamw-sdpa-ckpt-seq4096-mb1/measurement.json'`
- `allenai/OLMo-1B-hf::lora-single_gpu-adamw-sdpa-seq4096-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/single_gpu_80gb_ckpt_scaled_20260406/olmo1b_lora_seq4096_no_ckpt/lora-singlegpu-adamw-sdpa-seq4096-mb1/measurement.json'`
- `Qwen/Qwen2.5-14B-Instruct::lora-single_gpu-adamw-sdpa-ckpt-seq4096-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/single_gpu_80gb_ckpt_scaled_20260406/qwen25_14b_lora_seq4096_ckpt/lora-singlegpu-adamw-sdpa-ckpt-seq4096-mb1/measurement.json'`
- `Qwen/Qwen2.5-1.5B-Instruct::lora-single_gpu-adamw-sdpa-ckpt-seq4096-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/single_gpu_80gb_ckpt_scaled_20260406/qwen25_15b_lora_seq4096_ckpt/lora-singlegpu-adamw-sdpa-ckpt-seq4096-mb1/measurement.json'`
- `Qwen/Qwen2.5-1.5B-Instruct::lora-single_gpu-adamw-sdpa-seq4096-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/single_gpu_80gb_ckpt_scaled_20260406/qwen25_15b_lora_seq4096_no_ckpt/lora-singlegpu-adamw-sdpa-seq4096-mb1/measurement.json'`
- `Qwen/Qwen2.5-7B::full_ft-zero2-adamw-sdpa-seq4096-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/zero_all_configs_qwen25_7b_seq4096_20260406/fullft-zero2-adamw-sdpa-seq4096-mb1/measurement.json'`
- `Qwen/Qwen2.5-7B::full_ft-zero2-adamw-flash2-seq4096-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/zero_all_configs_qwen25_7b_seq4096_20260406/fullft-zero2-adamw-flash2-seq4096-mb1/measurement.json'`
- `Qwen/Qwen2.5-7B::full_ft-zero3-adamw-sdpa-seq4096-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/zero_all_configs_qwen25_7b_seq4096_20260406/fullft-zero3-adamw-sdpa-seq4096-mb1/measurement.json'`
- `Qwen/Qwen2.5-7B::full_ft-zero3-adamw-flash2-seq4096-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/zero_all_configs_qwen25_7b_seq4096_20260406/fullft-zero3-adamw-flash2-seq4096-mb1/measurement.json'`
- `Qwen/Qwen2.5-7B::lora-zero2-adamw-sdpa-seq4096-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/zero_all_configs_qwen25_7b_seq4096_20260406/lora-zero2-adamw-sdpa-seq4096-mb1/measurement.json'`
- `Qwen/Qwen2.5-7B::lora-zero2-adamw-flash2-seq4096-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/zero_all_configs_qwen25_7b_seq4096_20260406/lora-zero2-adamw-flash2-seq4096-mb1/measurement.json'`
- `Qwen/Qwen2.5-7B::lora-zero3-adamw-sdpa-seq4096-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/zero_all_configs_qwen25_7b_seq4096_20260406/lora-zero3-adamw-sdpa-seq4096-mb1/measurement.json'`
- `Qwen/Qwen2.5-7B::lora-zero3-adamw-flash2-seq4096-mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/zero_all_configs_qwen25_7b_seq4096_20260406/lora-zero3-adamw-flash2-seq4096-mb1/measurement.json'`
- `allenai/Olmo-3-1025-7B::full_ft_zero2_ckpt_flash2_seq16384_mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/olmo3_7b_a10080gb_ckpt_flash2_clean_20260406/full_ft_zero2_ckpt_flash2_seq16384_mb1/measurement.json'`
- `allenai/Olmo-3-1025-7B::full_ft_zero3_ckpt_flash2_seq16384_mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/olmo3_7b_a10080gb_ckpt_flash2_clean_20260406/full_ft_zero3_ckpt_flash2_seq16384_mb1/measurement.json'`
- `allenai/Olmo-3-1025-7B::full_ft_zero3_ckpt_flash2_seq24576_mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/olmo3_7b_a10080gb_ckpt_flash2_clean_20260406/full_ft_zero3_ckpt_flash2_seq24576_mb1/measurement.json'`
- `allenai/Olmo-3-1025-7B::lora_ddp_ckpt_flash2_seq32768_mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/olmo3_7b_a10080gb_ckpt_flash2_clean_20260406/lora_ddp_ckpt_flash2_seq32768_mb1/measurement.json'`
- `allenai/Olmo-3-1025-7B::lora_zero2_ckpt_flash2_seq32768_mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/olmo3_7b_a10080gb_ckpt_flash2_clean_20260406/lora_zero2_ckpt_flash2_seq32768_mb1/measurement.json'`
- `allenai/Olmo-3-1025-7B::lora_zero3_ckpt_flash2_seq24576_mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/olmo3_7b_a10080gb_ckpt_flash2_clean_20260406/lora_zero3_ckpt_flash2_seq24576_mb1/measurement.json'`
- `allenai/Olmo-3-1025-7B::lora_zero3_ckpt_flash2_seq32768_mb1::[Errno 2] No such file or directory: 'benchmark_artifacts/olmo3_7b_a10080gb_ckpt_flash2_clean_20260406/lora_zero3_ckpt_flash2_seq32768_mb1/measurement.json'`
- `allenai/Olmo-3.1-32B-Think::olmo31_32b_lora_zero2_seq2048_ckpt::[Errno 2] No such file or directory: 'benchmark_artifacts/zero2_h100_ckpt_20260406/olmo31_32b_lora_zero2_seq2048_ckpt/measurement.json'`
- `allenai/Olmo-3-1025-7B::olmo3_7b_lora_zero2_seq4096_ckpt::[Errno 2] No such file or directory: 'benchmark_artifacts/zero2_h100_ckpt_20260406/olmo3_7b_lora_zero2_seq4096_ckpt/measurement.json'`
- `Qwen/Qwen2.5-14B-Instruct::qwen25_14b_lora_zero2_seq4096_ckpt::[Errno 2] No such file or directory: 'benchmark_artifacts/zero2_h100_ckpt_20260406/qwen25_14b_lora_zero2_seq4096_ckpt/measurement.json'`
- `Qwen/Qwen2.5-7B-Instruct::qwen25_7b_fullft_zero2_seq2048_ckpt::[Errno 2] No such file or directory: 'benchmark_artifacts/zero2_h100_ckpt_20260406/qwen25_7b_fullft_zero2_seq2048_ckpt/measurement.json'`
- `Qwen/Qwen2.5-7B-Instruct::qwen25_7b_lora_zero2_seq4096_ckpt::[Errno 2] No such file or directory: 'benchmark_artifacts/zero2_h100_ckpt_20260406/qwen25_7b_lora_zero2_seq4096_ckpt/measurement.json'`
- `allenai/Olmo-3.1-32B-Think::olmo31_32b_lora_zero3_seq2048_ckpt::[Errno 2] No such file or directory: 'benchmark_artifacts/zero3_h100_ckpt_20260406/olmo31_32b_lora_zero3_seq2048_ckpt/measurement.json'`
- `allenai/Olmo-3-1025-7B::olmo3_7b_lora_zero3_seq4096_ckpt::[Errno 2] No such file or directory: 'benchmark_artifacts/zero3_h100_ckpt_20260406/olmo3_7b_lora_zero3_seq4096_ckpt/measurement.json'`
- `Qwen/Qwen2.5-14B-Instruct::qwen25_14b_lora_zero3_seq4096_ckpt::[Errno 2] No such file or directory: 'benchmark_artifacts/zero3_h100_ckpt_20260406/qwen25_14b_lora_zero3_seq4096_ckpt/measurement.json'`
- `Qwen/Qwen2.5-7B-Instruct::qwen25_7b_fullft_zero3_seq2048_ckpt::[Errno 2] No such file or directory: 'benchmark_artifacts/zero3_h100_ckpt_20260406/qwen25_7b_fullft_zero3_seq2048_ckpt/measurement.json'`
- `Qwen/Qwen2.5-7B-Instruct::qwen25_7b_lora_zero3_seq4096_ckpt::[Errno 2] No such file or directory: 'benchmark_artifacts/zero3_h100_ckpt_20260406/qwen25_7b_lora_zero3_seq4096_ckpt/measurement.json'`
