# Phase-Aligned Calibration Replay

- Replayed rows: `324`
- Failed rows: `6`
- Global MAE: `1.332 GiB`
- Mean absolute relative error: `9.1%`
- Median absolute relative error: `6.1%`
- Phase match rate: `63.6%`
- Retained-forward proxy MAE: `1.101 GiB`

## Component MAE

| Metric | Raw | Phase-Aligned |
| --- | ---: | ---: |
| parameter_bytes | 0.287 GiB | 0.287 GiB |
| gradient_bytes | 0.338 GiB | 0.338 GiB |
| optimizer_state_bytes | 0.124 GiB | 0.124 GiB |
| activation_bytes | 2.332 GiB | `n/a` |
| transient_bytes | 5.418 GiB | 5.742 GiB |
| runtime_reserve_bytes | 0.817 GiB | 0.817 GiB |


## Phase Confusion

| Measured | Estimated | Count |
| --- | --- | ---: |
| backward | backward | 104 |
| backward | optimizer_step | 19 |
| optimizer_step | backward | 99 |
| optimizer_step | optimizer_step | 102 |


## Targeted Slices

| Slice | Count | Global MAE | Proxy MAE | Phase Match |
| --- | ---: | ---: | ---: | ---: |
| Non-Eager Long-Seq Full FT | 12 | 4.408 GiB | 7.358 GiB | 83.3% |
| Non-Eager Long-Seq LoRA | 52 | 2.381 GiB | 2.287 GiB | 50.0% |
| ZeRO-2 | 111 | 1.390 GiB | 0.862 GiB | 92.8% |
| ZeRO-3 | 13 | 3.219 GiB | 6.504 GiB | 38.5% |


### By measured phase

| Group | Count | Global MAE | Proxy MAE | Phase Match |
| --- | ---: | ---: | ---: | ---: |
| backward | 123 | 1.521 GiB | 1.562 GiB | 84.6% |
| optimizer_step | 201 | 1.217 GiB | 0.818 GiB | 50.7% |


### By attention backend

| Group | Count | Global MAE | Proxy MAE | Phase Match |
| --- | ---: | ---: | ---: | ---: |
| flash2 | 117 | 1.711 GiB | 1.632 GiB | 67.5% |
| sdpa | 207 | 1.119 GiB | 0.801 GiB | 61.4% |


### By sequence length

| Group | Count | Global MAE | Proxy MAE | Phase Match |
| --- | ---: | ---: | ---: | ---: |
| 1024 | 29 | 0.957 GiB | 0.525 GiB | 51.7% |
| 128 | 16 | 1.495 GiB | 0.107 GiB | 100.0% |
| 16384 | 2 | 3.575 GiB | 10.660 GiB | 50.0% |
| 2048 | 93 | 1.107 GiB | 1.035 GiB | 58.1% |
| 24576 | 2 | 6.304 GiB | 11.590 GiB | 0.0% |
| 256 | 6 | 0.300 GiB | 0.087 GiB | 100.0% |
| 32768 | 3 | 2.646 GiB | 7.058 GiB | 33.3% |
| 3328 | 2 | 6.611 GiB | 2.631 GiB | 0.0% |
| 4096 | 32 | 2.384 GiB | 2.182 GiB | 53.1% |
| 4608 | 3 | 3.729 GiB | 2.894 GiB | 0.0% |
| 512 | 114 | 0.748 GiB | 0.268 GiB | 69.3% |
| 5120 | 1 | 0.067 GiB | 5.769 GiB | 0.0% |
| 7808 | 2 | 2.247 GiB | 4.753 GiB | 0.0% |
| 8192 | 14 | 3.171 GiB | 2.599 GiB | 100.0% |
| 8704 | 2 | 2.990 GiB | 5.402 GiB | 0.0% |
| 9216 | 2 | 2.622 GiB | 0.170 GiB | 100.0% |
| 9728 | 1 | 1.360 GiB | 0.220 GiB | 100.0% |


### By distributed mode

| Group | Count | Global MAE | Proxy MAE | Phase Match |
| --- | ---: | ---: | ---: | ---: |
| ddp | 29 | 1.136 GiB | 0.795 GiB | 82.8% |
| single_gpu | 171 | 1.185 GiB | 0.897 GiB | 43.3% |
| zero2 | 111 | 1.390 GiB | 0.862 GiB | 92.8% |
| zero3 | 13 | 3.219 GiB | 6.504 GiB | 38.5% |


### By tuning mode

| Group | Count | Global MAE | Proxy MAE | Phase Match |
| --- | ---: | ---: | ---: | ---: |
| full_ft | 177 | 1.185 GiB | 1.173 GiB | 74.0% |
| lora | 147 | 1.510 GiB | 1.014 GiB | 51.0% |


## Top 20 Overestimates

| Model | Backend | Seq | Dist | Tune | Measured | Estimated | Signed Error |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: |
| HuggingFaceTB/SmolLM2-1.7B | sdpa | 8192 | single_gpu | full_ft | 36.568 GiB | 50.019 GiB | 13.451 GiB |
| Qwen/Qwen2.5-14B | flash2 | 4096 | single_gpu | lora | 79.367 GiB | 90.557 GiB | 11.190 GiB |
| Qwen/Qwen2.5-14B | flash2 | 3328 | single_gpu | lora | 70.256 GiB | 78.863 GiB | 8.607 GiB |
| Qwen/Qwen2.5-14B | sdpa | 2048 | zero2 | lora | 46.922 GiB | 53.770 GiB | 6.848 GiB |
| Qwen/Qwen2.5-14B | sdpa | 4608 | single_gpu | lora | 86.020 GiB | 92.830 GiB | 6.810 GiB |
| Qwen/Qwen2.5-14B | sdpa | 4096 | single_gpu | lora | 79.371 GiB | 85.647 GiB | 6.276 GiB |
| HuggingFaceTB/SmolLM2-1.7B | sdpa | 2048 | single_gpu | full_ft | 17.672 GiB | 22.293 GiB | 4.621 GiB |
| Qwen/Qwen2.5-14B | sdpa | 3328 | single_gpu | lora | 70.258 GiB | 74.873 GiB | 4.616 GiB |
| allenai/OLMo-2-1124-13B | flash2 | 4608 | single_gpu | lora | 82.357 GiB | 86.635 GiB | 4.277 GiB |
| Qwen/Qwen2.5-7B-Instruct | sdpa | 2048 | zero3 | full_ft | 76.430 GiB | 80.619 GiB | 4.190 GiB |
| allenai/Olmo-3-1025-7B | flash2 | 32768 | zero2 | lora | 66.689 GiB | 70.780 GiB | 4.091 GiB |
| Qwen/Qwen2.5-7B | flash2 | 9216 | single_gpu | lora | 85.689 GiB | 89.622 GiB | 3.932 GiB |
| allenai/OLMo-2-1124-13B | flash2 | 4096 | single_gpu | lora | 75.994 GiB | 79.924 GiB | 3.930 GiB |
| allenai/Olmo-3-1025-7B | flash2 | 32768 | ddp | lora | 66.938 GiB | 70.768 GiB | 3.831 GiB |
| Qwen/Qwen2.5-7B | flash2 | 8192 | single_gpu | lora | 77.637 GiB | 81.290 GiB | 3.653 GiB |
| allenai/OLMo-1B-hf | sdpa | 2048 | single_gpu | full_ft | 11.762 GiB | 15.314 GiB | 3.552 GiB |
| Qwen/Qwen2.5-1.5B | sdpa | 1024 | zero2 | full_ft | 31.260 GiB | 34.811 GiB | 3.551 GiB |
| EleutherAI/pythia-1b-deduped | sdpa | 2048 | single_gpu | full_ft | 10.566 GiB | 13.944 GiB | 3.378 GiB |
| Qwen/Qwen2.5-1.5B | flash2 | 2048 | single_gpu | full_ft | 19.281 GiB | 22.583 GiB | 3.302 GiB |
| Qwen/Qwen2.5-1.5B | flash2 | 2048 | single_gpu | full_ft | 19.281 GiB | 22.583 GiB | 3.302 GiB |


## Top 20 Underestimates

| Model | Backend | Seq | Dist | Tune | Measured | Estimated | Signed Error |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: |
| allenai/Olmo-3-1025-7B | flash2 | 24576 | zero3 | full_ft | 71.723 GiB | 60.785 GiB | -10.937 GiB |
| allenai/Olmo-3-1025-7B | flash2 | 16384 | zero2 | full_ft | 66.877 GiB | 61.210 GiB | -5.667 GiB |
| Qwen/Qwen3-0.6B | flash2 | 8192 | zero2 | full_ft | 37.027 GiB | 32.855 GiB | -4.173 GiB |
| Qwen/Qwen3-0.6B | flash2 | 8192 | zero2 | full_ft | 35.906 GiB | 31.744 GiB | -4.162 GiB |
| Qwen/Qwen2.5-7B-Instruct | sdpa | 4096 | zero3 | lora | 21.320 GiB | 17.338 GiB | -3.982 GiB |
| allenai/Olmo-3-1025-7B | sdpa | 2048 | single_gpu | full_ft | 78.561 GiB | 74.935 GiB | -3.626 GiB |
| Qwen/Qwen2.5-14B-Instruct | sdpa | 4096 | zero3 | lora | 30.209 GiB | 26.600 GiB | -3.609 GiB |
| Qwen/Qwen3-0.6B | flash2 | 8192 | single_gpu | full_ft | 35.170 GiB | 31.800 GiB | -3.370 GiB |
| allenai/Olmo-3-1025-7B | sdpa | 4096 | zero3 | lora | 18.551 GiB | 15.212 GiB | -3.338 GiB |
| Qwen/Qwen3-0.6B | flash2 | 8192 | single_gpu | full_ft | 32.875 GiB | 29.581 GiB | -3.294 GiB |
| Qwen/Qwen2.5-7B | sdpa | 4096 | zero3 | lora | 45.654 GiB | 42.445 GiB | -3.209 GiB |
| Qwen/Qwen2.5-7B | flash2 | 4096 | zero3 | lora | 45.654 GiB | 42.445 GiB | -3.209 GiB |
| Qwen/Qwen2.5-1.5B-Instruct | sdpa | 4096 | single_gpu | lora | 11.885 GiB | 8.722 GiB | -3.162 GiB |
| Qwen/Qwen2.5-1.5B-Instruct | sdpa | 4096 | single_gpu | lora | 22.006 GiB | 18.962 GiB | -3.044 GiB |
| allenai/Olmo-3-1025-7B | sdpa | 8704 | single_gpu | lora | 90.303 GiB | 87.267 GiB | -3.036 GiB |
| Qwen/Qwen2.5-7B-Instruct | sdpa | 4096 | single_gpu | lora | 24.033 GiB | 21.217 GiB | -2.816 GiB |
| Qwen/Qwen2.5-7B-Instruct | flash2 | 4096 | single_gpu | lora | 24.033 GiB | 21.217 GiB | -2.816 GiB |
| allenai/Olmo-3.1-32B-Think | sdpa | 2048 | zero3 | lora | 39.730 GiB | 36.963 GiB | -2.767 GiB |
| allenai/Olmo-3-1025-7B | flash2 | 2048 | single_gpu | full_ft | 77.221 GiB | 74.935 GiB | -2.286 GiB |
| Qwen/Qwen2.5-1.5B | flash2 | 2048 | zero2 | full_ft | 31.688 GiB | 29.492 GiB | -2.196 GiB |


## Top 20 Phase Mismatches

| Model | Backend | Seq | Dist | Tune | Measured Phase | Estimated Phase | Global Error | Proxy Error |
| --- | --- | ---: | --- | --- | --- | --- | ---: | ---: |
| Qwen/Qwen2.5-14B | flash2 | 4096 | single_gpu | lora | optimizer_step | backward | 11.190 GiB | 5.452 GiB |
| allenai/Olmo-3-1025-7B | flash2 | 24576 | zero3 | full_ft | backward | optimizer_step | 10.937 GiB | 16.464 GiB |
| Qwen/Qwen2.5-14B | flash2 | 3328 | single_gpu | lora | optimizer_step | backward | 8.607 GiB | 4.155 GiB |
| Qwen/Qwen2.5-14B | sdpa | 4608 | single_gpu | lora | optimizer_step | backward | 6.810 GiB | 1.751 GiB |
| Qwen/Qwen2.5-14B | sdpa | 4096 | single_gpu | lora | optimizer_step | backward | 6.276 GiB | 1.702 GiB |
| HuggingFaceTB/SmolLM2-1.7B | sdpa | 2048 | single_gpu | full_ft | optimizer_step | backward | 4.621 GiB | 0.720 GiB |
| Qwen/Qwen2.5-14B | sdpa | 3328 | single_gpu | lora | optimizer_step | backward | 4.616 GiB | 1.108 GiB |
| allenai/OLMo-2-1124-13B | flash2 | 4608 | single_gpu | lora | optimizer_step | backward | 4.277 GiB | 1.708 GiB |
| allenai/Olmo-3-1025-7B | flash2 | 32768 | zero2 | lora | backward | optimizer_step | 4.091 GiB | 6.426 GiB |
| Qwen/Qwen2.5-7B-Instruct | sdpa | 4096 | zero3 | lora | backward | optimizer_step | 3.982 GiB | 3.075 GiB |
| allenai/OLMo-2-1124-13B | flash2 | 4096 | single_gpu | lora | optimizer_step | backward | 3.930 GiB | 1.397 GiB |
| allenai/Olmo-3-1025-7B | sdpa | 2048 | single_gpu | full_ft | optimizer_step | backward | 3.626 GiB | 3.505 GiB |
| Qwen/Qwen2.5-14B-Instruct | sdpa | 4096 | zero3 | lora | backward | optimizer_step | 3.609 GiB | 3.166 GiB |
| allenai/OLMo-1B-hf | sdpa | 2048 | single_gpu | full_ft | optimizer_step | backward | 3.552 GiB | 0.528 GiB |
| EleutherAI/pythia-1b-deduped | sdpa | 2048 | single_gpu | full_ft | optimizer_step | backward | 3.378 GiB | 1.531 GiB |
| allenai/Olmo-3-1025-7B | sdpa | 4096 | zero3 | lora | backward | optimizer_step | 3.338 GiB | 2.716 GiB |
| Qwen/Qwen2.5-1.5B | flash2 | 2048 | single_gpu | full_ft | optimizer_step | backward | 3.302 GiB | 2.604 GiB |
| Qwen/Qwen2.5-1.5B | flash2 | 2048 | single_gpu | full_ft | optimizer_step | backward | 3.302 GiB | 2.604 GiB |
| allenai/Olmo-3-1025-7B | sdpa | 8704 | single_gpu | lora | optimizer_step | backward | 3.036 GiB | 7.096 GiB |
| allenai/Olmo-3-1025-7B | flash2 | 8704 | single_gpu | lora | optimizer_step | backward | 2.943 GiB | 3.709 GiB |


## Failures

- `tiiuae/falcon-rw-1b::full_ft-single_gpu-adamw-standard-seq256-mb1::Unsupported model_type `falcon`. Missing explicit dense architecture manifest for this family.`
- `tiiuae/falcon-rw-1b::full_ft-single_gpu-adamw-standard-seq1024-mb1::Unsupported model_type `falcon`. Missing explicit dense architecture manifest for this family.`
- `tiiuae/falcon-rw-1b::full_ft-single_gpu-adamw-standard-seq2048-mb1::Unsupported model_type `falcon`. Missing explicit dense architecture manifest for this family.`
- `tiiuae/falcon-rw-1b::lora-single_gpu-adamw-standard-seq256-mb1::Unsupported model_type `falcon`. Missing explicit dense architecture manifest for this family.`
- `tiiuae/falcon-rw-1b::lora-single_gpu-adamw-standard-seq1024-mb1::Unsupported model_type `falcon`. Missing explicit dense architecture manifest for this family.`
- `tiiuae/falcon-rw-1b::lora-single_gpu-adamw-standard-seq2048-mb1::Unsupported model_type `falcon`. Missing explicit dense architecture manifest for this family.`
