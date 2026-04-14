"""Static assets for the SimpleSFT local web interface."""

from __future__ import annotations

from ..models.model_catalog import model_select_options_html


APP_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SimpleSFT Estimator</title>
  <style>
    :root {
      --bg: #f6f7f3;
      --panel: rgba(255,255,255,0.88);
      --ink: #162117;
      --muted: #617063;
      --accent: #1a7f56;
      --accent-2: #c96f2d;
      --line: rgba(22,33,23,0.12);
      --good: #1f7a45;
      --bad: #a33a2e;
      --shadow: 0 20px 60px rgba(27, 39, 29, 0.10);
      --radius: 18px;
      --mono: "IBM Plex Mono", "SFMono-Regular", Consolas, monospace;
      --sans: "Avenir Next", "Segoe UI", Helvetica, Arial, sans-serif;
    }
    * { box-sizing: border-box; }
    [hidden] { display: none !important; }
    body {
      margin: 0;
      font-family: var(--sans);
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(26,127,86,0.13), transparent 32%),
        radial-gradient(circle at top right, rgba(201,111,45,0.16), transparent 28%),
        linear-gradient(180deg, #fafbf7 0%, #eef1e9 100%);
      min-height: 100vh;
    }
    .shell {
      max-width: 1440px;
      margin: 0 auto;
      padding: 28px 20px 48px;
    }
    .hero {
      display: grid;
      gap: 14px;
      margin-bottom: 24px;
    }
    .eyebrow {
      font-size: 12px;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      color: var(--accent);
      font-weight: 700;
    }
    h1 {
      margin: 0;
      font-size: clamp(32px, 5vw, 58px);
      line-height: 0.95;
      letter-spacing: -0.04em;
      max-width: 18ch;
    }
    .hero p {
      margin: 0;
      max-width: 72ch;
      color: var(--muted);
      line-height: 1.55;
      font-size: 16px;
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(320px, 420px) minmax(0, 1fr);
      gap: 20px;
      align-items: start;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      backdrop-filter: blur(14px);
    }
    .panel-inner { padding: 18px; }
    form {
      display: grid;
      gap: 14px;
    }
    .form-section {
      display: grid;
      gap: 14px;
      padding: 16px;
      border: 1px solid var(--line);
      border-radius: 16px;
      background: rgba(255,255,255,0.58);
    }
    .section-header {
      display: grid;
      gap: 4px;
    }
    .form-title {
      margin: 0;
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }
    .section-copy {
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.45;
    }
    .field-grid {
      display: grid;
      gap: 12px;
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .field { display: grid; gap: 6px; }
    .field.full { grid-column: 1 / -1; }
    .hint {
      color: var(--muted);
      font-size: 12px;
      line-height: 1.4;
    }
    label {
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-weight: 700;
      color: var(--muted);
    }
    input, select {
      width: 100%;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.92);
      color: var(--ink);
      border-radius: 12px;
      padding: 12px 13px;
      font: inherit;
    }
    select[multiple] {
      min-height: 148px;
    }
    input[type="checkbox"] {
      width: 18px;
      height: 18px;
      margin: 0;
    }
    .check-row {
      display: flex;
      align-items: center;
      gap: 10px;
      padding-top: 4px;
    }
    .check-row label {
      font-size: 14px;
      letter-spacing: 0;
      text-transform: none;
      color: var(--ink);
    }
    .actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }
    button {
      appearance: none;
      border: none;
      border-radius: 999px;
      padding: 13px 18px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
      transition: transform 140ms ease, opacity 140ms ease;
    }
    button:hover { transform: translateY(-1px); }
    button:disabled { opacity: 0.6; cursor: wait; transform: none; }
    .primary { background: var(--ink); color: #fff; }
    .secondary { background: rgba(22,33,23,0.08); color: var(--ink); }
    details {
      border: 1px solid var(--line);
      border-radius: 16px;
      background: rgba(255,255,255,0.5);
    }
    summary {
      cursor: pointer;
      list-style: none;
      padding: 14px 16px;
      font-weight: 700;
      color: var(--ink);
    }
    summary::-webkit-details-marker { display: none; }
    .details-body {
      display: grid;
      gap: 14px;
      padding: 0 16px 16px;
    }
    .results {
      display: grid;
      gap: 18px;
    }
    .status {
      display: flex;
      gap: 12px;
      align-items: center;
      justify-content: space-between;
      padding: 16px 18px;
      border-radius: var(--radius);
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.72);
    }
    .status strong {
      display: block;
      font-size: 28px;
      letter-spacing: -0.04em;
    }
    .status small {
      display: block;
      color: var(--muted);
      margin-top: 4px;
    }
    .pill {
      padding: 9px 12px;
      border-radius: 999px;
      font-weight: 700;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }
    .ok { background: rgba(31,122,69,0.14); color: var(--good); }
    .oom { background: rgba(163,58,46,0.14); color: var(--bad); }
    .cards {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 12px;
    }
    .bar-panel {
      display: grid;
      gap: 18px;
      padding: 22px;
      border-radius: calc(var(--radius) - 2px);
      overflow: hidden;
      background:
        linear-gradient(180deg, rgba(255,255,255,0.96) 0%, rgba(243,247,241,0.94) 100%),
        radial-gradient(circle at top right, rgba(93,127,232,0.10), transparent 28%),
        radial-gradient(circle at top left, rgba(26,127,86,0.10), transparent 26%);
      border: 1px solid rgba(26,127,86,0.12);
    }
    .bar-group {
      display: grid;
      gap: 8px;
    }
    .bar-subtitle {
      margin: 0;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      font-weight: 700;
    }
    .bar-wrap {
      position: relative;
      display: grid;
      gap: 10px;
    }
    .stacked-bar {
      display: flex;
      width: 100%;
      min-height: 24px;
      overflow: hidden;
      border-radius: 999px;
      background: rgba(22,33,23,0.08);
      border: 1px solid var(--line);
      box-shadow: inset 0 1px 0 rgba(255,255,255,0.55);
    }
    .stacked-bar.summary {
      min-height: 26px;
    }
    .stacked-bar.detail {
      min-height: 26px;
    }
    .stacked-segment {
      min-width: 2px;
      cursor: help;
      transition: filter 140ms ease;
    }
    .stacked-segment:hover {
      filter: brightness(1.06);
    }
    .stacked-segment.phase {
      background-image: repeating-linear-gradient(
        135deg,
        rgba(255,255,255,0.18) 0,
        rgba(255,255,255,0.18) 6px,
        rgba(0,0,0,0.03) 6px,
        rgba(0,0,0,0.03) 12px
      );
      background-blend-mode: multiply;
    }
    .stacked-segment.free {
      background: rgba(31,122,69,0.12);
      border-left: 1px solid rgba(31,122,69,0.18);
    }
    .bar-marker {
      position: absolute;
      top: -3px;
      bottom: 18px;
      width: 2px;
      border-radius: 999px;
      transform: translateX(-1px);
      pointer-events: none;
    }
    .bar-marker.gpu {
      background: rgba(22,33,23,0.48);
    }
    .bar-marker.peak {
      background: rgba(201,111,45,0.82);
    }
    .bar-marker-label {
      position: absolute;
      top: -18px;
      transform: translateX(-50%);
      padding: 2px 6px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.92);
      font-size: 10px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--muted);
      white-space: nowrap;
      pointer-events: none;
    }
    .bar-scale {
      display: flex;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 12px;
      font-size: 12px;
      color: var(--muted);
    }
    .memory-tooltip {
      position: fixed;
      z-index: 40;
      max-width: min(260px, calc(100vw - 24px));
      padding: 7px 9px;
      border-radius: 10px;
      background: rgba(22,33,23,0.94);
      color: #f4f8f4;
      font-size: 12px;
      line-height: 1.35;
      box-shadow: 0 12px 30px rgba(22,33,23,0.18);
      pointer-events: none;
    }
    .segment-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 8px 16px;
    }
    .legend-item {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      font-size: 11px;
      color: var(--ink);
    }
    .legend-swatch {
      width: 10px;
      height: 10px;
      border-radius: 999px;
      flex: 0 0 10px;
    }
    .legend-label {
      font-weight: 600;
      display: inline-flex;
      align-items: center;
    }
    .card {
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
      background: rgba(255,255,255,0.76);
    }
    .card-title {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-weight: 700;
      margin-bottom: 10px;
    }
    .card-value {
      font-size: 28px;
      font-weight: 800;
      letter-spacing: -0.04em;
    }
    .section-title {
      margin: 0 0 12px;
      font-size: 14px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }
    th, td {
      padding: 11px 12px;
      border-top: 1px solid var(--line);
      vertical-align: top;
    }
    th {
      text-align: left;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.07em;
    }
    td:last-child, th:last-child { text-align: right; }
    td[data-col="export"], th[data-col="export"] { text-align: center; }
    pre {
      margin: 0;
      padding: 14px;
      overflow: auto;
      border-radius: 16px;
      background: #15211b;
      color: #e9f0ea;
      font-size: 12px;
      line-height: 1.5;
      font-family: var(--mono);
    }
    .empty {
      padding: 24px;
      text-align: center;
      color: var(--muted);
      border: 1px dashed var(--line);
      border-radius: 16px;
      background: rgba(255,255,255,0.55);
    }
    .stack { display: grid; gap: 16px; }
    .error {
      color: var(--bad);
      font-weight: 600;
      padding: 14px 16px;
      border-radius: 14px;
      background: rgba(163,58,46,0.10);
      border: 1px solid rgba(163,58,46,0.16);
    }
    @media (max-width: 980px) {
      .layout { grid-template-columns: 1fr; }
      .field-grid { grid-template-columns: 1fr; }
      .bar-panel { padding: 18px; }
    }
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="eyebrow">SimpleSFT</div>
      <h1>Memory estimate workbench</h1>
      <p>
        Inspect a model, choose the structural training setup, and get a peak-memory estimate with
        resident-state, activation, workspace, and phase-level detail. Hook-visible activations are
        shown as diagnostics only.
      </p>
    </section>

    <div class="layout">
      <aside class="panel">
        <div class="panel-inner">
          <form id="estimate-form">
            <section class="form-section">
              <div class="section-header">
                <h2 class="form-title">Core Setup</h2>
                <p class="section-copy">These are the inputs most users actually change.</p>
              </div>
              <div class="field-grid">
                <div class="field full">
                  <label for="model_select">Model</label>
                  <select id="model_select" name="model_select">
                    __MODEL_OPTIONS__
                  </select>
                  <div class="hint">Curated public models only. Organizations are alphabetical, and newer/trending entries sort first within each group. Use <code>Custom model…</code> for other officially supported repos.</div>
                </div>
                <div class="field full" id="custom-model-field" hidden>
                  <label for="model_custom">Custom model</label>
                  <input id="model_custom" name="model_custom" placeholder="org/model-name">
                </div>
                <div class="field">
                  <label for="tuning_mode">Tuning mode</label>
                  <select id="tuning_mode" name="tuning_mode">
                    <option value="full_ft">full_ft</option>
                    <option value="lora">lora</option>
                  </select>
                </div>
                <div class="field">
                  <label for="optimizer_name">Optimizer</label>
                  <select id="optimizer_name" name="optimizer_name">
                    <option value="adamw">adamw</option>
                    <option value="adam">adam</option>
                    <option value="sgd">sgd</option>
                    <option value="rmsprop">rmsprop</option>
                    <option value="adagrad">adagrad</option>
                    <option value="adafactor">adafactor</option>
                  </select>
                </div>
                <div class="field">
                  <label for="max_seq_len">Sequence length</label>
                  <input id="max_seq_len" name="max_seq_len" type="number" min="1" value="2048">
                </div>
                <div class="field">
                  <label for="micro_batch_size_per_gpu">Micro-batch / GPU</label>
                  <input id="micro_batch_size_per_gpu" name="micro_batch_size_per_gpu" type="number" min="1" value="1">
                </div>
              </div>
            </section>

            <section class="form-section">
              <div class="section-header">
                <h2 class="form-title">Hardware</h2>
                <p class="section-copy">Describe the device count and per-GPU memory budget.</p>
              </div>
              <div class="field-grid">
                <div class="field">
                  <label for="gpus_per_node">GPUs / node</label>
                  <input id="gpus_per_node" name="gpus_per_node" type="number" min="1" value="1">
                </div>
                <div class="field">
                  <label for="num_nodes">Nodes</label>
                  <input id="num_nodes" name="num_nodes" type="number" min="1" value="1">
                </div>
                <div class="field full">
                  <label for="gpu_memory_gb">GPU memory (GiB)</label>
                  <input id="gpu_memory_gb" name="gpu_memory_gb" type="number" min="1" step="0.1" value="80">
                </div>
              </div>
            </section>

            <section class="form-section" id="lora-section" hidden>
              <div class="section-header">
                <h2 class="form-title">LoRA</h2>
                <p class="section-copy">Only shown when the tuning mode is set to <code>lora</code>.</p>
              </div>
              <div class="field-grid">
                <div class="field">
                  <label for="lora_rank">LoRA rank</label>
                  <input id="lora_rank" name="lora_rank" type="number" min="1" value="16">
                </div>
              </div>
              <details id="lora-advanced">
                <summary>Advanced LoRA options</summary>
                <div class="details-body">
                  <p class="section-copy">Most users can leave these at their defaults.</p>
                  <div class="field-grid">
                    <div class="field">
                      <label for="lora_alpha">LoRA alpha</label>
                      <input id="lora_alpha" name="lora_alpha" type="number" min="1" step="1" value="32">
                    </div>
                    <div class="field">
                      <label for="lora_dropout">LoRA dropout</label>
                      <input id="lora_dropout" name="lora_dropout" type="number" min="0" max="1" step="0.01" value="0">
                    </div>
                    <div class="field">
                      <label for="lora_bias">LoRA bias</label>
                      <select id="lora_bias" name="lora_bias">
                        <option value="none">none</option>
                        <option value="all">all</option>
                        <option value="lora_only">lora_only</option>
                      </select>
                    </div>
                    <div class="field full">
                      <label for="lora_target_modules">LoRA target modules</label>
                      <select id="lora_target_modules" name="lora_target_modules" multiple>
                        <option value="q_proj" selected>q_proj</option>
                        <option value="k_proj" selected>k_proj</option>
                        <option value="v_proj" selected>v_proj</option>
                        <option value="o_proj" selected>o_proj</option>
                        <option value="gate_proj">gate_proj</option>
                        <option value="up_proj">up_proj</option>
                        <option value="down_proj">down_proj</option>
                      </select>
                      <div class="hint">Use Ctrl/Cmd-click to select more than one module.</div>
                    </div>
                  </div>
                </div>
              </details>
            </section>

            <details id="advanced-controls">
              <summary>Advanced estimator controls</summary>
              <div class="details-body">
                <p class="section-copy">These rarely change unless you are matching a specific runtime or debugging the estimator.</p>
                <div class="field-grid">
                  <div class="field">
                    <label for="attention_backend">Attention backend</label>
                    <select id="attention_backend" name="attention_backend">
                      <option value="standard">standard</option>
                      <option value="sdpa">sdpa</option>
                      <option value="flash2" selected>flash2 (FA2)</option>
                    </select>
                  </div>
                  <div class="field">
                    <label for="tensor_parallel_degree">Tensor parallel</label>
                    <select id="tensor_parallel_degree" name="tensor_parallel_degree">
                      <option value="">auto (recommended)</option>
                      <option value="1">1</option>
                      <option value="2">2</option>
                      <option value="4">4</option>
                      <option value="8">8</option>
                    </select>
                  </div>
                  <div class="field">
                    <label for="sequence_parallel">Sequence parallel over TP group</label>
                    <select id="sequence_parallel" name="sequence_parallel">
                      <option value="">auto (recommended)</option>
                      <option value="false">off</option>
                      <option value="true">on</option>
                    </select>
                  </div>
                  <div class="field">
                    <label for="weight_dtype">Weight dtype</label>
                    <select id="weight_dtype" name="weight_dtype">
                      <option value="bf16">bf16</option>
                      <option value="fp16">fp16</option>
                      <option value="fp32">fp32</option>
                    </select>
                  </div>
                  <div class="field">
                    <label for="grad_dtype">Gradient dtype</label>
                    <select id="grad_dtype" name="grad_dtype">
                      <option value="bf16">bf16</option>
                      <option value="fp16">fp16</option>
                      <option value="fp32">fp32</option>
                    </select>
                  </div>
                  <div class="field">
                    <label for="optimizer_state_dtype">Optimizer state dtype</label>
                    <select id="optimizer_state_dtype" name="optimizer_state_dtype">
                      <option value="auto">auto</option>
                      <option value="bf16">bf16</option>
                      <option value="fp16">fp16</option>
                      <option value="fp32">fp32</option>
                    </select>
                  </div>
                  <div class="field">
                    <label for="optimizer_update_dtype">Optimizer update dtype</label>
                    <select id="optimizer_update_dtype" name="optimizer_update_dtype">
                      <option value="auto">auto</option>
                      <option value="bf16">bf16</option>
                      <option value="fp16">fp16</option>
                      <option value="fp32">fp32</option>
                    </select>
                  </div>
                  <div class="field">
                    <label for="runtime_cuda_context_gb">CUDA context support (GiB)</label>
                    <input id="runtime_cuda_context_gb" name="runtime_cuda_context_gb" type="number" min="0" step="0.05" value="0.25">
                  </div>
                  <div class="field">
                    <label for="runtime_allocator_pool_gb">Allocator pool support (GiB)</label>
                    <input id="runtime_allocator_pool_gb" name="runtime_allocator_pool_gb" type="number" min="0" step="0.05" value="0.05">
                  </div>
                  <div class="field">
                    <label for="runtime_nccl_gb">NCCL support (GiB)</label>
                    <input id="runtime_nccl_gb" name="runtime_nccl_gb" type="number" min="0" step="0.05" value="0.0">
                  </div>
                  <div class="field">
                    <label for="runtime_deepspeed_gb">DeepSpeed support (GiB)</label>
                    <input id="runtime_deepspeed_gb" name="runtime_deepspeed_gb" type="number" min="0" step="0.05" value="0.0">
                  </div>
                  <div class="field">
                    <label for="runtime_support_gb_override">Runtime support override (GiB)</label>
                    <input id="runtime_support_gb_override" name="runtime_support_gb_override" type="number" min="0" step="0.1" placeholder="optional">
                  </div>
                  <div class="field">
                    <label for="ddp_bucket_elements">DDP bucket elements</label>
                    <input id="ddp_bucket_elements" name="ddp_bucket_elements" type="number" min="1" step="1" value="268435456">
                  </div>
                  <div class="field">
                    <label for="zero_bucket_elements">ZeRO bucket elements</label>
                    <input id="zero_bucket_elements" name="zero_bucket_elements" type="number" min="1" step="1" value="500000000">
                  </div>
                  <div class="field">
                    <label for="zero_prefetch_elements">ZeRO prefetch elements</label>
                    <input id="zero_prefetch_elements" name="zero_prefetch_elements" type="number" min="1" step="1" value="50000000">
                  </div>
                  <div class="field full">
                    <div class="check-row">
                      <input id="vocab_parallel_logits" name="vocab_parallel_logits" type="checkbox" checked>
                      <label for="vocab_parallel_logits">Shard embeddings / LM head with tensor parallel</label>
                    </div>
                  </div>
                </div>
              </div>
            </details>
            <div class="actions">
              <button class="primary" id="submit-btn" type="submit">Estimate</button>
              <button class="secondary" id="sample-btn" type="button">Load sample</button>
            </div>
          </form>
        </div>
      </aside>

      <main class="results">
        <div id="error-box"></div>
        <div id="result-root" class="empty">
          Submit a model and structural config to get a peak estimate and debug breakdown.
        </div>
      </main>
    </div>
  </div>

  <script>
    const form = document.getElementById("estimate-form");
    const resultRoot = document.getElementById("result-root");
    const errorBox = document.getElementById("error-box");
    const submitBtn = document.getElementById("submit-btn");
    const sampleBtn = document.getElementById("sample-btn");
    const loraSection = document.getElementById("lora-section");
    const tuningModeInput = document.getElementById("tuning_mode");
    const modelSelect = document.getElementById("model_select");

    let currentTrlConfigs = null;
    const customModelField = document.getElementById("custom-model-field");
    const customModelInput = document.getElementById("model_custom");
    const memoryTooltip = document.createElement("div");
    memoryTooltip.className = "memory-tooltip";
    memoryTooltip.hidden = true;
    document.body.appendChild(memoryTooltip);

    const gib = (bytes) => (bytes / (1024 ** 3)).toFixed(2);
    const breakdownColors = {
      parameters: "#1a7f56",
      gradients: "#2f9f79",
      optimizer: "#c96f2d",
      backend_buffers: "#8b5a2b",
      runtime_support: "#48555a",
      activations: "#5d7fe8",
      peak_overhead: "#59697a",
      transient: "#8e57c9",
      phase_other: "#a164cf",
      fixed_summary: "#1f6e4d",
      phase_summary: "#6a63e7",
      free: "#cfe7d5"
    };

    function escapeAttribute(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll('"', "&quot;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }

    function syncFormSections() {
      const isLora = tuningModeInput.value === "lora";
      loraSection.hidden = !isLora;
    }

    function syncModelField() {
      const usesCustomModel = modelSelect.value === "__custom__";
      customModelField.hidden = !usesCustomModel;
    }

    function selectedValues(selectElement) {
      return Array.from(selectElement.selectedOptions).map((option) => option.value);
    }

    function collectPayload() {
      const tuningMode = form.tuning_mode.value;
      const modelValue = modelSelect.value === "__custom__"
        ? customModelInput.value.trim()
        : modelSelect.value;
      const payload = {
        model: modelValue,
        tuning_mode: tuningMode,
        optimizer_name: form.optimizer_name.value,
        attention_backend: form.attention_backend.value,
        max_seq_len: Number(form.max_seq_len.value),
        micro_batch_size_per_gpu: Number(form.micro_batch_size_per_gpu.value),
        gpus_per_node: Number(form.gpus_per_node.value),
        num_nodes: Number(form.num_nodes.value),
        gpu_memory_gb: Number(form.gpu_memory_gb.value),
        weight_dtype: form.weight_dtype.value,
        grad_dtype: form.grad_dtype.value,
        optimizer_state_dtype: form.optimizer_state_dtype.value,
        optimizer_update_dtype: form.optimizer_update_dtype.value,
        runtime_cuda_context_gb: Number(form.runtime_cuda_context_gb.value),
        runtime_allocator_pool_gb: Number(form.runtime_allocator_pool_gb.value),
        runtime_nccl_gb: Number(form.runtime_nccl_gb.value),
        runtime_deepspeed_gb: Number(form.runtime_deepspeed_gb.value),
        ddp_bucket_elements: Number(form.ddp_bucket_elements.value),
        zero_bucket_elements: Number(form.zero_bucket_elements.value),
        zero_prefetch_elements: Number(form.zero_prefetch_elements.value),
        vocab_parallel_logits: form.vocab_parallel_logits.checked
      };
      const tensorParallelDegree = form.tensor_parallel_degree.value;
      if (tensorParallelDegree !== "") {
        payload.tensor_parallel_degree = Number(tensorParallelDegree);
      }
      const sequenceParallel = form.sequence_parallel.value;
      if (sequenceParallel !== "") {
        payload.sequence_parallel = sequenceParallel === "true";
      }
      const runtimeSupportOverride = form.runtime_support_gb_override.value.trim();
      if (runtimeSupportOverride !== "") {
        payload.runtime_support_gb_override = Number(runtimeSupportOverride);
      }
      if (tuningMode === "lora") {
        payload.lora = {
          rank: Number(form.lora_rank.value),
          alpha: Number(form.lora_alpha.value),
          dropout: Number(form.lora_dropout.value),
          target_modules: selectedValues(form.lora_target_modules),
          bias: form.lora_bias.value
        };
      }
      return payload;
    }

    function renderTable(title, rows) {
      const [headerRow, ...bodyRows] = rows;
      const headerCells = headerRow.map((value) =>
        `<th${value === "Export" ? ' style="text-align:center;"' : ""}>${value}</th>`
      ).join("");
      const header = `<thead><tr>${headerCells}</tr></thead>`;
      const body = bodyRows.map((row) => {
        const cells = row.map((value, colIndex) => {
          const isExport = colIndex === row.length - 1 && headerRow[colIndex] === "Export";
          return `<td${isExport ? ' style="text-align:center;"' : ""}>${value}</td>`;
        }).join("");
        return `<tr>${cells}</tr>`;
      }).join("");
      return `
        <section class="panel">
          <div class="panel-inner">
            <h2 class="section-title">${title}</h2>
            <table>
              ${header}
              <tbody>${body}</tbody>
            </table>
          </div>
        </section>
      `;
    }

    function peakPhaseKernelOverheadBytes(estimate) {
      const phase = estimate.peak_phase;
      if (phase === "forward") {
        return estimate.debug.workspace.attention_forward_workspace_bytes;
      }
      if (phase === "backward") {
        return estimate.debug.workspace.backward_kernel_workspace_bytes;
      }
      return 0;
    }

    function peakPhaseCommOverheadBytes(estimate) {
      const workspace = estimate.debug.workspace;
      const phase = estimate.peak_phase;
      const tpCommBytes = (
        workspace.tensor_parallel_comm_window_bytes
        + workspace.sequence_parallel_comm_window_bytes
      );
      if (phase === "forward") {
        return tpCommBytes;
      }
      if (phase === "backward") {
        return workspace.ddp_reducer_bucket_bytes + tpCommBytes;
      }
      if (phase !== "optimizer_step") {
        return 0;
      }
      const resident = estimate.debug.resident_state;
      const zeroFetchExtra = Math.max(0, workspace.zero_fetch_window_bytes - resident.parameter_bytes);
      const zeroCommExtra = Math.max(0, workspace.zero_comm_window_bytes - resident.gradient_bytes);
      return Math.max(workspace.ddp_reducer_bucket_bytes, zeroFetchExtra, zeroCommExtra);
    }

    function buildDisplayBreakdown(estimate) {
      const debug = estimate.debug;
      const gpuBytes = estimate.config.gpu_memory_gb * (1024 ** 3);
      const peakBytes = estimate.global_peak_bytes;
      const recomputeBytes = debug.workspace.recompute_workspace_bytes;
      const activationBytes = estimate.breakdown.activation_bytes + recomputeBytes;
      const kernelOverheadBytes = peakPhaseKernelOverheadBytes(estimate);
      const commOverheadBytes = peakPhaseCommOverheadBytes(estimate);
      const fixedOverheadBytes = debug.resident_state.runtime_support_bytes;
      const backendBufferBytes = debug.resident_state.persistent_backend_buffer_bytes;
      const phaseOverheadBytes = kernelOverheadBytes + commOverheadBytes;
      const optimizerBytes = (
        estimate.breakdown.optimizer_state_bytes
        + debug.resident_state.master_weight_bytes
      );
      const transientBytes = Math.max(
        0,
        estimate.breakdown.transient_bytes
        - recomputeBytes
        - kernelOverheadBytes
        - commOverheadBytes
      );
      const fixedItems = [
        {key: "parameters", label: "Parameters", bytes: estimate.breakdown.parameter_bytes, scope: "fixed"},
        {key: "gradients", label: "Gradients", bytes: estimate.breakdown.gradient_bytes, scope: "fixed"},
        {key: "optimizer", label: "Optimizer state", bytes: optimizerBytes, scope: "fixed"},
        {key: "backend_buffers", label: "Backend buffers", bytes: backendBufferBytes, scope: "fixed"},
        {key: "runtime_support", label: "Runtime support", bytes: fixedOverheadBytes, scope: "fixed"},
      ];
      const fixedBytes = fixedItems.reduce((sum, item) => sum + item.bytes, 0);
      const phaseBytes = Math.max(0, peakBytes - fixedBytes);
      const phaseItems = [
        {key: "activations", label: "Activations", bytes: activationBytes, scope: "phase"},
        {key: "peak_overhead", label: "Peak overhead", bytes: phaseOverheadBytes, scope: "phase"},
        {key: "transient", label: "Scratch", bytes: transientBytes, scope: "phase"},
      ];
      const phaseGapBytes = Math.max(
        0,
        phaseBytes - phaseItems.reduce((sum, item) => sum + item.bytes, 0),
      );
      if (phaseGapBytes > 0) {
        phaseItems.push({
          key: "phase_other",
          label: "Other peak bytes",
          bytes: phaseGapBytes,
          scope: "phase",
        });
      }
      const freeBytes = Math.max(0, gpuBytes - peakBytes);
      const totalScaleBytes = Math.max(gpuBytes, peakBytes);
      const summaryItems = [
        {key: "fixed_summary", label: "Fixed", bytes: fixedBytes, scope: "fixed"},
        {key: "phase_summary", label: "Peak", bytes: phaseBytes, scope: "phase"},
        {key: "free", label: "Free", bytes: freeBytes, scope: "free"},
      ];
      return {
        summaryItems,
        items: [
          ...fixedItems,
          ...phaseItems,
          {key: "free", label: "Free", bytes: freeBytes, scope: "free"},
        ],
        fixedBytes,
        phaseBytes,
        freeBytes,
        gpuBytes,
        peakBytes,
        totalScaleBytes,
        kernelOverheadBytes,
        commOverheadBytes,
        fixedOverheadBytes,
        backendBufferBytes,
      };
    }

    function renderBarScale(displayBreakdown) {
      return `
        <div class="bar-scale">
          <span>GPU memory: ${gib(displayBreakdown.gpuBytes)} GiB</span>
          <span>Peak: ${gib(displayBreakdown.peakBytes)} GiB</span>
        </div>
      `;
    }

    function displayPhaseName(phaseName) {
      if (phaseName === "optimizer_step") {
        return "optim";
      }
      return phaseName;
    }

    function tooltipSegment(target) {
      if (!(target instanceof Element)) {
        return null;
      }
      return target.closest(".stacked-segment");
    }

    function positionMemoryTooltip(clientX, clientY) {
      const offsetX = 14;
      const offsetY = 14;
      const tooltipWidth = memoryTooltip.offsetWidth;
      const tooltipHeight = memoryTooltip.offsetHeight;
      const maxLeft = Math.max(12, window.innerWidth - tooltipWidth - 12);
      const maxTop = Math.max(12, window.innerHeight - tooltipHeight - 12);
      const left = Math.min(maxLeft, Math.max(12, clientX + offsetX));
      const top = Math.min(maxTop, Math.max(12, clientY - tooltipHeight - offsetY));
      memoryTooltip.style.left = `${left}px`;
      memoryTooltip.style.top = `${top}px`;
    }

    function showMemoryTooltip(segment, clientX, clientY) {
      const tooltip = segment.dataset.tooltip;
      if (!tooltip) {
        memoryTooltip.hidden = true;
        return;
      }
      memoryTooltip.textContent = tooltip;
      memoryTooltip.hidden = false;
      positionMemoryTooltip(clientX, clientY);
    }

    function hideMemoryTooltip() {
      memoryTooltip.hidden = true;
    }

    function markerStyle(positionPercent) {
      const clampedPosition = Math.min(99.6, Math.max(0.4, positionPercent));
      return `left:${clampedPosition}%;`;
    }

    function markerLabelStyle(positionPercent) {
      const clampedPosition = Math.min(99.6, Math.max(0.4, positionPercent));
      if (clampedPosition <= 6) {
        return `left:${clampedPosition}%;transform:none;`;
      }
      if (clampedPosition >= 94) {
        return `left:${clampedPosition}%;transform:translateX(-100%);`;
      }
      return `left:${clampedPosition}%;transform:translateX(-50%);`;
    }

    function renderBarWrap(items, displayBreakdown, variant) {
      const totalBytes = displayBreakdown.totalScaleBytes;
      const gpuMarkerLeft = totalBytes === 0 ? 0 : (displayBreakdown.gpuBytes / totalBytes) * 100;
      const peakMarkerLeft = totalBytes === 0 ? 0 : (displayBreakdown.peakBytes / totalBytes) * 100;
      const segments = items
        .filter((item) => item.bytes > 0)
        .map((item) => {
          const width = totalBytes === 0 ? 0 : (item.bytes / totalBytes) * 100;
          const tooltip = `${item.label}: ${gib(item.bytes)} GiB`;
          return `<div class="stacked-segment ${item.scope}" tabindex="0" data-tooltip="${escapeAttribute(tooltip)}" aria-label="${escapeAttribute(tooltip)}" style="width:${width}%;background:${breakdownColors[item.key]};"></div>`;
        })
        .join("");
      return `
        <div class="bar-wrap ${variant}">
          <div class="stacked-bar ${variant}">${segments}</div>
          <div class="bar-marker gpu" style="${markerStyle(gpuMarkerLeft)}"></div>
          <div class="bar-marker-label" style="${markerLabelStyle(gpuMarkerLeft)}">GPU</div>
          <div class="bar-marker peak" style="${markerStyle(peakMarkerLeft)}"></div>
          <div class="bar-marker-label" style="${markerLabelStyle(peakMarkerLeft)}">Peak</div>
          ${renderBarScale(displayBreakdown)}
        </div>
      `;
    }

    function renderBreakdownLegend(items) {
      const legend = items
        .filter((item) => item.bytes > 0)
        .map((item) => `
          <div class="legend-item">
            <div class="legend-swatch" style="background:${breakdownColors[item.key]};"></div>
            <div class="legend-label">${item.label}</div>
          </div>
        `)
        .join("");
      return `<div class="segment-legend">${legend}</div>`;
    }

    function renderBreakdownBar(displayBreakdown) {
      const summaryBar = renderBarWrap(displayBreakdown.summaryItems, displayBreakdown, "summary");
      const summaryLegend = renderBreakdownLegend(displayBreakdown.summaryItems);
      const itemizedBar = renderBarWrap(displayBreakdown.items, displayBreakdown, "detail");
      const itemizedLegend = renderBreakdownLegend(displayBreakdown.items);
      return `
        <section class="panel">
          <div class="panel-inner bar-panel">
            <h2 class="section-title">Memory Composition</h2>
            <div class="bar-group">
              <p class="bar-subtitle">Fixed / Peak / Free</p>
              ${summaryBar}
              ${summaryLegend}
            </div>
            <div class="bar-group">
              <p class="bar-subtitle">Itemized view</p>
              ${itemizedBar}
              ${itemizedLegend}
            </div>
          </div>
        </section>
      `;
    }

    function renderResult(data) {
      const estimate = data.estimate;
      const recommendation = data.recommendation;
      const displayBreakdown = buildDisplayBreakdown(estimate);
      const model = data.model_spec;
      const top = `
        <section class="status panel">
          <div>
            <strong>${model.model_name}</strong>
            <small>${model.model_type} · ${model.num_layers} layers · hidden ${model.hidden_size} · ${model.total_params.toLocaleString()} params</small>
          </div>
          <div class="pill ${estimate.feasible ? "ok" : "oom"}">${estimate.feasible ? "Fits" : "OOM risk"}</div>
        </section>
      `;
      const cards = `
        <div class="cards">
          <div class="card"><div class="card-title">Peak</div><div class="card-value">${estimate.global_peak_gb.toFixed(2)} GiB</div></div>
          <div class="card"><div class="card-title">Headroom</div><div class="card-value">${estimate.headroom_gb.toFixed(2)} GiB</div></div>
          <div class="card"><div class="card-title">Peak phase</div><div class="card-value">${displayPhaseName(estimate.peak_phase)}</div></div>
          <div class="card"><div class="card-title">Strategy</div><div class="card-value">${recommendation.config.distributed_mode}</div></div>
          <div class="card"><div class="card-title">Data parallel</div><div class="card-value">${estimate.metadata.data_parallel_degree}x</div></div>
          <div class="card"><div class="card-title">Tensor parallel</div><div class="card-value">${recommendation.config.tensor_parallel_degree}x</div></div>
          <div class="card"><div class="card-title">Sequence parallel</div><div class="card-value">${recommendation.config.sequence_parallel ? `on (tp=${recommendation.config.tensor_parallel_degree})` : "off"}</div></div>
          <div class="card"><div class="card-title">Checkpointing</div><div class="card-value">${recommendation.config.gradient_checkpointing ? "on" : "off"}</div></div>
          <div class="card"><div class="card-title">World size</div><div class="card-value">${estimate.metadata.world_size}</div></div>
        </div>
      `;
      const candidateRows = recommendation.candidates.map((candidate, index) => [
        `#${index + 1}`,
        [
          candidate.distributed_mode,
          `dp=${candidate.data_parallel_degree}`,
          `tp=${candidate.tensor_parallel_degree}`,
          candidate.sequence_parallel ? "sp" : "no-sp",
          candidate.gradient_checkpointing ? "ckpt" : "no-ckpt"
        ].join(" · "),
        candidate.feasible ? "fits" : "oom",
        `${candidate.global_peak_gb.toFixed(2)} GiB`,
        `${candidate.headroom_gb.toFixed(2)} GiB`,
        `${candidate.estimated_slowdown_percent.toFixed(1)}%`,
        data.trl_configs && data.trl_configs[index]
          ? `<button
               class="secondary"
               data-trl-index="${index}"
               style="font-size:11px;padding:5px 10px;border-radius:999px;cursor:pointer;"
             >&#8659; TRL</button>`
          : ""
      ]);
      resultRoot.className = "stack";
      resultRoot.innerHTML = [
        top,
        renderBreakdownBar(displayBreakdown),
        cards,
        renderTable(
          "Candidate Strategies",
          [["Rank", "Configuration", "Fit", "Peak", "Headroom", "Est. slowdown", "Export"], ...candidateRows]
        )
      ].join("");
    }

    function clearError() {
      errorBox.innerHTML = "";
    }

    function showError(message) {
      errorBox.innerHTML = `<div class="error">${message}</div>`;
    }

    async function submitEstimate() {
      clearError();
      submitBtn.disabled = true;
      submitBtn.textContent = "Estimating...";
      currentTrlConfigs = null;
      try {
        const response = await fetch("/api/estimate", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(collectPayload())
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.error || "Estimate failed.");
        }
        currentTrlConfigs = payload.trl_configs || null;
        renderResult(payload);
      } catch (error) {
        showError(error.message);
      } finally {
        submitBtn.disabled = false;
        submitBtn.textContent = "Estimate";
      }
    }

    resultRoot.addEventListener("mousemove", (event) => {
      const segment = tooltipSegment(event.target);
      if (segment === null || memoryTooltip.hidden) {
        return;
      }
      positionMemoryTooltip(event.clientX, event.clientY);
    });

    resultRoot.addEventListener("click", async (event) => {
      const btn = event.target.closest("[data-trl-index]");
      if (!btn || !currentTrlConfigs) return;
      const index = parseInt(btn.dataset.trlIndex, 10);
      const cfg = currentTrlConfigs[index];
      if (!cfg) return;
      try {
        const response = await fetch("/api/trl-config-yaml", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({ config: cfg })
        });
        if (!response.ok) {
          const errPayload = await response.json().catch(() => ({}));
          throw new Error(errPayload.error || "YAML export failed.");
        }
        const yamlText = await response.text();
        const blob = new Blob([yamlText], { type: "text/yaml;charset=utf-8" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `trl_config_candidate_${index + 1}.yaml`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
      } catch (error) {
        showError(error.message);
      }
    });

    resultRoot.addEventListener("mouseover", (event) => {
      const segment = tooltipSegment(event.target);
      if (segment === null) {
        return;
      }
      showMemoryTooltip(segment, event.clientX, event.clientY);
    });

    resultRoot.addEventListener("mouseout", (event) => {
      const segment = tooltipSegment(event.target);
      const nextSegment = tooltipSegment(event.relatedTarget);
      if (segment === null || segment === nextSegment) {
        return;
      }
      hideMemoryTooltip();
    });

    resultRoot.addEventListener("focusin", (event) => {
      const segment = tooltipSegment(event.target);
      if (segment === null) {
        return;
      }
      const rect = segment.getBoundingClientRect();
      showMemoryTooltip(segment, rect.left + (rect.width / 2), rect.top);
    });

    resultRoot.addEventListener("focusout", (event) => {
      if (tooltipSegment(event.target) === null) {
        return;
      }
      hideMemoryTooltip();
    });

    window.addEventListener("scroll", () => {
      if (!memoryTooltip.hidden) {
        hideMemoryTooltip();
      }
    });

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      await submitEstimate();
    });

    sampleBtn.addEventListener("click", () => {
      modelSelect.value = "Qwen/Qwen2.5-7B-Instruct";
      form.tuning_mode.value = "full_ft";
      form.optimizer_name.value = "adamw";
      form.attention_backend.value = "flash2";
      form.max_seq_len.value = "4096";
      form.micro_batch_size_per_gpu.value = "1";
      form.gpus_per_node.value = "4";
      form.num_nodes.value = "1";
      form.gpu_memory_gb.value = "80";
      syncFormSections();
      syncModelField();
    });

    tuningModeInput.addEventListener("change", syncFormSections);
    modelSelect.addEventListener("change", syncModelField);
    syncFormSections();
    syncModelField();
    submitEstimate();
  </script>
</body>
</html>
"""


def build_app_html() -> str:
    """Return the rendered web application HTML with the catalog injected.

    Returns:
        Full HTML document for the local estimator web UI.
    """

    return APP_HTML_TEMPLATE.replace("__MODEL_OPTIONS__", model_select_options_html())


APP_HTML = build_app_html()
