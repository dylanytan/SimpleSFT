#!/usr/bin/env python3
"""Run Hugging Face TRL SFTTrainer using a SimpleSFT exported strategy config."""

import argparse
import sys
from pathlib import Path

import torch
import warnings

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

# Allow `python scripts/run_sft.py` without an editable install.
_src = Path(__file__).resolve().parents[1] / "src"
if _src.is_dir():
    sys.path.insert(0, str(_src))

from simplesft.results.export import load_trl_strategy_config
from simplesft.sft_auto_format import resolve_sft_data_format


def main():
    parser = argparse.ArgumentParser(description="Run SFT using a SimpleSFT exported strategy config")
    parser.add_argument("--config", required=True, help="Path to YAML config exported from SimpleSFT")
    parser.add_argument("--model", help="Model ID or path (otherwise read from config 'model')")
    parser.add_argument("--dataset", required=True, help="Dataset ID or path (e.g., 'timdettmers/openassistant-guanaco')")
    parser.add_argument(
        "--dataset-config",
        help="Dataset configuration/subset name (required for some datasets, e.g. 'en')",
    )
    parser.add_argument(
        "--dataset-text-field",
        default="text",
        help="Column to use as plain text (if present). Otherwise layout is inferred automatically.",
    )
    parser.add_argument(
        "--format-template",
        help=(
            "Optional format string with {Column} placeholders; overrides automatic layout. "
            "Example: '{Question}\\n\\n{Complex_CoT}\\n\\n{Response}'"
        ),
    )
    parser.add_argument("--output-dir", default="./sft_output", help="Output directory for model and checkpoints")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N optimizer steps (overrides epoch-based training; use for smoke tests)",
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Shortcut: run only 5 training steps (same as --max-steps 5)",
    )
    args = parser.parse_args()
    if args.test_run and args.max_steps is not None:
        parser.error("Use either --test-run or --max-steps, not both")
    max_steps = 5 if args.test_run else args.max_steps

    # Load config
    print(f"Loading strategy config from '{args.config}'...")
    config_data = load_trl_strategy_config(args.config)
    model_id = args.model or config_data.get("model")
    if not model_id:
        parser.error("Missing model. Provide --model or include a top-level 'model' in the config.")

    # Extract TRL SFTConfig arguments
    max_seq_length = config_data.get("max_seq_length", 1024)
    per_device_train_batch_size = config_data.get("per_device_train_batch_size", 1)
    gradient_checkpointing = config_data.get("gradient_checkpointing", False)
    optim = config_data.get("optim", "adamw_torch")
    fp16 = config_data.get("fp16", False)
    bf16 = config_data.get("bf16", False)
    
    use_peft = config_data.get("use_peft", False)
    peft_config = None
    if use_peft:
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=config_data.get("lora_r", 8),
            lora_alpha=config_data.get("lora_alpha", 16),
            lora_dropout=config_data.get("lora_dropout", 0.05),
            target_modules=config_data.get("lora_target_modules", ["q_proj", "v_proj"]),
            task_type="CAUSAL_LM",
        )

    print(f"Loading dataset '{args.dataset}'...")
    try:
        if args.dataset_config:
            dataset = load_dataset(args.dataset, args.dataset_config)
        else:
            dataset = load_dataset(args.dataset)
        if "train" in dataset:
            dataset = dataset["train"]
    except Exception as e:
        warnings.warn(f"Failed to load standard dataset: {e}. Trying to load as local file...")
        if args.dataset.endswith('.json') or args.dataset.endswith('.jsonl'):
            dataset = load_dataset("json", data_files=args.dataset)["train"]
        elif args.dataset.endswith('.csv'):
            dataset = load_dataset("csv", data_files=args.dataset)["train"]
        else:
            raise

    columns = list(dataset.column_names)
    formatting_func, sft_text_field, format_desc = resolve_sft_data_format(
        columns,
        dataset,
        dataset_text_field=args.dataset_text_field,
        format_template=args.format_template,
    )
    print(f"SFT dataset layout: {format_desc}")

    print(f"Loading tokenizer '{model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model '{model_id}'...")
    torch_dtype = torch.bfloat16 if bf16 else (torch.float16 if fp16 else torch.float32)
    
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": "auto",
    }
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **model_kwargs
    )
    if gradient_checkpointing:
        model.config.use_cache = False

    # Create TRL SFTConfig
    sft_kwargs = dict(
        output_dir=args.output_dir,
        dataset_text_field=sft_text_field,
        max_length=max_seq_length,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_checkpointing=gradient_checkpointing,
        optim=optim,
        fp16=fp16,
        bf16=bf16,
    )
    if max_steps is not None:
        sft_kwargs["max_steps"] = max_steps
        print(f"Limiting training to max_steps={max_steps}")
    sft_config = SFTConfig(**sft_kwargs)

    print("Initializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=sft_config,
        peft_config=peft_config,
        processing_class=tokenizer,
        formatting_func=formatting_func,
    )

    print("Starting training...")
    trainer.train()
    
    print(f"Training complete. Saving model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done!")

if __name__ == "__main__":
    main()
