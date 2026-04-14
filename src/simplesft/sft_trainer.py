"""Run Hugging Face TRL SFTTrainer using a SimpleSFT exported strategy config."""

import torch
import warnings
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

from .results.export import load_trl_strategy_config
from .sft_auto_format import resolve_sft_data_format


def run_sft(
    config_path: str,
    model_id: Optional[str],
    dataset_id: str,
    dataset_config: str = None,
    dataset_text_field: str = "text",
    format_template: Optional[str] = None,
    max_steps: Optional[int] = None,
    output_dir: str = "./sft_output",
):
    """Run SFT given a dataset and a SimpleSFT exported strategy config."""
    print(f"Loading strategy config from '{config_path}'...")
    config_data = load_trl_strategy_config(config_path)
    resolved_model_id = model_id or config_data.get("model")
    if not resolved_model_id:
        raise ValueError(
            "Missing model. Provide --model or include a top-level 'model' field in the config."
        )

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

    print(f"Loading dataset '{dataset_id}'" + (f" (config: {dataset_config})" if dataset_config else "") + "...")
    try:
        if dataset_config:
            dataset = load_dataset(dataset_id, dataset_config)
        else:
            dataset = load_dataset(dataset_id)
        if "train" in dataset:
            dataset = dataset["train"]
    except Exception as e:
        warnings.warn(f"Failed to load standard dataset: {e}. Trying to load as local file...")
        if dataset_id.endswith('.json') or dataset_id.endswith('.jsonl'):
            dataset = load_dataset("json", data_files=dataset_id)["train"]
        elif dataset_id.endswith('.csv'):
            dataset = load_dataset("csv", data_files=dataset_id)["train"]
        else:
            raise

    columns = list(dataset.column_names)
    formatting_func, sft_text_field, format_desc = resolve_sft_data_format(
        columns,
        dataset,
        dataset_text_field=dataset_text_field,
        format_template=format_template,
    )
    print(f"SFT dataset layout: {format_desc}")

    print(f"Loading tokenizer '{resolved_model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(resolved_model_id)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading model '{resolved_model_id}'...")
    torch_dtype = torch.bfloat16 if bf16 else (torch.float16 if fp16 else torch.float32)
    
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": "auto",
    }
    
    model = AutoModelForCausalLM.from_pretrained(
        resolved_model_id,
        **model_kwargs
    )
    if gradient_checkpointing:
        model.config.use_cache = False

    # Create TRL SFTConfig
    sft_kwargs = dict(
        output_dir=output_dir,
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
    
    print(f"Training complete. Saving model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Done!")
