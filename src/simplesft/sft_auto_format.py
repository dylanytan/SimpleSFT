"""Infer how to feed a Hugging Face dataset into TRL `SFTTrainer` (plain text, native TRL layouts, or auto-join)."""

from __future__ import annotations

import re
from typing import Callable, Optional, Tuple

from trl.data_utils import is_conversational

ResolvedFormat = Tuple[Optional[Callable[[dict], str]], str, str]
# (formatting_func | None, dataset_text_field for SFTConfig, description for logging)


def _formatting_func_from_template(template: str) -> Callable[[dict], str]:
    keys = re.findall(r"\{(\w+)\}", template)

    def fmt(example: dict) -> str:
        kwargs = {k: ("" if example.get(k) is None else str(example[k])) for k in keys}
        return template.format(**kwargs)

    return fmt


def _lower_map(columns: list[str]) -> dict[str, str]:
    return {c.lower(): c for c in columns}


def _first_example(dataset) -> Optional[dict]:
    try:
        if hasattr(dataset, "__len__") and len(dataset) == 0:
            return None
        return dataset[0]
    except Exception:
        return None


def _alpaca_format(example: dict, ins_k: str, out_k: str, in_k: Optional[str]) -> str:
    ins = example.get(ins_k) or ""
    out = example.get(out_k) or ""
    inp = (example.get(in_k) or "").strip() if in_k else ""
    if inp:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{ins}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
        )
    return (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{ins}\n\n### Response:\n{out}"
    )


def _join_labeled_columns(example: dict, columns: list[str]) -> str:
    parts: list[str] = []
    for k in columns:
        v = example.get(k)
        if v is None:
            continue
        parts.append(f"{k}: {v}")
    return "\n\n".join(parts)


def resolve_sft_data_format(
    columns: list[str],
    dataset,
    *,
    dataset_text_field: str,
    format_template: Optional[str] = None,
) -> ResolvedFormat:
    """
    Decide TRL `formatting_func` and `dataset_text_field`.

    Priority:
    1. Explicit `format_template` → custom formatter, TRL uses synthetic `text` column.
    2. If `dataset_text_field` exists on the dataset → train on that column as plain text (no formatter).
    3. Otherwise infer: TRL-native (`prompt`/`completion`, `messages`), common instruction layouts, or join-all fallback.
    """
    cols_set = set(columns)
    lower = _lower_map(columns)

    if format_template is not None:
        return (
            _formatting_func_from_template(format_template),
            "text",
            "explicit --format-template",
        )

    if dataset_text_field in cols_set:
        return None, dataset_text_field, f"column {dataset_text_field!r}"

    first = _first_example(dataset)

    # TRL native: prompt + completion (tokenize_fn uses these keys; no synthetic `text` column needed)
    if "prompt" in cols_set and "completion" in cols_set:
        if first is not None:
            if is_conversational(first):
                return None, "text", "conversational `prompt` / `completion` (TRL native)"
            return None, "text", "plain `prompt` / `completion` (TRL native)"
        return None, "text", "`prompt` / `completion` (TRL native, no sample row to inspect)"

    # TRL native: chat messages
    if "messages" in cols_set:
        if first is not None and is_conversational(first):
            return None, "text", "`messages` conversational (TRL native)"
        if first is None:
            return None, "text", "`messages` (TRL native, no sample row to inspect)"

    # Alpaca-style
    if "instruction" in lower and "output" in lower:
        ins_k, out_k = lower["instruction"], lower["output"]
        in_k = lower.get("input")

        def alpaca_fmt(ex: dict) -> str:
            return _alpaca_format(ex, ins_k, out_k, in_k)

        return alpaca_fmt, "text", "Alpaca-style instruction/input/output"

    # Medical / reasoning-style (e.g. FreedomIntelligence/medical-o1-reasoning-SFT)
    if all(k in lower for k in ("question", "complex_cot", "response")):
        q_k, cot_k, r_k = lower["question"], lower["complex_cot"], lower["response"]

        def medical_fmt(ex: dict) -> str:
            q = ex.get(q_k)
            cot = ex.get(cot_k)
            r = ex.get(r_k)
            return "\n\n".join(
                str(p) if p is not None else "" for p in (q, cot, r)
            )

        return medical_fmt, "text", "Question / Complex_CoT / Response"

    # query + response (common naming)
    if "query" in lower and "response" in lower:
        q_k, r_k = lower["query"], lower["response"]

        def qr_fmt(ex: dict) -> str:
            return "\n\n".join(
                str(p) if p is not None else ""
                for p in (ex.get(q_k), ex.get(r_k))
            )

        return qr_fmt, "text", "query / response"

    # input + output (no instruction)
    if "input" in lower and "output" in lower and "instruction" not in lower:
        i_k, o_k = lower["input"], lower["output"]

        def io_fmt(ex: dict) -> str:
            return f"{ex.get(i_k) or ''}\n\n{ex.get(o_k) or ''}"

        return io_fmt, "text", "input / output"

    # Last resort: one string per column, labeled (stable column order)
    def fallback_fmt(ex: dict) -> str:
        return _join_labeled_columns(ex, columns)

    return (
        fallback_fmt,
        "text",
        f"fallback: join all columns ({len(columns)} fields)",
    )
