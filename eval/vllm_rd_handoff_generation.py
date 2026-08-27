#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if "--no-local_files_only" not in sys.argv:
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _load_local_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


pt_utils = _load_local_module("pt_latent_safe_prefix_utils", REPO_ROOT / "scripts" / "pt-latent-safe-prefix-utils.py")

THINK_END_TEXT = pt_utils.THINK_END_TEXT
XML_THINK_PREFIX_TEXT = pt_utils.XML_THINK_PREFIX_TEXT
QWEN_THINK_PREFIX_TEXT = pt_utils.QWEN_THINK_PREFIX_TEXT
read_jsonl = pt_utils.read_jsonl
write_json = pt_utils.write_json
write_jsonl = pt_utils.write_jsonl


def parse_handoff_slots(raw: str) -> list[str]:
    allowed = {"intent", "risk", "decision"}
    aliases = {
        "i": "intent",
        "r": "risk",
        "d": "decision",
    }
    slots: list[str] = []
    for item in str(raw).split(","):
        key = item.strip().lower()
        if not key:
            continue
        slot = aliases.get(key, key)
        if slot not in allowed:
            raise ValueError(f"Unsupported handoff slot {item!r}; expected one of intent,risk,decision.")
        if slot not in slots:
            slots.append(slot)
    if not slots:
        raise ValueError("--handoff_slots cannot be empty.")
    return slots


def parse_separator(raw: str) -> str:
    if raw == r"\n":
        return "\n"
    if raw == r"\t":
        return "\t"
    return str(raw)


def normalize_prefix(row: dict[str, Any], *, slots: list[str], separator: str) -> str:
    parts = []
    key_by_slot = {"intent": "i", "risk": "r", "decision": "d"}
    for slot in slots:
        text = str(row.get(key_by_slot[slot], "") or "").strip()
        if text:
            parts.append(text)
    return str(separator).join(parts).strip()


def crop_at_first_marker(text: str, markers: list[str]) -> str:
    best = None
    for marker in markers:
        if not marker:
            continue
        idx = str(text).find(marker)
        if idx >= 0 and (best is None or idx < best):
            best = idx
    return str(text) if best is None else str(text)[:best]


def strip_control_tokens(text: str, markers: list[str]) -> str:
    cleaned = str(text)
    for marker in markers:
        if marker:
            cleaned = cleaned.replace(marker, "")
    return cleaned.strip()


def normalize_cot_for_prompt(text: str, markers: dict[str, list[str]]) -> str:
    cleaned = str(text)
    if THINK_END_TEXT in cleaned:
        cleaned = cleaned.split(THINK_END_TEXT, 1)[0]
    cleaned = strip_control_tokens(cleaned, [*markers["control"], XML_THINK_PREFIX_TEXT, QWEN_THINK_PREFIX_TEXT, THINK_END_TEXT])
    return cleaned + THINK_END_TEXT if cleaned else THINK_END_TEXT


def format_saved_cot(cot_for_prompt: str, *, prompt_format: str, decode_profile: str) -> str:
    prompt_format = pt_utils.resolve_prompt_format(None, prompt_format=prompt_format)
    decode_profile = pt_utils.resolve_decode_profile(decode_profile)
    if decode_profile == pt_utils.AUTO_DECODE_PROFILE:
        decode_profile = pt_utils.default_decode_profile_for_prompt_format(prompt_format)
    think_prefix = XML_THINK_PREFIX_TEXT if decode_profile == pt_utils.XML_COT_DECODE_PROFILE else QWEN_THINK_PREFIX_TEXT
    body = str(cot_for_prompt).strip()
    if body.startswith(think_prefix.strip()):
        return body
    if THINK_END_TEXT in body:
        body = body.split(THINK_END_TEXT, 1)[0].strip()
    return f"{think_prefix}{body}\n{THINK_END_TEXT}" if body else f"{think_prefix}{THINK_END_TEXT}"


def normalize_answer_text(text: str, markers: dict[str, list[str]]) -> str:
    cleaned = str(text)
    if THINK_END_TEXT in cleaned:
        cleaned = cleaned.split(THINK_END_TEXT)[-1]
    return strip_control_tokens(cleaned, [*markers["control"], THINK_END_TEXT])


def prompt_markers(prompt_format: str, decode_profile: str) -> dict[str, list[str]]:
    prompt_format = str(prompt_format).strip().lower()
    decode_profile = pt_utils.resolve_decode_profile(decode_profile)
    if decode_profile == pt_utils.AUTO_DECODE_PROFILE:
        decode_profile = pt_utils.default_decode_profile_for_prompt_format(prompt_format)

    common_control = [
        "<|im_start|>answer",
        "<|im_start|>think",
        "<|im_start|>",
        "<|im_end|>",
        "<｜User｜>",
        "<｜Assistant｜>",
        "<｜begin▁of▁sentence｜>",
        "<｜end▁of▁sentence｜>",
        "</answer>",
        "</ans>",
    ]
    llama_control = [
        "<|begin_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
    ]
    control = [*common_control]
    if prompt_format == pt_utils.LLAMA3_PROMPT_FORMAT:
        control.extend(llama_control)
    return {
        "think_stop": ["\n"],
        "control": control,
    }


def build_sampling_params(*, tokenizer, max_tokens: int, temperature: float, top_p: float, top_k: int, stop: list[str]):
    from vllm import SamplingParams

    kwargs: dict[str, Any] = {
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "top_p": float(top_p),
        "stop": [marker for marker in stop if marker],
    }
    if int(top_k) > 0:
        kwargs["top_k"] = int(top_k)
    return SamplingParams(**kwargs)


def resolve_model_arg(model_key: str, field: str) -> str:
    mapping = {
        "llama-r1-distill": {
            "model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
            "config_model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        },
        "qwen-r1-distill": {
            "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "config_model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        },
        "s1.1": {
            "model_name": "simplescaling/s1.1-7B",
            "config_model_name": "simplescaling/s1.1-7B",
        },
    }
    if model_key not in mapping:
        return model_key
    return mapping[model_key][field]


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM stage2 handoff generation from probe-PT ird.jsonl.")
    parser.add_argument("--ird_file", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--config_model_name", type=str, default="")
    parser.add_argument("--model_key", type=str, default="")
    parser.add_argument("--max_think_tokens", type=int, default=1000)
    parser.add_argument("--max_answer_tokens", type=int, default=200)
    parser.add_argument(
        "--handoff_slots",
        type=str,
        default="risk,decision",
        help="Comma-separated slots from ird.jsonl to prepend before handoff generation. Aliases: i,r,d.",
    )
    parser.add_argument(
        "--handoff_separator",
        type=str,
        default=r"\n",
        help=r"Separator used between selected handoff slots. Use '\n' for newline.",
    )
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--max_num_seqs", type=int, default=32)
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prompt_format", default="auto")
    parser.add_argument("--decode_profile", default="auto")
    args = parser.parse_args()

    model_name = str(args.model_name)
    config_model_name = str(args.config_model_name or args.model_name)
    if args.model_key:
        model_name = resolve_model_arg(args.model_key, "model_name")
        config_model_name = resolve_model_arg(args.model_key, "config_model_name")

    from transformers import AutoTokenizer, set_seed
    from vllm import LLM

    set_seed(int(args.seed))
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=bool(args.trust_remote_code),
        local_files_only=bool(args.local_files_only),
    )
    model_cfg = pt_utils.load_model_config(config_model_name) or {}
    system_prompt = str(model_cfg.get("system_prompt", ""))
    prompt_format = pt_utils.resolve_prompt_format(
        tokenizer,
        prompt_format=(model_cfg.get("prompt_format", "auto") if str(args.prompt_format) == "auto" else args.prompt_format),
    )
    decode_profile = pt_utils.resolve_decode_profile(
        model_cfg.get("decode_profile", "auto") if str(args.decode_profile) == "auto" else args.decode_profile
    )
    if decode_profile == pt_utils.AUTO_DECODE_PROFILE:
        decode_profile = pt_utils.default_decode_profile_for_prompt_format(prompt_format)
    markers = prompt_markers(prompt_format, decode_profile)

    rows = read_jsonl(args.ird_file)
    slots = parse_handoff_slots(args.handoff_slots)
    handoff_separator = parse_separator(args.handoff_separator)
    prompts = []
    prefixes = []
    for row in rows:
        raw_prompt = str(row.get("prompt", row.get("raw_prompt", "")) or "")
        prefix = normalize_prefix(row, slots=slots, separator=handoff_separator)
        spec = pt_utils.build_prompt_spec(
            tokenizer=tokenizer,
            raw_prompt=raw_prompt,
            system_prompt=system_prompt,
            prompt_format=prompt_format,
            decode_profile=decode_profile,
        )
        prompt_text = str(spec["prompt_text"]) + prefix
        if not prompt_text.endswith((" ", "\n", "\t")):
            prompt_text += " "
        prompts.append(prompt_text)
        prefixes.append(prefix)

    llm = LLM(
        model=model_name,
        tokenizer=model_name,
        dtype=str(args.dtype),
        tensor_parallel_size=int(args.tensor_parallel_size),
        seed=int(args.seed),
        trust_remote_code=bool(args.trust_remote_code),
        gpu_memory_utilization=float(args.gpu_memory_utilization),
        max_model_len=int(args.max_model_len),
        max_num_seqs=int(args.max_num_seqs),
    )

    think_params = build_sampling_params(
        tokenizer=tokenizer,
        max_tokens=int(args.max_think_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        stop=markers["think_stop"],
    )
    answer_params = build_sampling_params(
        tokenizer=tokenizer,
        max_tokens=int(args.max_answer_tokens),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=int(args.top_k),
        stop=[],
    )

    think_outputs = llm.generate(prompts, think_params, use_tqdm=True)
    result_rows = []
    answer_prompts = []
    for row, prefix, out in zip(rows, prefixes, think_outputs):
        think_text = out.outputs[0].text
        think_text = crop_at_first_marker(think_text, markers["think_stop"])
        generated_think = strip_control_tokens(think_text, markers["control"])
        cot_body = " ".join(part for part in [prefix, generated_think] if str(part).strip()).strip()
        cot = normalize_cot_for_prompt(cot_body, markers)
        raw_prompt = str(row.get("prompt", row.get("raw_prompt", "")) or "")
        answer_prompt = pt_utils.build_answer_prompt_text(
            tokenizer=tokenizer,
            raw_prompt=raw_prompt,
            cot=cot,
            system_prompt=system_prompt,
            prompt_format=prompt_format,
            decode_profile=decode_profile,
        )
        answer_prompts.append(answer_prompt)
        result_rows.append(
            {
                "raw_prompt": raw_prompt,
                "dataset": str(row.get("dataset", "")),
                "label": str(row.get("label", "")),
                "i": str(row.get("i", "")),
                "r": str(row.get("r", "")),
                "d": str(row.get("d", "")),
                "handoff_slots": list(slots),
                "handoff_separator": args.handoff_separator,
                "ird_handoff_prefix": prefix,
                "prefixed_cot": format_saved_cot(cot, prompt_format=prompt_format, decode_profile=decode_profile),
            }
        )

    answer_outputs = llm.generate(answer_prompts, answer_params, use_tqdm=True)
    for row, out in zip(result_rows, answer_outputs):
        answer_text = out.outputs[0].text
        row["prefixed_answer"] = normalize_answer_text(answer_text, markers)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "results.jsonl", result_rows)
    write_json(
        output_dir / "summary.json",
        {
            "num_examples": len(result_rows),
            "ird_file": str(args.ird_file),
            "model_name": model_name,
            "config_model_name": config_model_name,
            "prompt_format": prompt_format,
            "decode_profile": decode_profile,
            "rd_handoff_slots": slots,
            "rd_handoff_separator": args.handoff_separator,
            "max_think_tokens": int(args.max_think_tokens),
            "max_answer_tokens": int(args.max_answer_tokens),
            "temperature": float(args.temperature),
            "top_p": float(args.top_p),
            "top_k": int(args.top_k),
            "backend": "vllm",
        },
    )
    print(json.dumps({"output_dir": str(output_dir), "num_examples": len(result_rows)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
