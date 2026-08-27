from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import importlib.util
import json
import os
import random
import subprocess
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if "--no-local_files_only" not in sys.argv:
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import torch.nn.functional as F
from peft import PeftConfig, PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

def _load_local_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


pt_utils = _load_local_module("pt_latent_safe_prefix_utils", REPO_ROOT / "scripts" / "pt-latent-safe-prefix-utils.py")

THINK_END_TEXT = pt_utils.THINK_END_TEXT
build_answer_prompt_text = pt_utils.build_answer_prompt_text
build_prompt_spec = pt_utils.build_prompt_spec
DEFAULT_TRIPLET_SLOT_NAMES = pt_utils.DEFAULT_TRIPLET_SLOT_NAMES
load_model_config = pt_utils.load_model_config
parse_dtype = pt_utils.parse_dtype
read_jsonl = pt_utils.read_jsonl
resolve_decode_profile = pt_utils.resolve_decode_profile
resolve_prompt_format = pt_utils.resolve_prompt_format
resolve_model_input_device = pt_utils.resolve_model_input_device
set_seed = pt_utils.set_seed
write_json = pt_utils.write_json
write_jsonl = pt_utils.write_jsonl

TRIPLET_SLOT_NAMES = tuple(DEFAULT_TRIPLET_SLOT_NAMES)
DECISION_ALLOW_LABEL = "allow"
DECISION_REFUSE_LABEL = "refuse"

LLAMA3_CONTROL_MARKERS = [
    "<|begin_of_text|>",
    "<|start_header_id|>",
    "<|end_header_id|>",
    "<|eot_id|>",
]

def _load_tokenizer_for_adapter(args, *, base_model_name: str, adapter_dir: Path):
    tokenizer_source = adapter_dir if (adapter_dir / "tokenizer_config.json").exists() else base_model_name
    return AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=bool(args.trust_remote_code),
        local_files_only=bool(args.local_files_only),
    )


def _load_tokenizer_for_adapters(args, *, base_model_name: str, adapter_dirs: Sequence[Path]):
    adapter_dirs = [Path(path) for path in adapter_dirs]
    tokenizer = _load_tokenizer_for_adapter(
        args,
        base_model_name=base_model_name,
        adapter_dir=adapter_dirs[0],
    )
    existing = set(str(token) for token in getattr(tokenizer, "additional_special_tokens", []) or [])
    to_add: list[str] = []
    def add_missing_token(token_text: str) -> None:
        token_text = str(token_text)
        token_id = tokenizer.convert_tokens_to_ids(token_text)
        if token_id is not None and token_id >= 0 and token_id != tokenizer.unk_token_id:
            existing.add(token_text)
            return
        if token_text not in existing and token_text not in to_add:
            to_add.append(token_text)

    for adapter_dir in adapter_dirs[1:]:
        if not (adapter_dir / "tokenizer_config.json").exists():
            other = None
        else:
            other = AutoTokenizer.from_pretrained(
                adapter_dir,
                trust_remote_code=bool(args.trust_remote_code),
                local_files_only=bool(args.local_files_only),
            )
            for token in getattr(other, "additional_special_tokens", []) or []:
                add_missing_token(str(token))

        payload = _load_special_token_rows_payload(adapter_dir)
        id_to_token = payload.get("id_to_token", {}) if isinstance(payload, dict) else {}
        if isinstance(id_to_token, dict):
            for _, token_text in sorted(id_to_token.items(), key=lambda item: int(item[0])):
                add_missing_token(str(token_text))
    if to_add:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": to_add},
            replace_additional_special_tokens=False,
        )
    return tokenizer


def _load_special_token_rows_payload(adapter_dir: Path) -> dict[str, Any] | None:
    path = adapter_dir / "special_token_rows.pt"
    if not path.exists():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    token_ids = [int(x) for x in payload.get("token_ids", [])]
    if not token_ids:
        return None
    payload["token_ids"] = token_ids
    return payload


def _apply_special_token_rows(model, payload: dict[str, Any] | None) -> None:
    if not payload:
        return
    token_ids = [int(x) for x in payload.get("token_ids", [])]
    if not token_ids:
        return
    input_rows = payload.get("input_rows")
    output_rows = payload.get("output_rows")
    output_tied_to_input = bool(payload.get("output_tied_to_input", False))
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    input_vocab_size = int(input_embeddings.weight.shape[0])
    output_vocab_size = (
        int(output_embeddings.weight.shape[0])
        if output_embeddings is not None and hasattr(output_embeddings, "weight")
        else input_vocab_size
    )
    max_token_id = max(int(token_id) for token_id in token_ids)
    if int(max_token_id) >= min(input_vocab_size, output_vocab_size):
        raise ValueError(
            "special_token_rows.pt contains token ids outside the loaded tokenizer/model vocab: "
            f"max_token_id={max_token_id}, input_vocab={input_vocab_size}, output_vocab={output_vocab_size}. "
            "In triplet mode, make sure the eval tokenizer is built from the union of all slot adapter tokenizers."
        )
    row_index = torch.tensor(token_ids, dtype=torch.long, device=input_embeddings.weight.device)
    with torch.no_grad():
        if input_rows is not None:
            input_embeddings.weight.index_copy_(
                0,
                row_index,
                input_rows.to(device=input_embeddings.weight.device, dtype=input_embeddings.weight.dtype),
            )
        if output_embeddings is not None and hasattr(output_embeddings, "weight"):
            rows = input_rows if output_tied_to_input else output_rows
            if rows is not None:
                output_embeddings.weight.index_copy_(
                    0,
                    row_index.to(output_embeddings.weight.device),
                    rows.to(device=output_embeddings.weight.device, dtype=output_embeddings.weight.dtype),
                )


@dataclass(frozen=True)
class GenerationSettings:
    decoding_strategy: str = "greedy"
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    renormalize_logits: bool = False


DEFAULT_GENERATION_SETTINGS = GenerationSettings()


@dataclass(frozen=True)
class PromptFormatMarkers:
    prompt_format: str
    decode_profile: str
    think_stop: tuple[str, ...]
    control: tuple[str, ...]


def _prompt_format_markers(prompt_format: str, decode_profile: str) -> PromptFormatMarkers:
    normalized = str(prompt_format).strip().lower()
    normalized_decode = resolve_decode_profile(decode_profile)
    if normalized_decode == pt_utils.AUTO_DECODE_PROFILE:
        normalized_decode = pt_utils.default_decode_profile_for_prompt_format(normalized)
    if normalized_decode == pt_utils.XML_COT_DECODE_PROFILE:
        deepseek_control_markers = (
            "<｜User｜>",
            "<｜Assistant｜>",
            "<｜begin▁of▁sentence｜>",
            "<｜end▁of▁sentence｜>",
        )
        control_markers = (
            *(LLAMA3_CONTROL_MARKERS if normalized == pt_utils.LLAMA3_PROMPT_FORMAT else []),
            *(deepseek_control_markers if normalized == pt_utils.DEEPSEEK_R1_PROMPT_FORMAT else []),
            "<|im_start|>answer",
            "<|im_start|>think",
            "<|im_start|>",
            "<|im_end|>",
            "</answer>",
            "</ans>",
        )
        return PromptFormatMarkers(
            prompt_format=normalized,
            decode_profile=normalized_decode,
            think_stop=("\n",),
            control=tuple(control_markers),
        )
    if normalized == pt_utils.LLAMA3_PROMPT_FORMAT:
        control_markers = (*LLAMA3_CONTROL_MARKERS, "</answer>", "</ans>")
        return PromptFormatMarkers(
            prompt_format=normalized,
            decode_profile=normalized_decode,
            think_stop=("\n",),
            control=tuple(control_markers),
        )
    control_markers = (
        THINK_END_TEXT,
        "<|im_start|>answer",
        "<|im_start|>think",
        "<|im_start|>",
        "<|im_end|>",
        "</answer>",
        "</ans>",
    )
    return PromptFormatMarkers(
        prompt_format=pt_utils.QWEN_CHATML_PROMPT_FORMAT,
        decode_profile=normalized_decode,
        think_stop=("\n",),
        control=tuple(control_markers),
    )


@dataclass(frozen=True)
class PrefixStrengthSchedule:
    boundaries: tuple[int, ...]
    strengths: tuple[float, ...]


class _StopOnSubsequence(StoppingCriteria):
    def __init__(self, stop_sequences: list[list[int]]) -> None:
        super().__init__()
        self.stop_sequences = [seq for seq in stop_sequences if len(seq) > 0]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            return False
        row = input_ids[0].tolist()
        for seq in self.stop_sequences:
            if len(row) >= len(seq) and row[-len(seq) :] == seq:
                return True
        return False


def _resolve_device_map_arg(raw_value: str) -> str | None:
    value = str(raw_value).strip().lower()
    if value in {"", "none", "null"}:
        return None
    return raw_value


def _parse_prefix_horizon_arg(raw_value: str | None) -> int | None:
    if raw_value is None:
        return None
    value = str(raw_value).strip().lower()
    if value in {"", "none", "null", "full", "all"}:
        return None
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("--prefix_horizon must be >= 0, or one of: full, all, none.")
    return parsed


def _serialize_prefix_horizon(value: int | None) -> str | int:
    if value is None:
        return "full"
    return int(value)


def _parse_generation_budget_arg(raw_value: str | int | None, *, arg_name: str) -> int | None:
    if raw_value is None:
        return None
    value = str(raw_value).strip().lower()
    if value in {"", "none", "null", "full", "all", "unlimited", "inf", "max"}:
        return None
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            f"{arg_name} must be a positive integer, or one of: full, all, unlimited, max."
        )
    return parsed


def _serialize_generation_budget(value: int | None) -> str | int:
    if value is None:
        return "unlimited"
    return int(value)


def _resolve_requested_prompt_format(tokenizer, raw_prompt_format: str, model_cfg: dict[str, Any]) -> str:
    requested = str(raw_prompt_format).strip().lower()
    if requested == pt_utils.AUTO_PROMPT_FORMAT:
        configured = model_cfg.get("prompt_format", pt_utils.AUTO_PROMPT_FORMAT)
        return resolve_prompt_format(tokenizer, prompt_format=configured)
    return resolve_prompt_format(tokenizer, prompt_format=raw_prompt_format)


def _resolve_requested_decode_profile(*, model_cfg: dict[str, Any], prompt_format: str, raw_decode_profile: str) -> str:
    requested = resolve_decode_profile(raw_decode_profile)
    if requested != pt_utils.AUTO_DECODE_PROFILE:
        return requested
    configured = resolve_decode_profile(model_cfg.get("decode_profile", pt_utils.AUTO_DECODE_PROFILE))
    if configured != pt_utils.AUTO_DECODE_PROFILE:
        return configured
    return pt_utils.default_decode_profile_for_prompt_format(prompt_format)


def _resolve_generation_value(value, cfg: dict[str, Any], key: str, default):
    if value is not None and str(value) != "auto":
        return value
    return cfg.get(key, default)


def _resolve_generation_settings(args, model_cfg: dict[str, Any]) -> GenerationSettings:
    decoding_strategy = str(_resolve_generation_value(args.decoding_strategy, model_cfg, "decoding_strategy", "greedy"))
    if decoding_strategy == "auto":
        decoding_strategy = "greedy"
    if decoding_strategy != "greedy":
        raise ValueError(f"Unsupported decoding_strategy={decoding_strategy!r}")
    do_sample = _resolve_generation_value(args.do_sample, model_cfg, "do_sample", False)
    temperature = _resolve_generation_value(args.temperature, model_cfg, "temperature", 1.0)
    top_p = _resolve_generation_value(args.top_p, model_cfg, "top_p", 1.0)
    top_k = _resolve_generation_value(args.top_k, model_cfg, "top_k", 0)
    return GenerationSettings(
        decoding_strategy=decoding_strategy,
        do_sample=bool(do_sample),
        temperature=float(temperature),
        top_p=float(top_p),
        top_k=int(top_k),
        renormalize_logits=bool(args.renormalize_logits),
    )


def _generation_settings_summary(settings: GenerationSettings) -> dict[str, Any]:
    return {
        "decoding_strategy": str(settings.decoding_strategy),
        "do_sample": bool(settings.do_sample),
        "temperature": float(settings.temperature),
        "top_p": float(settings.top_p),
        "top_k": int(settings.top_k),
        "renormalize_logits": bool(settings.renormalize_logits),
    }


def _parse_positive_float_arg(raw_value: str | float, *, arg_name: str, min_value: float = 0.0) -> float:
    value = float(raw_value)
    if value <= float(min_value):
        raise argparse.ArgumentTypeError(f"{arg_name} must be > {min_value}, got {raw_value!r}")
    return float(value)


def _parse_nonnegative_int_arg(raw_value: str | int, *, arg_name: str) -> int:
    value = int(raw_value)
    if value < 0:
        raise argparse.ArgumentTypeError(f"{arg_name} must be >= 0, got {raw_value!r}")
    return int(value)


def _parse_slot_name_set(raw: str) -> list[str]:
    names = list(dict.fromkeys(chunk.strip().lower() for chunk in str(raw).split(",") if chunk.strip()))
    invalid = [name for name in names if name not in TRIPLET_SLOT_NAMES]
    if invalid:
        raise ValueError(f"Unsupported slot names in {raw!r}: {invalid}. Expected subset of {TRIPLET_SLOT_NAMES}.")
    return names


def _parse_slot_budget_ratio(raw: str, *, expected_len: int) -> list[int]:
    values = [int(chunk.strip()) for chunk in str(raw).split(",") if chunk.strip()]
    if any(value <= 0 for value in values):
        raise ValueError(f"slot budget ratios must be positive, got {values!r} from {raw!r}")
    if len(values) != int(expected_len):
        raise ValueError(f"Expected {expected_len} slot budget ratios, got {len(values)} from {raw!r}")
    return values


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    values = [int(chunk.strip()) for chunk in str(raw).split(",") if chunk.strip()]
    if any(value < 0 for value in values):
        raise ValueError(f"schedule boundaries must be >= 0, got {values!r} from {raw!r}")
    if any(values[idx] >= values[idx + 1] for idx in range(len(values) - 1)):
        raise ValueError(f"schedule boundaries must be strictly increasing, got {values!r}")
    return tuple(values)


def _parse_float_tuple(raw: str) -> tuple[float, ...]:
    values = [float(chunk.strip()) for chunk in str(raw).split(",") if chunk.strip()]
    if any(value < 0.0 for value in values):
        raise ValueError(f"schedule strengths must be >= 0, got {values!r} from {raw!r}")
    return tuple(values)


def _parse_prefix_strength_schedule(raw: str) -> PrefixStrengthSchedule | None:
    value = str(raw).strip()
    if not value or value.lower() in {"none", "null", "off"}:
        return None
    if ":" not in value:
        raise ValueError(
            f"Invalid prefix strength schedule {raw!r}. Expected BOUNDARIES:STRENGTHS, e.g. 80,120:1,0.1,0.01."
        )
    boundary_part, strength_part = value.split(":", 1)
    boundaries = _parse_int_tuple(boundary_part)
    strengths = _parse_float_tuple(strength_part)
    if len(strengths) != len(boundaries) + 1:
        raise ValueError(
            f"Schedule {raw!r} must provide exactly len(boundaries)+1 strengths; "
            f"got {len(boundaries)} boundaries and {len(strengths)} strengths."
        )
    return PrefixStrengthSchedule(boundaries=boundaries, strengths=strengths)


def _serialize_prefix_strength_schedule(schedule: PrefixStrengthSchedule | None) -> dict[str, Any] | None:
    if schedule is None:
        return None
    return {
        "boundaries": [int(x) for x in schedule.boundaries],
        "strengths": [float(x) for x in schedule.strengths],
    }


def _parse_slot_prefix_strength_schedules(raw: str, *, slot_names: Sequence[str]) -> dict[str, PrefixStrengthSchedule]:
    value = str(raw).strip()
    if not value or value.lower() in {"none", "null", "off"}:
        return {}
    schedules: dict[str, PrefixStrengthSchedule] = {}
    for spec in value.split(";"):
        spec = spec.strip()
        if not spec:
            continue
        if "=" not in spec:
            raise ValueError(
                f"Invalid slot prefix strength schedule {spec!r}. "
                "Expected slot=BOUNDARIES:STRENGTHS, e.g. intent=80,120:1,0.1,0.01."
            )
        slot_name, schedule_raw = spec.split("=", 1)
        slot_name = slot_name.strip().lower()
        if slot_name not in slot_names:
            raise ValueError(f"Unsupported slot {slot_name!r}; expected one of {tuple(slot_names)!r}.")
        schedule = _parse_prefix_strength_schedule(schedule_raw)
        if schedule is None:
            continue
        schedules[slot_name] = schedule
    return schedules


def _serialize_slot_prefix_strength_schedules(
    schedules: dict[str, PrefixStrengthSchedule],
) -> dict[str, dict[str, Any]]:
    return {
        str(slot_name): _serialize_prefix_strength_schedule(schedule)
        for slot_name, schedule in sorted(schedules.items())
        if schedule is not None
    }


def _allocate_slot_budgets(total_budget: int | None, ratios: list[int], slot_names: Sequence[str]) -> dict[str, int | None]:
    if total_budget is None:
        return {str(slot_name): None for slot_name in slot_names}
    total_budget = int(total_budget)
    if total_budget <= 0:
        raise ValueError(f"total_budget must be positive or None, got {total_budget}")
    ratio_sum = int(sum(int(x) for x in ratios))
    raw_budgets = [total_budget * int(ratio) / ratio_sum for ratio in ratios]
    budgets = [max(1, int(value)) for value in raw_budgets]
    current_sum = sum(budgets)
    order = sorted(
        range(len(ratios)),
        key=lambda idx: (raw_budgets[idx] - int(raw_budgets[idx])),
        reverse=True,
    )
    while current_sum < total_budget:
        for idx in order:
            if current_sum >= total_budget:
                break
            budgets[idx] += 1
            current_sum += 1
    while current_sum > total_budget:
        for idx in reversed(order):
            if current_sum <= total_budget:
                break
            if budgets[idx] > 1:
                budgets[idx] -= 1
                current_sum -= 1
    return {str(slot_name): int(budget) for slot_name, budget in zip(slot_names, budgets)}


def _parse_gpu_ids(raw_value: str) -> list[str]:
    return [chunk.strip() for chunk in str(raw_value).split(",") if chunk.strip()]


def _compute_contiguous_shards(total: int, num_shards: int) -> list[tuple[int, int]]:
    if num_shards <= 0:
        raise ValueError(f"num_shards must be positive, got {num_shards}")
    base = int(total // num_shards)
    remainder = int(total % num_shards)
    shards: list[tuple[int, int]] = []
    cursor = 0
    for shard_idx in range(num_shards):
        extra = 1 if remainder > 0 and shard_idx >= num_shards - remainder else 0
        next_cursor = cursor + base + extra
        shards.append((cursor, next_cursor))
        cursor = next_cursor
    return shards


def _append_boolean_flag(cmd: list[str], *, name: str, value: bool, default: bool) -> None:
    if bool(value) == bool(default):
        return
    if value:
        cmd.append(f"--{name}")
    else:
        cmd.append(f"--no-{name}")


def _write_prompt_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    write_jsonl(path, rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate IRD-prefixed outputs on local CSV/JSONL prompt files. Triplet mode uses cascade "
            "decoding with intent -> risk -> decision adapters."
        )
    )
    parser.add_argument("--intent_adapter_dir", type=str, required=True)
    parser.add_argument("--risk_adapter_dir", type=str, required=True)
    parser.add_argument("--decision_adapter_dir", type=str, required=True)
    parser.add_argument("--data_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--input_format",
        type=str,
        default="auto",
        choices=["auto", "csv", "jsonl"],
    )
    parser.add_argument(
        "--prompt_field",
        type=str,
        default="auto",
        help="Prompt column/key. Use auto to resolve raw_prompt/input_prompt/prompt.",
    )
    parser.add_argument("--config_model_name", type=str, default=None)
    parser.add_argument(
        "--prompt_format",
        type=str,
        default="auto",
        choices=["auto", "qwen_chatml", "qwen", "chatml", "llama3", "llama", "deepseek", "deepseek_r1", "ds_r1", "r1_distill"],
        help="Chat template family. auto infers from tokenizer.",
    )
    parser.add_argument(
        "--decode_profile",
        type=str,
        default="auto",
        choices=["auto", "s1", "qwen_s1", "xml_cot", "xml", "stratos", "open_r1"],
        help=(
            "Reasoning decode profile. auto uses pt-model-configs.json when present, otherwise llama3->xml_cot "
            "and qwen_chatml->s1."
        ),
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--max_prompts", type=int, default=None)
    parser.add_argument(
        "--max_allow_prompts",
        type=int,
        default=None,
        help="Optional cap after loading/filtering: keep at most this many expected allow rows.",
    )
    parser.add_argument(
        "--max_refuse_prompts",
        type=int,
        default=None,
        help="Optional cap after loading/filtering: keep at most this many expected refuse rows.",
    )
    parser.add_argument(
        "--label_sample_seed",
        type=int,
        default=42,
        help="Seed used when --max_allow_prompts/--max_refuse_prompts sample labeled rows.",
    )
    parser.add_argument(
        "--label_sample_shuffle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Randomly sample within each expected decision label before applying label caps.",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default="",
        help=(
            "Comma-separated physical GPU ids for parallel data sharding, for example 0,2,7. "
            "The script will split prompts evenly, launch one worker per GPU, and merge outputs."
        ),
    )
    parser.add_argument(
        "--case_kind_filter",
        type=str,
        default="",
        help="Optional CSV filter on case_kind, comma separated.",
    )
    parser.add_argument(
        "--manual_true_unsafe_filter",
        type=str,
        default="",
        help="Optional CSV filter on manual_true_unsafe, e.g. yes or no.",
    )
    parser.add_argument(
        "--max_think_tokens",
        type=lambda value: _parse_generation_budget_arg(value, arg_name="--max_think_tokens"),
        default=500,
    )
    parser.add_argument(
        "--max_answer_tokens",
        type=lambda value: _parse_generation_budget_arg(value, arg_name="--max_answer_tokens"),
        default=128,
    )
    parser.add_argument(
        "--prefix_horizon",
        type=_parse_prefix_horizon_arg,
        default=None,
        help=(
            "How many think tokens are adapter-controlled. Use an integer N for first-N control, or full/all "
            "for the full think phase."
        ),
    )
    parser.add_argument(
        "--ird_handoff",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "In triplet mode, first generate full Intent/Risk/Decision, then use only Risk/Decision as the "
            "plain handoff prefix for the final think+answer pass."
        ),
    )
    parser.add_argument(
        "--ird_handoff_max_think_tokens",
        type=lambda value: _parse_generation_budget_arg(value, arg_name="--ird_handoff_max_think_tokens"),
        default=1000,
    )
    parser.add_argument(
        "--ird_handoff_max_answer_tokens",
        type=lambda value: _parse_generation_budget_arg(value, arg_name="--ird_handoff_max_answer_tokens"),
        default=200,
    )
    parser.add_argument(
        "--ird_handoff_disable_adapters",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable prefix adapters for the handoff think+answer pass.",
    )
    parser.add_argument(
        "--ird_handoff_answer_kvreuse",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For --ird_handoff, generate the handoff think stage with returned past_key_values and reuse that "
            "KV state when decoding the answer stage."
        ),
    )
    parser.add_argument(
        "--decision_decode_mode",
        type=str,
        default="generate",
        choices=["generate", "forced_choice"],
        help="For triplet mode, choose whether Decision is freely generated or chosen by allow/refuse log-probs.",
    )
    parser.add_argument(
        "--route_source",
        type=str,
        default="decision",
        choices=["decision", "intent"],
        help="Triplet routing source for prefixed_decision_label. intent maps <safe_intent>/<harmful_intent> to allow/refuse.",
    )
    parser.add_argument("--intent_tag_safe", type=str, default="<safe_intent>")
    parser.add_argument("--intent_tag_harmful", type=str, default="<harmful_intent>")
    parser.add_argument(
        "--intent_decode_mode",
        type=str,
        default="auto",
        choices=["auto", "generate", "label_conditioned_body"],
        help=(
            "Triplet intent decoding. generate keeps the legacy free-generation path. "
            "label_conditioned_body first selects <safe_intent>/<harmful_intent> by forced-choice scores, "
            "then generates the intent body conditioned on the selected tag. auto enables this for adapters "
            "trained with intent_label_conditioned_body."
        ),
    )
    parser.add_argument(
        "--intent_margin",
        type=float,
        default=0.0,
        help="Forced-choice intent rule: safe iff score_safe - score_harmful > margin.",
    )
    parser.add_argument(
        "--intent_score_length_norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Length-normalize forced-choice Intent tag log-probabilities.",
    )
    parser.add_argument(
        "--decision_margin",
        type=float,
        default=0.0,
        help="Forced-choice rule: allow iff score_allow - score_refuse > margin; larger is more conservative.",
    )
    parser.add_argument(
        "--decision_score_length_norm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Length-normalize forced-choice Decision tag log-probabilities.",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", type=str, default="none")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
    )
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--worker_mode", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--disable_progress", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--progress_position", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--progress_desc", type=str, default="", help=argparse.SUPPRESS)
    parser.add_argument(
        "--ablate_slots",
        type=str,
        default="",
        help="Comma-separated triplet slots to disable during prefixed generation, e.g. intent or risk,decision.",
    )
    parser.add_argument(
        "--cascade_stage_max_tokens",
        type=lambda value: _parse_generation_budget_arg(value, arg_name="--cascade_stage_max_tokens"),
        default=64,
        help=(
            "Per-stage token budget for cascade triplet decoding. Use an integer or full/all/unlimited."
        ),
    )
    parser.add_argument(
        "--cascade_slot_budget_ratio",
        type=str,
        default="1,1,1",
        help="Comma-separated slot budget ratios for intent,risk,decision when --cascade_stage_max_tokens is finite.",
    )
    parser.add_argument(
        "--slot_prefix_strength_schedules",
        type=str,
        default="",
        help=(
            "Optional per-slot prefix strength schedules. Format: "
            "intent=80,120:1,0.1,0.01;risk=80,120:1,0.1,0.01;decision=40,60:1,0.2,0.05. "
            "Only affects triplet prefix cascade slots."
        ),
    )
    parser.add_argument(
        "--decoding_strategy",
        type=str,
        default="auto",
        choices=["auto", "greedy"],
        help="Token selection strategy used for generation.",
    )
    parser.add_argument(
        "--do_sample",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to sample tokens. Default is profile/model-config dependent.",
    )
    parser.add_argument(
        "--temperature",
        type=lambda value: _parse_positive_float_arg(value, arg_name="--temperature"),
        default=None,
        help="Sampling temperature. Default is profile/model-config dependent.",
    )
    parser.add_argument(
        "--top_p",
        type=lambda value: _parse_positive_float_arg(value, arg_name="--top_p"),
        default=None,
        help="Nucleus sampling top_p. Default is profile/model-config dependent.",
    )
    parser.add_argument(
        "--top_k",
        type=lambda value: _parse_nonnegative_int_arg(value, arg_name="--top_k"),
        default=None,
        help="Sampling top_k. Use 0 to disable. Default is profile/model-config dependent.",
    )
    parser.add_argument(
        "--renormalize_logits",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to renormalize logits after processors are applied.",
    )
    return parser


def _infer_input_format(path: Path, requested: str) -> str:
    if requested != "auto":
        return requested
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".jsonl":
        return "jsonl"
    raise ValueError(f"Unsupported input file suffix {suffix!r}; set --input_format explicitly.")


def _pick_prompt(row: dict[str, Any], prompt_field: str) -> str:
    if prompt_field != "auto":
        return str(row.get(prompt_field, "")).strip()
    for key in ["raw_prompt", "input_prompt", "prompt"]:
        value = str(row.get(key, "")).strip()
        if value:
            return value
    return ""


def _expected_decision_label_from_row(row: dict[str, Any]) -> str:
    explicit = str(row.get("expected_decision_label", "")).strip().lower()
    if explicit in {DECISION_ALLOW_LABEL, DECISION_REFUSE_LABEL}:
        return explicit

    decision_text = str(row.get("decision", "")).strip().lower()
    if decision_text.startswith(DECISION_ALLOW_LABEL):
        return DECISION_ALLOW_LABEL
    if decision_text.startswith(DECISION_REFUSE_LABEL):
        return DECISION_REFUSE_LABEL

    label = str(row.get("label", "")).strip().lower()
    if label in {"safe", "benign", "allow", "allowed", "0"}:
        return DECISION_ALLOW_LABEL
    if label in {"unsafe", "harmful", "refuse", "refused", "1"}:
        return DECISION_REFUSE_LABEL

    source_label = str(row.get("source_label", "")).strip().lower()
    if source_label == "0":
        return DECISION_ALLOW_LABEL
    if source_label == "1":
        return DECISION_REFUSE_LABEL

    manual_true_unsafe = str(row.get("manual_true_unsafe", "")).strip().lower()
    if manual_true_unsafe in {"no", "false", "0"}:
        return DECISION_ALLOW_LABEL
    if manual_true_unsafe in {"yes", "true", "1"}:
        return DECISION_REFUSE_LABEL

    return ""


def _slice_rows(rows: list[dict[str, Any]], *, start: int, max_prompts: int | None) -> list[dict[str, Any]]:
    sliced = rows[int(start) :]
    if max_prompts is not None:
        sliced = sliced[: int(max_prompts)]
    return sliced


def _sample_rows_by_expected_decision_label(
    rows: list[dict[str, Any]],
    *,
    max_allow_prompts: int | None,
    max_refuse_prompts: int | None,
    seed: int,
    shuffle: bool,
) -> list[dict[str, Any]]:
    limits = {"allow": max_allow_prompts, "refuse": max_refuse_prompts}
    if all(value is None for value in limits.values()):
        return rows
    indexed_by_label: dict[str, list[tuple[int, dict[str, Any]]]] = {"allow": [], "refuse": [], "other": []}
    for idx, row in enumerate(rows):
        label = str(row.get("expected_decision_label", "")).strip().lower()
        if label not in {"allow", "refuse"}:
            label = "other"
        indexed_by_label.setdefault(label, []).append((idx, row))

    rng = random.Random(int(seed))
    selected: list[tuple[int, dict[str, Any]]] = []
    for label in ("allow", "refuse"):
        group = list(indexed_by_label.get(label, []))
        limit = limits[label]
        if limit is not None:
            if int(limit) < 0:
                raise ValueError(f"max_{label}_prompts must be non-negative, got {limit}")
            if bool(shuffle):
                rng.shuffle(group)
            group = group[: int(limit)]
        selected.extend(group)

    selected.sort(key=lambda item: item[0])
    return [row for _, row in selected]


def _load_csv_rows(
    *,
    path: Path,
    prompt_field: str,
    case_kind_filter: str,
    manual_true_unsafe_filter: str,
    start: int,
    max_prompts: int | None,
) -> list[dict[str, Any]]:
    case_kinds = {chunk.strip() for chunk in str(case_kind_filter).split(",") if chunk.strip()}
    manual_filter = str(manual_true_unsafe_filter).strip().lower()
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as rf:
        reader = csv.DictReader(rf)
        for idx, row in enumerate(reader):
            prompt = _pick_prompt(row, prompt_field)
            if not prompt:
                continue
            case_kind = str(row.get("case_kind", "")).strip()
            if case_kinds and case_kind not in case_kinds:
                continue
            manual_true_unsafe = str(row.get("manual_true_unsafe", "")).strip().lower()
            if manual_filter and manual_true_unsafe != manual_filter:
                continue
            rows.append(
                {
                    "id": str(row.get("id", f"csv-{idx:04d}")),
                    "raw_prompt": prompt,
                    "dataset": str(row.get("dataset", "csv")),
                    "category": case_kind or str(row.get("type", row.get("category", ""))),
                    "source": str(row.get("dataset", "csv")),
                    "case_kind": case_kind,
                    "label": str(row.get("label", "")),
                    "type": str(row.get("type", "")),
                    "expected_decision_label": _expected_decision_label_from_row(row),
                    "manual_true_unsafe": str(row.get("manual_true_unsafe", "")),
                    "issue_types": str(row.get("issue_types", "")),
                    "manual_note": str(row.get("manual_note", "")),
                    "evidence_excerpt": str(row.get("evidence_excerpt", "")),
                }
            )
    return _slice_rows(rows, start=start, max_prompts=max_prompts)


def _load_jsonl_rows(
    *,
    path: Path,
    prompt_field: str,
    start: int,
    max_prompts: int | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    default_expected_decision_label = "refuse" if "strongreject" in path.name.lower() else ""
    for idx, row in enumerate(read_jsonl(path)):
        prompt = _pick_prompt(row, prompt_field)
        if not prompt:
            continue
        expected_decision_label = _expected_decision_label_from_row(row) or default_expected_decision_label
        rows.append(
            {
                "id": str(row.get("id", f"jsonl-{idx:04d}")),
                "raw_prompt": prompt,
                "dataset": str(row.get("dataset", "jsonl")),
                "category": str(row.get("category", row.get("subgroup", ""))),
                "source": str(row.get("source", "jsonl")),
                "label": str(row.get("label", "")),
                "source_label": str(row.get("source_label", "")),
                "subgroup": str(row.get("subgroup", "")),
                "decision": str(row.get("decision", "")),
                "expected_decision_label": expected_decision_label,
                "case_kind": str(row.get("case_kind", "")),
                "manual_true_unsafe": str(row.get("manual_true_unsafe", "")),
                "issue_types": str(row.get("issue_types", "")),
                "manual_note": str(row.get("manual_note", "")),
                "evidence_excerpt": str(row.get("evidence_excerpt", "")),
            }
        )
    return _slice_rows(rows, start=start, max_prompts=max_prompts)


def _crop_at_first_marker(text: str, markers: list[str]) -> str:
    cropped = str(text)
    indices = [idx for idx in (cropped.find(marker) for marker in markers) if idx >= 0]
    return cropped[: min(indices)] if indices else cropped


def _strip_control_tokens(
    text: str,
    *,
    markers: Sequence[str] | None = None,
) -> str:
    cleaned = str(text)
    control_markers = (
        list(markers)
        if markers is not None
        else list(_prompt_format_markers(pt_utils.QWEN_CHATML_PROMPT_FORMAT, pt_utils.S1_DECODE_PROFILE).control)
    )
    for marker in control_markers:
        if marker and marker in cleaned:
            cleaned = cleaned.replace(marker, "")
    return cleaned.strip()


def _normalize_cot(text: str, *, markers: PromptFormatMarkers) -> str:
    cleaned = _strip_control_tokens(text, markers=markers.control)
    if THINK_END_TEXT in cleaned:
        cleaned = cleaned.split(THINK_END_TEXT)[0].strip()
    return cleaned + THINK_END_TEXT


def _normalize_answer(text: str, *, markers: PromptFormatMarkers) -> str:
    cleaned = str(text)
    if THINK_END_TEXT in cleaned:
        cleaned = cleaned.split(THINK_END_TEXT)[-1].strip()
    return cleaned.strip()


def _generate_suffix(
    *,
    model,
    tokenizer,
    text: str,
    max_new_tokens: int | None,
    stop_markers: list[str] | None = None,
    generation_settings: GenerationSettings = DEFAULT_GENERATION_SETTINGS,
) -> str:
    input_device = resolve_model_input_device(model)
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    encoded = {k: v.to(input_device) for k, v in encoded.items()}
    effective_max_new_tokens = _resolve_effective_generation_budget(
        requested_max_new_tokens=max_new_tokens,
        prompt_len=int(encoded["input_ids"].shape[1]),
        model=model,
        tokenizer=tokenizer,
    )
    stopping_criteria = _build_stopping_criteria(tokenizer, stop_markers)

    with torch.inference_mode():
        output_ids = model.generate(
            **encoded,
            max_new_tokens=int(effective_max_new_tokens),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
            **_build_generate_kwargs(generation_settings=generation_settings),
        )
    new_ids = output_ids[0, encoded["input_ids"].shape[1] :]
    return tokenizer.decode(new_ids, skip_special_tokens=False)


def _tokenize_ids(tokenizer, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def _resolve_model_context_limit(model, tokenizer) -> int:
    candidates: list[int] = []
    config = getattr(model, "config", None)
    generation_config = getattr(model, "generation_config", None)
    for source in [tokenizer, config, generation_config]:
        if source is None:
            continue
        for attr in ["model_max_length", "max_position_embeddings", "n_positions", "max_sequence_length", "seq_length"]:
            value = getattr(source, attr, None)
            if isinstance(value, int) and 0 < int(value) < 10**9:
                candidates.append(int(value))
    if candidates:
        return max(candidates)
    return 32768


def _resolve_effective_generation_budget(
    *,
    requested_max_new_tokens: int | None,
    prompt_len: int,
    model,
    tokenizer,
    reserve_tokens: int = 1,
) -> int:
    if requested_max_new_tokens is not None:
        return max(1, int(requested_max_new_tokens))
    context_limit = _resolve_model_context_limit(model, tokenizer)
    available = max(1, int(context_limit) - int(prompt_len) - int(max(0, reserve_tokens)))
    return int(available)


def _build_stopping_criteria(tokenizer, stop_markers: list[str] | None):
    stop_markers = list(stop_markers) if stop_markers else []
    stop_sequences = [
        _tokenize_ids(tokenizer, str(marker))
        for marker in stop_markers
        if str(marker) != ""
    ]
    stop_sequences = [seq for seq in stop_sequences if len(seq) > 0]
    if not stop_sequences:
        return None
    return StoppingCriteriaList([_StopOnSubsequence(stop_sequences)])


def _build_generate_kwargs(*, generation_settings: GenerationSettings) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "do_sample": bool(generation_settings.do_sample),
        "renormalize_logits": bool(generation_settings.renormalize_logits),
    }
    if bool(generation_settings.do_sample):
        kwargs["temperature"] = float(generation_settings.temperature)
        kwargs["top_p"] = float(generation_settings.top_p)
        if int(generation_settings.top_k) > 0:
            kwargs["top_k"] = int(generation_settings.top_k)
    return kwargs


def _past_seq_length(past_key_values) -> int:
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    if isinstance(past_key_values, (tuple, list)) and past_key_values:
        key_states = past_key_values[0][0]
        return int(key_states.shape[-2])
    raise TypeError(f"Unsupported past_key_values type: {type(past_key_values)!r}")


def _generate_suffix_with_past(
    *,
    model,
    tokenizer,
    input_ids: list[int],
    max_new_tokens: int | None,
    stop_markers: list[str] | None = None,
    past_key_values=None,
    generation_settings: GenerationSettings = DEFAULT_GENERATION_SETTINGS,
) -> dict[str, Any]:
    prefix_ids = [int(x) for x in input_ids]
    if len(prefix_ids) == 0:
        raise ValueError("input_ids must be non-empty for KV-cache generation.")

    input_device = resolve_model_input_device(model)
    encoded_ids = torch.tensor([prefix_ids], dtype=torch.long, device=input_device)
    cache_start = _past_seq_length(past_key_values)
    attention_mask = torch.ones(
        (1, cache_start + int(encoded_ids.shape[1])),
        dtype=torch.long,
        device=input_device,
    )
    cache_position = torch.arange(
        cache_start,
        cache_start + int(encoded_ids.shape[1]),
        dtype=torch.long,
        device=input_device,
    )
    stopping_criteria = _build_stopping_criteria(tokenizer, stop_markers)
    generation_model = model
    if past_key_values is not None and hasattr(model, "base_model"):
        generation_model = model.base_model
    effective_max_new_tokens = _resolve_effective_generation_budget(
        requested_max_new_tokens=max_new_tokens,
        prompt_len=int(cache_start + encoded_ids.shape[1]),
        model=generation_model,
        tokenizer=tokenizer,
    )

    generate_kwargs = {
        "input_ids": encoded_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": int(effective_max_new_tokens),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "stopping_criteria": stopping_criteria,
        "return_dict_in_generate": True,
        "use_cache": True,
        **_build_generate_kwargs(generation_settings=generation_settings),
    }
    if past_key_values is not None:
        generate_kwargs["past_key_values"] = past_key_values
        generate_kwargs["cache_position"] = cache_position

    with torch.inference_mode():
        outputs = generation_model.generate(**generate_kwargs)

    sequences = outputs.sequences
    generated_ids = sequences[0, encoded_ids.shape[1] :].tolist()
    return {
        "generated_ids": generated_ids,
        "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=False),
        "past_key_values": outputs.past_key_values,
    }


def _answer_stage_suffix_text(*, markers: PromptFormatMarkers, source_has_think_end: bool) -> str:
    suffix = "\n\n" if str(markers.decode_profile) == pt_utils.XML_COT_DECODE_PROFILE else "\n<|im_start|>answer\n"
    if bool(source_has_think_end):
        return suffix
    return THINK_END_TEXT + suffix


def _extract_decision_label_from_text(text: str) -> str:
    normalized = str(text).strip().lower()
    if normalized.startswith(DECISION_ALLOW_LABEL):
        return DECISION_ALLOW_LABEL
    if normalized.startswith(DECISION_REFUSE_LABEL):
        return DECISION_REFUSE_LABEL
    return "unknown"


def _extract_intent_decision_label_from_text(
    text: str,
    *,
    safe_tag: str = "<safe_intent>",
    harmful_tag: str = "<harmful_intent>",
) -> tuple[str, str]:
    normalized = str(text).strip().lower()
    safe_normalized = str(safe_tag).strip().lower()
    harmful_normalized = str(harmful_tag).strip().lower()
    if safe_normalized and normalized.startswith(safe_normalized):
        return "safe", "allow"
    if harmful_normalized and normalized.startswith(harmful_normalized):
        return "harmful", "refuse"
    return "unknown", "unknown"


def _score_candidate_for_prompt(
    *,
    model,
    tokenizer,
    prompt_text: str,
    candidate_text: str,
    adapter_enabled: bool,
    length_norm: bool,
) -> dict[str, Any]:
    prompt_ids = _tokenize_ids(tokenizer, prompt_text)
    candidate_ids = _tokenize_ids(tokenizer, candidate_text)
    if not prompt_ids:
        raise ValueError("prompt_text must tokenize to at least one token for forced-choice scoring.")
    if not candidate_ids:
        raise ValueError("candidate_text must tokenize to at least one token for forced-choice scoring.")

    full_ids = list(prompt_ids) + list(candidate_ids)
    input_device = resolve_model_input_device(model)
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=input_device)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        with _adapter_context(model, enabled=bool(adapter_enabled)):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
            log_probs = F.log_softmax(outputs.logits, dim=-1)
            token_scores: list[torch.Tensor] = []
            prompt_len = int(len(prompt_ids))
            for candidate_idx, token_id in enumerate(candidate_ids):
                pred_pos = prompt_len + int(candidate_idx) - 1
                if pred_pos < 0:
                    raise ValueError("prompt_text must contain at least one token for candidate scoring.")
                token_scores.append(log_probs[0, pred_pos, int(token_id)])
            score = torch.stack(token_scores).sum()
            if bool(length_norm):
                score = score / float(len(candidate_ids))
    return {
        "score": float(score.detach().cpu().item()),
        "candidate_ids": list(candidate_ids),
        "full_ids": list(full_ids),
        "past_key_values": outputs.past_key_values,
    }


def _score_forced_choice_decision(
    *,
    model,
    tokenizer,
    prompt_text: str,
    adapter_enabled: bool,
    margin: float,
    length_norm: bool,
) -> dict[str, Any]:
    allow_result = _score_candidate_for_prompt(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        candidate_text=DECISION_ALLOW_LABEL,
        adapter_enabled=bool(adapter_enabled),
        length_norm=bool(length_norm),
    )
    refuse_result = _score_candidate_for_prompt(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        candidate_text=DECISION_REFUSE_LABEL,
        adapter_enabled=bool(adapter_enabled),
        length_norm=bool(length_norm),
    )
    score_allow = float(allow_result["score"])
    score_refuse = float(refuse_result["score"])
    score_delta = float(score_allow - score_refuse)
    selected_label = DECISION_ALLOW_LABEL if score_delta > float(margin) else DECISION_REFUSE_LABEL
    selected_result = allow_result if selected_label == "allow" else refuse_result
    selected_tag = DECISION_ALLOW_LABEL if selected_label == DECISION_ALLOW_LABEL else DECISION_REFUSE_LABEL
    return {
        "selected_label": selected_label,
        "selected_tag": selected_tag,
        "score_allow": score_allow,
        "score_refuse": score_refuse,
        "score_delta_allow_minus_refuse": score_delta,
        "margin": float(margin),
        "length_norm": bool(length_norm),
        "allow_token_count": int(len(allow_result["candidate_ids"])),
        "refuse_token_count": int(len(refuse_result["candidate_ids"])),
        "selected_candidate_ids": list(selected_result["candidate_ids"]),
        "full_ids": list(selected_result["full_ids"]),
        "past_key_values": selected_result["past_key_values"],
    }


def _score_forced_choice_intent(
    *,
    model,
    tokenizer,
    prompt_text: str,
    adapter_enabled: bool,
    safe_tag: str,
    harmful_tag: str,
    margin: float,
    length_norm: bool,
) -> dict[str, Any]:
    safe_result = _score_candidate_for_prompt(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        candidate_text=str(safe_tag),
        adapter_enabled=bool(adapter_enabled),
        length_norm=bool(length_norm),
    )
    harmful_result = _score_candidate_for_prompt(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        candidate_text=str(harmful_tag),
        adapter_enabled=bool(adapter_enabled),
        length_norm=bool(length_norm),
    )
    score_safe = float(safe_result["score"])
    score_harmful = float(harmful_result["score"])
    score_delta = float(score_safe - score_harmful)
    selected_label = "safe" if score_delta > float(margin) else "harmful"
    selected_result = safe_result if selected_label == "safe" else harmful_result
    selected_tag = str(safe_tag) if selected_label == "safe" else str(harmful_tag)
    return {
        "selected_label": selected_label,
        "selected_decision_label": "allow" if selected_label == "safe" else "refuse",
        "selected_tag": selected_tag,
        "score_safe": score_safe,
        "score_harmful": score_harmful,
        "score_delta_safe_minus_harmful": score_delta,
        "margin": float(margin),
        "length_norm": bool(length_norm),
        "safe_token_count": int(len(safe_result["candidate_ids"])),
        "harmful_token_count": int(len(harmful_result["candidate_ids"])),
        "selected_candidate_ids": list(selected_result["candidate_ids"]),
        "full_ids": list(selected_result["full_ids"]),
        "past_key_values": selected_result["past_key_values"],
    }


def _contains_stop_marker(text: str, markers: list[str]) -> bool:
    return any(marker in str(text) for marker in markers)


def _normalize_cascade_stage_text(
    text: str,
    *,
    stop_at_newline: bool,
    markers: PromptFormatMarkers,
) -> str:
    crop_markers = [*list(markers.think_stop)]
    if bool(stop_at_newline):
        crop_markers = ["\n", *crop_markers]
    cleaned = _strip_control_tokens(
        _crop_at_first_marker(str(text), markers=crop_markers),
        markers=markers.control,
    ).strip()
    return cleaned


def _build_cascade_prompt_text(*, base_prompt_text: str, prior_sections: list[str]) -> str:
    clean_sections = [str(section).strip() for section in prior_sections if str(section).strip()]
    if not clean_sections:
        return str(base_prompt_text)
    return str(base_prompt_text) + "\n".join(clean_sections) + "\n"


def _normalize_ird_handoff_prefix(sections: dict[str, str]) -> str:
    return "\n".join(
        text
        for text in (str(sections.get(slot_name, "")).strip() for slot_name in ("risk", "decision"))
        if text
    )


def _generate_ird_handoff_one(
    *,
    model,
    tokenizer,
    prompt: str,
    system_prompt: str,
    prompt_format: str,
    decode_profile: str,
    handoff_prefix: str,
    max_think_tokens: int | None,
    max_answer_tokens: int | None,
    generation_settings: GenerationSettings,
    disable_adapters: bool,
    enable_answer_kvreuse: bool = False,
) -> tuple[str, str, str, dict[str, Any]]:
    prefix = str(handoff_prefix).strip()
    spec = build_prompt_spec(
        tokenizer=tokenizer,
        raw_prompt=prompt,
        system_prompt=system_prompt,
        prompt_format=prompt_format,
        decode_profile=decode_profile,
    )
    format_markers = _prompt_format_markers(str(spec["prompt_format"]), str(spec["decode_profile"]))
    prompt_with_prefix = str(spec["prompt_text"]) + prefix
    if not prompt_with_prefix.endswith((" ", "\n", "\t")):
        prompt_with_prefix += " "

    handoff_debug: dict[str, Any] = {
        "answer_kvreuse_requested": bool(enable_answer_kvreuse),
        "answer_kvreuse_feasible": False,
        "answer_kvreuse_reuse_tokens": 0,
        "answer_kvreuse_suffix_tokens": 0,
        "answer_kvreuse_fallback_reason": "",
    }

    with (model.disable_adapter() if bool(disable_adapters) else nullcontext()):
        think_result = None
        think_prompt_ids = _tokenize_ids(tokenizer, prompt_with_prefix)
        if bool(enable_answer_kvreuse):
            think_result = _generate_suffix_with_past(
                model=model,
                tokenizer=tokenizer,
                input_ids=think_prompt_ids,
                max_new_tokens=max_think_tokens,
                stop_markers=["\n"],
                generation_settings=generation_settings,
            )
            think_raw = str(think_result["generated_text"])
        else:
            think_raw = _generate_suffix(
                model=model,
                tokenizer=tokenizer,
                text=prompt_with_prefix,
                max_new_tokens=max_think_tokens,
                stop_markers=["\n"],
                generation_settings=generation_settings,
            )

        think_cropped = _crop_at_first_marker(think_raw, markers=["\n"])
        generated_think = _strip_control_tokens(think_cropped, markers=format_markers.control).strip()
        cot_body = " ".join(part for part in [prefix, generated_think] if part).strip()
        cot = _normalize_cot(cot_body, markers=format_markers)

        answer_prompt = build_answer_prompt_text(
            tokenizer=tokenizer,
            raw_prompt=prompt,
            cot=cot,
            system_prompt=system_prompt,
            prompt_format=prompt_format,
            decode_profile=decode_profile,
        )
        if bool(enable_answer_kvreuse) and think_result is not None:
            think_full_ids = list(think_prompt_ids) + list(think_result["generated_ids"])
            answer_suffix_text = _answer_stage_suffix_text(
                markers=format_markers,
                source_has_think_end=bool(THINK_END_TEXT in str(think_raw)),
            )
            answer_suffix_ids = _tokenize_ids(tokenizer, answer_suffix_text)
            if answer_suffix_ids:
                handoff_debug.update(
                    {
                        "answer_kvreuse_desired_tokens": int(len(think_full_ids) + len(answer_suffix_ids)),
                        "answer_kvreuse_source_tokens": int(len(think_full_ids)),
                        "answer_kvreuse_common_prefix_tokens": int(len(think_full_ids)),
                        "answer_kvreuse_reuse_tokens": int(len(think_full_ids)),
                        "answer_kvreuse_suffix_tokens": int(len(answer_suffix_ids)),
                        "answer_kvreuse_feasible": True,
                        "answer_kvreuse_fallback_reason": "",
                    }
                )
                answer_result = _generate_suffix_with_past(
                    model=model,
                    tokenizer=tokenizer,
                    input_ids=answer_suffix_ids,
                    max_new_tokens=max_answer_tokens,
                    stop_markers=None,
                    past_key_values=think_result["past_key_values"],
                    generation_settings=generation_settings,
                )
                answer_raw = str(answer_result["generated_text"])
            else:
                handoff_debug["answer_kvreuse_fallback_reason"] = "empty_answer_suffix"
                answer_raw = _generate_suffix(
                    model=model,
                    tokenizer=tokenizer,
                    text=answer_prompt,
                    max_new_tokens=max_answer_tokens,
                    stop_markers=None,
                    generation_settings=generation_settings,
                )
        else:
            answer_raw = _generate_suffix(
                model=model,
                tokenizer=tokenizer,
                text=answer_prompt,
                max_new_tokens=max_answer_tokens,
                stop_markers=None,
                generation_settings=generation_settings,
            )

    answer = _normalize_answer(answer_raw, markers=format_markers)
    return cot, answer, prefix, handoff_debug


def _adapter_context(model, enabled: bool):
    return nullcontext() if bool(enabled) else model.disable_adapter()


@contextmanager
def _scaled_active_prefix_embedding(model, *, strength: float):
    strength = float(strength)
    if abs(strength - 1.0) < 1e-8:
        yield
        return
    active_adapter = str(model.active_adapter)
    prompt_encoder = model.prompt_encoder[active_adapter]
    embedding = prompt_encoder.embedding
    original_weight = embedding.weight.data
    scaled_weight = original_weight * strength
    embedding.weight.data = scaled_weight
    try:
        yield
    finally:
        embedding.weight.data = original_weight


def _build_schedule_segments(total_budget: int, schedule: PrefixStrengthSchedule | None) -> list[tuple[int, float]]:
    total_budget = int(total_budget)
    if total_budget <= 0:
        return []
    if schedule is None:
        return [(int(total_budget), 1.0)]
    boundaries = [min(int(total_budget), int(x)) for x in schedule.boundaries]
    strengths = [float(x) for x in schedule.strengths]
    segments: list[tuple[int, float]] = []
    prev = 0
    for boundary, strength in zip(boundaries, strengths):
        seg_len = max(0, int(boundary) - int(prev))
        if seg_len > 0:
            segments.append((seg_len, float(strength)))
        prev = int(boundary)
    final_len = max(0, int(total_budget) - int(prev))
    if final_len > 0:
        segments.append((final_len, float(strengths[-1])))
    return [(int(seg_len), float(alpha)) for seg_len, alpha in segments if int(seg_len) > 0]


def _generate_cascade_stage(
    *,
    model,
    tokenizer,
    stage_prompt_text: str,
    adapter_enabled: bool,
    prefix_horizon: int | None,
    max_new_tokens: int,
    stop_markers: list[str],
    generation_settings: GenerationSettings,
    prefix_strength_schedule: PrefixStrengthSchedule | None = None,
) -> dict[str, Any]:
    prompt_ids = _tokenize_ids(tokenizer, stage_prompt_text)
    effective_stage_budget = _resolve_effective_generation_budget(
        requested_max_new_tokens=max_new_tokens,
        prompt_len=len(prompt_ids),
        model=model,
        tokenizer=tokenizer,
    )
    effective_horizon = None
    if prefix_horizon is not None:
        effective_horizon = max(0, min(int(prefix_horizon), int(effective_stage_budget)))

    if (not bool(adapter_enabled)) or effective_horizon == 0:
        with model.disable_adapter():
            result = _generate_suffix_with_past(
                model=model,
                tokenizer=tokenizer,
                input_ids=prompt_ids,
                max_new_tokens=int(effective_stage_budget),
                stop_markers=stop_markers,
                generation_settings=generation_settings,
            )
        return {
            "raw_text": str(result["generated_text"]),
            "past_key_values": result["past_key_values"],
            "full_ids": list(prompt_ids) + list(result["generated_ids"]),
            "effective_horizon": int(effective_horizon or 0),
            "continued_without_adapter": False,
        }

    schedule_segments = _build_schedule_segments(
        total_budget=int(effective_stage_budget),
        schedule=prefix_strength_schedule if effective_horizon is None else None,
    )
    if effective_horizon is None and len(schedule_segments) > 1:
        aggregated_raw_parts: list[str] = []
        current_prompt_text = str(stage_prompt_text)
        final_past = None
        final_full_ids = _tokenize_ids(tokenizer, current_prompt_text)
        used_tokens = 0
        stop_hit = False
        for segment_budget, segment_strength in schedule_segments:
            with _adapter_context(model, enabled=True):
                with _scaled_active_prefix_embedding(model, strength=float(segment_strength)):
                    segment_result = _generate_suffix_with_past(
                        model=model,
                        tokenizer=tokenizer,
                        input_ids=_tokenize_ids(tokenizer, current_prompt_text),
                        max_new_tokens=int(segment_budget),
                        stop_markers=stop_markers,
                        generation_settings=generation_settings,
                    )
            segment_raw = str(segment_result["generated_text"])
            aggregated_raw_parts.append(segment_raw)
            segment_cropped = _crop_at_first_marker(segment_raw, markers=stop_markers)
            current_prompt_text = current_prompt_text + segment_cropped
            final_past = segment_result["past_key_values"]
            final_full_ids = _tokenize_ids(tokenizer, current_prompt_text)
            used_tokens += len(segment_result["generated_ids"])
            if _contains_stop_marker(segment_raw, stop_markers):
                stop_hit = True
                break
            if len(segment_result["generated_ids"]) < int(segment_budget):
                stop_hit = True
                break
        return {
            "raw_text": "".join(aggregated_raw_parts),
            "past_key_values": final_past,
            "full_ids": final_full_ids,
            "effective_horizon": None,
            "continued_without_adapter": False,
            "prefix_strength_schedule": _serialize_prefix_strength_schedule(prefix_strength_schedule),
            "used_generation_tokens": int(used_tokens),
            "stop_hit": bool(stop_hit),
        }

    if effective_horizon is None:
        with _adapter_context(model, enabled=True):
            result = _generate_suffix_with_past(
                model=model,
                tokenizer=tokenizer,
                input_ids=prompt_ids,
                max_new_tokens=int(effective_stage_budget),
                stop_markers=stop_markers,
                generation_settings=generation_settings,
            )
        return {
            "raw_text": str(result["generated_text"]),
            "past_key_values": result["past_key_values"],
            "full_ids": list(prompt_ids) + list(result["generated_ids"]),
            "effective_horizon": None,
            "continued_without_adapter": False,
            "prefix_strength_schedule": _serialize_prefix_strength_schedule(prefix_strength_schedule),
            "used_generation_tokens": int(len(result["generated_ids"])),
            "stop_hit": bool(_contains_stop_marker(str(result["generated_text"]), stop_markers)),
        }

    with _adapter_context(model, enabled=True):
        first_result = _generate_suffix_with_past(
            model=model,
            tokenizer=tokenizer,
            input_ids=prompt_ids,
            max_new_tokens=int(effective_horizon),
            stop_markers=stop_markers,
            generation_settings=generation_settings,
        )
    first_raw = str(first_result["generated_text"])
    first_cropped = _crop_at_first_marker(first_raw, markers=stop_markers)
    final_raw = first_cropped
    final_past = first_result["past_key_values"]
    final_full_ids = list(prompt_ids) + list(first_result["generated_ids"])
    continued_without_adapter = False

    if not _contains_stop_marker(first_raw, stop_markers) and int(effective_horizon) < int(effective_stage_budget):
        continued_without_adapter = True
        continue_prompt = stage_prompt_text + first_cropped
        continue_ids = _tokenize_ids(tokenizer, continue_prompt)
        with model.disable_adapter():
            second_result = _generate_suffix_with_past(
                model=model,
                tokenizer=tokenizer,
                input_ids=continue_ids,
                max_new_tokens=int(effective_stage_budget) - int(effective_horizon),
                stop_markers=stop_markers,
                generation_settings=generation_settings,
            )
        final_raw = first_cropped + str(second_result["generated_text"])
        final_past = second_result["past_key_values"]
        final_full_ids = list(continue_ids) + list(second_result["generated_ids"])

    return {
        "raw_text": str(final_raw),
        "past_key_values": final_past,
        "full_ids": final_full_ids,
        "effective_horizon": int(effective_horizon),
        "continued_without_adapter": bool(continued_without_adapter),
        "prefix_strength_schedule": _serialize_prefix_strength_schedule(prefix_strength_schedule),
        "used_generation_tokens": int(max(0, len(final_full_ids) - len(prompt_ids))),
        "stop_hit": bool(_contains_stop_marker(str(final_raw), stop_markers)),
    }


def _generate_triplet_cascade_one(
    *,
    model,
    tokenizer,
    prompt: str,
    system_prompt: str,
    prompt_format: str,
    decode_profile: str,
    prefix_horizon: int | None,
    decision_decode_mode: str,
    route_source: str,
    intent_tag_safe: str,
    intent_tag_harmful: str,
    intent_decode_mode: str,
    intent_margin: float,
    intent_score_length_norm: bool,
    decision_margin: float,
    decision_score_length_norm: bool,
    ablate_slots: set[str],
    cascade_stage_max_tokens: int | None,
    cascade_slot_budget_ratio: list[int],
    generation_settings: GenerationSettings,
    slot_prefix_strength_schedules: dict[str, PrefixStrengthSchedule],
    special_token_rows_by_slot: dict[str, dict[str, Any] | None] | None = None,
) -> tuple[str, str, dict[str, str], list[dict[str, Any]]]:
    spec = build_prompt_spec(
        tokenizer=tokenizer,
        raw_prompt=prompt,
        system_prompt=system_prompt,
        prompt_format=prompt_format,
        decode_profile=decode_profile,
    )
    format_markers = _prompt_format_markers(str(spec["prompt_format"]), str(spec["decode_profile"]))
    generated_sections: dict[str, str] = {}
    stage_debug: list[dict[str, Any]] = []
    prior_sections: list[str] = []
    decision_label = "unknown"
    intent_label = "unknown"
    intent_decision_label = "unknown"
    slot_budgets = _allocate_slot_budgets(
        total_budget=cascade_stage_max_tokens,
        ratios=list(cascade_slot_budget_ratio),
        slot_names=TRIPLET_SLOT_NAMES,
    )

    for slot_name in TRIPLET_SLOT_NAMES:
        stage_prompt_text = _build_cascade_prompt_text(
            base_prompt_text=str(spec["prompt_text"]),
            prior_sections=prior_sections,
        )
        stop_markers = ["\n"]
        adapter_enabled = str(slot_name) not in ablate_slots
        if adapter_enabled:
            model.set_adapter(str(slot_name))
            if special_token_rows_by_slot is not None:
                _apply_special_token_rows(model, special_token_rows_by_slot.get(str(slot_name)))
        if str(slot_name) == "intent" and str(intent_decode_mode) == "label_conditioned_body":
            intent_result = _score_forced_choice_intent(
                model=model,
                tokenizer=tokenizer,
                prompt_text=stage_prompt_text,
                adapter_enabled=bool(adapter_enabled),
                safe_tag=str(intent_tag_safe),
                harmful_tag=str(intent_tag_harmful),
                margin=float(intent_margin),
                length_norm=bool(intent_score_length_norm),
            )
            selected_tag = str(intent_result["selected_tag"])
            body_prompt_text = stage_prompt_text + selected_tag
            if not body_prompt_text.endswith((" ", "\n", "\t")):
                body_prompt_text += " "
            slot_budget = slot_budgets[str(slot_name)]
            body_budget = (
                None
                if slot_budget is None
                else max(1, int(slot_budget) - int(len(intent_result["selected_candidate_ids"])))
            )
            body_result = _generate_cascade_stage(
                model=model,
                tokenizer=tokenizer,
                stage_prompt_text=body_prompt_text,
                adapter_enabled=bool(adapter_enabled),
                prefix_horizon=prefix_horizon,
                max_new_tokens=body_budget,
                stop_markers=stop_markers,
                generation_settings=generation_settings,
                prefix_strength_schedule=slot_prefix_strength_schedules.get(str(slot_name)),
            )
            stage_result = {
                **body_result,
                "raw_text": selected_tag + " " + str(body_result["raw_text"]).lstrip(),
                "used_generation_tokens": int(len(intent_result["selected_candidate_ids"]))
                + int(body_result.get("used_generation_tokens", 0)),
                "intent_forced_choice": {
                    key: value
                    for key, value in intent_result.items()
                    if key not in {"past_key_values", "full_ids", "selected_candidate_ids"}
                },
                "intent_body_prompt_text": body_prompt_text,
            }
            intent_label = str(intent_result["selected_label"])
            intent_decision_label = str(intent_result["selected_decision_label"])
            if str(route_source) == "intent":
                decision_label = str(intent_decision_label)
        elif str(slot_name) == "decision" and str(decision_decode_mode) == "forced_choice":
            decision_result = _score_forced_choice_decision(
                model=model,
                tokenizer=tokenizer,
                prompt_text=stage_prompt_text,
                adapter_enabled=bool(adapter_enabled),
                margin=float(decision_margin),
                length_norm=bool(decision_score_length_norm),
            )
            stage_result = {
                "raw_text": str(decision_result["selected_tag"]),
                "past_key_values": decision_result["past_key_values"],
                "full_ids": list(decision_result["full_ids"]),
                "effective_horizon": None,
                "continued_without_adapter": False,
                "prefix_strength_schedule": None,
                "used_generation_tokens": int(len(decision_result["selected_candidate_ids"])),
                "stop_hit": True,
                "decision_forced_choice": {
                    key: value
                    for key, value in decision_result.items()
                    if key not in {"past_key_values", "full_ids", "selected_candidate_ids"}
                },
            }
            decision_label = str(decision_result["selected_label"])
        else:
            stage_result = _generate_cascade_stage(
                model=model,
                tokenizer=tokenizer,
                stage_prompt_text=stage_prompt_text,
                adapter_enabled=bool(adapter_enabled),
                prefix_horizon=prefix_horizon,
                max_new_tokens=slot_budgets[str(slot_name)],
                stop_markers=stop_markers,
                generation_settings=generation_settings,
                prefix_strength_schedule=slot_prefix_strength_schedules.get(str(slot_name)),
            )
        stage_raw = str(stage_result["raw_text"])
        stage_text = _normalize_cascade_stage_text(
            stage_raw,
            stop_at_newline=True,
            markers=format_markers,
        )
        generated_sections[str(slot_name)] = stage_text
        if stage_text:
            prior_sections.append(stage_text)
        if str(slot_name) == "intent":
            if str(intent_decode_mode) != "label_conditioned_body":
                intent_label, intent_decision_label = _extract_intent_decision_label_from_text(
                    stage_text,
                    safe_tag=str(intent_tag_safe),
                    harmful_tag=str(intent_tag_harmful),
                )
                if str(route_source) == "intent":
                    decision_label = str(intent_decision_label)
        if str(slot_name) == "decision" and str(decision_decode_mode) != "forced_choice":
            if str(route_source) != "intent":
                decision_label = _extract_decision_label_from_text(stage_text)
        debug_row = {
            "slot_name": str(slot_name),
            "slot_budget": _serialize_generation_budget(slot_budgets[str(slot_name)]),
            "adapter_enabled": bool(adapter_enabled),
            "prefix_horizon": (_serialize_prefix_horizon(prefix_horizon) if prefix_horizon is not None else "full"),
            "effective_horizon": stage_result["effective_horizon"],
            "continued_without_adapter": bool(stage_result["continued_without_adapter"]),
            "prefix_strength_schedule": stage_result.get("prefix_strength_schedule"),
            "used_generation_tokens": int(stage_result.get("used_generation_tokens", 0)),
            "stop_hit": bool(stage_result.get("stop_hit", False)),
            "prompt_text": stage_prompt_text,
            "raw_generation": str(stage_raw),
            "normalized_text": str(stage_text),
        }
        if "decision_forced_choice" in stage_result:
            debug_row["decision_decode_mode"] = "forced_choice"
            debug_row["decision_label"] = str(decision_label)
            debug_row["decision_forced_choice"] = dict(stage_result["decision_forced_choice"])
        elif str(slot_name) == "decision":
            debug_row["decision_decode_mode"] = "generate"
            debug_row["decision_label"] = str(decision_label)
        elif str(slot_name) == "intent":
            debug_row["intent_decode_mode"] = str(intent_decode_mode)
            debug_row["intent_label"] = str(intent_label)
            debug_row["intent_decision_label"] = str(intent_decision_label)
            debug_row["route_source"] = str(route_source)
            if "intent_forced_choice" in stage_result:
                debug_row["intent_forced_choice"] = dict(stage_result["intent_forced_choice"])
                debug_row["intent_body_prompt_text"] = str(stage_result.get("intent_body_prompt_text", ""))
        stage_debug.append(debug_row)

    cot_body = "\n".join(str(generated_sections.get(slot_name, "")).strip() for slot_name in TRIPLET_SLOT_NAMES).strip()
    cot = cot_body + THINK_END_TEXT if cot_body else THINK_END_TEXT
    return cot, "", generated_sections, stage_debug


def _extract_prefixed_route_debug(stage_debug: list[dict[str, Any]]) -> dict[str, Any]:
    route = {
        "prefixed_route_source": "decision",
        "prefixed_intent_label": "unknown",
        "prefixed_intent_decision_label": "unknown",
        "prefixed_decision_label": "unknown",
        "prefixed_decision_score_allow": None,
        "prefixed_decision_score_refuse": None,
        "prefixed_decision_score_delta_allow_minus_refuse": None,
        "prefixed_answer_kvreuse_requested": False,
        "prefixed_answer_kvreuse_feasible": False,
    }
    for item in stage_debug:
        if str(item.get("slot_name", "")) == "intent":
            route["prefixed_intent_label"] = str(item.get("intent_label", "unknown"))
            route["prefixed_intent_decision_label"] = str(item.get("intent_decision_label", "unknown"))
            route["prefixed_route_source"] = str(item.get("route_source", route["prefixed_route_source"]))
            if str(item.get("route_source", "")) == "intent":
                route["prefixed_decision_label"] = str(item.get("intent_decision_label", "unknown"))
        elif str(item.get("slot_name", "")) == "decision":
            if route.get("prefixed_route_source") != "intent":
                route["prefixed_decision_label"] = str(item.get("decision_label", "unknown"))
            forced_choice = item.get("decision_forced_choice", {})
            if isinstance(forced_choice, dict):
                route["prefixed_decision_score_allow"] = forced_choice.get("score_allow")
                route["prefixed_decision_score_refuse"] = forced_choice.get("score_refuse")
                route["prefixed_decision_score_delta_allow_minus_refuse"] = forced_choice.get(
                    "score_delta_allow_minus_refuse"
                )
    return route


def _build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "num_examples": 0,
            "prefixed_decision_label_counts": {},
            "prefixed_decision_accuracy": float("nan"),
            "prefixed_decision_num_labeled": 0,
            "prefixed_decision_expected_allow_count": 0,
            "prefixed_decision_overrefuse_count": 0,
            "prefixed_decision_overrefuse_rate": float("nan"),
            "prefixed_answer_kvreuse_requested_rate": float("nan"),
            "prefixed_answer_kvreuse_feasible_rate": float("nan"),
        }

    decision_label_counts: dict[str, int] = {}
    decision_labeled = 0
    decision_correct = 0
    decision_expected_allow = 0
    decision_overrefuse = 0
    for row in rows:
        decision_label = str(row.get("prefixed_decision_label", "")).strip() or "unknown"
        decision_label_counts[decision_label] = decision_label_counts.get(decision_label, 0) + 1
        expected_decision_label = str(row.get("expected_decision_label", "")).strip().lower()
        if expected_decision_label in {"allow", "refuse"}:
            decision_labeled += 1
            decision_correct += int(decision_label == expected_decision_label)
        if expected_decision_label == "allow":
            decision_expected_allow += 1
            decision_overrefuse += int(decision_label == "refuse")

    return {
        "num_examples": int(len(rows)),
        "prefixed_decision_label_counts": dict(sorted(decision_label_counts.items())),
        "prefixed_decision_accuracy": (
            float(decision_correct / decision_labeled) if decision_labeled else float("nan")
        ),
        "prefixed_decision_num_labeled": int(decision_labeled),
        "prefixed_decision_expected_allow_count": int(decision_expected_allow),
        "prefixed_decision_overrefuse_count": int(decision_overrefuse),
        "prefixed_decision_overrefuse_rate": (
            float(decision_overrefuse / decision_expected_allow)
            if decision_expected_allow
            else float("nan")
        ),
    }


def _build_minimal_result_row(
    *,
    row: dict[str, Any],
    prefixed_mode: str,
    prefixed_sections: dict[str, str],
    prefixed_route_debug: dict[str, Any],
    prefixed_cot: str = "",
    prefixed_answer: str = "",
    ird_handoff_prefix: str = "",
    ird_handoff_debug: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = {
        "id": row.get("id", ""),
        "dataset": row.get("dataset", ""),
        "category": row.get("category", ""),
        "subgroup": row.get("subgroup", ""),
        "raw_prompt": row.get("raw_prompt", ""),
        "expected_decision_label": row.get("expected_decision_label", ""),
        "prefixed_mode": prefixed_mode,
        "prefixed_route_source": prefixed_route_debug.get("prefixed_route_source", "decision"),
        "prefixed_intent_label": prefixed_route_debug.get("prefixed_intent_label", "unknown"),
        "prefixed_intent_decision_label": prefixed_route_debug.get("prefixed_intent_decision_label", "unknown"),
        "prefixed_decision_label": prefixed_route_debug.get("prefixed_decision_label", "unknown"),
        "generated_ird": dict(prefixed_sections),
        "prefixed_cot": str(prefixed_cot),
        "prefixed_answer": str(prefixed_answer),
        "ird_handoff_prefix": str(ird_handoff_prefix),
    }
    if isinstance(ird_handoff_debug, dict) and ird_handoff_debug:
        result["ird_handoff_debug"] = dict(ird_handoff_debug)
    return result


def _resolve_benign_harmful_label(row: dict[str, Any]) -> str:
    raw_label = str(row.get("label", "")).strip().lower()
    if raw_label in {"safe", "benign", "allow", "allowed", "harmless"}:
        return "benign"
    if raw_label in {"harmful", "unsafe", "refuse", "refusal"}:
        return "harmful"
    expected = str(row.get("expected_decision_label", "")).strip().lower()
    if expected == "allow":
        return "benign"
    if expected == "refuse":
        return "harmful"
    return "unknown"


def _build_ird_result_row(
    *,
    row: dict[str, Any],
    prefixed_sections: dict[str, str],
) -> dict[str, Any]:
    return {
        "prompt": str(row.get("raw_prompt", "")),
        "i": str(prefixed_sections.get("intent", "")),
        "r": str(prefixed_sections.get("risk", "")),
        "d": str(prefixed_sections.get("decision", "")),
        "dataset": str(row.get("dataset", "")),
        "label": _resolve_benign_harmful_label(row),
    }


def _resolve_adapter_dirs(args) -> dict[str, Path]:
    triplet_dirs = {
        "intent": str(args.intent_adapter_dir).strip(),
        "risk": str(args.risk_adapter_dir).strip(),
        "decision": str(args.decision_adapter_dir).strip(),
    }
    if not all(bool(path) for path in triplet_dirs.values()):
        raise ValueError("Provide all of --intent_adapter_dir, --risk_adapter_dir, --decision_adapter_dir.")
    return {slot: Path(path) for slot, path in triplet_dirs.items()}


def _load_triplet_configs(adapter_dirs: dict[str, Path]) -> tuple[str, dict[str, Any]]:
    configs = {slot: PeftConfig.from_pretrained(path) for slot, path in adapter_dirs.items()}
    base_model_names = {str(config.base_model_name_or_path) for config in configs.values()}
    if len(base_model_names) != 1:
        raise ValueError(f"All triplet adapters must share the same base model, got {base_model_names}.")
    for slot, config in configs.items():
        if str(getattr(config, "peft_type", "")) != "PeftType.PREFIX_TUNING":
            raise ValueError(f"Adapter {slot!r} at {adapter_dirs[slot]} is not a PREFIX_TUNING adapter.")
    return next(iter(base_model_names)), configs


def _resolve_intent_decode_mode(raw_mode: str, *, adapter_dirs: dict[str, Path]) -> str:
    mode = str(raw_mode).strip().lower()
    if mode != "auto":
        return mode
    if "intent" not in adapter_dirs:
        return "generate"
    training_config_path = Path(adapter_dirs["intent"]).parent / "training_config.json"
    if not training_config_path.exists():
        return "generate"
    try:
        payload = json.loads(training_config_path.read_text(encoding="utf-8"))
    except Exception:
        return "generate"
    if bool(payload.get("intent_label_conditioned_body_loss", False)):
        return "label_conditioned_body"
    if str(payload.get("slot_supervision_mode", "")).strip() == "intent_label_conditioned_body":
        return "label_conditioned_body"
    return "generate"


def _run_parallel_generation(
    *,
    args,
    adapter_dirs: dict[str, Path],
    input_path: Path,
    input_format: str,
    output_dir: Path,
    prompt_rows: list[dict[str, Any]],
) -> None:
    gpu_ids = _parse_gpu_ids(args.gpu_ids)
    if not gpu_ids:
        raise ValueError("Parallel generation requested without any gpu_ids.")

    worker_root = output_dir / "_parallel_workers"
    worker_root.mkdir(parents=True, exist_ok=True)
    shard_ranges = _compute_contiguous_shards(len(prompt_rows), len(gpu_ids))
    worker_device_map = str(args.device_map)
    if worker_device_map.strip().lower() == "auto":
        worker_device_map = "none"

    script_path = Path(__file__).resolve()
    worker_specs: list[dict[str, Any]] = []
    print(
        f"Launching {len(gpu_ids)} generation workers across GPUs {gpu_ids} for {len(prompt_rows)} prompts.",
        flush=True,
    )

    for worker_idx, (gpu_id, (start_idx, end_idx)) in enumerate(zip(gpu_ids, shard_ranges)):
        shard_rows = prompt_rows[start_idx:end_idx]
        if not shard_rows:
            continue

        worker_dir = worker_root / f"worker_{worker_idx:02d}_gpu{gpu_id}"
        worker_dir.mkdir(parents=True, exist_ok=True)
        shard_file = worker_dir / "shard_input.jsonl"
        _write_prompt_rows(shard_file, shard_rows)

        cmd = [
            sys.executable,
            str(script_path),
            "--worker_mode",
            "--data_file",
            str(shard_file),
            "--output_dir",
            str(worker_dir),
            "--input_format",
            "jsonl",
            "--prompt_field",
            "auto",
            "--start",
            "0",
            "--max_think_tokens",
            str(_serialize_generation_budget(args.max_think_tokens)),
            "--max_answer_tokens",
            str(_serialize_generation_budget(args.max_answer_tokens)),
            "--dtype",
            str(args.dtype),
            "--device_map",
            str(worker_device_map),
            "--attn_implementation",
            str(args.attn_implementation),
            "--seed",
            str(int(args.seed) + worker_idx),
            "--progress_position",
            str(worker_idx),
            "--progress_desc",
            f"strict[gpu{gpu_id}]",
        ]
        cmd.extend(["--intent_adapter_dir", str(adapter_dirs["intent"])])
        cmd.extend(["--risk_adapter_dir", str(adapter_dirs["risk"])])
        cmd.extend(["--decision_adapter_dir", str(adapter_dirs["decision"])])
        if args.config_model_name is not None:
            cmd.extend(["--config_model_name", str(args.config_model_name)])
        cmd.extend(["--prompt_format", str(args.prompt_format)])
        cmd.extend(["--decode_profile", str(args.decode_profile)])
        _append_boolean_flag(
            cmd,
            name="ird_handoff",
            value=bool(args.ird_handoff),
            default=False,
        )
        cmd.extend(
            [
                "--ird_handoff_max_think_tokens",
                str(_serialize_generation_budget(args.ird_handoff_max_think_tokens)),
            ]
        )
        cmd.extend(
            [
                "--ird_handoff_max_answer_tokens",
                str(_serialize_generation_budget(args.ird_handoff_max_answer_tokens)),
            ]
        )
        _append_boolean_flag(
            cmd,
            name="ird_handoff_disable_adapters",
            value=bool(args.ird_handoff_disable_adapters),
            default=True,
        )
        _append_boolean_flag(
            cmd,
            name="ird_handoff_answer_kvreuse",
            value=bool(args.ird_handoff_answer_kvreuse),
            default=False,
        )
        if args.prefix_horizon is not None:
            cmd.extend(["--prefix_horizon", str(args.prefix_horizon)])
        cmd.extend(["--decision_decode_mode", str(args.decision_decode_mode)])
        cmd.extend(["--route_source", str(args.route_source)])
        cmd.extend(["--intent_tag_safe", str(args.intent_tag_safe)])
        cmd.extend(["--intent_tag_harmful", str(args.intent_tag_harmful)])
        cmd.extend(["--intent_decode_mode", str(args.intent_decode_mode)])
        cmd.extend(["--intent_margin", str(args.intent_margin)])
        _append_boolean_flag(
            cmd,
            name="intent_score_length_norm",
            value=bool(args.intent_score_length_norm),
            default=True,
        )
        cmd.extend(["--decision_margin", str(args.decision_margin)])
        _append_boolean_flag(
            cmd,
            name="decision_score_length_norm",
            value=bool(args.decision_score_length_norm),
            default=True,
        )
        cmd.extend(["--ablate_slots", str(args.ablate_slots)])
        cmd.extend(["--cascade_stage_max_tokens", str(_serialize_generation_budget(args.cascade_stage_max_tokens))])
        cmd.extend(["--cascade_slot_budget_ratio", str(args.cascade_slot_budget_ratio)])
        if str(args.slot_prefix_strength_schedules).strip():
            cmd.extend(["--slot_prefix_strength_schedules", str(args.slot_prefix_strength_schedules)])
        cmd.extend(["--decoding_strategy", str(args.decoding_strategy)])
        if args.do_sample is not None:
            cmd.append("--do_sample" if bool(args.do_sample) else "--no-do_sample")
        if args.temperature is not None:
            cmd.extend(["--temperature", str(args.temperature)])
        if args.top_p is not None:
            cmd.extend(["--top_p", str(args.top_p)])
        if args.top_k is not None:
            cmd.extend(["--top_k", str(args.top_k)])
        _append_boolean_flag(
            cmd,
            name="renormalize_logits",
            value=bool(args.renormalize_logits),
            default=False,
        )
        _append_boolean_flag(
            cmd,
            name="trust_remote_code",
            value=bool(args.trust_remote_code),
            default=True,
        )
        _append_boolean_flag(
            cmd,
            name="local_files_only",
            value=bool(args.local_files_only),
            default=True,
        )
        _append_boolean_flag(
            cmd,
            name="disable_progress",
            value=bool(args.disable_progress),
            default=False,
        )

        worker_env = os.environ.copy()
        worker_env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        print(
            f"[worker {worker_idx}] GPU {gpu_id}: prompts {start_idx}:{end_idx} "
            f"({len(shard_rows)} examples, device_map={worker_device_map}) -> {worker_dir}",
            flush=True,
        )
        proc = subprocess.Popen(cmd, env=worker_env)
        worker_specs.append(
            {
                "worker_idx": int(worker_idx),
                "gpu_id": str(gpu_id),
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "num_examples": int(len(shard_rows)),
                "worker_dir": str(worker_dir),
                "process": proc,
            }
        )

    failures: list[dict[str, Any]] = []
    for spec in worker_specs:
        return_code = spec["process"].wait()
        if return_code != 0:
            failures.append(
                {
                    "worker_idx": spec["worker_idx"],
                    "gpu_id": spec["gpu_id"],
                    "return_code": int(return_code),
                    "worker_dir": str(spec["worker_dir"]),
                }
            )
    if failures:
        raise RuntimeError(f"One or more parallel generation workers failed: {failures}")

    merged_result_rows: list[dict[str, Any]] = []
    merged_ird_rows: list[dict[str, Any]] = []
    worker_summaries: list[dict[str, Any]] = []
    for spec in worker_specs:
        worker_dir = Path(spec["worker_dir"])
        merged_result_rows.extend(read_jsonl(worker_dir / "results.jsonl"))
        ird_path = worker_dir / "ird.jsonl"
        if ird_path.exists():
            merged_ird_rows.extend(read_jsonl(ird_path))
        with (worker_dir / "summary.json").open("r", encoding="utf-8") as rf:
            worker_summaries.append(json.load(rf))

    write_jsonl(output_dir / "results.jsonl", merged_result_rows)
    if merged_ird_rows:
        write_jsonl(output_dir / "ird.jsonl", merged_ird_rows)

    summary = _build_summary(merged_result_rows)
    first_worker_summary = worker_summaries[0] if worker_summaries else {}
    summary.update(
        {
            "adapter_mode": "triplet",
            "data_file": str(input_path),
            "config_model_name": str(first_worker_summary.get("config_model_name", "")),
            "route_source": str(args.route_source),
            "intent_tag_safe": str(args.intent_tag_safe),
            "intent_tag_harmful": str(args.intent_tag_harmful),
            "intent_decode_mode": str(args.intent_decode_mode),
            "cascade_stage_max_tokens": _serialize_generation_budget(args.cascade_stage_max_tokens),
            "ird_handoff": bool(args.ird_handoff),
            "rd_handoff_slots": ["risk", "decision"],
            "rd_handoff_separator": "\\n",
            "ird_handoff_max_think_tokens": _serialize_generation_budget(args.ird_handoff_max_think_tokens),
            "ird_handoff_max_answer_tokens": _serialize_generation_budget(args.ird_handoff_max_answer_tokens),
            "ird_handoff_disable_adapters": bool(args.ird_handoff_disable_adapters),
            "ird_handoff_answer_kvreuse": bool(args.ird_handoff_answer_kvreuse),
            "parallel_gpu_ids": [str(x) for x in gpu_ids],
            "parallel_worker_count": int(len(worker_specs)),
        }
    )
    summary["adapter_dirs"] = {slot: str(path) for slot, path in adapter_dirs.items()}
    write_json(output_dir / "summary.json", summary)


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(args.seed)

    if args.local_files_only:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    adapter_dirs = _resolve_adapter_dirs(args)
    for path in adapter_dirs.values():
        if not path.exists():
            raise ValueError(f"Adapter directory not found: {path}")
    args.intent_decode_mode = _resolve_intent_decode_mode(
        str(args.intent_decode_mode),
        adapter_dirs=adapter_dirs,
    )

    input_path = Path(args.data_file)
    if not input_path.exists():
        raise ValueError(f"Input file not found: {input_path}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_format = _infer_input_format(input_path, str(args.input_format))
    if input_format == "csv":
        prompt_rows = _load_csv_rows(
            path=input_path,
            prompt_field=str(args.prompt_field),
            case_kind_filter=str(args.case_kind_filter),
            manual_true_unsafe_filter=str(args.manual_true_unsafe_filter),
            start=args.start,
            max_prompts=args.max_prompts,
        )
    else:
        prompt_rows = _load_jsonl_rows(
            path=input_path,
            prompt_field=str(args.prompt_field),
            start=args.start,
            max_prompts=args.max_prompts,
        )
    prompt_rows = _sample_rows_by_expected_decision_label(
        prompt_rows,
        max_allow_prompts=args.max_allow_prompts,
        max_refuse_prompts=args.max_refuse_prompts,
        seed=int(args.label_sample_seed),
        shuffle=bool(args.label_sample_shuffle),
    )
    if not prompt_rows:
        raise ValueError("No prompts loaded after filtering/sampling.")
    if not bool(args.worker_mode) and _parse_gpu_ids(args.gpu_ids):
        _run_parallel_generation(
            args=args,
            adapter_dirs=adapter_dirs,
            input_path=input_path,
            input_format=input_format,
            output_dir=output_dir,
            prompt_rows=prompt_rows,
        )
        return

    base_model_name, triplet_configs = _load_triplet_configs(adapter_dirs)
    triplet_config_summary = {
        slot: {
            "adapter_dir": str(adapter_dirs[slot]),
            "num_virtual_tokens": int(triplet_configs[slot].num_virtual_tokens),
        }
        for slot in TRIPLET_SLOT_NAMES
    }

    config_model_name = args.config_model_name or base_model_name
    model_cfg = load_model_config(
        model_name=base_model_name,
        config_model_name=config_model_name,
    )
    system_prompt = str(model_cfg.get("system_prompt", ""))

    tokenizer = _load_tokenizer_for_adapters(
        args,
        base_model_name=base_model_name,
        adapter_dirs=[adapter_dirs[slot] for slot in TRIPLET_SLOT_NAMES],
    )
    prompt_format = _resolve_requested_prompt_format(
        tokenizer,
        str(args.prompt_format),
        model_cfg=model_cfg,
    )
    decode_profile = _resolve_requested_decode_profile(
        model_cfg=model_cfg,
        prompt_format=prompt_format,
        raw_decode_profile=str(args.decode_profile),
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id.")
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, Any] = {
        "dtype": parse_dtype(args.dtype),
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
        "attn_implementation": args.attn_implementation,
    }
    device_map = _resolve_device_map_arg(args.device_map)
    if device_map is not None:
        load_kwargs["device_map"] = device_map

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        **load_kwargs,
    )
    if len(tokenizer) != int(base_model.get_input_embeddings().num_embeddings):
        base_model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    if device_map is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_model.to(device)
    base_model.eval()

    ablate_slots = set(_parse_slot_name_set(args.ablate_slots))
    cascade_slot_budget_ratio = _parse_slot_budget_ratio(
        args.cascade_slot_budget_ratio,
        expected_len=len(TRIPLET_SLOT_NAMES),
    )
    slot_prefix_strength_schedules = _parse_slot_prefix_strength_schedules(
        args.slot_prefix_strength_schedules,
        slot_names=TRIPLET_SLOT_NAMES,
    )
    generation_settings = _resolve_generation_settings(args, model_cfg)
    generation_settings_payload = _generation_settings_summary(generation_settings)
    special_token_rows_by_slot = {
        slot: _load_special_token_rows_payload(adapter_dirs[slot])
        for slot in TRIPLET_SLOT_NAMES
    }
    model = PeftModel.from_pretrained(
        base_model,
        adapter_dirs["intent"],
        adapter_name="intent",
        is_trainable=False,
    )
    model.load_adapter(adapter_dirs["risk"], adapter_name="risk", is_trainable=False)
    model.load_adapter(adapter_dirs["decision"], adapter_name="decision", is_trainable=False)
    _apply_special_token_rows(model, special_token_rows_by_slot.get("intent"))
    prefixed_mode = "triplet_prefix_cascade_decode"
    triplet_generation_summary = {
        "mode": "cascade_section_decode",
        "slot_names": ["intent", "risk", "decision"],
        "routing_mode": (
            "intent_label"
            if str(args.route_source) == "intent"
            else ("decoder_decision" if str(args.decision_decode_mode) == "forced_choice" else None)
        ),
        "decision_decode_mode": str(args.decision_decode_mode),
        "decision_candidate_allow": DECISION_ALLOW_LABEL,
        "decision_candidate_refuse": DECISION_REFUSE_LABEL,
        "route_source": str(args.route_source),
        "intent_tag_safe": str(args.intent_tag_safe),
        "intent_tag_harmful": str(args.intent_tag_harmful),
        "intent_decode_mode": str(args.intent_decode_mode),
        "intent_margin": float(args.intent_margin),
        "intent_score_length_norm": bool(args.intent_score_length_norm),
        "decision_margin": float(args.decision_margin),
        "decision_score_length_norm": bool(args.decision_score_length_norm),
        "ablate_slots": sorted(ablate_slots),
        "cascade_stage_max_tokens": _serialize_generation_budget(args.cascade_stage_max_tokens),
        "cascade_slot_budget_ratio": [int(x) for x in cascade_slot_budget_ratio],
            "cascade_stop": "newline",
        "slot_prefix_strength_schedules": _serialize_slot_prefix_strength_schedules(slot_prefix_strength_schedules),
        "prompt_format": str(prompt_format),
        "decode_profile": str(decode_profile),
        **generation_settings_payload,
        "generation_settings": dict(generation_settings_payload),
    }

    model.eval()
    model.config.use_cache = True

    result_rows: list[dict[str, Any]] = []
    ird_rows: list[dict[str, Any]] = []
    result_path = output_dir / "results.jsonl"
    ird_path = output_dir / "ird.jsonl"
    progress_desc = str(args.progress_desc).strip() or "Generating prefixed outputs"
    with result_path.open("w", encoding="utf-8") as wf_result, ird_path.open("w", encoding="utf-8") as wf_ird:
        for row in tqdm(
            prompt_rows,
            desc=progress_desc,
            total=len(prompt_rows),
            disable=bool(args.disable_progress),
            position=int(args.progress_position),
        ):
            pref_cot, pref_answer, prefixed_sections, prefixed_stage_debug = _generate_triplet_cascade_one(
                model=model,
                tokenizer=tokenizer,
                prompt=row["raw_prompt"],
                system_prompt=system_prompt,
                prompt_format=prompt_format,
                decode_profile=decode_profile,
                prefix_horizon=args.prefix_horizon,
                decision_decode_mode=str(args.decision_decode_mode),
                route_source=str(args.route_source),
                intent_tag_safe=str(args.intent_tag_safe),
                intent_tag_harmful=str(args.intent_tag_harmful),
                intent_decode_mode=str(args.intent_decode_mode),
                intent_margin=float(args.intent_margin),
                intent_score_length_norm=bool(args.intent_score_length_norm),
                decision_margin=float(args.decision_margin),
                decision_score_length_norm=bool(args.decision_score_length_norm),
                ablate_slots=ablate_slots,
                cascade_stage_max_tokens=args.cascade_stage_max_tokens,
                cascade_slot_budget_ratio=cascade_slot_budget_ratio,
                generation_settings=generation_settings,
                slot_prefix_strength_schedules=slot_prefix_strength_schedules,
                special_token_rows_by_slot=special_token_rows_by_slot,
            )
            prefixed_route_debug = _extract_prefixed_route_debug(prefixed_stage_debug)
            ird_handoff_prefix = ""
            ird_handoff_debug = None
            if bool(args.ird_handoff):
                pref_cot, pref_answer, ird_handoff_prefix, ird_handoff_debug = _generate_ird_handoff_one(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=row["raw_prompt"],
                    system_prompt=system_prompt,
                    prompt_format=prompt_format,
                    decode_profile=decode_profile,
                    handoff_prefix=_normalize_ird_handoff_prefix(
                        prefixed_sections,
                    ),
                    max_think_tokens=args.ird_handoff_max_think_tokens,
                    max_answer_tokens=args.ird_handoff_max_answer_tokens,
                    generation_settings=generation_settings,
                    disable_adapters=bool(args.ird_handoff_disable_adapters),
                    enable_answer_kvreuse=bool(args.ird_handoff_answer_kvreuse),
                )
            result_row = _build_minimal_result_row(
                row=row,
                prefixed_mode=prefixed_mode,
                prefixed_sections=prefixed_sections,
                prefixed_route_debug=prefixed_route_debug,
                prefixed_cot=pref_cot,
                prefixed_answer=pref_answer,
                ird_handoff_prefix=ird_handoff_prefix,
                ird_handoff_debug=ird_handoff_debug,
            )

            result_rows.append(result_row)
            wf_result.write(json.dumps(result_row, ensure_ascii=False) + "\n")
            wf_result.flush()
            ird_row = _build_ird_result_row(row=row, prefixed_sections=prefixed_sections)
            ird_rows.append(ird_row)
            wf_ird.write(json.dumps(ird_row, ensure_ascii=False) + "\n")
            wf_ird.flush()

    summary = _build_summary(result_rows)
    summary.update(
        {
            "adapter_mode": "triplet",
            "data_file": str(input_path),
            "config_model_name": config_model_name,
            "route_source": str(args.route_source),
            "intent_tag_safe": str(args.intent_tag_safe),
            "intent_tag_harmful": str(args.intent_tag_harmful),
            "intent_decode_mode": str(args.intent_decode_mode),
            "triplet_generation": triplet_generation_summary,
            "cascade_stage_max_tokens": _serialize_generation_budget(args.cascade_stage_max_tokens),
            "ird_handoff": bool(args.ird_handoff),
            "rd_handoff_slots": ["risk", "decision"],
            "rd_handoff_separator": "\\n",
            "ird_handoff_max_think_tokens": _serialize_generation_budget(args.ird_handoff_max_think_tokens),
            "ird_handoff_max_answer_tokens": _serialize_generation_budget(args.ird_handoff_max_answer_tokens),
            "ird_handoff_disable_adapters": bool(args.ird_handoff_disable_adapters),
            "ird_handoff_answer_kvreuse": bool(args.ird_handoff_answer_kvreuse),
        }
    )
    summary["adapter_dirs"] = {slot: str(path) for slot, path in adapter_dirs.items()}
    summary["triplet_config"] = triplet_config_summary
    write_json(output_dir / "summary.json", summary)


if __name__ == "__main__":
    main()
