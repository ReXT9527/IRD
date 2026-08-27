from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DATA_ROOT = REPO_ROOT / "data"
PT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "PT"
LEGACY_EVAL_SCRIPT = REPO_ROOT / "eval" / "pt-eval-strict-slot-prefix-generation.py"

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from peft import PrefixTuningConfig, TaskType, get_peft_model


def _load_local_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


pt_utils = _load_local_module("pt_latent_safe_prefix_utils", REPO_ROOT / "scripts" / "pt-latent-safe-prefix-utils.py")

THINK_END_TEXT = pt_utils.THINK_END_TEXT
build_prompt_spec = pt_utils.build_prompt_spec
build_triplet_cascade_prompt_suffix = pt_utils.build_triplet_cascade_prompt_suffix
compose_triplet_cot_text = pt_utils.compose_triplet_cot_text
DEFAULT_TRIPLET_SLOT_NAMES = pt_utils.DEFAULT_TRIPLET_SLOT_NAMES
extract_triplet_section_texts = pt_utils.extract_triplet_section_texts
load_model_config = pt_utils.load_model_config
parse_dtype = pt_utils.parse_dtype
read_jsonl = pt_utils.read_jsonl
resolve_decode_profile = pt_utils.resolve_decode_profile
resolve_prompt_format = pt_utils.resolve_prompt_format
resolve_model_input_device = pt_utils.resolve_model_input_device
set_seed = pt_utils.set_seed
write_json = pt_utils.write_json
write_jsonl = pt_utils.write_jsonl

def _parse_marker_list(raw: str) -> list[str]:
    return list(dict.fromkeys(marker for marker in (chunk.strip() for chunk in str(raw or "").split(",")) if marker))


def _special_token_markers_for_training(args, *, slot_names: tuple[str, ...]) -> list[str]:
    markers = (
        str(args.intent_tag_safe),
        str(args.intent_tag_harmful),
        *_parse_marker_list(getattr(args, "special_token_extra_markers", "")),
    )
    return list(dict.fromkeys(marker for marker in (str(marker).strip() for marker in markers) if marker))


def _register_additional_special_tokens(tokenizer, markers: list[str]) -> dict[str, Any]:
    requested = [str(marker).strip() for marker in markers if str(marker).strip()]
    existing = set(str(token) for token in getattr(tokenizer, "additional_special_tokens", []) or [])
    to_add = [marker for marker in requested if marker not in existing]
    added = (
        int(tokenizer.add_special_tokens({"additional_special_tokens": to_add}, replace_additional_special_tokens=False))
        if to_add
        else 0
    )
    token_ids = {
        marker: tokenizer.convert_tokens_to_ids(marker)
        for marker in requested
        if tokenizer.convert_tokens_to_ids(marker) != tokenizer.unk_token_id
    }
    return {
        "requested_tokens": requested,
        "added_tokens": int(added),
        "token_ids": {str(k): int(v) for k, v in token_ids.items()},
    }


def _init_added_token_rows_from_old_pieces(model, tokenizer, old_piece_ids_by_token: dict[str, list[int]]) -> None:
    input_embeddings = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    with torch.no_grad():
        for token, old_piece_ids in old_piece_ids_by_token.items():
            token_id = tokenizer.convert_tokens_to_ids(str(token))
            if token_id is None or token_id < 0 or not old_piece_ids:
                continue
            piece_ids = torch.tensor([int(x) for x in old_piece_ids], device=input_embeddings.weight.device)
            input_embeddings.weight[int(token_id)].copy_(input_embeddings.weight.index_select(0, piece_ids).mean(dim=0))
            if output_embeddings is not None and hasattr(output_embeddings, "weight"):
                piece_ids_out = piece_ids.to(output_embeddings.weight.device)
                output_embeddings.weight[int(token_id)].copy_(
                    output_embeddings.weight.index_select(0, piece_ids_out).mean(dim=0)
                )


class TrainableSelectedTokenRows(nn.Module):
    def __init__(self, base_layer: nn.Module, token_ids: list[int]):
        super().__init__()
        if not hasattr(base_layer, "weight"):
            raise TypeError(f"Expected module with weight, got {type(base_layer).__name__}")
        self.base_layer = base_layer
        self.token_ids = [int(x) for x in token_ids]
        row_index = torch.tensor(self.token_ids, dtype=torch.long)
        self.register_buffer("row_index", row_index, persistent=True)
        with torch.no_grad():
            init_rows = base_layer.weight.detach().index_select(0, row_index.to(base_layer.weight.device)).clone()
        self.rows = nn.Parameter(init_rows)
        self.base_layer.weight.requires_grad_(False)

    @property
    def weight(self):
        if not self.token_ids or not torch.is_grad_enabled():
            return self.base_layer.weight
        row_index = self.row_index.to(self.base_layer.weight.device)
        rows = self.rows.to(device=self.base_layer.weight.device, dtype=self.base_layer.weight.dtype)
        return self.base_layer.weight.index_copy(0, row_index, rows)

    def forward(self, *args, **kwargs):
        if isinstance(self.base_layer, nn.Embedding):
            input_ids = args[0]
            output = F.embedding(
                input_ids,
                self.base_layer.weight,
                padding_idx=self.base_layer.padding_idx,
                max_norm=self.base_layer.max_norm,
                norm_type=self.base_layer.norm_type,
                scale_grad_by_freq=self.base_layer.scale_grad_by_freq,
                sparse=self.base_layer.sparse,
            )
            rows = self.rows.to(device=output.device, dtype=output.dtype)
            for row_pos, token_id in enumerate(self.token_ids):
                mask = input_ids.eq(int(token_id))
                if bool(mask.any()):
                    output = torch.where(mask.unsqueeze(-1), rows[int(row_pos)].view(*([1] * (output.ndim - 1)), -1), output)
            return output

        hidden_states = args[0]
        logits = F.linear(hidden_states, self.base_layer.weight, getattr(self.base_layer, "bias", None))
        if not self.token_ids:
            return logits
        row_index = self.row_index.to(logits.device)
        rows = self.rows.to(device=hidden_states.device, dtype=hidden_states.dtype)
        selected_logits = F.linear(hidden_states, rows, None).to(dtype=logits.dtype)
        return logits.index_copy(-1, row_index, selected_logits)


def _set_output_embeddings(model, module: nn.Module) -> None:
    if hasattr(model, "set_output_embeddings"):
        model.set_output_embeddings(module)
        return
    if hasattr(model, "lm_head"):
        model.lm_head = module
        return
    raise ValueError("Model does not expose set_output_embeddings or lm_head.")


def _attach_trainable_special_token_rows(model, tokenizer, markers: list[str]) -> dict[str, Any]:
    token_ids: list[int] = []
    for marker in markers:
        token_id = tokenizer.convert_tokens_to_ids(str(marker))
        if token_id is None or token_id < 0 or token_id == tokenizer.unk_token_id:
            continue
        token_ids.append(int(token_id))
    token_ids = sorted(set(token_ids))
    if not token_ids:
        return {"enabled": False, "token_ids": []}

    input_wrapper = TrainableSelectedTokenRows(model.get_input_embeddings(), token_ids)
    model.set_input_embeddings(input_wrapper)

    output_wrapper = None
    output_embeddings = model.get_output_embeddings()
    if output_embeddings is not None and hasattr(output_embeddings, "weight"):
        if output_embeddings.weight.data_ptr() == input_wrapper.base_layer.weight.data_ptr():
            output_wrapper = input_wrapper
        else:
            output_wrapper = TrainableSelectedTokenRows(output_embeddings, token_ids)
        _set_output_embeddings(model, output_wrapper)

    return {
        "enabled": True,
        "token_ids": [int(x) for x in token_ids],
        "input_module": input_wrapper,
        "output_module": output_wrapper,
    }


def _reenable_trainable_special_token_rows(token_row_state: dict[str, Any]) -> dict[str, Any]:
    if not token_row_state.get("enabled"):
        return {"num_row_parameters": 0, "num_trainable_row_parameters": 0}
    seen: set[int] = set()
    num_row_parameters = 0
    num_trainable_row_parameters = 0
    for module_key in ("input_module", "output_module"):
        module = token_row_state.get(module_key)
        if module is None or not hasattr(module, "rows"):
            continue
        rows = module.rows
        param_id = id(rows)
        if param_id in seen:
            continue
        seen.add(param_id)
        rows.requires_grad_(True)
        num_row_parameters += int(rows.numel())
        if bool(rows.requires_grad):
            num_trainable_row_parameters += int(rows.numel())
    return {
        "num_row_parameters": int(num_row_parameters),
        "num_trainable_row_parameters": int(num_trainable_row_parameters),
    }


def _save_trainable_special_token_rows(adapter_dir: Path, token_row_state: dict[str, Any], tokenizer) -> None:
    if not token_row_state.get("enabled"):
        return
    input_module = token_row_state.get("input_module")
    output_module = token_row_state.get("output_module")
    payload = {
        "format_version": 1,
        "token_ids": [int(x) for x in token_row_state.get("token_ids", [])],
        "id_to_token": {
            str(int(token_id)): str(tokenizer.convert_ids_to_tokens(int(token_id)))
            for token_id in token_row_state.get("token_ids", [])
        },
        "input_rows": input_module.rows.detach().cpu() if input_module is not None else None,
        "output_rows": (
            None
            if output_module is None or output_module is input_module
            else output_module.rows.detach().cpu()
        ),
        "output_tied_to_input": bool(output_module is input_module),
    }
    torch.save(payload, adapter_dir / "special_token_rows.pt")


def _init_distributed_from_env() -> dict[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world_size > 1
    if enabled and not dist.is_initialized():
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            try:
                dist.init_process_group(
                    backend=backend,
                    init_method="env://",
                    device_id=torch.device(f"cuda:{local_rank}"),
                )
            except TypeError:
                dist.init_process_group(backend=backend, init_method="env://")
        else:
            dist.init_process_group(backend=backend, init_method="env://")
    return {
        "enabled": bool(enabled),
        "rank": int(rank),
        "local_rank": int(local_rank),
        "world_size": int(world_size),
        "is_main": int(rank) == 0,
    }


def _is_main_process(distributed_ctx: dict[str, Any] | None) -> bool:
    return not distributed_ctx or bool(distributed_ctx.get("is_main", True))


def _distributed_barrier(distributed_ctx: dict[str, Any] | None) -> None:
    if distributed_ctx and bool(distributed_ctx.get("enabled", False)):
        if torch.cuda.is_available():
            try:
                dist.barrier(device_ids=[int(distributed_ctx["local_rank"])])
                return
            except TypeError:
                pass
        dist.barrier()


def _distributed_sum_values(
    distributed_ctx: dict[str, Any] | None,
    values: list[float],
    *,
    device: torch.device,
) -> list[float]:
    if not distributed_ctx or not bool(distributed_ctx.get("enabled", False)):
        return values
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return [float(x) for x in tensor.detach().cpu().tolist()]


def _unwrap_model(model):
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def _parse_target_lengths(raw: str) -> list[int] | None:
    normalized = str(raw).strip().lower()
    if normalized in {"", "full", "none", "null", "all"}:
        return None
    values = [int(chunk.strip()) for chunk in str(raw).split(",") if chunk.strip()]
    if any(value <= 0 for value in values):
        raise ValueError(f"target_lengths must be positive, got {values!r}")
    if not values:
        raise ValueError("target_lengths must contain at least one positive integer")
    return sorted(set(values))


def _serialize_target_lengths(target_lengths: list[int] | None) -> list[int] | None:
    return None if target_lengths is None else [int(x) for x in target_lengths]


def _parse_int_list(raw: str, *, expected_len: int | None = None) -> list[int]:
    values = [int(chunk.strip()) for chunk in str(raw).split(",") if chunk.strip()]
    if any(value <= 0 for value in values):
        raise ValueError(f"Expected positive integers, got {values!r} from {raw!r}")
    if expected_len is not None and len(values) != int(expected_len):
        raise ValueError(f"Expected {expected_len} integers, got {len(values)} from {raw!r}")
    if not values:
        raise ValueError(f"Expected at least one integer value, got {raw!r}")
    return values


def _parse_slot_names(raw: str) -> tuple[str, ...]:
    names = tuple(chunk.strip().lower() for chunk in str(raw).split(",") if chunk.strip())
    if not names:
        raise ValueError("slot_names must contain at least one non-empty value")
    if len(names) != len(set(names)):
        raise ValueError(f"slot_names must be unique, got {names!r}")
    expected = set(DEFAULT_TRIPLET_SLOT_NAMES)
    if set(names) != expected:
        raise ValueError(
            "slot_names must be a permutation of "
            f"{DEFAULT_TRIPLET_SLOT_NAMES}, got {names!r}."
        )
    return names


def _parse_train_slots(raw: str, *, slot_names: tuple[str, ...]) -> set[str]:
    value = str(raw).strip().lower()
    allowed = {str(slot_name) for slot_name in slot_names}
    if value in {"", "all", "full", "*"}:
        return allowed
    selected = {chunk.strip().lower() for chunk in str(raw).split(",") if chunk.strip()}
    unknown = sorted(selected.difference(allowed))
    if unknown:
        raise ValueError(f"Unknown train_slots {unknown!r}; expected subset of {slot_names!r} or all.")
    if not selected:
        raise ValueError("train_slots must be all or a non-empty comma-separated subset of slot_names.")
    return selected


def _resolve_slot_virtual_tokens(
    *,
    slot_lengths: list[int],
    slot_names: tuple[str, ...],
    num_virtual_tokens: int | None,
) -> tuple[int, list[int]]:
    for length in slot_lengths:
        if int(length) <= 0:
            raise ValueError(f"slot_lengths must be positive, got {slot_lengths!r}")
    if num_virtual_tokens is None:
        resolved = [int(x) for x in slot_lengths]
        return max(resolved), resolved
    shared = int(num_virtual_tokens)
    if shared <= 0:
        raise ValueError(f"num_virtual_tokens must be positive, got {shared}")
    return shared, [shared for _ in slot_names]


def _normalize_rows(rows: list[dict[str, Any]], max_train_samples: int | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        prompt = str(row.get("prompt", "")).strip()
        raw_sections = dict(row.get("safe_cot_sections", {}))
        explicit_sections: dict[str, str] = {}
        explicit_slot_fields_seen = False
        for slot_name in DEFAULT_TRIPLET_SLOT_NAMES:
            if str(slot_name) not in row:
                continue
            explicit_slot_fields_seen = True
            explicit_sections[str(slot_name)] = str(row.get(slot_name, "")).strip()
        section_source = "explicit_slot_fields" if explicit_slot_fields_seen else "safe_cot_sections"
        safe_cot_sections = explicit_sections if explicit_slot_fields_seen else dict(raw_sections)
        safe_cot = str(row.get("safe_cot", "")).strip()
        if explicit_slot_fields_seen:
            safe_cot = "\n".join(
                f"{str(slot_name).capitalize()}: {str(safe_cot_sections.get(slot_name, '')).strip()}"
                for slot_name in DEFAULT_TRIPLET_SLOT_NAMES
            ).strip()
        elif not safe_cot and safe_cot_sections:
            safe_cot = "\n".join(
                f"{str(slot_name).capitalize()}: {str(safe_cot_sections.get(slot_name, '')).strip()}"
                for slot_name in DEFAULT_TRIPLET_SLOT_NAMES
                if str(safe_cot_sections.get(slot_name, "")).strip()
            ).strip()
        if not prompt or not safe_cot:
            continue
        normalized.append(
            {
                "id": str(row.get("id", f"row-{idx}")),
                "prompt": prompt,
                "safe_cot": safe_cot,
                "safe_cot_sections": safe_cot_sections,
                "primary_category": str(row.get("primary_category", "")),
                "subgroup": str(row.get("subgroup", "")),
                "annotation_style": str(row.get("annotation_style", "")),
                "section_source": section_source,
                "split_group_id": str(row.get("split_group_id", row.get("source_id", row.get("id", f"row-{idx}")))),
            }
        )
        if max_train_samples is not None and len(normalized) >= int(max_train_samples):
            break
    if not normalized:
        raise ValueError("No usable rows found in the data file.")
    return normalized


def _split_indices_stratified(
    rows: list[dict[str, Any]],
    *,
    val_ratio: float,
    seed: int,
) -> dict[str, list[int]]:
    grouped: dict[str, dict[str, list[int]]] = {}
    for idx, row in enumerate(rows):
        category_key = str(row.get("primary_category") or row.get("subgroup") or "all")
        split_group_id = str(row.get("split_group_id") or row.get("id") or f"row-{idx}")
        grouped.setdefault(category_key, {}).setdefault(split_group_id, []).append(idx)

    rng = random.Random(int(seed))
    train_indices: list[int] = []
    val_indices: list[int] = []

    for _, group_map in sorted(grouped.items()):
        shuffled = list(group_map.keys())
        rng.shuffle(shuffled)
        if len(shuffled) == 1:
            train_indices.extend(group_map[shuffled[0]])
            continue
        val_count = max(1, int(round(len(shuffled) * float(val_ratio))))
        val_count = min(val_count, len(shuffled) - 1)
        for group_id in shuffled[:val_count]:
            val_indices.extend(group_map[group_id])
        for group_id in shuffled[val_count:]:
            train_indices.extend(group_map[group_id])

    if not train_indices:
        raise ValueError("Train split is empty after stratified split.")
    return {"train": sorted(train_indices), "val": sorted(val_indices)}


def _truncate_token_ids(token_ids: list[int], max_tokens: int) -> tuple[list[int], bool]:
    if max_tokens is None or len(token_ids) <= int(max_tokens):
        return token_ids, False
    return token_ids[-int(max_tokens) :], True


def _pick_eval_target(pool: list[list[int]], policy: str) -> list[int]:
    if len(pool) == 0:
        raise ValueError("target pool must be non-empty")
    if policy == "longest":
        return max(pool, key=len)
    if policy == "shortest":
        return min(pool, key=len)
    if policy == "middle":
        ordered = sorted(pool, key=len)
        return ordered[len(ordered) // 2]
    raise ValueError(f"Unsupported eval target policy: {policy}")


def _build_stage_target_suffix(
    *,
    slot_name: str,
    slot_names: tuple[str, ...],
    section_source: str = "",
) -> str:
    return "\n"


def _build_target_ids_with_suffix(
    *,
    tokenizer,
    target_text: str,
    suffix_text: str,
    max_target_tokens: int | None,
) -> list[int]:
    body_text = str(target_text).strip()
    suffix_text = str(suffix_text)
    suffix_ids = tokenizer(suffix_text, add_special_tokens=False)["input_ids"] if suffix_text else []
    if not body_text:
        if not suffix_ids:
            raise ValueError("target_text must be non-empty or have a non-empty suffix")
        target_ids = list(suffix_ids)
    else:
        suffix_clean = suffix_text.strip()
        if suffix_clean and body_text.endswith(suffix_clean):
            body_text = body_text[: -len(suffix_clean)].rstrip()
        body_ids = tokenizer(body_text, add_special_tokens=False)["input_ids"]
        target_ids = list(body_ids) + list(suffix_ids)

    if max_target_tokens is None or len(target_ids) <= int(max_target_tokens):
        return target_ids

    max_target_tokens = int(max_target_tokens)
    if max_target_tokens <= len(suffix_ids):
        raise ValueError(
            f"max_target_tokens={max_target_tokens} is too small to fit the required suffix of length {len(suffix_ids)}"
        )
    keep_body = max_target_tokens - len(suffix_ids)
    return list(body_ids[:keep_body]) + list(suffix_ids)


def _build_stage_target_pool(
    *,
    tokenizer,
    target_text: str,
    suffix_text: str,
    target_lengths: list[int] | None,
) -> list[list[int]]:
    clean_text = str(target_text).strip()
    target_pool: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    target_length_values = [None] if target_lengths is None else [int(length) for length in target_lengths]
    for length in target_length_values:
        target_ids = _build_target_ids_with_suffix(
            tokenizer=tokenizer,
            target_text=clean_text,
            suffix_text=suffix_text,
            max_target_tokens=None if length is None else int(length),
        )
        key = tuple(int(x) for x in target_ids)
        if key in seen:
            continue
        seen.add(key)
        target_pool.append(list(key))
    if not target_pool:
        raise ValueError("Failed to build a non-empty stage target pool.")
    return sorted(target_pool, key=len)


def _prepare_training_rows(
    *,
    rows: list[dict[str, Any]],
    tokenizer,
    system_prompt: str,
    prompt_format: str,
    decode_profile: str,
    slot_names: tuple[str, ...],
    train_slots: set[str] | None = None,
    target_lengths: list[int] | None,
    max_prompt_tokens: int,
    require_triplet_sections: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    active_train_slots = set(train_slots) if train_slots is not None else {str(slot_name) for slot_name in slot_names}
    prepared: list[dict[str, Any]] = []
    prompt_truncation_counts = {slot_name: 0 for slot_name in slot_names}
    parse_mode_counts: dict[str, int] = {}
    stage_example_counts = {slot_name: 0 for slot_name in slot_names}
    suffix_only_stage_examples = {slot_name: 0 for slot_name in slot_names}
    skipped_slot_examples = 0
    skipped_source_rows_without_stages = 0
    target_pool_sizes: list[int] = []

    for row in rows:
        base_spec = build_prompt_spec(
            tokenizer=tokenizer,
            raw_prompt=str(row["prompt"]),
            system_prompt=system_prompt,
            prompt_format=prompt_format,
            decode_profile=decode_profile,
        )
        section_texts, section_meta = extract_triplet_section_texts(
            safe_cot=str(row["safe_cot"]),
            safe_cot_sections=dict(row.get("safe_cot_sections", {})),
            slot_names=slot_names,
            require_triplet_sections=require_triplet_sections,
            strip_section_headers=True,
        )
        parse_mode = str(section_meta.get("parse_mode", "unknown"))
        parse_mode_counts[parse_mode] = parse_mode_counts.get(parse_mode, 0) + 1

        stage_payloads: dict[str, dict[str, Any]] = {}
        for slot_idx, slot_name in enumerate(slot_names):
            if str(slot_name) not in active_train_slots:
                continue
            slot_text = str(section_texts.get(slot_name, "")).strip()
            conditioning_slots = slot_names[:slot_idx]
            prompt_suffix_text = build_triplet_cascade_prompt_suffix(
                slot_name=slot_name,
                section_texts=section_texts,
                slot_names=slot_names,
            )
            stage_prompt_text = str(base_spec["prompt_text"]) + prompt_suffix_text
            prompt_ids = tokenizer(stage_prompt_text, add_special_tokens=False)["input_ids"]
            prompt_ids, prompt_truncated = _truncate_token_ids(prompt_ids, max_tokens=max_prompt_tokens)
            prompt_truncation_counts[slot_name] += int(prompt_truncated)

            target_suffix_text = _build_stage_target_suffix(
                slot_name=slot_name,
                slot_names=slot_names,
                section_source=str(row.get("section_source", "")),
            )
            if not slot_text and not str(target_suffix_text):
                skipped_slot_examples += 1
                continue
            target_pool = _build_stage_target_pool(
                tokenizer=tokenizer,
                target_text=slot_text,
                suffix_text=target_suffix_text,
                target_lengths=target_lengths,
            )
            target_pool_sizes.append(len(target_pool))
            stage_example_counts[slot_name] += 1
            if not slot_text and str(target_suffix_text):
                suffix_only_stage_examples[slot_name] += 1

            stage_payloads[slot_name] = {
                "slot_idx": int(slot_idx),
                "conditioning_slots": list(conditioning_slots),
                "conditioning_texts": {
                    prev_slot: str(section_texts.get(prev_slot, "")).strip() for prev_slot in conditioning_slots
                },
                "prompt_suffix_text": prompt_suffix_text,
                "prompt_text": stage_prompt_text,
                "prompt_ids": list(prompt_ids),
                "prompt_truncated": bool(prompt_truncated),
                "target_text": slot_text,
                "target_suffix_text": target_suffix_text,
                "target_pool_ids": target_pool,
                "target_pool_text": [
                    tokenizer.decode(target_ids, skip_special_tokens=False) for target_ids in target_pool
                ],
            }

        if len(stage_payloads) == 0:
            skipped_source_rows_without_stages += 1
            continue

        prepared.append(
            {
                **row,
                "parse_mode": parse_mode,
                "section_texts": {slot_name: str(section_texts.get(slot_name, "")).strip() for slot_name in slot_names},
                "stage_payloads": stage_payloads,
            }
        )

    metadata = {
        "cache_format_version": 13,
        "prompt_format": str(prompt_format),
        "decode_profile": str(decode_profile),
        "num_source_examples": int(len(rows)),
        "num_prepared_source_rows": int(len(prepared)),
        "slot_names": list(slot_names),
        "train_slots": sorted(active_train_slots),
        "stage_example_counts": {slot_name: int(count) for slot_name, count in stage_example_counts.items()},
        "suffix_only_stage_examples": {
            slot_name: int(count) for slot_name, count in suffix_only_stage_examples.items()
        },
        "skipped_slot_examples": int(skipped_slot_examples),
        "skipped_source_rows_without_stages": int(skipped_source_rows_without_stages),
        "prompt_truncation_counts": {slot_name: int(count) for slot_name, count in prompt_truncation_counts.items()},
        "target_lengths": _serialize_target_lengths(target_lengths),
        "target_boundary_text": "\\n",
        "section_source_counts": {
            source: sum(1 for row in rows if str(row.get("section_source", "")) == source)
            for source in sorted({str(row.get("section_source", "")) for row in rows})
        },
        "avg_target_pool_size": float(sum(target_pool_sizes) / max(1, len(target_pool_sizes))),
        "min_target_pool_size": int(min(target_pool_sizes)) if target_pool_sizes else 0,
        "max_target_pool_size": int(max(target_pool_sizes)) if target_pool_sizes else 0,
        "parse_mode_counts": parse_mode_counts,
        "slot_supervision_mode": "triplet_cascade_prefix_pt",
    }
    return prepared, metadata


class CascadeSectionPrefixDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        *,
        slot_name: str,
        train: bool,
        eval_target_policy: str,
    ) -> None:
        self.slot_name = str(slot_name)
        self.rows = [row for row in rows if self.slot_name in row.get("stage_payloads", {})]
        self.train = bool(train)
        self.eval_target_policy = str(eval_target_policy)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        stage_payload = row["stage_payloads"][self.slot_name]
        pool = stage_payload["target_pool_ids"]
        if self.train:
            target_ids = list(random.choice(pool))
        else:
            target_ids = list(_pick_eval_target(pool=pool, policy=self.eval_target_policy))
        return {
            "id": str(row["id"]),
            "prompt": str(row["prompt"]),
            "prompt_ids": list(stage_payload["prompt_ids"]),
            "target_ids": target_ids,
            "slot_name": self.slot_name,
            "slot_text": str(stage_payload["target_text"]),
            "target_suffix_text": str(stage_payload["target_suffix_text"]),
            "conditioning_slots": list(stage_payload["conditioning_slots"]),
            "conditioning_texts": dict(stage_payload["conditioning_texts"]),
            "primary_category": str(row["primary_category"]),
            "subgroup": str(row["subgroup"]),
            "annotation_style": str(row["annotation_style"]),
            "prompt_truncated": bool(stage_payload["prompt_truncated"]),
            "parse_mode": str(row["parse_mode"]),
            "target_pool_size": int(len(pool)),
        }


def _collate_batch(batch: list[dict[str, Any]], *, pad_token_id: int) -> dict[str, Any]:
    batch_size = len(batch)
    max_prompt_len = max(len(row["prompt_ids"]) for row in batch)
    max_target_len = max(len(row["target_ids"]) for row in batch)
    max_seq_len = max_prompt_len + max_target_len

    input_ids = torch.full((batch_size, max_seq_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    labels = torch.full((batch_size, max_seq_len), -100, dtype=torch.long)

    ids: list[str] = []
    prompts: list[str] = []
    slot_names: list[str] = []
    slot_texts: list[str] = []
    target_suffix_texts: list[str] = []
    conditioning_slots_list: list[list[str]] = []
    conditioning_texts_list: list[dict[str, str]] = []
    primary_categories: list[str] = []
    subgroups: list[str] = []
    parse_modes: list[str] = []
    target_lengths: list[int] = []
    target_pool_sizes: list[int] = []
    prompt_lengths: list[int] = []

    for row_idx, row in enumerate(batch):
        prompt_ids = list(row["prompt_ids"])
        target_ids = list(row["target_ids"])
        seq_ids = prompt_ids + target_ids
        seq_len = len(seq_ids)
        prompt_len = len(prompt_ids)

        input_ids[row_idx, :seq_len] = torch.tensor(seq_ids, dtype=torch.long)
        attention_mask[row_idx, :seq_len] = 1
        labels[row_idx, prompt_len:seq_len] = torch.tensor(target_ids, dtype=torch.long)

        ids.append(str(row["id"]))
        prompts.append(str(row["prompt"]))
        slot_names.append(str(row["slot_name"]))
        slot_texts.append(str(row["slot_text"]))
        target_suffix_texts.append(str(row["target_suffix_text"]))
        conditioning_slots_list.append(list(row["conditioning_slots"]))
        conditioning_texts_list.append(dict(row["conditioning_texts"]))
        primary_categories.append(str(row["primary_category"]))
        subgroups.append(str(row["subgroup"]))
        parse_modes.append(str(row["parse_mode"]))
        target_lengths.append(int(len(target_ids)))
        target_pool_sizes.append(int(row["target_pool_size"]))
        prompt_lengths.append(int(prompt_len))

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "ids": ids,
        "prompts": prompts,
        "slot_names": slot_names,
        "slot_texts": slot_texts,
        "target_suffix_texts": target_suffix_texts,
        "conditioning_slots_list": conditioning_slots_list,
        "conditioning_texts_list": conditioning_texts_list,
        "primary_categories": primary_categories,
        "subgroups": subgroups,
        "parse_modes": parse_modes,
        "target_lengths": target_lengths,
        "target_pool_sizes": target_pool_sizes,
        "prompt_lengths": prompt_lengths,
    }


def _run_step(
    *,
    model,
    batch: dict[str, Any],
    input_device: torch.device,
) -> dict[str, Any]:
    labels = batch["labels"].to(input_device)
    min_prompt_len = max(1, min(int(x) for x in batch.get("prompt_lengths", [1])))
    earliest_supervised_logit_index = int(min_prompt_len - 1)
    logits_to_keep = int(batch["input_ids"].shape[1] - earliest_supervised_logit_index)
    outputs = model(
        input_ids=batch["input_ids"].to(input_device),
        attention_mask=batch["attention_mask"].to(input_device),
        use_cache=False,
        return_dict=True,
        logits_to_keep=logits_to_keep,
    )
    shifted_labels = F.pad(labels, (0, 1), value=-100)[..., 1:]
    shifted_labels = shifted_labels[..., -logits_to_keep:].contiguous()
    loss = F.cross_entropy(
        outputs.logits.float().reshape(-1, int(outputs.logits.shape[-1])),
        shifted_labels.view(-1),
        ignore_index=-100,
    )
    return {
        "loss": loss,
        "lm_loss": loss.detach(),
        "total_loss": loss.detach(),
    }


def _evaluate(
    *,
    model,
    data_loader,
    input_device: torch.device,
    distributed_ctx: dict[str, Any] | None = None,
) -> dict[str, float]:
    model.eval()
    total_examples = 0
    total_loss = 0.0
    total_lm_loss = 0.0

    with torch.no_grad():
        for batch in data_loader:
            step = _run_step(
                model=model,
                batch=batch,
                input_device=input_device,
            )
            batch_size = int(batch["input_ids"].shape[0])
            total_examples += batch_size
            total_loss += float(step["total_loss"].item()) * batch_size
            total_lm_loss += float(step["lm_loss"].item()) * batch_size

    totals = _distributed_sum_values(
        distributed_ctx,
        [
            float(total_examples),
            float(total_loss),
            float(total_lm_loss),
        ],
        device=input_device,
    )
    total_examples = int(totals[0])
    total_loss = float(totals[1])
    total_lm_loss = float(totals[2])

    if total_examples == 0:
        return {
            "num_examples": 0,
            "lm_loss": float("nan"),
            "total_loss": float("nan"),
        }

    avg_loss = float(total_loss / total_examples)
    return {
        "num_examples": int(total_examples),
        "lm_loss": float(total_lm_loss / total_examples),
        "total_loss": avg_loss,
    }


def _train_epoch(
    *,
    model,
    trainable_parameters: list[torch.nn.Parameter],
    data_loader,
    optimizer,
    scheduler,
    grad_accum: int,
    grad_clip: float,
    input_device: torch.device,
    distributed_ctx: dict[str, Any] | None = None,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_examples = 0
    total_loss = 0.0
    total_lm_loss = 0.0

    iterator = tqdm(
        data_loader,
        desc="Training epoch",
        leave=False,
        disable=not _is_main_process(distributed_ctx),
    )
    for step_idx, batch in enumerate(iterator, start=1):
        should_sync = step_idx % int(max(1, grad_accum)) == 0 or step_idx == len(data_loader)
        sync_context = (
            model.no_sync()
            if isinstance(model, DistributedDataParallel) and not should_sync
            else contextlib.nullcontext()
        )
        with sync_context:
            step = _run_step(
                model=model,
                batch=batch,
                input_device=input_device,
            )
            (step["loss"] / float(max(1, grad_accum))).backward()

        if should_sync:
            torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=float(grad_clip))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        batch_size = int(batch["input_ids"].shape[0])
        total_examples += batch_size
        total_loss += float(step["total_loss"].item()) * batch_size
        total_lm_loss += float(step["lm_loss"].item()) * batch_size

    totals = _distributed_sum_values(
        distributed_ctx,
        [
            float(total_examples),
            float(total_loss),
            float(total_lm_loss),
        ],
        device=input_device,
    )
    total_examples = int(totals[0])
    total_loss = float(totals[1])
    total_lm_loss = float(totals[2])

    return {
        "num_examples": int(total_examples),
        "lm_loss": float(total_lm_loss / max(1, total_examples)),
        "total_loss": float(total_loss / max(1, total_examples)),
    }


def _resolve_device_map_arg(raw_value: str) -> str | None:
    value = str(raw_value).strip().lower()
    if value in {"", "none", "null"}:
        return None
    return raw_value


def _count_parameters(model) -> dict[str, int]:
    total = 0
    trainable = 0
    for param in model.parameters():
        cur = int(param.numel())
        total += cur
        if param.requires_grad:
            trainable += cur
    return {
        "total": int(total),
        "trainable": int(trainable),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train three official-PEFT Prefix-Tuning adapters for safe CoT triplets using true cascade section "
            "training with targets P(i|p), P(r|p,i), P(d|p,i,r)."
        )
    )
    parser.add_argument("--model_name", type=str, default="simplescaling/s1.1-7B")
    parser.add_argument("--config_model_name", type=str, default=None)
    parser.add_argument(
        "--data_file",
        type=str,
        default=str(DATA_ROOT / "wildjailbreak_safe_cot_triplet_curated_1k.line_pool_v1.jsonl"),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(PT_OUTPUT_ROOT / "wj_triplet_cascade_prefix_safe_cot"),
    )
    parser.add_argument(
        "--target_lengths",
        type=str,
        default="full",
        help="Comma-separated target lengths, or full to supervise each section without truncation.",
    )
    parser.add_argument(
        "--eval_target_policy",
        type=str,
        default="longest",
        choices=["longest", "shortest", "middle"],
    )
    parser.add_argument("--max_prompt_tokens", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument(
        "--save_epoch_adapters",
        action="store_true",
        help="Save an adapter snapshot after every epoch under epoch_adapters/epoch_XXXX.",
    )
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device_map", type=str, default="none")
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--slot_names", type=str, default="intent,risk,decision")
    parser.add_argument(
        "--train_slots",
        type=str,
        default="all",
        help="Train stage slots: all/intent,risk,decision, risk,decision, or decision.",
    )
    parser.add_argument(
        "--allow_missing_skipped_slots",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--slot_lengths",
        type=str,
        default="16,16,16",
        help="Per-slot virtual token counts, e.g. intent,risk,decision = 4,16,8.",
    )
    parser.add_argument(
        "--num_virtual_tokens",
        type=int,
        default=None,
        help="Shared virtual token count used by each cascade adapter. If set, overrides --slot_lengths.",
    )
    parser.add_argument(
        "--require_triplet_sections",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require recoverable Intent/Risk/Decision spans in each safe CoT.",
    )
    parser.add_argument(
        "--prefix_projection",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the standard PEFT Prefix-Tuning MLP projection.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable transformer gradient checkpointing to reduce activation memory at the cost of speed.",
    )
    parser.add_argument("--encoder_hidden_size", type=int, default=512)
    parser.add_argument("--intent_tag_safe", type=str, default="<safe_intent>")
    parser.add_argument("--intent_tag_harmful", type=str, default="<harmful_intent>")
    parser.add_argument(
        "--add_intent_special_tokens",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Register intent tags and extra markers as additional tokenizer special tokens before training.",
    )
    parser.add_argument(
        "--special_token_extra_markers",
        type=str,
        default="",
        help="Comma-separated extra markers to register when --add_intent_special_tokens is enabled, e.g. </think>.",
    )
    parser.add_argument(
        "--train_special_token_rows",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Train only the newly added special-token rows in input/output embeddings alongside prefix tuning.",
    )
    parser.add_argument("--trust_remote_code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local_files_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force_recompute_cache", action="store_true")
    parser.add_argument("--preview_rows", type=int, default=0)
    return parser


def _train_single_slot_adapter(
    *,
    args,
    tokenizer,
    system_prompt: str,
    prompt_format: str,
    decode_profile: str,
    slot_name: str,
    slot_index: int,
    prepared_rows: list[dict[str, Any]],
    split: dict[str, list[int]],
    output_dir: Path,
    slot_num_virtual_tokens: int,
    target_lengths: list[int] | None,
    prepare_meta: dict[str, Any],
    slot_names: tuple[str, ...],
    special_token_old_piece_ids: dict[str, list[int]] | None = None,
    special_token_summary: dict[str, Any] | None = None,
    distributed_ctx: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rank_offset = int(distributed_ctx.get("rank", 0)) if distributed_ctx else 0
    set_seed(int(args.seed) + int(slot_index) + rank_offset)

    slot_output_dir = output_dir / str(slot_name)
    slot_output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = [prepared_rows[idx] for idx in split["train"] if slot_name in prepared_rows[idx]["stage_payloads"]]
    val_rows = [prepared_rows[idx] for idx in split["val"] if slot_name in prepared_rows[idx]["stage_payloads"]]
    if not train_rows:
        raise ValueError(f"No training rows available for slot {slot_name!r}.")

    train_dataset = CascadeSectionPrefixDataset(
        train_rows,
        slot_name=slot_name,
        train=True,
        eval_target_policy=args.eval_target_policy,
    )
    val_dataset = CascadeSectionPrefixDataset(
        val_rows,
        slot_name=slot_name,
        train=False,
        eval_target_policy=args.eval_target_policy,
    )
    collate_fn = lambda batch: _collate_batch(batch=batch, pad_token_id=int(tokenizer.pad_token_id))
    train_sampler = None
    if distributed_ctx and bool(distributed_ctx.get("enabled", False)):
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=int(distributed_ctx["world_size"]),
            rank=int(distributed_ctx["rank"]),
            shuffle=True,
            seed=int(args.seed) + int(slot_index),
            drop_last=False,
        )
        val_indices = list(range(len(val_dataset)))[int(distributed_ctx["rank"]) :: int(distributed_ctx["world_size"])]
        val_dataset = Subset(val_dataset, val_indices)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

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
        args.model_name,
        **load_kwargs,
    )
    if bool(args.add_intent_special_tokens) and len(tokenizer) != int(base_model.get_input_embeddings().num_embeddings):
        base_model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
        _init_added_token_rows_from_old_pieces(base_model, tokenizer, special_token_old_piece_ids or {})
    trainable_special_token_rows = (
        _attach_trainable_special_token_rows(
            base_model,
            tokenizer,
            list((special_token_old_piece_ids or {}).keys()),
        )
        if bool(args.train_special_token_rows)
        else {"enabled": False}
    )
    if device_map is None:
        if distributed_ctx and bool(distributed_ctx.get("enabled", False)) and torch.cuda.is_available():
            device = torch.device(f"cuda:{int(distributed_ctx['local_rank'])}")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        base_model.to(device)

    prefix_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=int(slot_num_virtual_tokens),
        prefix_projection=bool(args.prefix_projection),
        encoder_hidden_size=int(args.encoder_hidden_size),
    )
    model = get_peft_model(base_model, prefix_config)
    model.config.use_cache = False
    trainable_special_token_row_counts = _reenable_trainable_special_token_rows(trainable_special_token_rows)
    if bool(args.gradient_checkpointing):
        try:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    if distributed_ctx and bool(distributed_ctx.get("enabled", False)):
        if device_map is not None:
            raise ValueError("DDP training requires --device_map none so each rank owns one full model replica.")
        if not torch.cuda.is_available():
            raise ValueError("DDP training for this script requires CUDA.")
        model = DistributedDataParallel(
            model,
            device_ids=[int(distributed_ctx["local_rank"])],
            output_device=int(distributed_ctx["local_rank"]),
            find_unused_parameters=False,
        )

    unwrapped_model = _unwrap_model(model)
    input_device = resolve_model_input_device(unwrapped_model)
    parameter_counts = _count_parameters(unwrapped_model)
    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    if bool(trainable_special_token_rows.get("enabled")):
        for module_key in ("input_module", "output_module"):
            module = trainable_special_token_rows.get(module_key)
            if module is not None:
                for param in module.parameters():
                    if param.requires_grad and not any(param is existing for existing in trainable_parameters):
                        trainable_parameters.append(param)
    if len(trainable_parameters) == 0:
        raise ValueError(f"No trainable parameters found after applying Prefix-Tuning for slot {slot_name!r}.")

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = max(1, math.ceil(len(train_loader) / max(1, args.grad_accum)))
    total_steps = max(1, updates_per_epoch * int(args.epochs))
    warmup_steps = int(round(total_steps * float(args.warmup_ratio)))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    conditioning_slots = list(slot_names[: slot_names.index(slot_name)])
    target_suffix_text = _build_stage_target_suffix(
        slot_name=slot_name,
        slot_names=slot_names,
        section_source=str(train_rows[0].get("section_source", "")) if train_rows else "",
    )
    training_config = {
        "method": "official_peft_triplet_cascade_prefix_tuning",
        "target_section": str(slot_name),
        "conditioning_slots": conditioning_slots,
        "model_name": str(args.model_name),
        "config_model_name": args.config_model_name,
        "prompt_format": str(prompt_format),
        "decode_profile": str(decode_profile),
        "system_prompt": system_prompt,
        "data_file": str(args.data_file),
        "target_lengths": _serialize_target_lengths(target_lengths),
        "target_boundary_text": "\\n",
        "eval_target_policy": str(args.eval_target_policy),
        "max_prompt_tokens": int(args.max_prompt_tokens),
        "num_virtual_tokens": int(slot_num_virtual_tokens),
        "prefix_projection": bool(args.prefix_projection),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "encoder_hidden_size": int(args.encoder_hidden_size),
        "train_size": int(len(train_rows)),
        "val_size": int(len(val_rows)),
        "seed": int(args.seed) + int(slot_index),
        "dtype": str(args.dtype),
        "device_map": device_map,
        "attn_implementation": str(args.attn_implementation),
        "parameter_counts": parameter_counts,
        "prepare_meta": prepare_meta,
        "slot_supervision_mode": "cascade_section_prefix_only",
        "prompt_suffix_template": build_triplet_cascade_prompt_suffix(
            slot_name=slot_name,
            section_texts={prev_slot: f"<{prev_slot}>" for prev_slot in conditioning_slots},
            slot_names=slot_names,
        ),
        "target_suffix_text": target_suffix_text,
        "answer_supervision": False,
    }
    if _is_main_process(distributed_ctx):
        write_json(slot_output_dir / "training_config.json", training_config)

    log_path = slot_output_dir / "train_log.jsonl"
    train_log_rows: list[dict[str, Any]] = []
    best_metric = float("inf")
    best_epoch = -1
    best_val_metrics: dict[str, Any] | None = None
    patience_counter = 0

    for epoch in range(1, int(args.epochs) + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(int(epoch))
        train_metrics = _train_epoch(
            model=model,
            trainable_parameters=trainable_parameters,
            data_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_accum=args.grad_accum,
            grad_clip=args.grad_clip,
            input_device=input_device,
            distributed_ctx=distributed_ctx,
        )

        if len(val_rows) > 0:
            val_metrics = _evaluate(
                model=model,
                data_loader=val_loader,
                input_device=input_device,
                distributed_ctx=distributed_ctx,
            )
            monitor_metric = float(val_metrics["total_loss"])
        else:
            val_metrics = {
                "num_examples": 0,
                "lm_loss": float("nan"),
                "total_loss": float("nan"),
            }
            monitor_metric = float(train_metrics["total_loss"])

        log_row = {
            "epoch": int(epoch),
            "train": train_metrics,
            "val": val_metrics,
            "lr": float(scheduler.get_last_lr()[0]),
        }
        if _is_main_process(distributed_ctx):
            train_log_rows.append(log_row)
            write_jsonl(log_path, train_log_rows)

        if monitor_metric < best_metric:
            best_metric = monitor_metric
            best_epoch = int(epoch)
            best_val_metrics = val_metrics
            patience_counter = 0
            if _is_main_process(distributed_ctx):
                adapter_dir = slot_output_dir / "best_adapter"
                adapter_dir.mkdir(parents=True, exist_ok=True)
                save_kwargs = {"save_embedding_layers": False} if bool(args.add_intent_special_tokens) else {}
                _unwrap_model(model).save_pretrained(adapter_dir, **save_kwargs)
                _save_trainable_special_token_rows(adapter_dir, trainable_special_token_rows, tokenizer)
                tokenizer.save_pretrained(adapter_dir)
        else:
            patience_counter += 1

        if _is_main_process(distributed_ctx) and bool(args.save_epoch_adapters):
            epoch_adapter_dir = slot_output_dir / "epoch_adapters" / f"epoch_{int(epoch):04d}"
            epoch_adapter_dir.mkdir(parents=True, exist_ok=True)
            save_kwargs = {"save_embedding_layers": False} if bool(args.add_intent_special_tokens) else {}
            _unwrap_model(model).save_pretrained(epoch_adapter_dir, **save_kwargs)
            _save_trainable_special_token_rows(epoch_adapter_dir, trainable_special_token_rows, tokenizer)
            tokenizer.save_pretrained(epoch_adapter_dir)

        _distributed_barrier(distributed_ctx)
        if patience_counter >= int(args.patience):
            break

    if best_val_metrics is None:
        best_val_metrics = {
            "num_examples": 0,
            "lm_loss": float("nan"),
            "total_loss": float("nan"),
        }

    active_peft_config = _unwrap_model(model).active_peft_config
    num_layers = int(active_peft_config.num_layers)
    num_attention_heads = int(active_peft_config.num_attention_heads)
    token_dim = int(active_peft_config.token_dim)
    num_transformer_submodules = int(active_peft_config.num_transformer_submodules)

    summary = {
        "target_section": str(slot_name),
        "conditioning_slots": conditioning_slots,
        "best_epoch": int(best_epoch),
        "best_metric": float(best_metric),
        "best_val_metrics": best_val_metrics,
        "num_train_rows": int(len(train_rows)),
        "num_val_rows": int(len(val_rows)),
        "num_source_rows": int(len(prepared_rows)),
        "num_virtual_tokens": int(slot_num_virtual_tokens),
        "num_layers": num_layers,
        "num_attention_heads": num_attention_heads,
        "token_dim": token_dim,
        "num_transformer_submodules": num_transformer_submodules,
        "adapter_dir": str((slot_output_dir / "best_adapter").resolve()),
        "target_suffix_text": target_suffix_text,
        "special_tokens": dict(special_token_summary or {}),
        "train_special_token_rows": {
            "enabled": bool(trainable_special_token_rows.get("enabled", False)),
            "token_ids": [int(x) for x in trainable_special_token_rows.get("token_ids", [])],
            **dict(trainable_special_token_row_counts),
        },
    }
    if _is_main_process(distributed_ctx):
        write_json(slot_output_dir / "summary.json", summary)
    _distributed_barrier(distributed_ctx)

    del optimizer
    del scheduler
    del trainable_parameters
    del model
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summary


def _train_cascade_triplet_adapters(
    *,
    args,
    tokenizer,
    system_prompt: str,
    prompt_format: str,
    decode_profile: str,
    prepared_rows: list[dict[str, Any]],
    split: dict[str, list[int]],
    output_dir: Path,
    slot_lengths: list[int],
    target_lengths: list[int] | None,
    prepare_meta: dict[str, Any],
    slot_names: tuple[str, ...],
    special_token_old_piece_ids: dict[str, list[int]] | None = None,
    special_token_summary: dict[str, Any] | None = None,
    distributed_ctx: dict[str, Any] | None = None,
) -> dict[str, Any]:
    slot_summaries: dict[str, Any] = {}
    train_slots = _parse_train_slots(args.train_slots, slot_names=slot_names)
    for slot_index, slot_name in enumerate(slot_names):
        if str(slot_name) not in train_slots:
            summary_path = output_dir / str(slot_name) / "summary.json"
            adapter_path = output_dir / str(slot_name) / "best_adapter"
            if not summary_path.exists() or not adapter_path.exists():
                if bool(getattr(args, "allow_missing_skipped_slots", False)):
                    slot_summaries[str(slot_name)] = {
                        "target_section": str(slot_name),
                        "skipped_existing_adapter": True,
                        "missing_skipped_adapter_allowed": True,
                        "adapter_dir": str(adapter_path.resolve()),
                    }
                    continue
                raise ValueError(
                    f"Skipping slot {slot_name!r} requires existing {summary_path} and {adapter_path}."
                )
            with summary_path.open("r", encoding="utf-8") as rf:
                existing_summary = json.load(rf)
            existing_summary["skipped_existing_adapter"] = True
            slot_summaries[str(slot_name)] = existing_summary
            continue
        slot_summaries[str(slot_name)] = _train_single_slot_adapter(
            args=args,
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            prompt_format=prompt_format,
            decode_profile=decode_profile,
            slot_name=str(slot_name),
            slot_index=int(slot_index),
            prepared_rows=prepared_rows,
            split=split,
            output_dir=output_dir,
            slot_num_virtual_tokens=int(slot_lengths[int(slot_index)]),
            target_lengths=target_lengths,
            prepare_meta=prepare_meta,
            slot_names=slot_names,
            special_token_old_piece_ids=special_token_old_piece_ids,
            special_token_summary=special_token_summary,
            distributed_ctx=distributed_ctx,
        )

    return {
        "training_mode": "cascade_section_prefix_only",
        "slot_summaries": slot_summaries,
    }


def main() -> None:
    args = build_arg_parser().parse_args()
    distributed_ctx = _init_distributed_from_env()
    set_seed(int(args.seed) + int(distributed_ctx.get("rank", 0)))

    slot_names = _parse_slot_names(args.slot_names)
    parsed_slot_lengths = _parse_int_list(args.slot_lengths, expected_len=len(slot_names))
    shared_num_virtual_tokens, slot_lengths = _resolve_slot_virtual_tokens(
        slot_lengths=parsed_slot_lengths,
        slot_names=slot_names,
        num_virtual_tokens=args.num_virtual_tokens,
    )
    target_lengths = _parse_target_lengths(args.target_lengths)
    if args.local_files_only:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    output_dir = Path(args.output_dir)
    if _is_main_process(distributed_ctx):
        output_dir.mkdir(parents=True, exist_ok=True)
    _distributed_barrier(distributed_ctx)

    model_cfg = load_model_config(
        model_name=args.model_name,
        config_model_name=args.config_model_name,
    )
    system_prompt = str(model_cfg.get("system_prompt", ""))

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    special_token_markers: list[str] = []
    special_token_old_piece_ids: dict[str, list[int]] = {}
    special_token_summary: dict[str, Any] = {"enabled": False}
    if bool(args.add_intent_special_tokens):
        special_token_markers = _special_token_markers_for_training(args, slot_names=slot_names)
        special_token_old_piece_ids = {
            marker: list(tokenizer(marker, add_special_tokens=False)["input_ids"])
            for marker in special_token_markers
        }
        special_token_summary = {
            "enabled": True,
            **_register_additional_special_tokens(tokenizer, special_token_markers),
            "old_piece_ids": {
                str(marker): [int(x) for x in ids] for marker, ids in special_token_old_piece_ids.items()
            },
        }
    prompt_format = resolve_prompt_format(tokenizer, prompt_format=model_cfg.get("prompt_format", None))
    decode_profile = resolve_decode_profile(model_cfg.get("decode_profile", None))
    if decode_profile == pt_utils.AUTO_DECODE_PROFILE:
        decode_profile = pt_utils.default_decode_profile_for_prompt_format(prompt_format)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has neither pad_token_id nor eos_token_id.")
        tokenizer.pad_token = tokenizer.eos_token

    train_slots = _parse_train_slots(args.train_slots, slot_names=slot_names)
    prepared_cache_path = output_dir / "prepared_rows.pt"
    cache_signature = {
        "cache_format_version": 13,
        "data_file": str(args.data_file),
        "model_name": str(args.model_name),
        "config_model_name": str(args.config_model_name),
        "prompt_format": str(prompt_format),
        "decode_profile": str(decode_profile),
        "max_prompt_tokens": int(args.max_prompt_tokens),
        "target_lengths": _serialize_target_lengths(target_lengths),
        "target_boundary_text": "\\n",
        "slot_names": list(slot_names),
        "train_slots": sorted(train_slots),
        "require_triplet_sections": bool(args.require_triplet_sections),
        "add_intent_special_tokens": bool(args.add_intent_special_tokens),
        "special_token_markers": list(special_token_markers),
        "train_special_token_rows": bool(args.train_special_token_rows),
        "tokenizer_vocab_size": int(len(tokenizer)),
        "training_mode": "cascade_section_prefix_only",
    }

    prepared_rows: list[dict[str, Any]] | None = None
    prepare_meta: dict[str, Any] | None = None
    if prepared_cache_path.exists() and not args.force_recompute_cache:
        cache_payload = torch.load(prepared_cache_path, map_location="cpu", weights_only=False)
        if cache_payload.get("signature") == cache_signature:
            prepared_rows = cache_payload["rows"]
            prepare_meta = cache_payload["meta"]

    if prepared_rows is None:
        if _is_main_process(distributed_ctx):
            print(f"[prep] building prepared_rows cache from {args.data_file}", flush=True)
            raw_rows = _normalize_rows(
                read_jsonl(args.data_file),
                max_train_samples=args.max_train_samples,
            )
            prepared_rows, prepare_meta = _prepare_training_rows(
                rows=raw_rows,
                tokenizer=tokenizer,
                system_prompt=system_prompt,
                prompt_format=prompt_format,
                decode_profile=decode_profile,
                slot_names=slot_names,
                train_slots=train_slots,
                target_lengths=target_lengths,
                max_prompt_tokens=args.max_prompt_tokens,
                require_triplet_sections=bool(args.require_triplet_sections),
            )
            torch.save(
                {
                    "signature": cache_signature,
                    "rows": prepared_rows,
                    "meta": prepare_meta,
                },
                prepared_cache_path,
            )
            print(f"[prep] saved prepared_rows cache -> {prepared_cache_path}", flush=True)
        _distributed_barrier(distributed_ctx)

    if prepared_rows is None:
        cache_payload = torch.load(prepared_cache_path, map_location="cpu", weights_only=False)
        if cache_payload.get("signature") != cache_signature:
            raise ValueError(f"Prepared rows cache signature mismatch for {prepared_cache_path}")
        prepared_rows = cache_payload["rows"]
        prepare_meta = cache_payload["meta"]

    assert prepare_meta is not None

    if not prepared_rows:
        raise ValueError("No prepared rows were created for cascade_section_prefix_only training.")

    if int(args.preview_rows) > 0:
        preview_rows: list[dict[str, Any]] = []
        for row in prepared_rows[: int(args.preview_rows)]:
            preview_row = {
                "id": row["id"],
                "primary_category": row["primary_category"],
                "subgroup": row["subgroup"],
                "parse_mode": row["parse_mode"],
                "full_triplet_cot": compose_triplet_cot_text(section_texts=row["section_texts"], slot_names=slot_names),
                "section_texts": dict(row["section_texts"]),
            }
            preview_row["stage_payloads"] = {
                str(slot_name): {
                    "target_text": payload["target_text"],
                    "target_suffix_text": payload["target_suffix_text"],
                    "prompt_truncated": bool(payload["prompt_truncated"]),
                    "conditioning_slots": list(payload["conditioning_slots"]),
                    "target_pool_text": payload["target_pool_text"],
                }
                for slot_name, payload in row["stage_payloads"].items()
            }
            preview_rows.append(preview_row)
        if _is_main_process(distributed_ctx):
            write_jsonl(output_dir / "prepared_preview.jsonl", preview_rows)

    split = _split_indices_stratified(
        rows=prepared_rows,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    if _is_main_process(distributed_ctx):
        write_json(
            output_dir / "split_summary.json",
            {
                "train_source_rows": int(len(split["train"])),
                "val_source_rows": int(len(split["val"])),
            },
        )

    cascade_training_summary = _train_cascade_triplet_adapters(
        args=args,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        prompt_format=prompt_format,
        decode_profile=decode_profile,
        prepared_rows=prepared_rows,
        split=split,
        output_dir=output_dir,
        slot_lengths=slot_lengths,
        target_lengths=target_lengths,
        prepare_meta=prepare_meta,
        slot_names=slot_names,
        special_token_old_piece_ids=special_token_old_piece_ids,
        special_token_summary=special_token_summary,
        distributed_ctx=distributed_ctx,
    )
    slot_summaries = dict(cascade_training_summary["slot_summaries"])

    top_level_training_config = {
        "method": "official_peft_triplet_cascade_prefix_tuning",
        "training_mode": "cascade_section_prefix_only",
        "routing_mode": None,
        "model_name": str(args.model_name),
        "config_model_name": args.config_model_name,
        "prompt_format": str(prompt_format),
        "decode_profile": str(decode_profile),
        "system_prompt": system_prompt,
        "data_file": str(args.data_file),
        "target_lengths": _serialize_target_lengths(target_lengths),
        "target_boundary_text": "\\n",
        "eval_target_policy": str(args.eval_target_policy),
        "max_prompt_tokens": int(args.max_prompt_tokens),
        "slot_names": list(slot_names),
        "slot_lengths": [int(x) for x in slot_lengths],
        "num_virtual_tokens": int(shared_num_virtual_tokens),
        "prefix_projection": bool(args.prefix_projection),
        "special_tokens": dict(special_token_summary),
        "train_special_token_rows": bool(args.train_special_token_rows),
        "gradient_checkpointing": bool(args.gradient_checkpointing),
        "encoder_hidden_size": int(args.encoder_hidden_size),
        "require_triplet_sections": bool(args.require_triplet_sections),
        "seed": int(args.seed),
        "dtype": str(args.dtype),
        "device_map": _resolve_device_map_arg(args.device_map),
        "attn_implementation": str(args.attn_implementation),
        "prepare_meta": prepare_meta,
        "safe_cot_supervision_scope": "cascade_sections_only",
        "answer_supervision": False,
        "intent_tag_safe": str(args.intent_tag_safe),
        "intent_tag_harmful": str(args.intent_tag_harmful),
        "recommended_triplet_inference_mode": "cascade_section_decode",
        "slot_summaries": slot_summaries,
    }
    adapter_manifest = {
        "mode": "cascade_section_decode",
        "training_mode": "cascade_section_prefix_only",
        "routing_mode": None,
        "model_name": str(args.model_name),
        "config_model_name": args.config_model_name,
        "prompt_format": str(prompt_format),
        "decode_profile": str(decode_profile),
        "slot_names": list(slot_names),
        "intent_adapter_dir": str((output_dir / "intent" / "best_adapter").resolve()),
        "risk_adapter_dir": str((output_dir / "risk" / "best_adapter").resolve()),
        "decision_adapter_dir": str((output_dir / "decision" / "best_adapter").resolve()),
        "target_boundary_text": "\\n",
        "eval_script": str(LEGACY_EVAL_SCRIPT.resolve()),
    }
    summary_payload = {
        "training_mode": "cascade_section_prefix_only",
        "routing_mode": None,
        "prompt_format": str(prompt_format),
        "decode_profile": str(decode_profile),
        "slot_summaries": slot_summaries,
        "num_source_rows": int(prepare_meta.get("num_source_examples", len(prepared_rows))),
        "num_prepared_source_rows": int(len(prepared_rows)),
        "train_source_rows": int(len(split["train"])),
        "val_source_rows": int(len(split["val"])),
        "slot_names": list(slot_names),
        "slot_lengths": [int(x) for x in slot_lengths],
        "num_virtual_tokens": int(shared_num_virtual_tokens),
        "triplet_inference_config": str((output_dir / "triplet_inference_config.json").resolve()),
    }

    if _is_main_process(distributed_ctx):
        write_json(output_dir / "training_config.json", top_level_training_config)
        write_json(output_dir / "triplet_inference_config.json", adapter_manifest)
        write_json(output_dir / "summary.json", summary_payload)
    _distributed_barrier(distributed_ctx)
    if bool(distributed_ctx.get("enabled", False)) and dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
