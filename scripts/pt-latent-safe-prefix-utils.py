from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn


MODEL_CONFIG_PATH = Path(__file__).resolve().with_name("pt-model-configs.json")
QWEN_CHATML_PROMPT_FORMAT = "qwen_chatml"
LLAMA3_PROMPT_FORMAT = "llama3"
DEEPSEEK_R1_PROMPT_FORMAT = "deepseek_r1"
AUTO_PROMPT_FORMAT = "auto"
S1_DECODE_PROFILE = "s1"
XML_COT_DECODE_PROFILE = "xml_cot"
AUTO_DECODE_PROFILE = "auto"
QWEN_ASSISTANT_PREFIX_TEXT = "<|im_start|>assistant\n"
QWEN_THINK_PREFIX_TEXT = "<|im_start|>think\n"
XML_THINK_PREFIX_TEXT = "<think>\n"
LLAMA3_ASSISTANT_PREFIX_TEXT = "<|start_header_id|>assistant<|end_header_id|>\n\n"
LLAMA3_THINK_PREFIX_TEXT = XML_THINK_PREFIX_TEXT
THINK_START_TEXT = QWEN_ASSISTANT_PREFIX_TEXT + QWEN_THINK_PREFIX_TEXT
THINK_END_TEXT = "</think>"

DEFAULT_TRIPLET_SLOT_NAMES = ("intent", "risk", "decision")
TRIPLET_SLOT_HEADERS = {
    "intent": "Intent",
    "risk": "Risk",
    "decision": "Decision",
}


def build_slot_index(slot_names: Sequence[str]) -> dict[str, int]:
    return {str(name): idx for idx, name in enumerate(slot_names)}


def normalize_triplet_section_name(raw: str) -> str:
    value = str(raw).strip().lower()
    if value not in {"full", *DEFAULT_TRIPLET_SLOT_NAMES}:
        raise ValueError(f"Unsupported target section {raw!r}. Expected one of: full, intent, risk, decision.")
    return value


def _strip_section_header(segment_text: str, slot_name: str) -> str:
    text = str(segment_text)
    header = TRIPLET_SLOT_HEADERS.get(str(slot_name).strip().lower())
    return re.sub(rf"^\s*{header}:\s*", "", text, count=1) if header else text


def _extract_triplet_segments(
    *,
    safe_cot: str,
    safe_cot_sections: dict[str, Any],
    slot_names: Sequence[str],
    require_triplet_sections: bool,
) -> tuple[list[tuple[str, int]], str]:
    slot_label_to_index = build_slot_index(slot_names)
    text = str(safe_cot).strip()

    risk_marker = "\nRisk:"
    decision_marker = "\nDecision:"
    if text.startswith("Intent:") and risk_marker in text and decision_marker in text:
        risk_idx = text.find(risk_marker)
        decision_idx = text.find(decision_marker)
        segments = [
            (text[: risk_idx + 1], slot_label_to_index["intent"]),
            (text[risk_idx + 1 : decision_idx + 1], slot_label_to_index["risk"]),
            (text[decision_idx + 1 :], slot_label_to_index["decision"]),
        ]
        return segments, "marker_split"

    sections = dict(safe_cot_sections)
    intent = str(sections.get("intent", "")).strip()
    risk = str(sections.get("risk", "")).strip()
    decision = str(sections.get("decision", "")).strip()
    if intent and risk and decision:
        segments = [
            (f"Intent: {intent}\n", slot_label_to_index["intent"]),
            (f"Risk: {risk}\n", slot_label_to_index["risk"]),
            (f"Decision: {decision}", slot_label_to_index["decision"]),
        ]
        return segments, "sections_dict"

    sentences = [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+", text) if chunk.strip()]
    if (not require_triplet_sections) and len(sentences) >= 3:
        segments = [
            (sentences[0] + " ", slot_label_to_index["intent"]),
            (sentences[1] + " ", slot_label_to_index["risk"]),
            (" ".join(sentences[2:]), slot_label_to_index["decision"]),
        ]
        return segments, "sentence_split"

    if require_triplet_sections:
        raise ValueError(f"Failed to recover Intent/Risk/Decision sections from safe_cot={safe_cot!r}")

    fallback_slot = slot_label_to_index.get("decision", int(len(slot_names) - 1))
    return [(text, fallback_slot)], "decision_fallback"


def extract_triplet_section_texts(
    *,
    safe_cot: str,
    safe_cot_sections: dict[str, Any],
    slot_names: Sequence[str] = DEFAULT_TRIPLET_SLOT_NAMES,
    require_triplet_sections: bool = True,
    strip_section_headers: bool = True,
) -> tuple[dict[str, str], dict[str, Any]]:
    segments, parse_mode = _extract_triplet_segments(
        safe_cot=safe_cot,
        safe_cot_sections=safe_cot_sections,
        slot_names=slot_names,
        require_triplet_sections=require_triplet_sections,
    )
    section_texts: dict[str, str] = {}
    for segment_text, section_idx in segments:
        slot_name = str(slot_names[int(section_idx)])
        text = str(segment_text)
        if strip_section_headers:
            text = _strip_section_header(text, slot_name)
        section_texts[slot_name] = str(text).strip()
    return section_texts, {"parse_mode": str(parse_mode)}


def get_triplet_slot_header(slot_name: str) -> str:
    slot = str(slot_name).strip().lower()
    if slot not in TRIPLET_SLOT_HEADERS:
        raise ValueError(f"Unsupported triplet slot {slot_name!r}.")
    return TRIPLET_SLOT_HEADERS[slot]


def build_triplet_cascade_prompt_suffix(
    *,
    slot_name: str,
    section_texts: dict[str, str],
    slot_names: Sequence[str] = DEFAULT_TRIPLET_SLOT_NAMES,
) -> str:
    ordered_slot_names = tuple(str(name) for name in slot_names)
    target_slot = str(slot_name).strip().lower()
    if target_slot not in ordered_slot_names:
        raise ValueError(f"slot_name={slot_name!r} is not present in slot_names={ordered_slot_names!r}.")

    target_idx = ordered_slot_names.index(target_slot)
    prefix = "\n".join(
        text
        for text in (str(section_texts.get(prev_slot, "")).strip() for prev_slot in ordered_slot_names[:target_idx])
        if text
    )
    return f"{prefix}\n" if prefix else ""


def compose_triplet_cot_text(
    *,
    section_texts: dict[str, str],
    slot_names: Sequence[str] = DEFAULT_TRIPLET_SLOT_NAMES,
) -> str:
    ordered_slot_names = tuple(str(name) for name in slot_names)
    return "\n".join(
        text
        for text in (str(section_texts.get(slot_name, "")).strip() for slot_name in ordered_slot_names)
        if text
    )


def parse_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as wf:
        for row in rows:
            wf.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_model_config(
    model_name: str,
    config_model_name: str | None = None,
    model_config_path: str | Path = MODEL_CONFIG_PATH,
) -> dict[str, Any]:
    with Path(model_config_path).open("r", encoding="utf-8") as rf:
        model_configs = json.load(rf)
    lookup_name = config_model_name or model_name
    cfg = model_configs.get(lookup_name, {})
    if len(cfg) == 0:
        raise ValueError(f"Model configuration not found for {lookup_name} in {model_config_path}")
    return cfg


def infer_prompt_format(tokenizer) -> str:
    chat_template = str(getattr(tokenizer, "chat_template", "") or "")
    if "<｜User｜>" in chat_template and "<｜Assistant｜>" in chat_template:
        return DEEPSEEK_R1_PROMPT_FORMAT
    if "<|start_header_id|>" in chat_template and "<|eot_id|>" in chat_template:
        return LLAMA3_PROMPT_FORMAT
    if "<|im_start|>" in chat_template:
        return QWEN_CHATML_PROMPT_FORMAT

    special_tokens = getattr(tokenizer, "special_tokens_map", {}) or {}
    special_token_text = json.dumps(special_tokens, ensure_ascii=False)
    if "<|start_header_id|>" in special_token_text and "<|eot_id|>" in special_token_text:
        return LLAMA3_PROMPT_FORMAT
    return QWEN_CHATML_PROMPT_FORMAT


def resolve_prompt_format(tokenizer, prompt_format: str | None = AUTO_PROMPT_FORMAT) -> str:
    requested = str(prompt_format or AUTO_PROMPT_FORMAT).strip().lower()
    aliases = {
        "auto": AUTO_PROMPT_FORMAT,
        "qwen": QWEN_CHATML_PROMPT_FORMAT,
        "qwen2": QWEN_CHATML_PROMPT_FORMAT,
        "qwen_chatml": QWEN_CHATML_PROMPT_FORMAT,
        "chatml": QWEN_CHATML_PROMPT_FORMAT,
        "s1": QWEN_CHATML_PROMPT_FORMAT,
        "llama": LLAMA3_PROMPT_FORMAT,
        "llama3": LLAMA3_PROMPT_FORMAT,
        "llama_3": LLAMA3_PROMPT_FORMAT,
        "deepseek": DEEPSEEK_R1_PROMPT_FORMAT,
        "deepseek_r1": DEEPSEEK_R1_PROMPT_FORMAT,
        "ds_r1": DEEPSEEK_R1_PROMPT_FORMAT,
        "r1_distill": DEEPSEEK_R1_PROMPT_FORMAT,
    }
    resolved = aliases.get(requested, requested)
    if resolved == AUTO_PROMPT_FORMAT:
        resolved = infer_prompt_format(tokenizer)
    if resolved not in {QWEN_CHATML_PROMPT_FORMAT, LLAMA3_PROMPT_FORMAT, DEEPSEEK_R1_PROMPT_FORMAT}:
        raise ValueError(f"Unsupported prompt_format={prompt_format!r}")
    return resolved


def default_decode_profile_for_prompt_format(prompt_format: str) -> str:
    return XML_COT_DECODE_PROFILE if prompt_format in {LLAMA3_PROMPT_FORMAT, DEEPSEEK_R1_PROMPT_FORMAT} else S1_DECODE_PROFILE


def resolve_decode_profile(decode_profile: str | None = AUTO_DECODE_PROFILE) -> str:
    requested = str(decode_profile or AUTO_DECODE_PROFILE).strip().lower()
    aliases = {
        "auto": AUTO_DECODE_PROFILE,
        "s1": S1_DECODE_PROFILE,
        "qwen_s1": S1_DECODE_PROFILE,
        "strict_slot": S1_DECODE_PROFILE,
        "xml": XML_COT_DECODE_PROFILE,
        "xml_cot": XML_COT_DECODE_PROFILE,
        "think_xml": XML_COT_DECODE_PROFILE,
        "open_r1": XML_COT_DECODE_PROFILE,
        "stratos": XML_COT_DECODE_PROFILE,
    }
    resolved = aliases.get(requested, requested)
    if resolved not in {AUTO_DECODE_PROFILE, S1_DECODE_PROFILE, XML_COT_DECODE_PROFILE}:
        raise ValueError(f"Unsupported decode_profile={decode_profile!r}")
    return resolved


def _apply_chat_template_text(tokenizer, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> str | None:
    if not hasattr(tokenizer, "apply_chat_template") or not getattr(tokenizer, "chat_template", None):
        return None
    return str(
        tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=bool(add_generation_prompt),
        )
    )


def _chat_template_prefix_pair(tokenizer, raw_prompt: str, system_prompt: str) -> tuple[str, str] | None:
    messages = [
        {"role": "system", "content": str(system_prompt)},
        {"role": "user", "content": str(raw_prompt).strip()},
    ]
    user_prefix_text = _apply_chat_template_text(tokenizer, messages, add_generation_prompt=False)
    assistant_prompt_text = _apply_chat_template_text(tokenizer, messages, add_generation_prompt=True)
    if user_prefix_text is None or assistant_prompt_text is None:
        return None
    if not assistant_prompt_text.startswith(user_prefix_text):
        return None
    return user_prefix_text, assistant_prompt_text[len(user_prefix_text) :]


def build_s1_user_prefix_text(raw_prompt: str, system_prompt: str) -> str:
    prompt = raw_prompt.strip()
    if not prompt:
        raise ValueError("raw_prompt must be a non-empty string")
    return (
        "<|im_start|>system\n"
        f"{system_prompt}<|im_end|>\n"
        "<|im_start|>user\n"
        f"{prompt}<|im_end|>\n"
    )


def build_s1_prompt_text(raw_prompt: str, system_prompt: str) -> str:
    return build_s1_user_prefix_text(raw_prompt=raw_prompt, system_prompt=system_prompt) + THINK_START_TEXT


def build_s1_answer_prompt_text(raw_prompt: str, cot: str, system_prompt: str) -> str:
    return build_s1_prompt_text(raw_prompt=raw_prompt, system_prompt=system_prompt) + cot + "\n<|im_start|>answer\n"


def build_s1_prompt_spec(
    tokenizer,
    raw_prompt: str,
    system_prompt: str,
) -> dict[str, Any]:
    return build_prompt_spec(
        tokenizer=tokenizer,
        raw_prompt=raw_prompt,
        system_prompt=system_prompt,
        prompt_format=QWEN_CHATML_PROMPT_FORMAT,
    )


def build_user_prefix_text(
    tokenizer,
    raw_prompt: str,
    system_prompt: str,
    prompt_format: str | None = AUTO_PROMPT_FORMAT,
) -> str:
    resolved_format = resolve_prompt_format(tokenizer, prompt_format=prompt_format)
    prompt = raw_prompt.strip()
    if not prompt:
        raise ValueError("raw_prompt must be a non-empty string")
    if resolved_format == QWEN_CHATML_PROMPT_FORMAT:
        return build_s1_user_prefix_text(raw_prompt=prompt, system_prompt=system_prompt)

    template_pair = _chat_template_prefix_pair(
        tokenizer,
        raw_prompt=prompt,
        system_prompt=system_prompt,
    )
    if template_pair is not None:
        return template_pair[0]
    if resolved_format == DEEPSEEK_R1_PROMPT_FORMAT:
        bos = str(getattr(tokenizer, "bos_token", "") or "")
        return f"{bos}<｜User｜>{prompt}"
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{prompt}<|eot_id|>"
    )


def _assistant_prefix_text(
    tokenizer,
    raw_prompt: str,
    system_prompt: str,
    prompt_format: str | None = AUTO_PROMPT_FORMAT,
) -> str:
    resolved_format = resolve_prompt_format(tokenizer, prompt_format=prompt_format)
    if resolved_format == QWEN_CHATML_PROMPT_FORMAT:
        return QWEN_ASSISTANT_PREFIX_TEXT
    template_pair = _chat_template_prefix_pair(
        tokenizer,
        raw_prompt=raw_prompt,
        system_prompt=system_prompt,
    )
    if template_pair is not None and template_pair[1]:
        suffix = template_pair[1]
        if resolved_format == DEEPSEEK_R1_PROMPT_FORMAT and suffix.endswith(XML_THINK_PREFIX_TEXT):
            return suffix[: -len(XML_THINK_PREFIX_TEXT)]
        return suffix
    if resolved_format == DEEPSEEK_R1_PROMPT_FORMAT:
        return "<｜Assistant｜>"
    return LLAMA3_ASSISTANT_PREFIX_TEXT


def _think_prefix_text(prompt_format: str, decode_profile: str) -> str:
    if decode_profile == XML_COT_DECODE_PROFILE:
        return XML_THINK_PREFIX_TEXT
    if prompt_format == LLAMA3_PROMPT_FORMAT:
        return XML_THINK_PREFIX_TEXT
    return QWEN_THINK_PREFIX_TEXT


def build_prompt_text(
    tokenizer,
    raw_prompt: str,
    system_prompt: str,
    prompt_format: str | None = AUTO_PROMPT_FORMAT,
    decode_profile: str | None = S1_DECODE_PROFILE,
) -> str:
    resolved_format = resolve_prompt_format(tokenizer, prompt_format=prompt_format)
    resolved_decode_profile = resolve_decode_profile(decode_profile)
    if resolved_decode_profile == AUTO_DECODE_PROFILE:
        resolved_decode_profile = default_decode_profile_for_prompt_format(resolved_format)
    user_prefix_text = build_user_prefix_text(
        tokenizer=tokenizer,
        raw_prompt=raw_prompt,
        system_prompt=system_prompt,
        prompt_format=resolved_format,
    )
    assistant_prefix_text = _assistant_prefix_text(
        tokenizer=tokenizer,
        raw_prompt=raw_prompt,
        system_prompt=system_prompt,
        prompt_format=resolved_format,
    )
    return user_prefix_text + assistant_prefix_text + _think_prefix_text(resolved_format, resolved_decode_profile)


def build_answer_prompt_text(
    tokenizer,
    raw_prompt: str,
    cot: str,
    system_prompt: str,
    prompt_format: str | None = AUTO_PROMPT_FORMAT,
    decode_profile: str | None = S1_DECODE_PROFILE,
) -> str:
    resolved_format = resolve_prompt_format(tokenizer, prompt_format=prompt_format)
    resolved_decode_profile = resolve_decode_profile(decode_profile)
    if resolved_decode_profile == AUTO_DECODE_PROFILE:
        resolved_decode_profile = default_decode_profile_for_prompt_format(resolved_format)
    prompt_text = build_prompt_text(
        tokenizer=tokenizer,
        raw_prompt=raw_prompt,
        system_prompt=system_prompt,
        prompt_format=resolved_format,
        decode_profile=resolved_decode_profile,
    )
    if resolved_decode_profile == XML_COT_DECODE_PROFILE:
        return prompt_text + str(cot).strip() + "\n\n"
    return prompt_text + str(cot).strip() + "\n<|im_start|>answer\n"


def build_prompt_spec(
    tokenizer,
    raw_prompt: str,
    system_prompt: str,
    prompt_format: str | None = AUTO_PROMPT_FORMAT,
    decode_profile: str | None = S1_DECODE_PROFILE,
) -> dict[str, Any]:
    resolved_format = resolve_prompt_format(tokenizer, prompt_format=prompt_format)
    resolved_decode_profile = resolve_decode_profile(decode_profile)
    if resolved_decode_profile == AUTO_DECODE_PROFILE:
        resolved_decode_profile = default_decode_profile_for_prompt_format(resolved_format)
    if resolved_format == QWEN_CHATML_PROMPT_FORMAT:
        user_prefix_text = build_s1_user_prefix_text(raw_prompt=raw_prompt, system_prompt=system_prompt)
        assistant_prefix_text = QWEN_ASSISTANT_PREFIX_TEXT
    else:
        user_prefix_text = build_user_prefix_text(
            tokenizer=tokenizer,
            raw_prompt=raw_prompt,
            system_prompt=system_prompt,
            prompt_format=resolved_format,
        )
        assistant_prefix_text = _assistant_prefix_text(
            tokenizer=tokenizer,
            raw_prompt=raw_prompt,
            system_prompt=system_prompt,
            prompt_format=resolved_format,
        )
    think_prefix_text = _think_prefix_text(resolved_format, resolved_decode_profile)
    prompt_text = user_prefix_text + assistant_prefix_text + think_prefix_text

    user_prefix_ids = tokenizer(user_prefix_text, add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    if len(user_prefix_ids) == 0:
        raise ValueError(f"Failed to tokenize user prefix for {resolved_format} prompt.")
    if len(prompt_ids) <= len(user_prefix_ids):
        raise ValueError("Prompt ids must include think prefix tokens beyond the user prefix.")

    return {
        "raw_prompt": raw_prompt,
        "user_prefix_text": user_prefix_text,
        "prompt_text": prompt_text,
        "user_prefix_ids": user_prefix_ids,
        "prompt_ids": prompt_ids,
        "t_inst_idx": int(len(user_prefix_ids) - 1),
        "prompt_format": resolved_format,
        "decode_profile": resolved_decode_profile,
        "assistant_prefix_text": assistant_prefix_text,
        "think_prefix_text": think_prefix_text,
    }


def build_target_ids(
    tokenizer,
    safe_cot: str,
    include_think_end: bool = True,
    max_target_tokens: int | None = None,
) -> list[int]:
    cot_text = safe_cot.strip()
    if not cot_text:
        raise ValueError("safe_cot must be a non-empty string")

    cot_ids = tokenizer(cot_text, add_special_tokens=False)["input_ids"]
    boundary_ids = tokenizer(THINK_END_TEXT, add_special_tokens=False)["input_ids"] if include_think_end else []
    target_ids = cot_ids + boundary_ids

    if max_target_tokens is None or len(target_ids) <= int(max_target_tokens):
        return target_ids

    max_target_tokens = int(max_target_tokens)
    if max_target_tokens <= len(boundary_ids):
        raise ValueError(
            f"max_target_tokens={max_target_tokens} is too small to fit the boundary of length {len(boundary_ids)}"
        )
    keep_cot = max_target_tokens - len(boundary_ids)
    return cot_ids[:keep_cot] + boundary_ids


def resolve_model_input_device(model) -> torch.device:
    embed = model.get_input_embeddings()
    if embed is not None and hasattr(embed, "weight"):
        return embed.weight.device

    hf_device_map = getattr(model, "hf_device_map", None)
    if isinstance(hf_device_map, dict):
        for _, value in hf_device_map.items():
            if isinstance(value, str) and value.startswith("cuda"):
                return torch.device(value)
            if isinstance(value, int):
                return torch.device(f"cuda:{value}")
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _normalize_probe_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "layer_index": int(payload["layer_index"]),
        "layer_number": int(payload["layer_number"]),
        "mean": payload["mean"].detach().cpu().to(dtype=torch.float32),
        "scale": payload["scale"].detach().cpu().to(dtype=torch.float32),
        "U": payload["U"].detach().cpu().to(dtype=torch.float32),
        "classifier_weight": payload["classifier_weight"].detach().cpu().to(dtype=torch.float32),
        "classifier_bias": payload["classifier_bias"].detach().cpu().to(dtype=torch.float32),
        "score_normalization": dict(payload.get("score_normalization", {}))
        if isinstance(payload.get("score_normalization"), dict)
        else payload.get("score_normalization"),
    }


def _extract_selected_layer_index(summary: dict[str, Any], artifacts: dict[str, Any]) -> int:
    candidate_paths = [
        summary.get("stage2_sjb_onset_erosion", {}).get("selected_layer"),
        summary.get("final_target_layer_selection", {}).get("selected_layer"),
        artifacts.get("stage2_sjb_onset_erosion", {}).get("selected_layer"),
        artifacts.get("final_target_layer_selection", {}).get("selected_layer"),
    ]
    for candidate in candidate_paths:
        if isinstance(candidate, dict) and "layer_index" in candidate:
            return int(candidate["layer_index"])
    raise ValueError("Failed to infer selected_layer.layer_index from harmfulness subspace artifacts.")


def load_probe_artifacts(
    probe_dir: str | Path,
    layer_index: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    probe_dir = Path(probe_dir)
    artifacts = torch.load(probe_dir / "layer_subspaces.pt", map_location="cpu", weights_only=False)
    summary = read_json(probe_dir / "summary.json")

    if layer_index is None:
        layer_index = _extract_selected_layer_index(summary=summary, artifacts=artifacts)
    layer_index = int(layer_index)

    payload = next(
        (row for row in artifacts["layer_payloads"] if int(row["layer_index"]) == layer_index),
        None,
    )
    if payload is None:
        raise ValueError(f"Layer index {layer_index} not found in {probe_dir / 'layer_subspaces.pt'}")
    return _normalize_probe_payload(payload), summary, artifacts


def compute_latent(
    hidden_state: torch.Tensor,
    payload: dict[str, Any],
    eps: float = 1e-6,
) -> torch.Tensor:
    hidden_state = hidden_state.to(dtype=torch.float32)
    mean = payload["mean"].to(dtype=torch.float32, device=hidden_state.device)
    scale = payload["scale"].to(dtype=torch.float32, device=hidden_state.device).clamp_min(eps)
    basis = payload["U"].to(dtype=torch.float32, device=hidden_state.device)
    return ((hidden_state - mean) / scale) @ basis


def compute_probe_logits_from_latent(
    latent: torch.Tensor,
    payload: dict[str, Any],
) -> torch.Tensor:
    weight = payload["classifier_weight"].to(dtype=torch.float32, device=latent.device)
    bias = payload["classifier_bias"].to(dtype=torch.float32, device=latent.device)
    return latent @ weight + bias


def apply_probe_score_normalization_to_logits(
    logits: torch.Tensor,
    payload: dict[str, Any],
    eps: float = 1e-6,
) -> torch.Tensor:
    normalization = payload.get("score_normalization")
    if not isinstance(normalization, dict) or not bool(normalization.get("implemented", False)):
        return logits

    method = str(normalization.get("method", "none"))
    if method == "relative_anchor":
        midpoint = float(normalization.get("midpoint_logit", 0.0))
        half_gap = max(float(normalization.get("half_gap_logit", 1.0)), float(eps))
        direction = float(normalization.get("direction", 1.0))
        return direction * (logits - midpoint) / half_gap
    if method == "none":
        return logits
    raise ValueError(f"Unsupported score normalization method in payload: {method}")


def compute_probe_probs_from_latent(
    latent: torch.Tensor,
    payload: dict[str, Any],
    eps: float = 1e-6,
) -> torch.Tensor:
    logits = compute_probe_logits_from_latent(latent=latent, payload=payload)
    return torch.sigmoid(apply_probe_score_normalization_to_logits(logits=logits, payload=payload, eps=eps))


def compute_probe_prob_from_hidden_state(
    hidden_state: torch.Tensor,
    payload: dict[str, Any],
    eps: float = 1e-6,
) -> torch.Tensor:
    latent = compute_latent(hidden_state=hidden_state, payload=payload, eps=eps)
    return compute_probe_probs_from_latent(latent=latent, payload=payload, eps=eps)


def compute_latent_and_prob(
    hidden_state: torch.Tensor,
    payload: dict[str, Any],
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    latent = compute_latent(hidden_state=hidden_state, payload=payload, eps=eps)
    prob = compute_probe_probs_from_latent(latent=latent, payload=payload, eps=eps)
    return latent, prob


class LatentSafePrefixController(nn.Module):
    def __init__(
        self,
        *,
        rank: int,
        hidden_size: int,
        prefix_len: int,
        num_strategies: int,
        strategy_hidden_dim: int = 128,
        residual_hidden_dim: int = 256,
        residual_scale_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.rank = int(rank)
        self.hidden_size = int(hidden_size)
        self.prefix_len = int(prefix_len)
        self.num_strategies = int(num_strategies)

        self.z_norm = nn.LayerNorm(self.rank)
        self.strategy_prototypes = nn.Parameter(
            torch.empty(self.num_strategies, self.prefix_len, self.hidden_size)
        )
        self.strategy_head = nn.Sequential(
            nn.Linear(self.rank, int(strategy_hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(strategy_hidden_dim), self.num_strategies),
        )
        self.residual_mlp = nn.Sequential(
            nn.Linear(self.rank, int(residual_hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(residual_hidden_dim), self.prefix_len * self.hidden_size),
        )
        self.alpha = nn.Parameter(torch.tensor(float(residual_scale_init), dtype=torch.float32))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.strategy_prototypes, mean=0.0, std=0.02)

    def forward(
        self,
        z_h: torch.Tensor,
        *,
        route_temperature: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        if z_h.ndim != 2 or int(z_h.shape[-1]) != self.rank:
            raise ValueError(f"Expected z_h with shape [B, {self.rank}], got {tuple(z_h.shape)}")

        z = self.z_norm(z_h.to(dtype=torch.float32))
        strategy_logits = self.strategy_head(z)
        delta = self.residual_mlp(z).view(-1, self.prefix_len, self.hidden_size)
        route_temperature = float(max(route_temperature, 1e-5))
        strategy_probs = torch.softmax(strategy_logits / route_temperature, dim=-1)
        soft_prototype = torch.einsum("bs,skd->bkd", strategy_probs, self.strategy_prototypes)
        strategy_pred_idx = torch.argmax(strategy_logits, dim=-1)
        prefix = soft_prototype + self.alpha * delta
        return {
            "strategy_logits": strategy_logits,
            "strategy_probs": strategy_probs,
            "delta": delta,
            "soft_prototype": soft_prototype,
            "strategy_pred_idx": strategy_pred_idx,
            "prefix": prefix,
        }


class StaticSafePrefixController(nn.Module):
    def __init__(
        self,
        *,
        prefix_len: int,
        hidden_size: int,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        self.prefix_len = int(prefix_len)
        self.hidden_size = int(hidden_size)
        self.prefix_embeddings = nn.Parameter(torch.empty(self.prefix_len, self.hidden_size))
        self.init_std = float(init_std)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.prefix_embeddings, mean=0.0, std=self.init_std)

    def forward(self, batch_size: int) -> dict[str, torch.Tensor]:
        if int(batch_size) <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        prefix = self.prefix_embeddings.unsqueeze(0).expand(int(batch_size), -1, -1)
        return {"prefix": prefix}


@dataclass(frozen=True)
class DecodeState:
    past_key_values: Any
    attention_mask: torch.Tensor
    cache_position: torch.Tensor
    next_logits: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...] | None = None


def _encode_text(tokenizer, text: str, device: torch.device) -> dict[str, torch.Tensor]:
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    out = {k: v.to(device) for k, v in inputs.items()}
    if "attention_mask" not in out:
        out["attention_mask"] = torch.ones_like(out["input_ids"], device=device)
    return out


def _sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    if float(temperature) <= 0.0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    scores = logits / float(max(temperature, 1e-5))
    if float(top_p) < 1.0:
        sorted_scores, sorted_indices = torch.sort(scores, dim=-1, descending=True)
        sorted_probs = torch.softmax(sorted_scores, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_remove = cumulative_probs > float(top_p)
        sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
        sorted_remove[..., 0] = False
        sorted_scores = sorted_scores.masked_fill(sorted_remove, float("-inf"))

        filtered_scores = torch.full_like(scores, float("-inf"))
        filtered_scores.scatter_(dim=-1, index=sorted_indices, src=sorted_scores)
        scores = filtered_scores

    probs = torch.softmax(scores, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _has_stop_suffix(token_ids: list[int], stop_sequences: list[list[int]]) -> bool:
    for seq in stop_sequences:
        m = len(seq)
        if m > 0 and len(token_ids) >= m and token_ids[-m:] == seq:
            return True
    return False


def _strip_trailing_stop_ids(token_ids: list[int], stop_sequences: list[list[int]]) -> list[int]:
    for seq in stop_sequences:
        m = len(seq)
        if m > 0 and len(token_ids) >= m and token_ids[-m:] == seq:
            return token_ids[:-m]
    return token_ids


def prefill_decode_state(
    model,
    tokenizer,
    *,
    text: str,
    output_hidden_states: bool = False,
) -> DecodeState:
    device = resolve_model_input_device(model)
    encoded = _encode_text(tokenizer=tokenizer, text=text, device=device)
    cache_position = torch.arange(
        int(encoded["input_ids"].shape[1]),
        device=encoded["input_ids"].device,
    )
    with torch.inference_mode():
        model_inputs = model.prepare_inputs_for_generation(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            cache_position=cache_position,
            use_cache=True,
        )
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_hidden_states=output_hidden_states,
        )
    return DecodeState(
        past_key_values=outputs.past_key_values,
        attention_mask=encoded["attention_mask"],
        cache_position=cache_position,
        next_logits=outputs.logits[:, -1, :],
        hidden_states=outputs.hidden_states if output_hidden_states else None,
    )


def prefill_with_prefix_embeddings(
    model,
    prompt_ids: list[int],
    prefix_embeddings: torch.Tensor,
    *,
    prefix_position: str = "after_prompt",
) -> DecodeState:
    input_device = resolve_model_input_device(model)
    input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=input_device).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids, device=input_device)
    token_embeds = model.get_input_embeddings()(input_ids)

    if prefix_embeddings.ndim == 2:
        prefix_embeddings = prefix_embeddings.unsqueeze(0)
    prefix_embeddings = prefix_embeddings.to(device=token_embeds.device, dtype=token_embeds.dtype)
    prefix_mask = torch.ones(
        (1, int(prefix_embeddings.shape[1])),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    if prefix_position == "prepend":
        full_attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        inputs_embeds = torch.cat([prefix_embeddings, token_embeds], dim=1)
    elif prefix_position == "after_prompt":
        full_attention_mask = torch.cat([attention_mask, prefix_mask], dim=1)
        inputs_embeds = torch.cat([token_embeds, prefix_embeddings], dim=1)
    else:
        raise ValueError(f"Unsupported prefix_position: {prefix_position}")
    cache_position = torch.arange(int(inputs_embeds.shape[1]), device=input_device)
    position_ids = full_attention_mask.cumsum(dim=1) - 1

    with torch.inference_mode():
        outputs = model(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            position_ids=position_ids,
            use_cache=True,
            return_dict=True,
        )
    return DecodeState(
        past_key_values=outputs.past_key_values,
        attention_mask=full_attention_mask,
        cache_position=cache_position,
        next_logits=outputs.logits[:, -1, :],
        hidden_states=None,
    )


def advance_decode_state(
    model,
    state: DecodeState,
    new_input_ids: torch.Tensor,
) -> DecodeState:
    if new_input_ids.ndim == 1:
        new_input_ids = new_input_ids.unsqueeze(0)
    new_input_ids = new_input_ids.to(state.attention_mask.device)

    batch_size = int(new_input_ids.shape[0])
    new_mask = torch.ones(
        (batch_size, int(new_input_ids.shape[1])),
        dtype=state.attention_mask.dtype,
        device=state.attention_mask.device,
    )
    attention_mask = torch.cat([state.attention_mask, new_mask], dim=1)
    start_pos = int(state.cache_position[-1].item()) + 1
    new_cache_position = torch.arange(
        start_pos,
        start_pos + int(new_input_ids.shape[1]),
        device=new_input_ids.device,
    )

    with torch.inference_mode():
        model_inputs = model.prepare_inputs_for_generation(
            input_ids=new_input_ids,
            attention_mask=attention_mask,
            past_key_values=state.past_key_values,
            cache_position=new_cache_position,
            use_cache=True,
        )
        outputs = model(
            **model_inputs,
            return_dict=True,
        )

    return DecodeState(
        past_key_values=outputs.past_key_values,
        attention_mask=attention_mask,
        cache_position=torch.cat([state.cache_position, new_cache_position], dim=0),
        next_logits=outputs.logits[:, -1, :],
        hidden_states=None,
    )


def decode_with_past_key_values(
    model,
    state: DecodeState,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    stop_sequences: list[list[int]],
) -> list[int]:
    generated_ids, _, _ = decode_with_past_key_values_and_state(
        model=model,
        state=state,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        stop_sequences=stop_sequences,
    )
    return generated_ids


def decode_with_past_key_values_and_state(
    model,
    state: DecodeState,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    stop_sequences: list[list[int]],
) -> tuple[list[int], DecodeState, bool]:
    generated_ids: list[int] = []
    cur_state = state
    hit_stop = False

    for _ in range(int(max_new_tokens)):
        next_token = _sample_next_token(
            cur_state.next_logits,
            temperature=temperature,
            top_p=top_p,
        )
        token_id = int(next_token.item())
        generated_ids.append(token_id)
        cur_state = advance_decode_state(model=model, state=cur_state, new_input_ids=next_token)
        if _has_stop_suffix(generated_ids, stop_sequences):
            hit_stop = True
            break

    return _strip_trailing_stop_ids(generated_ids, stop_sequences), cur_state, hit_stop
