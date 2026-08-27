# IRD

This repository contains the minimal code and data needed to train and evaluate IRD (Intent, Risk, and Decision) prefix adapters for reasoning models. It intentionally excludes baseline implementations, generated outputs, checkpoints, paper sources, and jailbreak-attack data files.

## Contents

- `scripts/pt-train-strict-slot-prefix-safe-cot.py`: trains the three IRD prefix adapters.
- `scripts/pt-latent-safe-prefix-utils.py`: shared prompt, decoding, and JSONL utilities.
- `scripts/pt-model-configs.json`: model-specific prompt and decoding defaults.
- `eval/pt-eval-strict-slot-prefix-generation.py`: generates IRD outputs with the Intent -> Risk -> Decision cascade.
- `eval/vllm_rd_handoff_generation.py`: optional vLLM handoff generation from saved IRD prefixes.
- `eval/pt-judge-harmful-outputs-with-llama-guard.py`: judges generated outputs with Llama-Guard-3.
- `eval/eval_guard_prompt_safety.py`: prompt-level guard-model evaluation.
- `data/ird.jsonl`: IRD training data.
- `data/eval-data/`: non-jailbreak evaluation data used by the release scripts.

## Installation

```bash
conda create -n ird python=3.11 -y
conda activate ird
pip install -r requirements.txt
```

For gated Hugging Face models such as Llama-Guard-3, authenticate before running:

```bash
huggingface-cli login
```

By default, scripts use `--local_files_only`. Add `--no-local_files_only` if models should be downloaded from Hugging Face.

## Train IRD

The training script writes one adapter per slot under the output directory:

```bash
torchrun --nproc_per_node 8 scripts/pt-train-strict-slot-prefix-safe-cot.py \
  --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --config_model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --data_file data/ird.jsonl \
  --output_dir outputs/PT/ird-r1-llama \
  --train_slots all \
  --slot_lengths 16,16,16 \
  --batch_size 1 \
  --grad_accum 2 \
  --epochs 10 \
  --patience 2 \
  --lr 3e-5 \
  --val_ratio 0.1 \
  --dtype bfloat16 \
  --attn_implementation sdpa \
  --no-local_files_only
```

After training, the expected adapter layout is:

```text
outputs/PT/ird-r1-llama/
  intent/best_adapter/
  risk/best_adapter/
  decision/best_adapter/
  triplet_inference_config.json
```

## Generate IRD Outputs

Run the cascade generation script on an evaluation set:

```bash
python eval/pt-eval-strict-slot-prefix-generation.py \
  --intent_adapter_dir outputs/PT/ird-r1-llama/intent/best_adapter \
  --risk_adapter_dir outputs/PT/ird-r1-llama/risk/best_adapter \
  --decision_adapter_dir outputs/PT/ird-r1-llama/decision/best_adapter \
  --data_file data/eval-data/harmbench.jsonl \
  --output_dir outputs/eval/ird-r1-llama/harmbench \
  --input_format jsonl \
  --prompt_field auto \
  --config_model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --cascade_stage_max_tokens 1546 \
  --cascade_slot_budget_ratio 1,32,32 \
  --cascade_stop_at_newline \
  --ird_handoff \
  --ird_handoff_answer_kvreuse \
  --max_think_tokens 4096 \
  --max_answer_tokens 4096 \
  --ird_handoff_max_think_tokens 4096 \
  --ird_handoff_max_answer_tokens 4096 \
  --route_source intent \
  --intent_decode_mode auto \
  --decision_decode_mode generate \
  --dtype bfloat16 \
  --gpu_ids 0,1,2,3
```

Repeat by changing `--data_file` and `--output_dir` for the other files under `data/eval-data/`.

## Judge Harmful Outputs

Use Llama-Guard-3 on generated `results.jsonl`:

```bash
python eval/pt-judge-harmful-outputs-with-llama-guard.py \
  --input_file outputs/eval/ird-r1-llama/harmbench/results.jsonl \
  --output_dir outputs/eval/ird-r1-llama/harmbench/llama_guard \
  --model_id meta-llama/Llama-Guard-3-8B \
  --backend local \
  --batch_size 8 \
  --dtype bfloat16 \
  --no-local_files_only
```

The script writes `judged_outputs.jsonl` and `summary.json`.

## Prompt-Level Guard Evaluation

To evaluate a guard model directly on prompts:

```bash
python eval/eval_guard_prompt_safety.py \
  --models llama-guard-3-8b=meta-llama/Llama-Guard-3-8B \
  --datasets harmbench=data/eval-data/harmbench.jsonl strongreject=data/eval-data/strongreject.jsonl wj-eval=data/eval-data/wj_eval_harmful.jsonl \
  --output_dir outputs/eval/guard_prompt_safety \
  --backend vllm \
  --tensor_parallel_size 2 \
  --no-local_files_only
```

## Data Notes

The release includes `data/ird.jsonl` for IRD training and non-jailbreak evaluation files under `data/eval-data/`. Jailbreak-attack files are intentionally not included in this package.

## Anonymity Notes

This package does not include model checkpoints, experiment outputs, baseline implementations, local cache paths, paper build artifacts, or user-specific logs. Generated adapters and evaluation outputs should remain under `outputs/`, which is ignored by `.gitignore`.
