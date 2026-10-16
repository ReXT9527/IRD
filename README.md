# IRD

Official training and evaluation code for Intent--Risk--Decision (IRD) prefix
tuning.

## Environment

```bash
conda create -n ird python=3.11 -y
conda activate ird
pip install -r requirements.txt
huggingface-cli login
```

## Train IRD adapters

Provide an IRD training JSONL file and run:

```bash
torchrun --nproc_per_node 8 scripts/pt-train-strict-slot-prefix-safe-cot.py \
  --model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --config_model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --data_file /path/to/ird_train.jsonl \
  --output_dir outputs/PT/ird-r1-llama \
  --train_slots all \
  --slot_lengths 4,32,32 \
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

The output contains one adapter for each slot:

```text
outputs/PT/ird-r1-llama/
  intent/best_adapter/
  risk/best_adapter/
  decision/best_adapter/
```

## Generate outputs

```bash
python eval/pt-eval-strict-slot-prefix-generation.py \
  --intent_adapter_dir outputs/PT/ird-r1-llama/intent/best_adapter \
  --risk_adapter_dir outputs/PT/ird-r1-llama/risk/best_adapter \
  --decision_adapter_dir outputs/PT/ird-r1-llama/decision/best_adapter \
  --data_file /path/to/eval.jsonl \
  --output_dir outputs/eval/ird-r1-llama \
  --input_format jsonl \
  --prompt_field auto \
  --config_model_name deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --cascade_stage_max_tokens 1546 \
  --cascade_slot_budget_ratio 1,32,32 \
  --ird_handoff \
  --ird_handoff_disable_adapters \
  --ird_handoff_answer_kvreuse \
  --max_think_tokens 4096 \
  --max_answer_tokens 1 \
  --ird_handoff_max_think_tokens 4096 \
  --ird_handoff_max_answer_tokens 4096 \
  --route_source intent \
  --intent_decode_mode auto \
  --decision_decode_mode generate \
  --dtype bfloat16 \
  --gpu_ids 0,1,2,3
```

## Judge generated responses

```bash
cd eval/ThinkSafe
python eval_safety.py \
  --input_file ../../outputs/eval/ird-r1-llama/results.jsonl \
  --output_dir ../../outputs/eval/ird-r1-llama/judge \
  --mode harmbench \
  --llamaguard_model meta-llama/Llama-Guard-3-8B \
  --wildguard_model allenai/wildguard \
  --tensor_parallel_size 2
```
