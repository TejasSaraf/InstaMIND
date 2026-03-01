#!/usr/bin/env bash
# End-to-end LoRA fine-tuning for shoplifting detection in video.
# Run from backend root: ./scripts/finetune_shoplifting_lora.sh
# Requires: annotations.jsonl, then GPU with CUDA for step 3.

set -e
BACKEND_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$BACKEND_ROOT"

DATASET_DIR="${DATASET_DIR:-app/dataset/poselift}"
ANNOTATIONS="${DATASET_DIR}/annotations.jsonl"
WINDOWS_NPZ="${DATASET_DIR}/windows.npz"
SFT_JSONL="${DATASET_DIR}/gemma_sft.jsonl"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/gemma-shoplifting-lora}"
MODEL_NAME="${MODEL_NAME:-google/gemma-2-2b-it}"

echo "=== 1. Build pose windows from annotations ==="
if [[ ! -f "$ANNOTATIONS" ]]; then
  echo "Missing $ANNOTATIONS. Add PoseLift-style JSONL with 'label' and 'frames' (labels: shoplifting, none, etc.)."
  exit 1
fi
python -m app.training.prepare_poselift_windows \
  --input "$ANNOTATIONS" \
  --output "$WINDOWS_NPZ" \
  --window 32 --stride 8

echo "=== 2. Build SFT dataset (shoplifting-focused) ==="
python -m app.training.build_gemma_sft_dataset \
  --dataset "$WINDOWS_NPZ" \
  --output "$SFT_JSONL" \
  --synthetic-none 400 \
  --synthetic-shoplifting 300 \
  --balance

echo "=== 3. QLoRA fine-tune (requires GPU with CUDA) ==="
python -m app.training.finetune_gemma_qlora \
  --model_name "$MODEL_NAME" \
  --train_file "$SFT_JSONL" \
  --output_dir "$OUTPUT_DIR" \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --bf16

echo "Done. Adapter saved to: $OUTPUT_DIR/adapter"
echo "To use with Ollama: python -m app.training.export_to_ollama --adapter-dir $OUTPUT_DIR/adapter --model-name gemma-shoplifting --create"
echo "Then set in .env: MODEL_MODE=local_gemma, LOCAL_GEMMA_MODEL_NAME=gemma-shoplifting"
