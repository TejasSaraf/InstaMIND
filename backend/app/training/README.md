# LoRA fine-tuning for incident classification

Fine-tune Gemma with **LoRA** so that:
- **Shoplifting** activity in video → output `shoplifting`
- **Fainting** (collapse) only when the person is actually collapsed → output `fainting`
- Otherwise → output `none` or `suspicious_activity` (no false fainting)

The tuned model is used as the **primary classifier** when `MODEL_MODE=local_gemma` (same prompt format as training).

## One-command: fine-tune for shoplifting detection (LoRA)

**Run on your GPU VM** (see **`scripts/GPU_FINETUNE_RUNBOOK.md`** for copy-paste GPU-only steps). From backend root on the VM:

```bash
./scripts/finetune_shoplifting_lora.sh
```

This runs: (1) build pose windows from `app/dataset/poselift/annotations.jsonl`, (2) build SFT JSONL with synthetic shoplifting + none + balance, (3) QLoRA fine-tune on GPU. Adapter is saved to `outputs/gemma-shoplifting-lora/adapter`. Optional env overrides: `MODEL_NAME`, `OUTPUT_DIR`, `DATASET_DIR`. Do not use `--no_cuda` on the VM.

## 1. Prepare labels and pose windows

- **annotations.jsonl**: One JSON object per line with `label` and `frames`.  
  Labels: `shoplifting`, `fainting`, `choking`, `violent_activity`, `suspicious_activity`, `none`.  
  Each frame: `keypoints` (17×3), `motion`, `audio_distress`.
- **label_map.csv**: Optional; maps filenames to labels if you build JSONL from clips.

Create PoseLift-style JSONL, then build windowed NPZ:

```bash
cd backend  # or app, depending on where you run
python -m app.training.prepare_poselift_windows \
  --input app/dataset/poselift/annotations.jsonl \
  --output app/dataset/poselift/windows.npz \
  --window 32 --stride 8
```

## 2. Build SFT dataset (match inference prompt)

Build the JSONL used for LoRA. The prompt format matches the runtime classifier (Gemini / local Gemma).

```bash
python -m app.training.build_gemma_sft_dataset \
  --dataset app/dataset/poselift/windows.npz \
  --output app/dataset/poselift/gemma_sft.jsonl \
  --synthetic-none 500 \
  --balance
```

- **--synthetic-none N**: Add N synthetic “normal activity” examples (low horizontal posture, moderate motion) so the model does not default to fainting.
- **--balance**: Oversample so shoplifting/fainting have at least as many samples as the majority class.

## 3. Run QLoRA fine-tuning (GPU)

Requires CUDA (e.g. on a GCP GPU VM). Install training deps:  
`pip install torch peft transformers datasets bitsandbytes accelerate`

Then run:

```bash
python -m app.training.finetune_gemma_qlora \
  --model_name google/gemma-2-2b-it \
  --train_file app/dataset/poselift/gemma_sft.jsonl \
  --output_dir outputs/gemma-incident-qlora \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --bf16 \
  --lora_r 16 --lora_alpha 32
```

Use `google/gemma-3n-e4b-it` or another Hugging Face model ID if you prefer.

**If you get 403 (Gemma not in authorized list):** use a non-gated model — no Hugging Face approval needed:

```bash
python -m app.training.finetune_gemma_qlora \
  --model_name google/gemma-2-2b-it \
  --train_file app/dataset/poselift/gemma_sft.jsonl \
  --output_dir outputs/gemma-shoplifting-lora \
  --num_train_epochs 3 \
  --no_cuda \
  --use_open_model
```

This uses `Qwen/Qwen2-0.5B-Instruct` (small, open, no gate). Adapter is still saved and can be loaded for inference; use the same prompt format at runtime.

Optional: **--merge_and_save** to merge LoRA into the base model and save a full checkpoint.

## 4. Export to Ollama (optional)

If you use Ollama to serve the model:

```bash
python -m app.training.export_to_ollama \
  --adapter-dir outputs/gemma-incident-qlora/adapter \
  --model-name gemma-incident \
  --create
```

Then in backend `.env`:

```env
MODEL_MODE=local_gemma
LOCAL_GEMMA_ENDPOINT=http://127.0.0.1:11434/api/generate
LOCAL_GEMMA_MODEL_NAME=gemma-incident
```

When `MODEL_MODE=local_gemma`, the backend uses the LoRA-tuned model as the **primary classifier** with the same prompt as in SFT (video/pose/audio summary → one of `fainting`, `shoplifting`, `none`, etc.).

## Tips

- Add plenty of **shoplifting** and **none** (or **suspicious_activity**) examples in `annotations.jsonl` so the model learns to distinguish them from fainting.
- Use **--synthetic-none** to reduce false “fainting” on normal or retail activity.
- For more capacity, use a larger base model (e.g. `google/gemma-2-9b-it`) and adjust batch size / gradient accumulation to fit GPU memory.
