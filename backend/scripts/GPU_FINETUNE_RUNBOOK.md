# GPU-only LoRA fine-tuning (run on VM)

Use these steps **on your GPU VM** (CUDA). Do not use `--no_cuda`.

---

## 1. Sync project and data to VM

From your Mac (backend has `annotations.jsonl`, and optionally pre-built `windows.npz` and `gemma_sft.jsonl`):

```bash
cd /Users/tejassaraf/Projects/instaMIND
rsync -avz --exclude '.venv' --exclude '__pycache__' --exclude 'data' --exclude 'outputs' \
  backend/ hackathon@35.194.63.161:~/InstaMIND/backend/
```

If you already built the SFT JSONL locally, sync it so the VM can skip steps 1–2:

```bash
scp backend/app/dataset/poselift/annotations.jsonl hackathon@35.194.63.161:~/InstaMIND/backend/app/dataset/poselift/
scp backend/app/dataset/poselift/windows.npz hackathon@35.194.63.161:~/InstaMIND/backend/app/dataset/poselift/ 2>/dev/null || true
scp backend/app/dataset/poselift/gemma_sft.jsonl hackathon@35.194.63.161:~/InstaMIND/backend/app/dataset/poselift/ 2>/dev/null || true
```

---

## 2. On the VM: install deps (once)

```bash
ssh hackathon@35.194.63.161
cd ~/InstaMIND/backend
source .venv/bin/activate
pip install torch peft transformers datasets bitsandbytes accelerate
```

---

## 3. On the VM: build windows + SFT (if not synced)

If you didn’t copy `windows.npz` and `gemma_sft.jsonl`:

```bash
cd ~/InstaMIND/backend
source .venv/bin/activate

python -m app.training.prepare_poselift_windows \
  --input app/dataset/poselift/annotations.jsonl \
  --output app/dataset/poselift/windows.npz \
  --window 32 --stride 8

python -m app.training.build_gemma_sft_dataset \
  --dataset app/dataset/poselift/windows.npz \
  --output app/dataset/poselift/gemma_sft.jsonl \
  --synthetic-none 400 --balance
```

If your VM has the updated script, add `--synthetic-shoplifting 300` before `--balance`.

---

## 4. On the VM: QLoRA fine-tune (GPU only)

**Option A – Gemma (need HF access + token):**

```bash
cd ~/InstaMIND/backend
source .venv/bin/activate
export HF_TOKEN=your_huggingface_token   # or: huggingface-cli login

python -m app.training.finetune_gemma_qlora \
  --model_name google/gemma-2-2b-it \
  --train_file app/dataset/poselift/gemma_sft.jsonl \
  --output_dir outputs/gemma-shoplifting-lora \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --bf16
```

**Option B – Open model (no Gemma access):**

```bash
python -m app.training.finetune_gemma_qlora \
  --model_name google/gemma-2-2b-it \
  --train_file app/dataset/poselift/gemma_sft.jsonl \
  --output_dir outputs/gemma-shoplifting-lora \
  --num_train_epochs 3 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 8 \
  --bf16 \
  --use_open_model
```

Do **not** use `--no_cuda` on the VM.

---

## 5. After training

Adapter path on VM: `~/InstaMIND/backend/outputs/gemma-shoplifting-lora/adapter`

To use with Ollama on the VM:

```bash
python -m app.training.export_to_ollama \
  --adapter-dir outputs/gemma-shoplifting-lora/adapter \
  --model-name gemma-shoplifting \
  --create
```

Then in backend `.env` on the VM: `MODEL_MODE=local_gemma`, `LOCAL_GEMMA_MODEL_NAME=gemma-shoplifting`.

To copy the adapter back to your Mac (optional):

```bash
# From Mac
scp -r hackathon@35.194.63.161:~/InstaMIND/backend/outputs/gemma-shoplifting-lora ./backend/outputs/
```
