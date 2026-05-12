"""
=============================================================================
finetune_gemma3n_unsloth.py — Gemma 3 4B QLoRA vision+language SFT (v23)
=============================================================================
Full QLoRA fine-tuning of Gemma 3 (vision + language) via Unsloth on Modal.

v23: Switched from Gemma 3n E2B to Gemma 3 4B.
  Gemma 3 uses SigLIP (ViT) vision encoder instead of MobileNetV5 (Conv2d).
  SigLIP layers are nn.Linear → "all-linear" wraps them with LoRA natively.
  No manual vision injection needed.  Vision LoRA works out of the box.

  Benefits over Gemma 3n:
  - Vision LoRA works (SigLIP ViT is standard PEFT target)
  - Better vision understanding (SigLIP-400M > MobileNetV5)
  - GGUF export supports vision (Gemma 3n GGUF is text-only)
  - Mature deployment: Ollama, LM Studio, llama.cpp, MLX all work

Unsloth handles:
  * SigLIP + Gemma 2 language model patching
  * 4-bit quantization via BitsAndBytes
  * Single-GPU bypass of distributed init (no NCCL hang on Modal kernels)

Ingests JSONL annotations:
    {"messages": [
        {"role": "system", "content": "..."},
        {"role": "user",   "content": [
            {"type": "text",  "text":  "..."},
            {"type": "image", "image": "/data/.../frame_0001.jpg"}
        ]},
        {"role": "assistant", "content": "..."}
    ]}

Key CLI args:
    --annotations_dir  DIR   directory with train.jsonl + val.jsonl
    --frames_root      DIR   fallback root for stale absolute image paths
    --output_dir       DIR   where to save the LoRA adapter
    --warmup_steps     N     warmup steps (default 10)
    --lr               F     learning rate (default 2e-4)
    --lora_r           N     LoRA rank (default 32)
    --lora_alpha       N     LoRA alpha (default 32)
    --load_in_4bit           enable 4-bit quantization (default True)
    --merge_adapter          merge LoRA into base before save
"""



from __future__ import annotations

from unsloth import FastVisionModel

from trl import SFTConfig, SFTTrainer

from transformers.trainer_callback import TrainerCallback

from datasets import Dataset

from PIL import Image

import torch

from typing import Any

from pathlib import Path

import time

import sys

import logging

import json

import copy

import argparse

from unsloth.trainer import UnslothVisionDataCollator

from unsloth import get_chat_template

import logging as _logging



import os

import warnings











os.environ.setdefault("UNSLOTH_COMPILE_DISABLE", "1")

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

os.environ.setdefault("UNSLOTH_DISABLE_FAST_GENERATION", "1")



warnings.filterwarnings(

    "ignore", message="Model has `tie_word_embeddings=True`")

warnings.filterwarnings(

    "ignore", message=".*kwargs.*were applied for backward compatibility")

warnings.filterwarnings("ignore", message="Detected kernel version")





class _SquelchKnownNoise(_logging.Filter):

    """Drop log records matching known-harmless patterns."""

    _PATTERNS = (

        "kwargs",

        "Skipping lm_head",

        "no quant_state found",

    )



    def filter(self, record: _logging.LogRecord) -> bool:

        msg = record.getMessage()

        return not any(p in msg for p in self._PATTERNS)





_logging.getLogger().addFilter(_SquelchKnownNoise())











try:

    import unsloth.models._utils as _unsloth_utils

    _unsloth_utils._get_statistics = lambda *a, **kw: None

    _unsloth_utils.get_statistics = lambda *a, **kw: None

except Exception:

    pass





if os.getenv("ENABLE_TORCH_COMPILE") == "1":

    torch._dynamo.config.suppress_errors = True

else:

    torch._dynamo.config.suppress_errors = True

    torch._dynamo.config.disable = True





logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s [%(levelname)s] %(message)s",

)

log = logging.getLogger("gemma3")



DEFAULT_MODEL = "unsloth/gemma-3-4b-pt"



INSTRUCTION_PART = "<start_of_turn>user\n"

RESPONSE_PART = "<start_of_turn>model\n"





def parse_args() -> argparse.Namespace:

    p = argparse.ArgumentParser(

        description="Gemma 3 4B QLoRA vision+language SFT via Unsloth (v23).",

        formatter_class=argparse.ArgumentDefaultsHelpFormatter,

    )

    p.add_argument("--annotations_dir", required=True, type=Path)

    p.add_argument("--frames_root",     default=None,  type=Path)

    p.add_argument("--extra_frames_roots", nargs="*", default=[], type=Path,

                   help="Additional directories to search for images (e.g. augmented data).")

    p.add_argument("--output_dir",      required=True, type=Path)



    p.add_argument("--model_name", default=DEFAULT_MODEL)

    p.add_argument("--hf_token",   default="")

    p.add_argument("--load_in_4bit",    dest="load_in_4bit",

                   action="store_true",  default=True)

    p.add_argument("--no_load_in_4bit", dest="load_in_4bit",

                   action="store_false")

    p.add_argument("--bf16",     dest="bf16",

                   action="store_true",  default=True)

    p.add_argument("--no_bf16",  dest="bf16", action="store_false")







    p.add_argument("--smoke_test", action="store_true")

    p.add_argument("--lr",         type=float, default=2e-4)

    p.add_argument("--batch_size", type=int,   default=1)

    p.add_argument("--grad_accum", type=int,   default=4)

    p.add_argument("--num_epochs",    type=int,   default=5)

    p.add_argument("--warmup_steps",  type=int,   default=10)

    p.add_argument("--weight_decay",  type=float, default=0.001)

    p.add_argument("--max_seq_length", type=int,  default=2048)

    p.add_argument("--optim", default="adamw_torch_fused",

                   help="adamw_torch_fused | adamw_8bit | paged_adamw_32bit")

    p.add_argument("--early_stopping_patience", type=int, default=3,

                   help="Stop if eval_loss stalls for N epochs. 0 = disabled.")





    p.add_argument("--lora_r",       type=int,   default=32)

    p.add_argument("--lora_alpha",   type=int,   default=32)

    p.add_argument("--lora_dropout", type=float, default=0.0)

    p.add_argument("--use_rslora",   action="store_true", default=False)





    p.add_argument("--max_images_per_sample", type=int, default=6,

                   help="Cap images per sample (0 = unlimited).")





    p.add_argument("--merge_adapter", action="store_true")





    p.add_argument("--print_modules", action="store_true",

                   help="Log every trainable module name.")



    return p.parse_args()





_FRAMES_INDEX_CACHE: dict[Path, dict[str, Path]] = {}





def _build_frames_index(frames_root: Path) -> dict[str, Path]:

    """Index every image file under frames_root by basename for stale-path fallback."""

    if frames_root in _FRAMES_INDEX_CACHE:

        return _FRAMES_INDEX_CACHE[frames_root]

    index: dict[str, Path] = {}

    for ext in ("*.jpg", "*.jpeg", "*.png", "*.webp"):

        for p in frames_root.rglob(ext):

            index.setdefault(p.name, p)

    log.info("frames index built: %d unique files in %s",

             len(index), frames_root)

    _FRAMES_INDEX_CACHE[frames_root] = index

    return index





def _resolve_image_path(img_path: str, frames_roots: list[Path]) -> Path:

    """Resolve an image path, falling back to basename lookup in frames_roots."""

    p = Path(img_path)

    if p.exists():

        return p

    for root in frames_roots:

        if root is not None:

            idx = _build_frames_index(root)

            hit = idx.get(p.name)

            if hit is not None:

                return hit

    raise FileNotFoundError(

        f"Image not found: {img_path}  (searched {len(frames_roots)} root(s): {frames_roots})"

    )





def _trim_messages_to_max_images(messages: list[dict], max_images: int) -> list[dict]:

    if max_images <= 0:

        return messages

    out = copy.deepcopy(messages)

    kept = 0

    for msg in out:

        content = msg.get("content")

        if not isinstance(content, list):

            continue

        new_content: list[dict] = []

        for part in content:

            if isinstance(part, dict) and part.get("type") == "image":

                if kept < max_images:

                    new_content.append(part)

                    kept += 1

            else:

                new_content.append(part)

        msg["content"] = new_content

    return out





def _to_unsloth_messages(messages: list[dict], frames_roots: list[Path]) -> list[dict]:

    """Convert {"image": path-string} → {"image": PIL.Image} and wrap str content.
    Eagerly opens images — use for inference only (small batch)."""

    out: list[dict] = []

    for msg in messages:

        role = msg["role"]

        content = msg.get("content")



        if isinstance(content, str):

            out.append({"role": role, "content": [

                       {"type": "text", "text": content}]})

            continue



        new_content: list[dict] = []

        for part in content:

            if not isinstance(part, dict):

                continue

            ptype = part.get("type")

            if ptype == "text":

                new_content.append({"type": "text", "text": part["text"]})

            elif ptype == "image":

                src = part["image"]

                if isinstance(src, Image.Image):

                    pil = src

                else:

                    path = _resolve_image_path(str(src), frames_roots)

                    pil = Image.open(path).convert("RGB")

                new_content.append({"type": "image", "image": pil})

        out.append({"role": role, "content": new_content})

    return out





def _to_lazy_messages(messages: list[dict], frames_roots: list[Path]) -> list[dict]:

    """Like _to_unsloth_messages but stores resolved path strings instead of
    PIL images. This avoids holding thousands of uncompressed images in RAM
    during dataset construction. Images are opened lazily via _materialize_images."""

    out: list[dict] = []

    for msg in messages:

        role = msg["role"]

        content = msg.get("content")



        if isinstance(content, str):

            out.append({"role": role, "content": [

                       {"type": "text", "text": content}]})

            continue



        new_content: list[dict] = []

        for part in content:

            if not isinstance(part, dict):

                continue

            ptype = part.get("type")

            if ptype == "text":

                new_content.append({"type": "text", "text": part["text"]})

            elif ptype == "image":

                src = part["image"]

                resolved = str(_resolve_image_path(str(src), frames_roots))

                new_content.append({"type": "image", "image": resolved})

        out.append({"role": role, "content": new_content})

    return out





def _materialize_images(batch: dict) -> dict:

    """set_transform callback: open PIL images on-the-fly from stored paths.
    Called by HF Dataset.__getitem__ — only the current batch is in memory."""

    new_batch: dict = {}

    for key, values in batch.items():

        if key == "messages":

            new_messages_list = []

            for messages in values:

                materialized = []

                for msg in messages:

                    role = msg["role"]

                    content = msg.get("content")

                    if not isinstance(content, list):

                        materialized.append(msg)

                        continue

                    new_content = []

                    for part in content:

                        if isinstance(part, dict) and part.get("type") == "image":

                            img_val = part["image"]

                            if isinstance(img_val, str):

                                pil = Image.open(img_val).convert("RGB")

                                new_content.append(

                                    {"type": "image", "image": pil})

                            else:

                                new_content.append(part)

                        else:

                            new_content.append(part)

                    materialized.append({"role": role, "content": new_content})

                new_messages_list.append(materialized)

            new_batch[key] = new_messages_list

        else:

            new_batch[key] = values

    return new_batch





def _load_jsonl(path: Path) -> list[dict]:

    samples: list[dict] = []

    for line in path.read_text(encoding="utf-8").splitlines():

        line = line.strip()

        if line:

            samples.append(json.loads(line))

    if not samples:

        raise ValueError(f"Empty dataset: {path}")

    log.info("Loaded %d samples from %s", len(samples), path)

    return samples





def _build_hf_dataset(

    raw: list[dict], frames_roots: list[Path], max_images: int

) -> Dataset:

    """Build dataset with lazy image loading. Stores resolved file paths as
    strings; PIL images are opened on-the-fly via set_transform() during
    collation, keeping system RAM usage constant regardless of dataset size."""

    def _nonempty_text(parts: list[dict]) -> bool:

        for p in parts:

            if p.get("type") == "text" and str(p.get("text", "")).strip():

                return True

        return False



    rows: list[dict] = []

    skipped = 0

    n_total = len(raw)

    t0 = time.time()

    log.info(

        "Building dataset: %d samples (resolving image paths, lazy PIL load)...", n_total)

    for i, sample in enumerate(raw):

        msgs = _trim_messages_to_max_images(sample["messages"], max_images)

        converted = _to_lazy_messages(msgs, frames_roots)

        has_user_text = any(

            m.get("role") == "user" and isinstance(

                m.get("content"), list) and _nonempty_text(m["content"])

            for m in converted

        )

        has_assistant_text = any(

            m.get("role") == "assistant" and isinstance(

                m.get("content"), list) and _nonempty_text(m["content"])

            for m in converted

        )

        if not (has_user_text and has_assistant_text):

            skipped += 1

            continue

        rows.append({"messages": converted})

        if (i + 1) % 200 == 0 or (i + 1) == n_total:

            elapsed = time.time() - t0

            rate = (i + 1) / elapsed if elapsed > 0 else 0

            eta = (n_total - i - 1) / rate if rate > 0 else 0

            log.info(

                "  dataset progress: %d/%d (%.0f%%) — %.1f samples/s, ETA %.0fs",

                i + 1, n_total, 100.0 * (i + 1) / n_total, rate, eta,

            )

    elapsed = time.time() - t0

    log.info("Dataset build complete: %d rows in %.1fs (skipped %d).",

             len(rows), elapsed, skipped)

    if skipped:

        log.warning("Dropped %d rows with empty user/assistant text.", skipped)

    if not rows:

        raise ValueError(

            "All rows were dropped as invalid (empty user/assistant text).")

    ds = Dataset.from_list(rows)

    ds.set_transform(_materialize_images)

    return ds





class EarlyLossDiagnosticCallback(TrainerCallback):

    """v5 fix: detect broken training early (first 10 steps).

    Gemma 3 multimodal training should see loss drop from ~10-12 to ~7-8
    within the first 10 steps. If loss hasn't dropped sufficiently by step 10,
    something is wrong (bad data, frozen adapters, numerical issue) and
    continuing wastes H100 time. Log a big warning — user can CTRL-C to abort.

    Adaptive threshold = max(8.5, initial_loss * 0.8).  Captures the
    actual initial loss at step 1 instead of assuming a fixed 9.5 ceiling,
    making the check reliable across different datasets and model configs.
    """

    _DEFAULT_FLOOR: float = 8.5

    _DECAY_FACTOR: float = 0.8



    def __init__(self, check_step: int = 10):

        self.check_step = check_step

        self.initial_loss: float | None = None

        self.checked = False



    def on_log(self, args, state, control, logs=None, **kwargs):

        if logs is None or "loss" not in logs:

            return

        loss = logs["loss"]



        if self.initial_loss is None and state.global_step >= 1:

            self.initial_loss = loss

            log.info(

                "DIAGNOSTIC — initial loss captured at step %d: %.3f",

                state.global_step, loss,

            )

            return



        if self.checked or state.global_step < self.check_step:

            return



        self.checked = True

        threshold = max(

            self._DEFAULT_FLOOR,

            (self.initial_loss or 12.0) * self._DECAY_FACTOR,

        )



        if loss > threshold:

            log.error(

                "=" * 70 + "\n"

                "DIAGNOSTIC — Loss at step %d is %.3f (expected <%.1f, "

                "adaptive from initial=%.3f).\n"

                "Training may be broken (bad labels, frozen adapters, NaN grads).\n"

                "Continuing will waste H100 time. CTRL-C NOW to investigate.\n"

                "Common causes:\n"

                "  - Label masking removed all supervised tokens → check dry-run output\n"

                "  - Vision LoRA not wrapping enough layers → re-run verify_paths\n"

                "  - Dataset has broken image paths → check frames_index size\n"

                + "=" * 70,

                state.global_step, loss, threshold,

                self.initial_loss or 0.0,

            )

        else:

            log.info(

                "DIAGNOSTIC — Loss at step %d is %.3f (threshold <%.1f, "

                "initial=%.3f) ✓",

                state.global_step, loss, threshold,

                self.initial_loss or 0.0,

            )





def _extract_sample_class(row: dict) -> str:

    """Extract incident_type from a dataset sample's assistant JSON response."""

    for msg in row.get("messages", []):

        if msg.get("role") != "assistant":

            continue

        content = msg.get("content")

        text = ""

        if isinstance(content, str):

            text = content

        elif isinstance(content, list):

            for part in content:

                if isinstance(part, dict) and part.get("type") == "text":

                    text = part.get("text", "")

                    break

        if not text.strip():

            continue

        try:

            data = json.loads(text)

            if isinstance(data, dict):

                return str(data.get("incident_type", "unknown"))

        except (json.JSONDecodeError, TypeError, ValueError):

            pass

    return "unknown"





class PerClassMetricsCallback(TrainerCallback):

    """Generation-based per-class eval at each evaluation point.

    Samples up to ``max_samples`` from the validation set, runs
    model.generate(), parses the JSON output, and computes:
      - per-class recall
      - per-class precision & F1
      - macro F1
      - confusion matrix

    Explicitly highlights shooting and fainting recall.
    """



    _MAX_NEW_TOKENS: int = 256



    def __init__(

        self,

        model: Any,

        tokenizer: Any,

        val_dataset: Any,

        max_samples: int = 30,

    ):

        self.model = model

        self.tokenizer = tokenizer

        self.val_dataset = val_dataset

        self.max_samples = max_samples



    def on_evaluate(self, args, state, control, **kwargs):

        if self.val_dataset is None or len(self.val_dataset) == 0:

            return

        n = min(self.max_samples, len(self.val_dataset))

        start = time.time()



        y_true: list[str] = []

        y_pred: list[str] = []



        was_training = self.model.training

        FastVisionModel.for_inference(self.model)

        self.model.eval()



        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):

            for idx in range(n):

                sample = self.val_dataset[idx]

                true_cls = _extract_sample_class(sample)



                prompt_msgs = [

                    m for m in sample["messages"]

                    if m.get("role") != "assistant"

                ]



                try:

                    inputs = self.tokenizer.apply_chat_template(

                        prompt_msgs,

                        tokenize=True,

                        add_generation_prompt=True,

                        return_dict=True,

                        return_tensors="pt",

                    )

                    if isinstance(inputs, str):

                        raise TypeError(

                            "apply_chat_template returned str; expected tokenized dict. "

                            "Check processor/chat template compatibility."

                        )

                    inputs = {

                        k: (v.to(device=self.model.device, dtype=torch.bfloat16)

                            if hasattr(v, "to") and v.is_floating_point()

                            else v.to(self.model.device) if hasattr(v, "to")

                            else v)

                        for k, v in inputs.items()

                    }



                    outputs = self.model.generate(

                        **inputs,

                        max_new_tokens=self._MAX_NEW_TOKENS,

                        do_sample=False,

                        temperature=1.0,

                        use_cache=False,

                    )



                    input_len = inputs["input_ids"].shape[1]

                    generated = self.tokenizer.decode(

                        outputs[0][input_len:], skip_special_tokens=True,

                    )



                    pred_cls = "unknown"

                    try:

                        data = json.loads(generated.strip())

                        if isinstance(data, dict):

                            pred_cls = str(

                                data.get("incident_type", "unknown"))

                    except (json.JSONDecodeError, TypeError, ValueError):

                        pass



                    y_true.append(true_cls)

                    y_pred.append(pred_cls)



                except Exception as e:

                    if idx < 3:

                        log.warning(

                            "PerClassMetrics: generation failed for sample %d (%s): %s",

                            idx, type(e).__name__, e,

                        )

                    else:

                        log.debug(

                            "PerClassMetrics: generation failed for sample %d: %s", idx, e)

                    continue



        if was_training:

            FastVisionModel.for_training(self.model)

            self.model.train()



        elapsed = time.time() - start



        if not y_true:

            log.warning("PerClassMetrics: 0 samples evaluated successfully.")

            return



        self._log_metrics(y_true, y_pred, elapsed, state.global_step)



    def _log_metrics(

        self,

        y_true: list[str],

        y_pred: list[str],

        elapsed: float,

        step: int,

    ) -> None:

        all_classes = sorted(set(y_true + y_pred))



        confusion: dict[str, dict[str, int]] = {

            t: {p: 0 for p in all_classes} for t in all_classes

        }

        for t, p in zip(y_true, y_pred):

            confusion[t][p] += 1



        recalls: dict[str, float] = {}

        precisions: dict[str, float] = {}

        f1s: dict[str, float] = {}



        for cls in all_classes:

            tp = confusion[cls][cls]

            row_sum = sum(confusion[cls].values())

            recalls[cls] = tp / row_sum if row_sum > 0 else 0.0

            col_sum = sum(confusion[t][cls] for t in all_classes)

            precisions[cls] = tp / col_sum if col_sum > 0 else 0.0

            p, r = precisions[cls], recalls[cls]

            f1s[cls] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0



        macro_f1 = sum(f1s.values()) / len(f1s) if f1s else 0.0



        log.info("=" * 60)

        log.info(

            "PER-CLASS EVAL (step %d, %d samples, %.0fs):",

            step, len(y_true), elapsed,

        )

        log.info("  Macro F1: %.3f", macro_f1)

        log.info("  %-20s  %8s  %8s  %8s", "class", "recall", "prec", "F1")

        for cls in all_classes:

            tag = "  <<<" if cls in ("shooting", "fainting") else ""

            log.info(

                "  %-20s  %8.3f  %8.3f  %8.3f%s",

                cls, recalls[cls], precisions[cls], f1s[cls], tag,

            )



        for rare in ("shooting", "fainting"):

            if rare in recalls:

                log.info("  ** %s recall: %.3f **", rare, recalls[rare])



        log.info("  Confusion matrix (rows=true, cols=pred):")

        hdr = "  %-14s" + "".join("%-10s" % c[:9] for c in all_classes)

        log.info(hdr, "")

        for t in all_classes:

            row = "  %-14s" + "".join("%-10d" %

                                      confusion[t][p] for p in all_classes)

            log.info(row, t[:13])

        log.info("=" * 60)





def _save_training_summary(

    args: argparse.Namespace,

    train_result: Any,

    class_counts: dict[str, int],

    train_ds: Any,

    val_ds: Any,

) -> None:

    """Persist training hyperparameters and final metrics as JSON."""

    summary = {

        "version": "v18",

        "hyperparameters": {

            "model_name": args.model_name,

            "load_in_4bit": args.load_in_4bit,

            "lr": args.lr,

            "batch_size": args.batch_size,

            "grad_accum": args.grad_accum,

            "effective_batch_size": args.batch_size * args.grad_accum,

            "num_epochs": args.num_epochs,

            "warmup_steps": args.warmup_steps,

            "lora_r": args.lora_r,

            "lora_alpha": args.lora_alpha,

            "lora_dropout": args.lora_dropout,

            "weight_decay": args.weight_decay,

            "max_seq_length": args.max_seq_length,

            "max_images_per_sample": args.max_images_per_sample,

            "optim": args.optim,

        },

        "dataset": {

            "train_samples": len(train_ds),

            "val_samples": len(val_ds) if val_ds else 0,

            "class_counts": class_counts,

        },

        "final_metrics": train_result.metrics if train_result else {},

    }

    path = args.output_dir / "training_summary.json"

    try:

        path.write_text(json.dumps(summary, indent=2, default=str))

        log.info("Training summary saved -> %s", path)

    except Exception as e:

        log.warning("Could not save training summary: %s", e)





def _log_class_distribution(dataset: Any) -> dict[str, int]:

    """Log incident-class distribution. Returns class_counts dict."""

    class_counts: dict[str, int] = {}

    for row in dataset:

        cls = _extract_sample_class(row)

        class_counts[cls] = class_counts.get(cls, 0) + 1



    total = sum(class_counts.values())

    if not class_counts:

        log.info("Class distribution: could not parse any assistant JSON.")

        return {}



    log.info("CLASS DISTRIBUTION (%d samples):", total)

    for cls in sorted(class_counts, key=class_counts.get, reverse=True):

        pct = 100 * class_counts[cls] / total

        log.info("  %-20s %4d  (%5.1f%%)", cls, class_counts[cls], pct)



    if len(class_counts) >= 2:

        max_count = max(class_counts.values())

        min_count = min(class_counts.values())

        ratio = max_count / min_count if min_count > 0 else float("inf")

        if ratio > 3:

            minority = min(class_counts, key=class_counts.get)

            majority = max(class_counts, key=class_counts.get)

            log.warning(

                "CLASS IMBALANCE: '%s' (%d) vs '%s' (%d) = %.1fx ratio.",

                majority, max_count, minority, min_count, ratio,

            )



    return class_counts





def _verify_lora_targets(model: Any, print_all: bool = False) -> None:

    """Count LoRA / fully-trained / frozen params, separating vision vs language.

    Hard-fails if 0 total trainable params (Unsloth/PEFT bug).
    Logs a warning if 0 vision LoRA params.

    Gemma 3 vision architecture uses SigLIP (ViT) with nn.Linear layers.
    Unsloth with finetune_vision_layers=True + "all-linear" wraps them with LoRA.
    """

    _VISION_PATH_KEYWORDS = (

        "vision_tower", "vision_model", "multi_modal_projector",

        "embed_vision", "siglip",

        "vision", "visual", "image"

    )



    _LANGUAGE_PATH_KEYWORDS = (

        "language_model", "lm_head", "embed_tokens",

    )



    _AUDIO_PATH_KEYWORDS = (

        "audio_tower", "embed_audio",

    )



    def _is_vision(name: str) -> bool:

        n = name.lower()

        if any(prefix in n for prefix in _LANGUAGE_PATH_KEYWORDS):

            return False

        if any(prefix in n for prefix in _AUDIO_PATH_KEYWORDS):

            return False

        if any(kw in n for kw in _VISION_PATH_KEYWORDS):

            return True

        return False



    lora_lang: list[str] = []

    lora_vision: list[str] = []

    fully_trained: list[str] = []

    frozen_count: int = 0



    for name, param in model.named_parameters():

        if param.requires_grad:

            if "lora" in name.lower():

                if _is_vision(name):

                    lora_vision.append(name)

                else:

                    lora_lang.append(name)

            else:

                fully_trained.append(name)

        else:

            frozen_count += 1



    total_lora = len(lora_lang) + len(lora_vision)

    total_trainable = total_lora + len(fully_trained)



    log.info(

        "LoRA verification (post-PEFT):\n"

        "  LoRA-adapted (language): %d\n"

        "  LoRA-adapted (vision):   %d\n"

        "  Fully-trained params:    %d\n"

        "  Frozen (4-bit) params:   %d",

        len(lora_lang), len(lora_vision), len(fully_trained), frozen_count,

    )



    if print_all:

        log.info("── All trainable modules (--print_modules) ──")

        for name in lora_vision:

            log.info("  [LoRA-V] %s", name)

        for name in lora_lang:

            log.info("  [LoRA-L] %s", name)

        for name in fully_trained:

            log.info("  [FULL]   %s", name)



    if total_trainable == 0:

        raise RuntimeError(

            "FATAL: 0 trainable parameters after get_peft_model(). "

            "Known issue (unsloth#3222). "

            "Try: pip install --upgrade unsloth peft transformers trl"

        )



    if len(lora_vision) == 0:

        log.warning(

            "0 vision LoRA params detected. Check that finetune_vision_layers=True "

            "is set in get_peft_model(). (%d language LoRA tensors active).",

            len(lora_lang),

        )

    else:

        log.info(

            "Verified: %d vision LoRA + %d language LoRA tensors ✓",

            len(lora_vision), len(lora_lang),

        )





def main() -> None:

    args = parse_args()



    for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):

        os.environ.pop(k, None)

    os.environ.setdefault("ACCELERATE_USE_DEEPSPEED", "false")

    os.environ.setdefault("ACCELERATE_USE_FSDP",      "false")



    if torch.cuda.is_available():

        log.info(

            "GPU: %s  (%.0f GB VRAM)",

            torch.cuda.get_device_name(0),

            torch.cuda.get_device_properties(0).total_memory / 1e9,

        )

    else:

        log.warning("CUDA not available — will fall back to CPU (very slow).")



    hf_token: str | None = (

        args.hf_token.strip() or os.environ.get("HF_TOKEN", "").strip() or None

    )



    log.info(

        "Loading %s (load_in_4bit=%s, max_seq=%d)",

        args.model_name, args.load_in_4bit, args.max_seq_length,

    )



    model, processor = FastVisionModel.from_pretrained(

        args.model_name,

        load_in_4bit=args.load_in_4bit,

        use_gradient_checkpointing="unsloth",

        token=hf_token,

    )



    eff_batch = args.batch_size * args.grad_accum

    log.info("=" * 60)

    log.info("TRAINING CONFIG — Full QLoRA (vision + language)")

    log.info("  Model:                 %s", args.model_name)

    log.info("  4-bit quantization:    %s", args.load_in_4bit)

    log.info("  Effective batch size:  %d  (batch=%d x grad_accum=%d)",

             eff_batch, args.batch_size, args.grad_accum)

    log.info("  Epochs:                %d", args.num_epochs)

    log.info("  Warmup steps:          %d", args.warmup_steps)

    log.info("  LR:                    %.1e", args.lr)

    log.info("  LoRA:                  r=%d alpha=%d dropout=%.2f",

             args.lora_r, args.lora_alpha, args.lora_dropout)

    log.info("  LoRA targets:          all-linear (vision + language)")

    log.info("  Optimizer:             %s", args.optim)

    log.info("  Max grad norm:         %.1f", 0.3)

    log.info("  Max seq length:        %d", args.max_seq_length)

    log.info("  Max images/sample:     %d", args.max_images_per_sample)

    log.info("=" * 60)



    log.info(

        "Attaching QLoRA: r=%d alpha=%d dropout=%.2f targets=all-linear 4bit=%s",

        args.lora_r, args.lora_alpha, args.lora_dropout, args.load_in_4bit,

    )

    model = FastVisionModel.get_peft_model(

        model,

        finetune_vision_layers=True,

        finetune_language_layers=True,

        finetune_attention_modules=True,

        finetune_mlp_modules=True,

        target_modules="all-linear",

        r=args.lora_r,

        lora_alpha=args.lora_alpha,

        lora_dropout=args.lora_dropout,

        bias="none",

        use_rslora=args.use_rslora,

        random_state=3407,

        loftq_config=None,

    )

    model.print_trainable_parameters()

    _verify_lora_targets(model, print_all=args.print_modules)



    processor = get_chat_template(processor, "gemma-3")

    log.info("Applied Gemma 3 chat template via get_chat_template().")



    model.config.use_cache = False



    log.info("TorchDynamo disabled: %s", torch._dynamo.config.disable)

    if "unsloth_compiled_cache" in str(getattr(model, "__module__", "")):

        log.warning("Unsloth compiled cache still active — forcing eager mode.")

        torch._dynamo.config.disable = True



    train_path = args.annotations_dir / "train.jsonl"

    val_path = args.annotations_dir / "val.jsonl"

    if not train_path.exists():

        raise FileNotFoundError(f"Missing train.jsonl: {train_path}")



    raw_train = _load_jsonl(train_path)

    raw_val = _load_jsonl(val_path) if val_path.exists() else []



    if args.smoke_test:

        log.warning("SMOKE TEST: 3 train / 0 val, max_steps=1.")

        raw_train = raw_train[:3]

        raw_val = []

        if args.max_images_per_sample == 0:

            args.max_images_per_sample = 2



    frames_roots: list[Path] = []

    if args.frames_root is not None:

        frames_roots.append(args.frames_root)

    frames_roots.extend(args.extra_frames_roots)

    if frames_roots:

        log.info("Image search roots: %s", frames_roots)



    train_ds = _build_hf_dataset(

        raw_train, frames_roots, args.max_images_per_sample)

    val_ds = (_build_hf_dataset(raw_val, frames_roots, args.max_images_per_sample)

              if raw_val and not args.smoke_test else None)



    log.info("Dataset: %d train, %s val",

             len(train_ds), len(val_ds) if val_ds is not None else "—")



    class_counts = _log_class_distribution(train_ds)



    args.output_dir.mkdir(parents=True, exist_ok=True)



    _has_early_stop = (

        args.early_stopping_patience > 0

        and val_ds is not None

        and not args.smoke_test

    )



    sft_config_kwargs: dict[str, Any] = dict(

        output_dir=str(args.output_dir),

        per_device_train_batch_size=args.batch_size,

        gradient_accumulation_steps=args.grad_accum,

        gradient_checkpointing=True,

        gradient_checkpointing_kwargs={"use_reentrant": False},

        max_grad_norm=0.3,

        warmup_steps=args.warmup_steps,

        num_train_epochs=args.num_epochs,

        learning_rate=args.lr,

        weight_decay=args.weight_decay,

        lr_scheduler_type="cosine",

        optim=args.optim,

        logging_steps=1,

        save_strategy="epoch",

        save_total_limit=3,

        seed=3407,

        fp16=False,

        bf16=True,

        report_to="none",

        remove_unused_columns=False,

        dataset_text_field="",

        dataset_kwargs={"skip_prepare_dataset": True},

        max_seq_length=args.max_seq_length,

    )

    log.info("Training for %d epochs.", args.num_epochs)



    if _has_early_stop:

        sft_config_kwargs["eval_strategy"] = "epoch"

        sft_config_kwargs["load_best_model_at_end"] = True

        sft_config_kwargs["metric_for_best_model"] = "eval_loss"

        sft_config_kwargs["greater_is_better"] = False

        log.info(

            "Early stopping: patience=%d epochs (will train until overfit then restore best).",

            args.early_stopping_patience,

        )



    if args.smoke_test:

        sft_config_kwargs["max_steps"] = 1



    callbacks = [EarlyLossDiagnosticCallback()]

    if _has_early_stop:

        from transformers import EarlyStoppingCallback

        callbacks.append(

            EarlyStoppingCallback(

                early_stopping_patience=args.early_stopping_patience)

        )

    if val_ds is not None and not args.smoke_test:

        callbacks.append(

            PerClassMetricsCallback(

                model=model, tokenizer=processor,

                val_dataset=val_ds, max_samples=15,

            )

        )



    FastVisionModel.for_training(model)



    _use_response_masking = not args.smoke_test

    _instr_part = INSTRUCTION_PART

    _resp_part = RESPONSE_PART

    if _use_response_masking:

        log.info(

            "UnslothVisionDataCollator: train_on_responses_only=True "

            "(instruction_part=%r, response_part=%r)",

            _instr_part, _resp_part,

        )



    _data_collator = UnslothVisionDataCollator(

        model, processor,

        train_on_responses_only=_use_response_masking,

        instruction_part=_instr_part if _use_response_masking else None,

        response_part=_resp_part if _use_response_masking else None,

    )



    trainer = SFTTrainer(

        model=model,

        processing_class=processor.tokenizer,

        data_collator=_data_collator,

        train_dataset=train_ds,

        eval_dataset=val_ds,

        args=SFTConfig(**sft_config_kwargs),

        callbacks=callbacks,

    )

    log.info(

        "SFTTrainer ready: %d train, epochs=%d, optim=%s, lr=%.1e.",

        len(train_ds), args.num_epochs, args.optim, args.lr,

    )



    post_trainable = sum(1 for p in model.parameters() if p.requires_grad)

    if post_trainable == 0:

        raise RuntimeError(

            "FATAL: trainable params dropped to 0 after SFTTrainer init. "

            "Try: pip install --upgrade unsloth peft"

        )

    post_lora = sum(

        1 for n, p in model.named_parameters()

        if p.requires_grad and "lora" in n.lower()

    )

    log.info("Post-init check: %d trainable (%d LoRA) ✓",

             post_trainable, post_lora)



    if not args.smoke_test:

        log.info("Running masking sanity check on first training sample...")

        try:

            _check_sample = train_ds[0]

            _check_batch = trainer.data_collator([_check_sample])

            _check_labels = _check_batch["labels"][0]

            _total_tokens = _check_labels.numel()

            _masked_tokens = (_check_labels == -100).sum().item()

            _masked_pct = _masked_tokens / _total_tokens if _total_tokens > 0 else 0

            _supervised_tokens = _total_tokens - _masked_tokens



            log.info(

                "MASKING CHECK: %d/%d tokens masked (%.1f%%), "

                "%d supervised tokens.",

                _masked_tokens, _total_tokens, 100 * _masked_pct,

                _supervised_tokens,

            )



            if _masked_pct < 0.20:

                raise RuntimeError(

                    f"MASKING BROKEN: only {_masked_pct:.1%} of tokens masked "

                    f"({_masked_tokens}/{_total_tokens}). Expected >50%. "

                    "Aborting to save GPU time."

                )

            if _supervised_tokens < 10:

                raise RuntimeError(

                    f"MASKING TOO AGGRESSIVE: only {_supervised_tokens} "

                    "supervised tokens. Check instruction/response markers."

                )

            log.info("Masking sanity check PASSED ✓")



        except RuntimeError:

            raise

        except Exception as e:

            log.warning(

                "Masking sanity check failed with %s: %s — continuing.", type(e).__name__, e)



    log.info("Starting training (%d train, %s val) ...",

             len(train_ds), len(val_ds) if val_ds is not None else "—")

    try:

        train_result = trainer.train()

    except RuntimeError as e:

        if "Dynamo" in str(e) or "FakeTensor" in str(e):

            log.warning(

                "Dynamo error detected — retrying with compile fully disabled.")

            os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

            torch._dynamo.config.disable = True

            train_result = trainer.train()

        else:

            raise

    log.info("Training done. metrics=%s", train_result.metrics)



    if args.smoke_test:

        log.info("Smoke test complete — adapter not saved.")

        return



    adapter_dir = args.output_dir / "adapter"

    adapter_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(adapter_dir))

    processor.save_pretrained(str(adapter_dir))

    log.info("LoRA adapter saved → %s", adapter_dir)



    _save_training_summary(args, train_result, class_counts, train_ds, val_ds)



    if args.merge_adapter:

        merged_dir = args.output_dir / "merged"

        merged_dir.mkdir(parents=True, exist_ok=True)

        log.info("Merging LoRA into base model → %s", merged_dir)

        model.save_pretrained_merged(

            str(merged_dir), processor, save_method="merged_16bit",

        )

        log.info("Merged model saved → %s", merged_dir)





if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        log.warning("Interrupted by user.")

        sys.exit(130)
