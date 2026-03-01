import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Gemma with QLoRA for incident reasoning.")
    parser.add_argument("--model_name", default="google/gemma-2-2b-it", help="Base HF model ID, e.g. google/gemma-3n-e4b-it")
    parser.add_argument("--train_file", required=True, help="JSONL dataset path (messages or prompt/response)")
    parser.add_argument("--output_dir", default="outputs/gemma-shoplifting-qlora", help="Output directory")
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true", help="Enable bf16 if GPU supports it")
    parser.add_argument("--merge_and_save", action="store_true", help="Merge LoRA into base model and save")
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha (scaling)")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Run on CPU or MPS (Mac) without NVIDIA GPU; uses full precision or bf16, no 4-bit quant",
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=os.environ.get("HF_TOKEN", ""),
        help="Hugging Face token for gated models (e.g. Gemma). Or set HF_TOKEN or run: huggingface-cli login",
    )
    parser.add_argument(
        "--use_open_model",
        action="store_true",
        help="Use non-gated model (Qwen2-0.5B-Instruct) if you don't have Gemma access. No HF login needed.",
    )
    return parser.parse_args()


def _to_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    if "messages" in row and isinstance(row["messages"], list):
        return row["messages"]

    prompt = str(row.get("prompt", "")).strip()
    response = str(row.get("response", "")).strip()
    system = row.get("system")

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": str(system)})
    else:
        messages.append(
            {
                "role": "system",
                "content": "You are a security incident reasoning model. Return strict JSON only.",
            }
        )
    messages.append({"role": "user", "content": prompt})
    messages.append({"role": "assistant", "content": response})
    return messages


def load_jsonl_dataset(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"Empty training file: {path}")
    return rows


def format_chat(tokenizer: AutoTokenizer, messages: list[dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    rendered = []
    for msg in messages:
        role = msg.get("role", "user").upper()
        content = msg.get("content", "")
        rendered.append(f"{role}: {content}")
    return "\n".join(rendered)


def build_dataset(tokenizer: AutoTokenizer, rows: list[dict[str, Any]], max_seq_length: int) -> Dataset:
    texts = [format_chat(tokenizer, _to_messages(row)) for row in rows]
    ds = Dataset.from_dict({"text": texts})

    def _tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        tokens = tokenizer(
            batch["text"],
            truncation=True,
            padding=False,
            max_length=max_seq_length,
        )
        return tokens

    return ds.map(_tokenize, batched=True, remove_columns=["text"])


def main() -> None:
    args = parse_args()
    train_path = Path(args.train_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_cuda = torch.cuda.is_available() and not args.no_cuda
    use_mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available() and not use_cuda
    use_cpu = not use_cuda and not use_mps

    if use_cuda:
        device_map = "auto"
        load_dtype = torch.bfloat16 if args.bf16 else torch.float16
    elif use_mps:
        device_map = "mps"
        load_dtype = torch.bfloat16 if args.bf16 else torch.float16
    else:
        device_map = "cpu"
        load_dtype = torch.float32

    if not use_cuda and not args.no_cuda:
        raise RuntimeError(
            "No CUDA GPU found. Add --no_cuda to run on CPU or Mac MPS (slower, more RAM)."
        )

    if use_mps:
        print("Using MPS (Apple Silicon). Training may be slower than CUDA.")
    elif use_cpu:
        print("Using CPU. Training will be slow; consider fewer epochs or a smaller model.")

    if args.use_open_model:
        args.model_name = "Qwen/Qwen2-0.5B-Instruct"
        print("Using open model (no gated access): Qwen2-0.5B-Instruct")

    hf_token = args.hf_token if args.hf_token else (None if args.use_open_model else True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if use_cuda:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            token=hf_token,
        )
        model = prepare_model_for_kbit_training(model)
        optim_name = "paged_adamw_8bit"
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=load_dtype,
            device_map=device_map,
            low_cpu_mem_usage=True,
            token=hf_token,
        )
        optim_name = "adamw_torch"

    model.config.use_cache = False

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    rows = load_jsonl_dataset(train_path)
    dataset = build_dataset(tokenizer, rows, args.max_seq_length)

    # bf16/fp16: use bf16 on MPS if requested, else fp32 on CPU
    use_bf16 = args.bf16 and (use_cuda or use_mps)
    use_fp16 = use_cuda and not args.bf16

    train_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=use_fp16,
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        optim=optim_name,
        report_to="none",
        gradient_checkpointing=True,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    trainer.train()

    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    print(f"Saved LoRA adapter to: {adapter_dir}")

    if args.merge_and_save:
        merged_dir = output_dir / "merged"
        merged_dir.mkdir(parents=True, exist_ok=True)
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)
        print(f"Saved merged model to: {merged_dir}")

    meta = {
        "model_name": args.model_name,
        "train_file": str(train_path),
        "output_dir": str(output_dir),
        "max_seq_length": args.max_seq_length,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "merge_and_save": args.merge_and_save,
        "cuda_available": torch.cuda.is_available(),
        "no_cuda": args.no_cuda,
    }
    (output_dir / "train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote training metadata to: {output_dir / 'train_meta.json'}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
