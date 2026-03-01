"""
Lightweight Ollama-compatible server for PEFT LoRA adapters.

Loads base model + LoRA adapter via transformers/peft and exposes
POST /api/generate with the same request/response shape as Ollama,
so the existing LocalGemmaClient works unchanged.

Usage:
    pip install torch peft transformers accelerate
    python scripts/serve_peft_model.py \
        --base-model Qwen/Qwen2-0.5B-Instruct \
        --adapter-dir ./adapter \
        --port 11434
"""

import argparse
import json
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Lock

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve PEFT LoRA model with Ollama-compatible API")
    p.add_argument("--base-model", default="Qwen/Qwen2-0.5B-Instruct", help="HuggingFace base model ID")
    p.add_argument("--adapter-dir", default="./adapter", help="Path to PEFT adapter directory")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=11434)
    p.add_argument("--max-tokens", type=int, default=512)
    return p.parse_args()


_model = None
_tokenizer = None
_lock = Lock()


def load_model(base_model: str, adapter_dir: str):
    global _model, _tokenizer

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device != "cpu" else torch.float32

    print(f"Loading base model: {base_model} (device={device}, dtype={dtype})")
    _tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=dtype, device_map=device, low_cpu_mem_usage=True,
    )
    print(f"Loading LoRA adapter from: {adapter_dir}")
    _model = PeftModel.from_pretrained(base, adapter_dir)
    _model.eval()
    print("Model ready.")


class OllamaHandler(BaseHTTPRequestHandler):
    max_tokens = 512

    def do_POST(self):
        if self.path != "/api/generate":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        prompt = body.get("prompt", "")
        temperature = body.get("options", {}).get("temperature", 0.1)
        max_tokens = body.get("options", {}).get("num_predict", self.max_tokens)

        with _lock:
            response_text = self._generate(prompt, temperature, max_tokens)

        payload = json.dumps({
            "model": body.get("model", "peft-local"),
            "response": response_text,
            "done": True,
        }).encode()

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _generate(self, prompt: str, temperature: float, max_tokens: int) -> str:
        inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)
        t0 = time.perf_counter()

        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            do_sample=temperature > 0,
            pad_token_id=_tokenizer.pad_token_id,
        )
        if temperature > 0:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            output_ids = _model.generate(**inputs, **gen_kwargs)

        new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
        text = _tokenizer.decode(new_tokens, skip_special_tokens=True)
        elapsed = time.perf_counter() - t0
        print(f"Generated {len(new_tokens)} tokens in {elapsed:.2f}s")
        return text

    def log_message(self, fmt, *args):
        print(f"[serve] {fmt % args}")


def main():
    args = parse_args()
    OllamaHandler.max_tokens = args.max_tokens
    load_model(args.base_model, args.adapter_dir)

    server = HTTPServer((args.host, args.port), OllamaHandler)
    print(f"Serving on http://{args.host}:{args.port}/api/generate")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
