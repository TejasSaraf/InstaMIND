import argparse
import shutil
import subprocess
from pathlib import Path


DEFAULT_SYSTEM = (
    "You are InstaMIND incident reasoning model. "
    "Always return strict JSON list with fields: "
    "incident_type, confidence, timestamp_seconds, evidence, recommended_action."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create/register Ollama model from Gemma LoRA adapter.")
    parser.add_argument("--base-model", default="gemma3n:e4b", help="Base Ollama model tag")
    parser.add_argument("--adapter-dir", required=True, help="Path to PEFT adapter directory")
    parser.add_argument("--model-name", default="gemma3n:e4b-shoplifting", help="New Ollama model tag")
    parser.add_argument("--out-dir", default="outputs/ollama", help="Directory to write Modelfile")
    parser.add_argument("--temperature", type=float, default=0.1, help="Default runtime temperature")
    parser.add_argument("--create", action="store_true", help="Run `ollama create` automatically")
    return parser.parse_args()


def build_modelfile_text(base_model: str, adapter_dir: Path, temperature: float) -> str:
    return (
        f"FROM {base_model}\n"
        f"ADAPTER {adapter_dir.resolve()}\n"
        f"SYSTEM \"{DEFAULT_SYSTEM}\"\n"
        f"PARAMETER temperature {temperature}\n"
    )


def main() -> None:
    args = parse_args()
    adapter_dir = Path(args.adapter_dir)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter dir not found: {adapter_dir}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    modelfile = out_dir / "Modelfile"
    modelfile.write_text(
        build_modelfile_text(
            base_model=args.base_model,
            adapter_dir=adapter_dir,
            temperature=args.temperature,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {modelfile}")
    print(f"Suggested command:\n  ollama create {args.model_name} -f {modelfile}")

    if args.create:
        if shutil.which("ollama") is None:
            raise RuntimeError("Ollama not found on PATH. Install/start Ollama first.")
        subprocess.run(
            ["ollama", "create", args.model_name, "-f", str(modelfile)],
            check=True,
        )
        print(f"Created model tag: {args.model_name}")
        print("Set this in backend .env:")
        print(f"  MODEL_MODE=local_gemma")
        print(f"  LOCAL_GEMMA_MODEL_NAME={args.model_name}")


if __name__ == "__main__":
    main()
