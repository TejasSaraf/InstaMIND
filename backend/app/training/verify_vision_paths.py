"""
=============================================================================
GEMMA 3 VISION PATH VERIFIER
=============================================================================
Run this BEFORE full training to confirm that _VISION_PATH_KEYWORDS in
gemma3.py actually matches the vision encoder modules in the
real Gemma 3 4B checkpoint.

Why this matters
----------------
gemma3.py logs a warning if zero vision LoRA params are detected.
This script diagnoses the mismatch BEFORE you wait 1+ hours and discover
zero vision adaptation.

Usage
-----
  export HF_TOKEN=hf_xxxx
  python verify_vision_paths.py
  python verify_vision_paths.py --model google/gemma-3-4b-pt

Output
------
  Prints every adapter-capable module path (Linear / Conv2d), grouped into
  LANGUAGE / VISION / UNMATCHED buckets based on the current
  _VISION_PATH_KEYWORDS, then emits a clear PASS / FAIL verdict.
=============================================================================
"""



from __future__ import annotations



import argparse

import os

import sys

import threading



import torch

import transformers

from transformers import AutoModelForImageTextToText





_VISION_PATH_KEYWORDS: tuple[str, ...] = (

    "vision_tower", "vision_model", "multi_modal_projector",

    "embed_vision", "siglip",

    "vision", "visual", "image",

)



_LANGUAGE_PATH_KEYWORDS: tuple[str, ...] = (

    "language_model", "lm_head", "embed_tokens",

)





def parse_args() -> argparse.Namespace:

    p = argparse.ArgumentParser(

        description="Verify _VISION_PATH_KEYWORDS against real Gemma 3 module paths.",

        formatter_class=argparse.ArgumentDefaultsHelpFormatter,

    )

    p.add_argument(

        "--model",

        default="google/gemma-3-4b-pt",

        help="HuggingFace model ID to inspect.",

    )

    p.add_argument(

        "--max_depth",

        type=int,

        default=3,

        help="Depth of path prefix to show in summary (1 = top-level, 2 = first two segments, …).",

    )

    return p.parse_args()





def classify_path(path: str) -> str:

    """Return 'vision', 'language', or 'other' for a module path."""

    if any(kw in path for kw in _LANGUAGE_PATH_KEYWORDS):

        return "language"

    if any(kw in path for kw in _VISION_PATH_KEYWORDS):

        return "vision"

    return "other"





def main() -> None:

    args = parse_args()



    hf_token = os.environ.get("HF_TOKEN") or os.environ.get(

        "HUGGING_FACE_HUB_TOKEN")

    if not hf_token:

        print(

            "\n[WARN] HF_TOKEN not set. If the model is gated you will get a 401 error.\n"

            "       Set: export HF_TOKEN=hf_xxxx\n",

            file=sys.stderr,

        )



    print(f"\ntransformers {transformers.__version__}")

    print(

        f"Loading {args.model!r} in bfloat16 on CPU (inspection only — no quantization)...")

    print("This takes ~60-120 seconds on an A100 instance.\n")



    if not torch.cuda.is_available():

        print("[WARN] CUDA not available — loading on CPU.", file=sys.stderr)



    if torch.cuda.is_available():

        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

        gpu_name = torch.cuda.get_device_name(0)

        print(f"GPU: {gpu_name}  ({vram_gb:.1f} GB VRAM)")

        device = "cuda" if vram_gb >= 10.0 else "cpu"

        if device == "cuda":

            print(

                f"Loading in bfloat16 on GPU — will use ~8 GB of {vram_gb:.1f} GB\n")

        else:

            print(

                f"[WARN] GPU VRAM ({vram_gb:.1f} GB) is below the 10 GB threshold "

                f"for bfloat16 load of Gemma 3 4B (~8 GB needed). "

                f"Falling back to CPU to avoid OOM.\n",

                file=sys.stderr,

            )

    else:

        device = "cpu"



    print(f"device_map: {device!r}", flush=True)



    os.environ.setdefault("HF_HUB_ENABLE_PROGRESS_BARS", "1")



    def _load_heartbeat(interval_s: float, stop: threading.Event) -> None:

        n = 0

        while not stop.wait(interval_s):

            n += 1

            print(

                f"\n[verify] still in from_pretrained() — {n * interval_s:.0f}s "

                f"(first cold load: HF cache + weights + vision init; "

                f"often 2–5 min, not frozen)\n",

                flush=True,

            )



    _stop = threading.Event()

    _hb = threading.Thread(

        target=_load_heartbeat,

        args=(25.0, _stop),

        daemon=True,

        name="verify-hb",

    )



    print(

        "Starting from_pretrained (no intermediate lines until this returns; "

        "heartbeat every 25s)…",

        flush=True,

    )

    _hb.start()

    try:

        def _from_pretrained(device_map: str):

            try:

                return AutoModelForImageTextToText.from_pretrained(

                    args.model,

                    dtype=torch.bfloat16,

                    device_map=device_map,

                    token=hf_token,

                    low_cpu_mem_usage=True,

                )

            except TypeError as e:

                if "dtype" not in str(e) and "torch_dtype" not in str(e):

                    raise

                return AutoModelForImageTextToText.from_pretrained(

                    args.model,

                    torch_dtype=torch.bfloat16,

                    device_map=device_map,

                    token=hf_token,

                    low_cpu_mem_usage=True,

                )



        try:

            model = _from_pretrained(device)

        except torch.cuda.OutOfMemoryError:

            print(

                "\n[OOM] CUDA out of memory during from_pretrained(). "

                "Retrying on CPU (slower but never OOMs)...\n",

                file=sys.stderr,

                flush=True,

            )

            torch.cuda.empty_cache()

            model = _from_pretrained("cpu")

    finally:

        _stop.set()

    model.eval()



    all_adapter_layers: list[tuple[str, str]] = [

        (name, name.split(".")[-1])

        for name, mod in model.named_modules()

        if isinstance(mod, (torch.nn.Linear, torch.nn.Conv2d))

    ]



    lang_paths = [(p, leaf)

                  for p, leaf in all_adapter_layers if classify_path(p) == "language"]

    vision_paths = [(p, leaf)

                    for p, leaf in all_adapter_layers if classify_path(p) == "vision"]

    other_paths = [(p, leaf)

                   for p, leaf in all_adapter_layers if classify_path(p) == "other"]



    print("=" * 70)

    print(f"  Model   : {args.model}")

    print(f"  Total adapter-capable layers : {len(all_adapter_layers)}")

    print(f"  Language (matched)  : {len(lang_paths)}")

    print(f"  Vision   (matched)  : {len(vision_paths)}")

    print(f"  Unclassified        : {len(other_paths)}")

    print("=" * 70)



    def unique_prefixes(paths: list[tuple[str, str]], depth: int) -> list[str]:

        seen: set[str] = set()

        out = []

        for p, _ in paths:

            prefix = ".".join(p.split(".")[:depth])

            if prefix not in seen:

                seen.add(prefix)

                out.append(prefix)

        return sorted(out)



    print(

        f"\n--- LANGUAGE adapter paths (depth-{args.max_depth} prefixes) ---")

    for pfx in unique_prefixes(lang_paths, args.max_depth):

        print(f"  {pfx}")



    print(f"\n--- VISION adapter paths (depth-{args.max_depth} prefixes) ---")

    if vision_paths:

        for pfx in unique_prefixes(vision_paths, args.max_depth):

            print(f"  {pfx}")

    else:

        print("  (none — see UNCLASSIFIED below)")



    print(

        f"\n--- UNCLASSIFIED adapter paths (depth-{args.max_depth} prefixes) ---")

    if other_paths:

        for pfx in unique_prefixes(other_paths, args.max_depth):

            print(f"  {pfx}")

    else:

        print("  (none — all adapter-capable layers are classified)")



    all_leaf_names = sorted({leaf for _, leaf in all_adapter_layers})

    vision_leaf_names = sorted({leaf for _, leaf in vision_paths})

    print(f"\n--- All unique adapter leaf names ({len(all_leaf_names)}) ---")

    print(f"  {all_leaf_names}")

    print(f"\n--- Vision adapter leaf names matched by _VISION_PATH_KEYWORDS ---")

    print(f"  {vision_leaf_names}")



    LORA_TARGETS = all_leaf_names

    matched_targets = [t for t in LORA_TARGETS if t in all_leaf_names]

    unmatched_targets = [t for t in LORA_TARGETS if t not in all_leaf_names]



    print(f"\n--- LoRA LORA_TARGET_MODULES validation ---")

    print(f"  Requested  : {LORA_TARGETS}")

    print(f"  Matched    : {matched_targets}")

    if unmatched_targets:

        print(

            f"  NOT FOUND  : {unmatched_targets}  ← PEFT will silently skip these")



    print(f"\n--- all-linear coverage ---")

    print(f"  Total nn.Linear layers : {len(all_adapter_layers)}")

    print(f"  Vision layers          : {len(vision_paths)}")

    print(f"  Language layers        : {len(lang_paths)}")



    print("\n" + "=" * 70)

    if vision_paths:

        print("  PASS — _VISION_PATH_KEYWORDS correctly identifies vision modules.")

        print(

            f"    {len(vision_paths)} vision adapter-capable layers will receive LoRA adapters.")

        print("    No changes needed to _VISION_PATH_KEYWORDS in gemma3.py.")

    else:

        print("  FAIL — _VISION_PATH_KEYWORDS matched ZERO vision adapter-capable layers.")

        print("    Full training will raise RuntimeError unless --allow_no_vision_lora is set.")

        print("\n    HOW TO FIX:")

        print("    1. Look at the UNCLASSIFIED paths above.")

        print("    2. Identify which prefix belongs to the vision encoder")

        print("       (look for 'image', 'vision', 'encoder', etc.).")

        print("    3. Open gemma3.py and find:")

        print("         _VISION_PATH_KEYWORDS: tuple[str, ...] = (")

        print("    4. Add the correct prefix keyword to that tuple.")

        print("    5. Re-run this script to confirm the fix.")



        unclassified_prefixes = unique_prefixes(other_paths, 2)

        if unclassified_prefixes:

            print("\n    Candidate prefixes from UNCLASSIFIED (inspect these):")

            for pfx in unclassified_prefixes:

                print(f"      '{pfx.split('.')[-1]}',  # from path: {pfx}")



    if unmatched_targets:

        print(

            f"\n⚠️   WARN — {len(unmatched_targets)} LoRA target modules not found as leaf names:")

        print(f"    {unmatched_targets}")

        print("    PEFT will skip them. This may reduce coverage. Run --print_modules")

        print("    during smoke test to see the full module tree and find correct names.")

    else:

        print(

            f"\n  All {len(matched_targets)} LoRA target leaf names exist in the model.")



    print("=" * 70 + "\n")



    sys.exit(0 if vision_paths else 1)





if __name__ == "__main__":

    main()
