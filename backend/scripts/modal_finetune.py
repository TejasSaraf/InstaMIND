"""
=============================================================================
INSTAMIND GEMMA 3 4B — MODAL TRAINING SCRIPT (v7 — Unsloth)
=============================================================================
Wraps finetune_gemma3n_unsloth.py to run on Modal's H200 141GB cloud GPUs.
Handles data upload, path rewriting, smoke testing, and adapter download.

v7 change: Switched from Gemma 3n E2B to Gemma 3 4B.
Gemma 3 uses SigLIP (ViT) vision encoder — all layers are nn.Linear, so
vision LoRA works natively via "all-linear".  GGUF export supports vision.
Deployment: Ollama, LM Studio, llama.cpp, MLX all work with Gemma 3.

Why Modal over other providers
------------------------------
  • No GPU quota approvals required (unlike GCP)
  • Auto-terminates container — no billing surprises if you forget to stop
  • H200 141GB HBM3e at ~$6.95/hr (~1.3× faster than H100, more VRAM headroom)
  • Secrets managed via Modal dashboard (HF token never in code)
  • Modal Volumes persist adapter between runs; no re-download needed

ONE-TIME SETUP (local machine)
-------------------------------
  # 1. Install Modal and authenticate
  pip install --upgrade modal
  modal token new   # opens browser for one-time login
  # (older Modal CLI versions used `modal setup`; still works but deprecated)

  # 2. Add HuggingFace token as a Modal Secret
  #    Go to: https://modal.com/secrets  →  New Secret
  #    Name: huggingface-secret   Key: HF_TOKEN   Value: hf_xxxx
  #    (Accept Gemma license first: https://huggingface.co/google/gemma-3-4b-it)

  # 3. Create Modal Volumes (one-time)
  modal volume create instamind-data
  modal volume create instamind-outputs

  # 4. Upload dataset to the data volume (run from repo root — takes ~5 min)
  #    annotations/ is small (~5 MB); processed/frames/ is ~546 MB
  modal volume put instamind-data data/annotations       /data/annotations
  modal volume put instamind-data data/processed/frames  /data/processed/frames

USAGE
-----
  # Verify vision LoRA paths match BEFORE training (critical check, ~2 min on H200)
  modal run backend/scripts/modal_finetune.py::verify_paths

  # Smoke test — end-to-end pipeline check, 20 samples, 1 epoch (<10 min on H200)
  modal run backend/scripts/modal_finetune.py::smoke_test

  # Full training run (~1–1.5 hours on H200 141GB)
  modal run backend/scripts/modal_finetune.py

  # Download the trained adapter to your local machine
  modal volume get instamind-outputs gemma3-incident-qlora/adapter \\
      backend/outputs/gemma3-incident-qlora/adapter

COST ESTIMATES (all endpoints on H200 141GB @ ~$6.95/hr)
--------------------------------------------------------
  First-run model download : +$0.35   (16GB from HuggingFace, ~3 min cold start)
  verify_paths  : ~$0.23   (2 min on H200 — module-tree inspection only)
  smoke_test    : ~$0.58-$0.81   (5-7 min on H200; `--print_modules` adds log overhead)
  full training : ~$8.70   (1–1.25 hr on H200)
  Data upload   : ~$0.10   (5 min, one-time per dataset)

  Total (first full run incl. verification + smoke): ~$10.00
=============================================================================
"""


from __future__ import annotations


from pathlib import Path


import modal


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


_TRAIN_SCRIPT = _REPO_ROOT / "backend" / "app" / "training" / "gemma3.py"

_VERIFY_SCRIPT = _REPO_ROOT / "backend" / \
    "app" / "training" / "verify_vision_paths.py"


_REMOTE_TRAIN = "/workspace/gemma3.py"

_REMOTE_VERIFY = "/workspace/verify_vision_paths.py"

_REMOTE_DATA = "/data/data/annotations"

_REMOTE_FRAMES = "/data/data/processed/frames"

_REMOTE_AUGMENTED = "/data/data/processed/augmented"

_REMOTE_OUTPUT = "/workspace/outputs/gemma3-incident-qlora"


app = modal.App("instamind-gemma3-finetune")


data_volume = modal.Volume.from_name(
    "instamind-data",    create_if_missing=True)

output_volume = modal.Volume.from_name(
    "instamind-outputs", create_if_missing=True)


HF_SECRET = modal.Secret.from_name("huggingface-secret")


image = (

    modal.Image.debian_slim(python_version="3.11")



    .pip_install(

        "unsloth",

        "unsloth_zoo",

        "timm",

        "torchcodec",

        "torchao>=0.16.0",

        "transformers>=4.56.2",

        "trl>=0.22.2",

        "tensorboard",

        "pillow",

        "packaging",

        "ninja",

        "wheel",

        "setuptools",

    )

    .add_local_file(str(_TRAIN_SCRIPT),  remote_path=_REMOTE_TRAIN)

    .add_local_file(str(_VERIFY_SCRIPT), remote_path=_REMOTE_VERIFY)

)


def _run(cmd: list[str], label: str) -> None:
    """Run a subprocess command with timing output.

    Streams stdout/stderr to the parent process. Raises RuntimeError on
    non-zero exit. Logs start/end timestamps and elapsed time so cold-start
    model downloads are visible in Modal logs and easy to diagnose.

    Issue 5 fix: added start timestamp and elapsed time so HF model download
    stalls (can take 10+ min on cold container) are immediately visible
    rather than leaving the user watching a silent log.
    """

    import os

    import subprocess

    import sys

    import time

    if cmd and (len(cmd) < 2 or cmd[1] != "-u"):

        exe = os.path.basename(cmd[0])

        if exe.startswith("python"):

            cmd = [cmd[0], "-u", *cmd[1:]]

    start = time.time()

    print(f"\n[{label}] Running: {' '.join(cmd)}", flush=True)

    print(f"[{label}] Start time: {time.strftime('%H:%M:%S UTC', time.gmtime())}",

          flush=True)

    result = subprocess.run(cmd, check=False)

    elapsed_min = (time.time() - start) / 60.0

    print(f"\n[{label}] Elapsed: {elapsed_min:.1f} min  "

          f"(exit code {result.returncode})", flush=True)

    if result.returncode != 0:

        raise RuntimeError(

            f"[{label}] FAILED — exit code {result.returncode} "

            f"after {elapsed_min:.1f} min.\n"

            "Check the subprocess output above for the error message."

        )


@app.function(

    image=image,

    gpu="H200:1",

    timeout=60 * 15,

    secrets=[HF_SECRET],

)
def verify_paths() -> None:
    """
    Load Gemma 3 4B in bfloat16 on an H200 GPU (141 GB VRAM) and
    check that _VISION_PATH_KEYWORDS in gemma3.py matches the
    actual vision encoder module paths. Raises RuntimeError on mismatch.

    Uses H200 (same as smoke/train) for consistency.
    ~2 min inspection ≈ $0.23 on H200 @ $6.95/hr.

    Run before smoke test or full training:
        modal run backend/scripts/modal_finetune.py::verify_paths
    """

    import sys

    _run([sys.executable, _REMOTE_VERIFY], label="verify_paths")


@app.function(

    image=image,

    gpu="H200:1",





    timeout=60 * 30,

    secrets=[HF_SECRET],

    volumes={

        "/data":             data_volume,

        "/workspace/outputs": output_volume,

    },

)
def smoke_test(print_modules: bool = False) -> None:
    """
    Run the training pipeline on 20 train / 10 val samples for 1 epoch.
    Validates collator, pixel_values batching, label masking, and LoRA setup
    without committing to a full 1–1.25 hour H200 run.

    Run with:
        modal run backend/scripts/modal_finetune.py::smoke_test
    """

    data_volume.reload()

    import sys

    cmd = [

        sys.executable, _REMOTE_TRAIN,

        "--annotations_dir", _REMOTE_DATA,

        "--frames_root",     _REMOTE_FRAMES,

        "--output_dir",      _REMOTE_OUTPUT,

        "--smoke_test",

        "--load_in_4bit",

        "--batch_size", "1",

        "--max_seq_length", "1024",

        "--max_images_per_sample", "1",

        "--grad_accum", "1",

        "--optim", "adamw_torch",

    ]

    import pathlib as _pl

    if _pl.Path(_REMOTE_AUGMENTED).exists():

        cmd.extend(["--extra_frames_roots", _REMOTE_AUGMENTED])

    if print_modules:

        print(

            "[smoke_test] Deep module dump enabled via --print_modules. "

            "This can add several minutes of Modal log overhead."

        )

        cmd.append("--print_modules")

    else:

        print(

            "[smoke_test] Running lean smoke mode without --print_modules. "

            "verify_paths already validates the module-path coverage first."

        )

    _run(cmd, label="smoke_test")

    print("\nSmoke test complete. Check output above for:")

    print("  1. 'Vision modules matched: N paths'  where N > 0")

    print("  2. 'supervised tokens: X / Y  (Z%)'   where Z > 0 for all 3 samples")

    print("  3. 'Smoke test complete. Adapter NOT saved' at the end")


@app.function(

    image=image,

    gpu="H200:1",



    timeout=86400,

    secrets=[HF_SECRET],

    volumes={

        "/data":             data_volume,

        "/workspace/outputs": output_volume,

    },

)
def train(

    lr: float = 2e-4,

    batch_size: int = 1,

    grad_accum: int = 16,

    lora_r: int = 64,

    lora_alpha: int = 64,

    lora_dropout: float = 0.0,

    num_epochs: int = 3,

    warmup_steps: int = 20,

    max_seq_length: int = 4096,

    early_stopping_patience: int = 3,

    optim: str = "adamw_torch_fused",

    merge_adapter: bool = False,

    load_in_4bit: bool = True,

    max_images_per_sample: int = 6,

) -> None:
    """
    Full QLoRA fine-tuning of Gemma 3 4B on the instaMIND
    incident-detection dataset. Adapter is saved to the instamind-outputs
    Modal Volume automatically.

    Run with:
        modal run backend/scripts/modal_finetune.py

    Download adapter after training:
        modal volume get instamind-outputs gemma3-incident-qlora/adapter \\
            backend/outputs/gemma3-incident-qlora/adapter
    """

    data_volume.reload()

    import pathlib

    import sys

    output_path = pathlib.Path(_REMOTE_OUTPUT)

    ann_dir = pathlib.Path(_REMOTE_DATA)

    frames_dir = pathlib.Path(_REMOTE_FRAMES)

    for required in [ann_dir / "train.jsonl", ann_dir / "val.jsonl"]:

        if not required.exists():

            raise FileNotFoundError(

                f"Required file not found: {required}\n"

                "Did you upload the dataset to the Modal Volume?\n"

                "  modal volume put instamind-data data/annotations /data/annotations"

            )

    if not frames_dir.exists() or not any(frames_dir.iterdir()):

        raise FileNotFoundError(

            f"Frames directory empty or missing: {frames_dir}\n"

            "Upload with:\n"

            "  modal volume put instamind-data data/processed/frames /data/processed/frames"

        )

    n_train = sum(1 for _ in (ann_dir / "train.jsonl").open())

    n_val = sum(1 for _ in (ann_dir / "val.jsonl").open())

    print(f"\nDataset: {n_train} train samples, {n_val} val samples")

    print(
        f"Frames root: {frames_dir}  ({sum(1 for _ in frames_dir.rglob('*.jpg'))} jpg files)\n")

    cmd = [

        sys.executable, _REMOTE_TRAIN,

        "--annotations_dir",           _REMOTE_DATA,

        "--frames_root",               _REMOTE_FRAMES,

        "--output_dir",                _REMOTE_OUTPUT,

        "--lr",                        str(lr),

        "--batch_size",                str(batch_size),

        "--grad_accum",                str(grad_accum),

        "--lora_r",                    str(lora_r),

        "--lora_alpha",                str(lora_alpha),

        "--lora_dropout",              str(lora_dropout),

        "--num_epochs",                str(num_epochs),

        "--warmup_steps",              str(warmup_steps),

        "--max_seq_length",            str(max_seq_length),

        "--early_stopping_patience",   str(early_stopping_patience),

        "--optim",                     optim,

        "--max_images_per_sample",     str(max_images_per_sample),

    ]

    import pathlib as _pl

    if _pl.Path(_REMOTE_AUGMENTED).exists():

        cmd.extend(["--extra_frames_roots", _REMOTE_AUGMENTED])

    if load_in_4bit:

        cmd.append("--load_in_4bit")

    if merge_adapter:

        cmd.append("--merge_adapter")

    try:

        _run(cmd, label="train")

    except Exception as e:

        print(f"\n[train] Training failed ({type(e).__name__}: {e})")

        print(
            "[train] Committing volume to preserve any partial/interrupted checkpoint ...")

        try:

            output_volume.commit()

            print("[train] Volume committed. Recover any checkpoint with:")

            print(
                "  modal volume get instamind-outputs gemma3-incident-qlora/interrupted_checkpoint \\")

            print("      backend/outputs/gemma3-incident-qlora/interrupted_checkpoint")

        except Exception as commit_err:

            print(f"[train] Volume commit ALSO failed: {commit_err}")

        raise

    adapter_dir = output_path / "adapter"

    adapter_files = sorted(adapter_dir.rglob(
        "*")) if adapter_dir.exists() else []

    if not adapter_files:

        output_volume.commit()

        raise RuntimeError(

            "Training completed but no adapter files found at "

            f"{adapter_dir}. Check that --smoke_test was NOT passed and "

            "training ran to completion."

        )

    print(f"\nAdapter saved ({len(adapter_files)} files):")

    for f in adapter_files[:10]:

        print(f"  {f.relative_to(_REMOTE_OUTPUT)}")

    output_volume.commit()

    print("\n✅ Adapter committed to Modal Volume 'instamind-outputs'")

    print("\nDownload with (run from your local machine):")

    print(f"  modal volume get instamind-outputs gemma3-incident-qlora/adapter \\")

    print(f"      backend/outputs/gemma3-incident-qlora/adapter")


gguf_image = (

    modal.Image.debian_slim(python_version="3.11")

    .apt_install("git", "build-essential", "cmake", "curl", "libcurl4-openssl-dev")

    .pip_install(

        "torch",

        "transformers",

        "sentencepiece",

        "protobuf",

        "numpy",

        "huggingface_hub",

        "safetensors",

        "gguf",

    )

    .run_commands(

        "git clone --depth 1 https://github.com/ggerganov/llama.cpp /opt/llama.cpp",

        "cmake -S /opt/llama.cpp -B /opt/llama.cpp/build "

        "-DLLAMA_CURL=ON -DGGML_NATIVE=OFF",

        "cmake --build /opt/llama.cpp/build --config Release -j "

        "--target llama-quantize llama-cli",

    )

)


@app.function(

    image=image,

    cpu=4.0,

    timeout=60 * 60,

    secrets=[HF_SECRET],

    volumes={

        "/workspace/outputs": output_volume,

    },

)
def push_to_hub(

    repo_id: str,

    merged_subdir: str = "merged",

    private: bool = True,

) -> None:
    """Upload the merged HF model on the volume to a HuggingFace Hub repo.

    Datacenter-to-datacenter upload is much faster than `modal volume get`
    over residential broadband. After this completes, you can:

      * Pull locally:        huggingface-cli download <repo_id>
      * Convert with MLX:    mlx_vlm.convert --hf-path <repo_id> --mlx-path ...
      * Reuse on any GPU box without re-running the merge.

    Run with:
        modal run backend/scripts/modal_finetune.py::push_to_hub \\
            --repo-id  your-username/instamind-gemma3-incident
    """

    import os

    from pathlib import Path

    from huggingface_hub import HfApi

    output_volume.reload()

    src = Path(_REMOTE_OUTPUT) / merged_subdir

    if not src.exists():

        raise FileNotFoundError(

            f"Merged model not found at {src}. Run merge_adapter first."

        )

    token = os.environ.get("HF_TOKEN")

    if not token:

        raise RuntimeError("HF_TOKEN not set in huggingface-secret.")

    api = HfApi(token=token)

    print(f"Creating / updating repo: {repo_id}  (private={private})")

    api.create_repo(repo_id=repo_id, private=private, exist_ok=True)

    print(f"Uploading {src} ...")

    api.upload_folder(

        folder_path=str(src),

        repo_id=repo_id,

        commit_message="instaMIND Gemma 3 incident-detection LoRA (merged)",

    )

    print(f"\n Pushed to https://huggingface.co/{repo_id}")

    print(f"\nNext (on your local Mac):")

    print(f"  python -m mlx_vlm convert \\")

    print(f"      --hf-path  {repo_id} \\")

    print(f"      --mlx-path backend/outputs/gemma3-incident-mlx-q4 \\")

    print(f"      -q --q-bits 4 --q-group-size 64")


@app.function(

    image=gguf_image,

    cpu=8.0,

    memory=32768,

    timeout=60 * 60,

    volumes={

        "/workspace/outputs": output_volume,

    },

)
def quantize_to_gguf(

    merged_subdir: str = "merged",

    output_subdir: str = "gguf",

    quant: str = "Q4_K_M",

) -> None:
    """Convert the merged HF model to GGUF and quantize for llama.cpp / Ollama.

    For Gemma 3 the convert script also emits an `mmproj-*.gguf` sidecar for
    the vision tower; both files are needed for image-input inference with
    `llama-cli --mmproj ...` or `llama-server --mmproj ...`.

    Output (in volume `instamind-outputs`):
        gemma3-incident-qlora/gguf/model-Q4_K_M.gguf      (~4 GB)
        gemma3-incident-qlora/gguf/mmproj-f16.gguf        (~600 MB)

    Run with:
        modal run backend/scripts/modal_finetune.py::quantize_to_gguf
    """

    import shutil

    import subprocess

    import sys

    from pathlib import Path

    import json

    output_volume.reload()

    src = Path(_REMOTE_OUTPUT) / merged_subdir

    if not src.exists():

        raise FileNotFoundError(

            f"Merged model not found at {src}. Run merge_adapter first."

        )

    out_dir = Path(_REMOTE_OUTPUT) / output_subdir

    out_dir.mkdir(parents=True, exist_ok=True)

    f16_path = Path("/tmp/model-f16.gguf")

    work_dir = Path("/tmp/gemma3-vision-only")

    if work_dir.exists():

        shutil.rmtree(work_dir)

    work_dir.mkdir(parents=True)

    DROP_TOKENS = (

        "language_conformer",

        "audio_tower",

        "embed_audio",

        "audio_embedding",

    )

    for f in src.iterdir():

        if f.name == "config.json":

            continue

        if f.name == "model.safetensors.index.json":

            continue

        if f.suffix == ".safetensors":

            continue

        (work_dir / f.name).symlink_to(f.resolve())

    from safetensors import safe_open

    from safetensors.torch import save_file

    new_weight_map: dict[str, str] = {}

    new_total_size = 0

    n_kept = 0

    n_drop = 0

    shard_paths = sorted(src.glob("*.safetensors"))

    print(
        f"[gguf] Rewriting {len(shard_paths)} shards without audio tensors ...")

    for shard in shard_paths:

        out_shard = work_dir / shard.name

        kept: dict = {}

        with safe_open(str(shard), framework="pt") as f:

            for key in f.keys():

                if any(tok in key for tok in DROP_TOKENS):

                    n_drop += 1

                    continue

                t = f.get_tensor(key)

                kept[key] = t

                new_weight_map[key] = shard.name

                new_total_size += t.numel() * t.element_size()

                n_kept += 1

        if not kept:

            print(f"[gguf]   {shard.name}: 100% audio, skipping")

            continue

        save_file(kept, str(out_shard))

        kept_size_gb = sum(t.numel() * t.element_size()
                           for t in kept.values()) / 1e9

        print(
            f"[gguf]   {shard.name}: kept {len(kept)} tensors ({kept_size_gb:.2f} GB)")

    print(
        f"[gguf] Total: kept {n_kept} tensors, dropped {n_drop} audio tensors")

    new_index = {

        "metadata": {"total_size": new_total_size},

        "weight_map": new_weight_map,

    }

    (work_dir / "model.safetensors.index.json").write_text(

        json.dumps(new_index, indent=2)

    )

    cfg_path = src / "config.json"

    cfg = json.loads(cfg_path.read_text())

    removed_keys = []

    for k in ("audio_config", "audio_token_id", "audio_token_index"):

        if k in cfg:

            cfg.pop(k)

            removed_keys.append(k)

    print(f"[gguf] Stripped audio config keys: {removed_keys or '(none)'}")

    (work_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    sample_keys = list(new_weight_map.keys())[
        :8] + list(new_weight_map.keys())[-8:]

    print("[gguf] Sample tensor names being handed to the converter:")

    for k in sample_keys:

        print(f"  {k}")

    vision_count = sum(1 for k in new_weight_map if "vision_tower" in k)

    print(f"[gguf] Vision-tower tensors in filtered set: {vision_count}")

    convert_cmd_lm = [

        sys.executable,

        "/opt/llama.cpp/convert_hf_to_gguf.py",

        str(work_dir),

        "--outfile",   str(f16_path),

        "--outtype",   "f16",

    ]

    _run(convert_cmd_lm, label="hf->gguf-f16-lm")

    mmproj_f16 = Path("/tmp/mmproj-f16.gguf")

    convert_cmd_mm = [

        sys.executable,

        "/opt/llama.cpp/convert_hf_to_gguf.py",

        str(work_dir),

        "--outfile",   str(mmproj_f16),

        "--outtype",   "f16",

        "--mmproj",

    ]

    _run(convert_cmd_mm, label="hf->gguf-f16-mmproj")

    if mmproj_f16.exists():

        size_mb = mmproj_f16.stat().st_size / 1e6

        if size_mb < 50:

            print(f"[gguf] WARNING: mmproj output is only {size_mb:.1f} MB "

                  "— vision tensors may not have been recognized. The LM "

                  "GGUF should still be usable for text-only inference.")

        dst = out_dir / "mmproj-f16.gguf"

        shutil.move(str(mmproj_f16), str(dst))

        print(f"[gguf] mmproj saved: {dst} ({size_mb:.1f} MB)")

    quant_path = out_dir / f"model-{quant}.gguf"

    quant_cmd = [

        "/opt/llama.cpp/build/bin/llama-quantize",

        str(f16_path),

        str(quant_path),

        quant,

    ]

    _run(quant_cmd, label=f"quantize-{quant}")

    f16_path.unlink(missing_ok=True)

    files = sorted(out_dir.iterdir())

    total_gb = sum(f.stat().st_size for f in files) / 1e9

    print(f"\n[gguf] Output ({total_gb:.2f} GB total):")

    for f in files:

        print(f"  {f.name}  ({f.stat().st_size / 1e9:.2f} GB)")

    output_volume.commit()

    print(f"\nGGUF artifacts committed to volume 'instamind-outputs'")

    print(f"\nDownload (much smaller — typically ~4 GB):")

    print(
        f"  modal volume get instamind-outputs gemma3-incident-qlora/{output_subdir}/ \\")

    print(f"      backend/outputs/gemma3-incident-qlora/{output_subdir}/")

    print(f"\nRun on-device with llama.cpp (vision via --mmproj):")

    print(f"  llama-cli -m model-{quant}.gguf --mmproj mmproj-f16.gguf \\")

    print(f"      --image frame.jpg -p 'Describe the incident.'")


@app.function(

    image=image,

    gpu="H200:1",

    timeout=60 * 60,

    secrets=[HF_SECRET],

    volumes={

        "/data":              data_volume,

        "/workspace/outputs": output_volume,

    },

)
def test_inference(

    base_model: str = "unsloth/gemma-3-4b-pt",

    adapter_subdir: str = "adapter",

    n_samples: int = 5,

    max_new_tokens: int = 256,

    max_seq_length: int = 4096,

    base_only: bool = False,

) -> None:
    """Validate that the trained adapter actually produces sensible outputs.

    Loads the base model via Unsloth's FastModel (same wrapper as training),
    attaches the LoRA adapter from the volume, and runs generation on the
    first `n_samples` rows of `val.jsonl`. For each sample, prints:

      * the system + user text + image paths the model was given
      * the gold assistant response from the dataset
      * the model's actual generation

    This bypasses the merge step entirely, so it works even when the
    `merged/` checkpoint on the volume is stale or malformed.

    Run with:
        modal run backend/scripts/modal_finetune.py::test_inference
        modal run backend/scripts/modal_finetune.py::test_inference --n-samples 10
    """

    import json

    import os

    import sys

    from pathlib import Path

    sys.path.insert(0, str(Path(_REMOTE_TRAIN).parent))

    output_volume.reload()

    data_volume.reload()

    adapter_dir = Path(_REMOTE_OUTPUT) / adapter_subdir

    val_path = Path(_REMOTE_DATA) / "val.jsonl"

    frames_root = Path(_REMOTE_FRAMES)

    if not base_only and not adapter_dir.exists():

        raise FileNotFoundError(

            f"Adapter not found at {adapter_dir}. Run train first."

        )

    if not val_path.exists():

        raise FileNotFoundError(f"val.jsonl not found at {val_path}")

    print(f"[test] Loading {base_model} via Unsloth FastVisionModel ...")

    from unsloth import FastVisionModel

    from unsloth import get_chat_template

    import torch

    import gemma3 as ft

    hf_token = os.environ.get("HF_TOKEN") or None

    model, tokenizer = FastVisionModel.from_pretrained(

        model_name=base_model,

        max_seq_length=max_seq_length,

        load_in_4bit=True,

        token=hf_token,

    )

    tokenizer = get_chat_template(tokenizer, "gemma-3")

    print("[test] Applied Gemma 3 chat template.")

    print(f"[test] Adapter directory contents:")

    for f in sorted(adapter_dir.iterdir()):

        if f.is_file():

            print(f"  {f.name}  ({f.stat().st_size/1e6:.1f} MB)")

    adapter_cfg_path = adapter_dir / "adapter_config.json"

    if adapter_cfg_path.exists():

        import json as _json

        _acfg = _json.loads(adapter_cfg_path.read_text())

        adapter_base = _acfg.get('base_model_name_or_path', '')

        print(f"[test] adapter_config.json base_model: {adapter_base}")

        def _model_family(name: str) -> str:

            name = name.lower()

            if "gemma-3-4b" in name or "gemma3-4b" in name:
                return "gemma3-4b"

            if "gemma-3-12b" in name or "gemma3-12b" in name:
                return "gemma3-12b"

            if "e4b" in name:
                return "e4b"

            if "e2b" in name:
                return "e2b"

            return "unknown"

        adapter_family = _model_family(adapter_base)

        base_family = _model_family(base_model)

        if adapter_family != "unknown" and base_family != "unknown" and adapter_family != base_family:

            raise RuntimeError(

                f"Model family mismatch: adapter was trained on {adapter_base} "

                f"but base_model is {base_model}. "

                f"Pass --base-model matching the adapter's base model to fix."

            )

    _diag = dict(model.named_parameters())

    _probe = "model.language_model.layers.0.mlp.gate_proj.weight"

    if _probe in _diag:

        print(
            f"[test] Base model MLP gate_proj shape: {tuple(_diag[_probe].shape)}")

    if not base_only:

        print(f"[test] Attaching adapter from {adapter_dir} ...")

        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir))

        import re

        import statistics

        _LANG_PATS = [

            re.compile(r"language_model"),

            re.compile(r"\blm_head\b"),

            re.compile(r"embed_tokens"),

        ]

        _VISION_PATS = [

            re.compile(r"vision_tower"),

            re.compile(r"vision_model"),

            re.compile(r"embed_vision"),

            re.compile(r"multi_modal_projector"),

            re.compile(r"embedding_projection"),

            re.compile(r"siglip"),

            re.compile(r"vision"),

            re.compile(r"visual"),

        ]

        _AUDIO_PATS = [

            re.compile(r"audio_tower"),

            re.compile(r"embed_audio"),

        ]

        def _classify_lora(name: str) -> str:

            if any(p.search(name.lower()) for p in _VISION_PATS):

                return "vision"

            if any(p.search(name.lower()) for p in _AUDIO_PATS):

                return "audio"

            if any(p.search(name.lower()) for p in _LANG_PATS):

                return "language"

            return "other"

        a_norms: list[float] = []

        b_norms: list[float] = []

        counts = {"vision": 0, "language": 0, "audio": 0, "other": 0}

        sample_names = []

        for name, p in model.named_parameters():

            if "lora_A" in name:

                a_norms.append(float(p.detach().float().norm().item()))

            elif "lora_B" in name:

                b_norms.append(float(p.detach().float().norm().item()))

            if "lora_" in name:

                cls = _classify_lora(name)

                counts[cls] += 1

                if len(sample_names) < 10:

                    sample_names.append(f"{cls:8} | {name}")

        n_lora = len(a_norms) + len(b_norms)

        if a_norms and b_norms:

            a_mean = statistics.mean(a_norms)

            b_mean = statistics.mean(b_norms)

            ratio = b_mean / a_mean if a_mean > 0 else 0.0

            print(f"[test] LoRA tensors: {n_lora}  "

                  f"(language={counts['language']}, vision={counts['vision']}, "

                  f"audio={counts['audio']}, other={counts['other']})")

            if counts["vision"] == 0:

                print(
                    "[test] ⚠️  No vision LoRA tensors detected! Listing sample names:")

                for s in sample_names:

                    print(f"  {s}")

            print(f"[test] mean ‖lora_A‖={a_mean:.4f}   "

                  f"mean ‖lora_B‖={b_mean:.4f}   ratio={ratio:.4f}")

            if ratio < 0.05:

                print("[test] ⚠️  lora_B/lora_A ratio < 0.05 — adapter is essentially "

                      "untrained (lora_B barely moved off zero init).")

            elif ratio < 0.20:

                print("[test] ⚠️  lora_B/lora_A ratio < 0.20 — adapter may be "

                      "under-trained.")

            else:

                print("[test] ✓ lora_B/lora_A ratio looks reasonable.")

        else:

            print(f"[test] ⚠️  No LoRA tensors found (n_lora={n_lora})")

    FastVisionModel.for_inference(model)

    model.eval()

    print("[test] Model set to inference mode.")

    print("\n[test] === Sample 0 Inference Check ===")

    sys.path.insert(0, str(Path(_REMOTE_TRAIN).parent))

    raw_val_for_compare = ft._load_jsonl(val_path)[:1]

    s0 = raw_val_for_compare[0]

    prompt_msgs0 = [m for m in s0["messages"] if m.get("role") != "assistant"]

    converted0 = ft._to_unsloth_messages(prompt_msgs0, [frames_root])

    text0 = tokenizer.apply_chat_template(

        converted0, tokenize=False, add_generation_prompt=True,

    )

    images0 = []

    for m in converted0:

        if isinstance(m.get("content"), list):

            for part in m["content"]:

                if isinstance(part, dict) and part.get("type") == "image":

                    if part.get("image") is not None:

                        images0.append(part["image"])

    inputs0 = tokenizer(

        text=[text0], images=images0 or None,

        return_tensors="pt", padding=True,

    ).to("cuda")

    def _gen_short() -> str:

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):

            out = model.generate(

                **inputs0, max_new_tokens=80,

                do_sample=False, temperature=1.0, top_p=1.0,

            )

        plen = inputs0["input_ids"].shape[1]

        return tokenizer.decode(out[0, plen:], skip_special_tokens=True).strip()

    if not base_only:

        try:

            print("[test] Running with adapter DISABLED ...")

            with model.disable_adapter():

                base_pred = _gen_short()

            print(f"[test]   BASE  : {base_pred[:200]}")

            print("[test] Running with adapter ENABLED ...")

            adap_pred = _gen_short()

            print(f"[test]   ADAPT : {adap_pred[:200]}")

            if base_pred == adap_pred:

                print("[test] ⚠️  BASE and ADAPTER outputs are IDENTICAL — adapter is "

                      "loaded but NOT being applied at inference. This is a runtime "

                      "bug, not a training bug.")

            else:

                print("[test] ✓ Base and adapter outputs differ — adapter IS active.")

        except Exception as e:

            print(
                f"[test] ⚠️  Base-vs-adapter diagnostic skipped ({type(e).__name__}: {e})")

    else:

        print("[test] Running BASE MODEL only ...")

        base_pred = _gen_short()

        print(f"[test]   BASE  : {base_pred[:200]}")

    print("=" * 78 + "\n")

    raw_val = ft._load_jsonl(val_path)

    samples = raw_val[:n_samples]

    print(f"[test] Running inference on {len(samples)} val samples ...\n")

    correct = 0

    y_true = []

    y_pred = []

    errors = []

    for i, sample in enumerate(samples):

        msgs = sample["messages"]

        gold_assistant = ""

        prompt_msgs = []

        for m in msgs:

            if m.get("role") == "assistant":

                content = m.get("content")

                if isinstance(content, str):

                    gold_assistant = content

                elif isinstance(content, list):

                    for part in content:

                        if isinstance(part, dict) and part.get("type") == "text":

                            gold_assistant += part.get("text", "")

                continue

            prompt_msgs.append(m)

        converted = ft._to_unsloth_messages(prompt_msgs, [frames_root])

        text = tokenizer.apply_chat_template(

            converted,

            tokenize=False,

            add_generation_prompt=True,

        )

        images = []

        for m in converted:

            content = m.get("content")

            if isinstance(content, list):

                for part in content:

                    if isinstance(part, dict) and part.get("type") == "image":

                        img = part.get("image")

                        if img is not None:

                            images.append(img)

        inputs = tokenizer(

            text=[text],

            images=images if images else None,

            return_tensors="pt",

            padding=True,

        ).to("cuda")

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):

            out_ids = model.generate(

                **inputs,

                max_new_tokens=max_new_tokens,

                do_sample=False,

                temperature=1.0,

                top_p=1.0,

            )

        prompt_len = inputs["input_ids"].shape[1]

        gen_ids = out_ids[0, prompt_len:]

        prediction = tokenizer.decode(
            gen_ids, skip_special_tokens=True).strip()

        print("=" * 78)

        print(f"Sample {i+1}/{len(samples)}  (n_images={len(images)})")

        for m in converted:

            if m.get("role") != "user":

                continue

            for part in m.get("content", []):

                if isinstance(part, dict) and part.get("type") == "text":

                    txt = part.get("text", "").strip()

                    if txt:

                        print(f"USER     : {txt[:300]}")

        print(f"GOLD     : {gold_assistant.strip()[:300]}")

        print(f"PREDICT  : {prediction[:300]}")

        import re

        def _extract_incident_type(s: str) -> str:

            m = re.search(r'"incident_type"\s*:\s*"([^"]+)"', s)

            return m.group(1).strip().lower() if m else "unknown"

        gold_label = _extract_incident_type(gold_assistant)

        pred_label = _extract_incident_type(prediction)

        y_true.append(gold_label)

        y_pred.append(pred_label)

        if gold_label and gold_label == pred_label:

            print(f"MATCH    : ✓  ({gold_label})")

            correct += 1

        else:

            print(f"MATCH    : ✗  (gold={gold_label!r}  pred={pred_label!r})")

            errors.append({

                "sample_idx": i + 1,

                "gold": gold_label,

                "pred": pred_label,

                "gold_summary": gold_assistant,

                "pred_summary": prediction,

            })

    print("=" * 78)

    print(
        f"\n[test] Accuracy: {correct}/{len(samples)} ({correct/len(samples):.1%})")

    all_classes = sorted(set(y_true + y_pred))

    confusion = {t: {p: 0 for p in all_classes} for t in all_classes}

    for t, p in zip(y_true, y_pred):

        confusion[t][p] += 1

    recalls = {}

    precisions = {}

    f1s = {}

    for cls in all_classes:

        tp = confusion[cls].get(cls, 0)

        row_sum = sum(confusion[cls].values())

        recalls[cls] = tp / row_sum if row_sum > 0 else 0.0

        col_sum = sum(confusion[t].get(cls, 0) for t in all_classes)

        precisions[cls] = tp / col_sum if col_sum > 0 else 0.0

        p, r = precisions[cls], recalls[cls]

        f1s[cls] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    macro_f1 = sum(f1s.values()) / len(f1s) if f1s else 0.0

    print("\n[test] PER-CLASS METRICS:")

    print(f"  Macro F1: {macro_f1:.3f}")

    print(f"  {'class':<20}  {'recall':>8}  {'prec':>8}  {'F1':>8}")

    for cls in all_classes:

        print(
            f"  {cls:<20}  {recalls[cls]:8.3f}  {precisions[cls]:8.3f}  {f1s[cls]:8.3f}")

    print("\n[test] ERROR ANALYSIS:")

    if not errors:

        print("  No errors found!")

    else:

        for err in errors[:10]:

            print(
                f"  Sample {err['sample_idx']}: Expected '{err['gold']}', got '{err['pred']}'")

            print(f"    Gold: {err['gold_summary'].strip()[:100]}...")

            print(f"    Pred: {err['pred_summary'].strip()[:100]}...\n")

        if len(errors) > 10:

            print(f"  ... and {len(errors) - 10} more errors.")

    print("\n[test] If predictions look correct, the adapter is healthy. "

          "Proceed with merge_adapter / push_to_hub.")


@app.function(

    image=image,

    gpu="H200:1",

    timeout=60 * 30,

    secrets=[HF_SECRET],

    volumes={

        "/workspace/outputs": output_volume,

    },

)
def merge_adapter(

    base_model: str = "unsloth/gemma-3-4b-pt",

    adapter_subdir: str = "adapter",

    output_subdir: str = "merged",

) -> None:
    """Merge the trained LoRA adapter into the base model and save a full
    HF checkpoint to the output volume for downstream conversion (MLX, GGUF).

    Run with:
        modal run backend/scripts/modal_finetune.py::merge_adapter

    Download merged model afterwards:
        modal volume get instamind-outputs gemma3n-incident-qlora/merged \\
            backend/outputs/gemma3n-incident-qlora/merged
    """

    import os

    import time

    from pathlib import Path

    output_volume.reload()

    adapter_dir = Path(f"{_REMOTE_OUTPUT}/{adapter_subdir}")

    output_dir = Path(f"{_REMOTE_OUTPUT}/{output_subdir}")

    if not adapter_dir.exists():

        raise FileNotFoundError(

            f"Adapter not found at {adapter_dir}. Run training first."

        )

    print(f"[merge] Base model:  {base_model}")

    print(f"[merge] Adapter dir: {adapter_dir}")

    print(f"[merge] Output dir:  {output_dir}")

    start = time.time()

    import torch

    from unsloth import FastVisionModel

    token = os.environ.get("HF_TOKEN")

    print("[merge] Loading base model ...")

    model, tokenizer = FastVisionModel.from_pretrained(

        model_name=base_model,

        load_in_4bit=False,

        token=token,

    )

    print(f"[merge] Loading adapter from {adapter_dir} ...")

    from peft import PeftModel

    model = PeftModel.from_pretrained(model, str(adapter_dir))

    print(f"[merge] Merging LoRA weights ...")

    model = model.merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[merge] Saving merged model to {output_dir} ...")

    model.save_pretrained(str(output_dir))

    tokenizer.save_pretrained(str(output_dir))

    elapsed = (time.time() - start) / 60.0

    print(f"[merge] Done in {elapsed:.1f} min")

    output_volume.commit()

    print("\n\u2705 Merged model committed to volume 'instamind-outputs'")

    print(f"\nDownload (run from your local machine):")

    print(
        f"  modal volume get instamind-outputs gemma3-incident-qlora/{output_subdir} \\")

    print(f"      backend/outputs/gemma3-incident-qlora/{output_subdir}")


@app.local_entrypoint()
def main(

    num_epochs: int = 5,

    lr: float = 1e-4,

    smoke: bool = False,

    verify: bool = False,

    skip_verify: bool = False,

) -> None:
    """
    Default entrypoint: vision path verification → full training.

    By default, full training and smoke test ALWAYS run verify_paths first.
    This prevents wasting 1.5 hours if _VISION_PATH_KEYWORDS doesn't match
    the real Gemma 3 4B module paths.

    Examples:
        modal run backend/scripts/modal_finetune.py --verify          # path check only
        modal run backend/scripts/modal_finetune.py --smoke           # verify + smoke test
        modal run backend/scripts/modal_finetune.py                   # verify + full training
        modal run backend/scripts/modal_finetune.py --skip-verify     # skip the path check
        modal run backend/scripts/modal_finetune.py --max-steps 100 --lr 1e-4
    """

    if verify:

        verify_paths.remote()

        return

    if not skip_verify:

        print("Step 1/2: Verifying vision LoRA paths before training ...")

        print("(Pass --skip-verify to skip this check after a confirmed pass)\n")

        verify_paths.remote()

        print("\nVision paths verified  Proceeding to training ...\n")

    if smoke:

        smoke_test.remote()

    else:

        train.remote(num_epochs=num_epochs, lr=lr)
