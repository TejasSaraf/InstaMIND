"""
Verify that JSONL annotations are correct for Gemma 3n fine-tuning.

Replays the exact same data-loading path as finetune_gemma3n_unsloth.py:
  _load_jsonl → _trim_messages_to_max_images → _to_unsloth_messages → _build_hf_dataset

Three verification levels:
  Level 1 (--level 1, default): Structure + JSON schema validation only (fast, no deps)
  Level 2 (--level 2): + Open every image as PIL RGB (catches corrupt/missing files)
  Level 3 (--level 3): + Tokenizer dry-run with Gemma 3n chat template (needs unsloth)

Usage:
  python verify_annotations.py                          # Level 1
  python verify_annotations.py --level 2                # Level 1 + 2
  python verify_annotations.py --level 3 --max-samples 50  # Full (slow)
  python verify_annotations.py --split train            # Single split only
"""


from __future__ import annotations
from lib.classes import INSTAMIND_CLASSES
from lib.paths import ANNOTATIONS, FRAMES_ROOT


import argparse

import json

import sys

from collections import Counter

from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))


KNOWN_CLASSES = frozenset(INSTAMIND_CLASSES)

REQUIRED_RESPONSE_KEYS = {"incident_type", "confidence",
                          "timestamp_seconds", "evidence", "recommended_action"}

SPLITS = ["train", "val", "test"]


def validate_structure(samples: list[dict], split: str) -> list[str]:
    """Validate JSONL structure matches finetune_gemma3n_unsloth.py expectations."""

    errors: list[str] = []

    class_counts: Counter = Counter()

    image_count_hist: Counter = Counter()

    confidence_values: list[float] = []

    for i, sample in enumerate(samples):

        prefix = f"{split}[{i}]"

        msgs = sample.get("messages")

        if not msgs or not isinstance(msgs, list):

            errors.append(f"{prefix}: missing or invalid 'messages' key")

            continue

        if len(msgs) != 3:

            errors.append(
                f"{prefix}: expected 3 messages (system/user/assistant), got {len(msgs)}")

            continue

        roles = [m.get("role") for m in msgs]

        if roles != ["system", "user", "assistant"]:

            errors.append(
                f"{prefix}: roles are {roles}, expected [system, user, assistant]")

            continue

        sys_content = msgs[0].get("content")

        if not isinstance(sys_content, str) or not sys_content.strip():

            errors.append(f"{prefix}: system content must be non-empty string")

        user_content = msgs[1].get("content")

        if not isinstance(user_content, list):

            errors.append(
                f"{prefix}: user content must be a list, got {type(user_content).__name__}")

            continue

        text_parts = [p for p in user_content if isinstance(
            p, dict) and p.get("type") == "text"]

        image_parts = [p for p in user_content if isinstance(
            p, dict) and p.get("type") == "image"]

        if not text_parts:

            errors.append(f"{prefix}: user content has no text part")

        elif not text_parts[0].get("text", "").strip():

            errors.append(f"{prefix}: user text part is empty")

        if not image_parts:

            errors.append(f"{prefix}: user content has no image parts")

        for j, img in enumerate(image_parts):

            img_val = img.get("image")

            if not img_val or not isinstance(img_val, str):

                errors.append(
                    f"{prefix}: image[{j}] has no 'image' path string")

            elif not Path(img_val).is_absolute():

                errors.append(
                    f"{prefix}: image[{j}] path is not absolute: {img_val}")

        image_count_hist[len(image_parts)] += 1

        for p in user_content:

            if isinstance(p, dict) and p.get("type") not in ("text", "image"):

                errors.append(
                    f"{prefix}: unknown content type '{p.get('type')}'")

        asst_content = msgs[2].get("content")

        if not isinstance(asst_content, str) or not asst_content.strip():

            errors.append(
                f"{prefix}: assistant content must be non-empty string")

            continue

        try:

            resp = json.loads(asst_content)

        except json.JSONDecodeError as e:

            errors.append(
                f"{prefix}: assistant content is not valid JSON: {e}")

            continue

        missing = REQUIRED_RESPONSE_KEYS - set(resp.keys())

        if missing:

            errors.append(f"{prefix}: assistant JSON missing keys: {missing}")

            continue

        inc_type = resp["incident_type"]

        if inc_type not in KNOWN_CLASSES:

            errors.append(f"{prefix}: unknown incident_type '{inc_type}'")

        else:

            class_counts[inc_type] += 1

        conf = resp["confidence"]

        if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):

            errors.append(f"{prefix}: confidence {conf} not in [0.0, 1.0]")

        else:

            confidence_values.append(float(conf))

        evidence = resp.get("evidence")

        if not evidence or not isinstance(evidence, str) or not evidence.strip():

            errors.append(f"{prefix}: evidence is empty or not a string")

        action = resp.get("recommended_action")

        if not action or not isinstance(action, str):

            errors.append(
                f"{prefix}: recommended_action is empty or not a string")

    return errors, class_counts, image_count_hist, confidence_values


def validate_images(samples: list[dict], split: str, max_samples: int) -> list[str]:
    """Open every referenced image as PIL RGB — same as _to_unsloth_messages."""

    from PIL import Image

    errors: list[str] = []

    checked = 0

    total_images = 0

    subset = samples[:max_samples] if max_samples > 0 else samples

    for i, sample in enumerate(subset):

        msgs = sample.get("messages", [])

        if len(msgs) < 2:

            continue

        user_content = msgs[1].get("content", [])

        if not isinstance(user_content, list):

            continue

        for part in user_content:

            if not isinstance(part, dict) or part.get("type") != "image":

                continue

            img_path = part.get("image", "")

            total_images += 1

            p = Path(img_path)

            if not p.exists():

                errors.append(f"{split}[{i}]: image not found: {img_path}")

                continue

            try:

                img = Image.open(p).convert("RGB")

                w, h = img.size

                if w < 16 or h < 16:

                    errors.append(
                        f"{split}[{i}]: image too small ({w}x{h}): {img_path}")

                img.close()

            except Exception as e:

                errors.append(
                    f"{split}[{i}]: cannot open image: {img_path} ({e})")

        checked += 1

    return errors, checked, total_images


def validate_tokenizer(samples: list[dict], split: str, max_samples: int) -> list[str]:
    """Dry-run through Gemma 3n tokenizer chat template."""

    try:

        from unsloth import FastModel

    except ImportError:

        return ["unsloth not installed — skip level 3 (pip install unsloth)"], 0

    from PIL import Image

    import copy

    errors: list[str] = []

    print(f"    Loading tokenizer (this may take a moment)...")

    _, tokenizer = FastModel.from_pretrained(

        "unsloth/gemma-3n-E4B-it",

        max_seq_length=4096,

        load_in_4bit=False,

    )

    subset = samples[:max_samples] if max_samples > 0 else samples

    rendered = 0

    for i, sample in enumerate(subset):

        msgs = copy.deepcopy(sample["messages"])

        converted = []

        for msg in msgs:

            role = msg["role"]

            content = msg.get("content")

            if isinstance(content, str):

                converted.append({"role": role, "content": [
                                 {"type": "text", "text": content}]})

                continue

            new_content = []

            for part in content:

                if not isinstance(part, dict):

                    continue

                if part.get("type") == "text":

                    new_content.append({"type": "text", "text": part["text"]})

                elif part.get("type") == "image":

                    try:

                        pil = Image.open(part["image"]).convert("RGB")

                        new_content.append({"type": "image", "image": pil})

                    except Exception as e:

                        errors.append(f"{split}[{i}]: PIL open failed: {e}")

                        continue

            converted.append({"role": role, "content": new_content})

        try:

            text = tokenizer.apply_chat_template(

                converted, tokenize=False, add_generation_prompt=False,

            )

            if "<start_of_turn>model\n" not in text:

                errors.append(
                    f"{split}[{i}]: rendered template missing '<start_of_turn>model\\n' marker")

            tokens = tokenizer(text, return_tensors="pt")

            seq_len = tokens["input_ids"].shape[1]

            if seq_len > 4096:

                errors.append(
                    f"{split}[{i}]: sequence length {seq_len} exceeds 4096")

            rendered += 1

        except Exception as e:

            errors.append(f"{split}[{i}]: tokenizer error: {e}")

    return errors, rendered


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL — same as finetune script's _load_jsonl."""

    samples = []

    for line in path.read_text(encoding="utf-8").splitlines():

        line = line.strip()

        if line:

            samples.append(json.loads(line))

    return samples


def print_report(title: str, errors: list[str], max_show: int = 20):

    if not errors:

        print(f"    ✅ {title}: PASS")

    else:

        print(f"    ❌ {title}: {len(errors)} errors")

        for e in errors[:max_show]:

            print(f"       • {e}")

        if len(errors) > max_show:

            print(f"       ... and {len(errors) - max_show} more")


def main():

    parser = argparse.ArgumentParser(
        description="Verify JSONL annotations for Gemma 3n")

    parser.add_argument("--annotations-dir", type=Path, default=ANNOTATIONS)

    parser.add_argument("--level", type=int, default=1, choices=[1, 2, 3],

                        help="1=structure, 2=+images, 3=+tokenizer")

    parser.add_argument("--split", type=str, default=None, choices=SPLITS,

                        help="Verify single split only")

    parser.add_argument("--max-samples", type=int, default=0,

                        help="Max samples for level 2/3 (0=all)")

    args = parser.parse_args()

    splits = [args.split] if args.split else SPLITS

    ann_dir = args.annotations_dir

    print(f"═══ Verify Annotations for Gemma 3n Fine-tuning ═══")

    print(f"  Directory: {ann_dir}")

    print(f"  Level:     {args.level}")

    print(f"  Splits:    {', '.join(splits)}")

    if args.max_samples > 0:

        print(f"  Max samples (L2/L3): {args.max_samples}")

    all_pass = True

    for split in splits:

        path = ann_dir / f"{split}.jsonl"

        print(f"\n── {split} ──")

        if not path.exists():

            print(f"  ⚠ {path.name} not found — skipping")

            all_pass = False

            continue

        samples = load_jsonl(path)

        print(f"  Loaded {len(samples)} samples")

        if not samples:

            print(f"  ⚠ Empty file")

            continue

        print(f"\n  Level 1: Structure validation")

        errors, class_counts, img_hist, conf_vals = validate_structure(
            samples, split)

        print_report("Structure", errors)

        if class_counts:

            print(f"\n    Class distribution:")

            for cls in INSTAMIND_CLASSES:

                count = class_counts.get(cls, 0)

                pct = 100 * count / len(samples) if samples else 0

                tag = " ⚠ MISSING" if count == 0 and split == "train" else ""

                print(f"      {cls:<15}: {count:4d} ({pct:5.1f}%){tag}")

        if img_hist:

            print(
                f"\n    Images per sample: min={min(img_hist)}, max={max(img_hist)}")

            for n_imgs in sorted(img_hist):

                print(f"      {n_imgs} images: {img_hist[n_imgs]} samples")

        if conf_vals:

            print(f"\n    Confidence: min={min(conf_vals):.2f}, max={max(conf_vals):.2f}, "

                  f"mean={sum(conf_vals)/len(conf_vals):.2f}")

            unique_confs = len(set(conf_vals))

            if unique_confs == 1:

                print(f"    ⚠ All confidence values are identical ({conf_vals[0]}) — "

                      f"model will learn to always predict this value")

            else:

                print(
                    f"    ✓ {unique_confs} unique confidence values (jitter working)")

        if errors:

            all_pass = False

        if args.level >= 2:

            print(f"\n  Level 2: Image validation")

            cap = args.max_samples if args.max_samples > 0 else len(samples)

            img_errors, checked, total_imgs = validate_images(
                samples, split, cap)

            print_report("Images", img_errors)

            print(f"    Checked {checked} samples, {total_imgs} images")

            if img_errors:

                all_pass = False

        if args.level >= 3:

            print(f"\n  Level 3: Tokenizer dry-run")

            cap = args.max_samples if args.max_samples > 0 else min(
                10, len(samples))

            tok_errors, rendered = validate_tokenizer(samples, split, cap)

            print_report("Tokenizer", tok_errors)

            print(f"    Rendered {rendered} samples through chat template")

            if tok_errors:

                all_pass = False

    if len(splits) > 1:

        print(f"\n── Cross-split checks ──")

        all_fingerprints: dict[str, set[str]] = {}

        for split in splits:

            path = ann_dir / f"{split}.jsonl"

            if not path.exists():

                continue

            samples = load_jsonl(path)

            fps = set()

            for s in samples:

                imgs = []

                user = s.get("messages", [{}])[1] if len(
                    s.get("messages", [])) > 1 else {}

                content = user.get("content", [])

                if isinstance(content, list):

                    for p in content:

                        if isinstance(p, dict) and p.get("type") == "image":

                            imgs.append(p["image"])

                fps.add("|".join(imgs))

            all_fingerprints[split] = fps

        train_fps = all_fingerprints.get("train", set())

        for other in ["val", "test"]:

            other_fps = all_fingerprints.get(other, set())

            overlap = train_fps & other_fps

            if overlap:

                print(
                    f"  ❌ {len(overlap)} samples share identical images between train and {other}")

                all_pass = False

            else:

                print(f"  ✅ No image overlap between train and {other}")

    meta_path = ann_dir / "build_meta.json"

    if meta_path.exists():

        print(f"\n── Build metadata ──")

        meta = json.loads(meta_path.read_text())

        print(f"  Seed: {meta.get('seed')}")

        print(f"  Max frames/sample: {meta.get('max_frames_per_sample')}")

        print(f"  Total samples: {meta.get('total_samples')}")

        for split, stats in meta.get("splits", {}).items():

            ve = stats.get("validation_errors", 0)

            tag = "✅" if ve == 0 else f"❌ {ve} errors"

            print(f"  {split}: {stats.get('samples', 0)} samples, "

                  f"dupes={stats.get('duplicates_removed', 0)}, "

                  f"missing={stats.get('frames_missing', 0)}, "

                  f"validation={tag}")

    else:

        print(f"\n  ⚠ build_meta.json not found")

    print(f"\n{'═' * 55}")

    if all_pass:

        print("✅ ALL CHECKS PASSED — annotations are ready for Gemma 3n fine-tuning")

    else:

        print("❌ SOME CHECKS FAILED — review errors above before training")

    print(f"{'═' * 55}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":

    main()
