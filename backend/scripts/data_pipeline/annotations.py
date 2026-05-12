

"""
Phase 7: Build final JSONL files for Gemma 3n fine-tuning.

Groups frames by their temporal window, so each training sample
contains up to MAX_FRAMES_PER_SAMPLE frames from the same incident
window.  This teaches the model temporal reasoning across a sequence.

Prompt template rotates across 6 variations to prevent the model
from memorizing prompt phrasing (which was a failure mode in the
previous training run — you saw "collision" from a normal frame
partially because it saw that specific prompt many times).

Output format matches finetune_gemma3n_unsloth.py exactly:
  {
    "messages": [
      {"role": "system",    "content": "..."},
      {"role": "user",      "content": [
        {"type": "text",  "text": "..."},
        {"type": "image", "image": "/abs/path/frame.jpg"},
        ...
      ]},
      {"role": "assistant", "content": "{\"incident_type\": ...}"}
    ]
  }

Edge cases handled:
  - Augmented frames excluded from val/test by frame_path pattern (not source)
  - Frame paths resolved to absolute (finetune script uses _resolve_image_path)
  - NaN/empty descriptions → fallback text
  - Very large windows capped via evenly-spaced selection
  - Duplicate samples deduplicated by frame-set fingerprint
  - Missing frame files skipped with warning
  - Template selection reproducible across sessions (hashlib, not hash())
  - Per-class confidence with deterministic jitter (no fixed-value overfitting)
  - Deterministic shuffle with seed for reproducibility
  - Post-write JSONL validation (structure + keys + known classes)
  - Build metadata JSON saved for reproducibility

Input:  manifests/frames_final.parquet
Output: annotations/train.jsonl
        annotations/val.jsonl
        annotations/test.jsonl
        annotations/build_meta.json
"""


from __future__ import annotations


import argparse

import hashlib

import json

import random

import sys

import pandas as pd

from pathlib import Path


from lib.paths import MANIFESTS, ANNOTATIONS

from lib.classes import INSTAMIND_CLASSES

from lib.config import PipelineConfig


INPUT = MANIFESTS / "frames_final.parquet"

DEFAULT_SEED = 42

DEFAULT_MAX_FRAMES = 8

MIN_FRAMES_PER_SAMPLE = 1


_OPENAI_SUMMARIES = {}

_openai_json = ANNOTATIONS / "openai_summaries.json"

if _openai_json.exists():

    with open(_openai_json, "r") as f:

        _OPENAI_SUMMARIES = json.load(f)

    print(f"  Loaded {_openai_json.name} ({len(_OPENAI_SUMMARIES)} summaries)")


SYSTEM_PROMPT = (

    "You are a surveillance security analyst. Given a temporal sequence of "

    "surveillance camera frames, classify the observed activity into exactly "

    "one of these incident types: fighting, robbery, shoplifting, shooting, "

    "fainting, normal. "

    "Respond ONLY with valid JSON containing exactly these keys: "

    "incident_type (string), confidence (float 0.0-1.0), "

    "timestamp_seconds (float), "

    "evidence (brief description of what is observed), "

    "recommended_action (string)."

)


_CLASS_LIST = "fighting, robbery, shoplifting, shooting, fainting, normal"


def _frames_ref(n: int) -> str:
    """Grammatically correct frame reference for singular/plural."""

    return "this surveillance frame" if n == 1 else f"these {n} surveillance frames"


def _frames_ref_short(n: int) -> str:

    return "this frame" if n == 1 else f"these {n} frames"


USER_TEMPLATES = [

    "Given {frames_ref}, determine if a security event is taking place. "

    f"The incident_type MUST be one of: {_CLASS_LIST}. "

    "Output your analysis as JSON: incident_type, confidence, timestamp_seconds, evidence, recommended_action.",



    "{frames_ref_cap} come from a security camera. Analyse for any abnormal or "

    f"threatening behaviour. Classify as one of: {_CLASS_LIST}. "

    "Return a JSON response with keys: incident_type, confidence, timestamp_seconds, evidence, recommended_action.",



    "You are reviewing {n} sequential frame(s) from a surveillance feed. Classify the "

    f"observed activity into exactly one of: {_CLASS_LIST}. "

    "Respond only with JSON containing: incident_type, confidence, timestamp_seconds, evidence, recommended_action.",



    "As a surveillance analyst, classify the activity in {frames_ref_short}. "

    f"Choose incident_type from: {_CLASS_LIST}. "

    "Provide your assessment as JSON with: incident_type, confidence, timestamp_seconds, evidence, recommended_action.",



    "What security event, if any, is shown in {frames_ref}? "

    f"incident_type must be one of: {_CLASS_LIST}. "

    "Output a JSON with keys: incident_type, confidence, timestamp_seconds, evidence, recommended_action.",



    "Analyse the following {n} surveillance frame(s) and classify any security incident. "

    f"Use exactly one of these incident types: {_CLASS_LIST}. "

    "Return a JSON object with exactly these keys: incident_type, confidence, timestamp_seconds, evidence, "

    "recommended_action.",



    "Review {frames_ref_short} from CCTV footage. Identify whether a security incident is "

    f"occurring. incident_type must be exactly one of: {_CLASS_LIST}. "

    "Respond with a JSON object containing: incident_type, confidence, timestamp_seconds, evidence, "

    "recommended_action.",



    "Inspect the following {n} frame(s) captured by a surveillance camera. Determine if "

    f"any incident is present. Classify as: {_CLASS_LIST}. "

    "Reply with a JSON object: incident_type, confidence, timestamp_seconds, evidence, recommended_action.",

]


RECOMMENDED_ACTIONS = {

    "fighting":    "dispatch_security_immediately",

    "robbery":     "alert_police_and_lock_exits",

    "shoplifting": "intercept_at_exit_and_review_footage",

    "shooting":    "initiate_lockdown_and_alert_emergency_services",

    "fainting":  "dispatch_medical_assistance",

    "normal":      "none",

}


CONFIDENCE_RANGES = {

    "fighting":    (0.40, 0.98),

    "robbery":     (0.40, 0.98),

    "shoplifting": (0.40, 0.95),

    "shooting":    (0.40, 0.98),

    "fainting":    (0.40, 0.98),

    "normal":      (0.45, 0.99),

}


_FAINTING_SUMMARIES = [

    "Person collapsed suddenly and is lying still on the ground.",

    "Individual has fallen and is not moving — possible loss of consciousness.",

    "Subject dropped to the floor without warning and remains motionless.",

    "Person crumpled to the ground and has not gotten back up.",

    "A sudden collapse was observed; the individual is unresponsive on the floor.",

    "One person fell abruptly and is lying flat on the ground.",

    "Person lost balance and collapsed — currently motionless.",

    "Individual slumped to the ground and appears unconscious.",

    "A person fell forward and is now lying face-down on the surface.",

    "Subject fell sideways and is motionless on the floor.",

    "An individual staggered briefly and then collapsed.",

    "Person suddenly buckled at the knees and went down.",

    "A standing person dropped to the ground unexpectedly.",

    "Individual fell and is lying in an awkward position without movement.",

    "Person lost consciousness and collapsed onto the floor.",

    "A sudden fall was detected — the person is not attempting to get up.",

    "Subject went limp and fell to the ground.",

    "Person toppled over and is lying still on the pavement.",

    "An apparent fainting episode — individual is on the ground unresponsive.",

    "One person fell to the ground and has remained there without moving.",

    "Individual swayed and then collapsed suddenly.",

    "Person fell backward and hit the ground — no further movement detected.",

    "A collapse event occurred; the subject is prone on the floor.",

    "Person stumbled and fell — currently lying still with no visible movement.",

    "Subject is on the ground after an apparent loss of balance or consciousness.",

    "A person in the scene has fallen and is lying motionless.",

    "Individual went down hard and is not moving.",

    "Fall event detected — person is on the ground and not moving.",

    "Person has fallen to the ground and is motionless.",

    "Subject has fallen and may require medical assistance.",

]


_NORMAL_SUMMARIES = [

    "Routine foot traffic — no security concern observed.",

    "People walking through the area normally.",

    "Standard activity in the scene — no anomalies detected.",

    "No unusual behaviour observed in the surveillance footage.",

    "Regular daily activity; nothing appears out of the ordinary.",

    "Normal pedestrian movement — no incident visible.",

    "The scene shows typical activity with no security events.",

    "Individuals are going about their regular business.",

    "Calm environment with normal human movement.",

    "No suspicious or threatening activity detected.",

    "People are walking, standing, or sitting in an orderly manner.",

    "Normal scene — individuals appear to be conducting routine tasks.",

    "The area is quiet with standard levels of activity.",

    "Regular patron or staff movement — nothing alarming.",

    "Scene appears safe with no signs of distress or criminal behaviour.",

    "Typical surveillance footage — no event of interest.",

    "People are moving through the space without any notable incidents.",

    "No abnormal motion or behaviour detected in the frames.",

    "Routine operations visible — no intervention required.",

    "The footage shows normal everyday activity.",

    "Standard low-risk scene with no threatening behaviour.",

    "Ordinary pedestrian and vehicle movement.",

    "No aggressive, suspicious, or medical events detected.",

    "The environment appears calm and controlled.",

    "Individuals are engaged in normal conversations or transactions.",

    "People entering and exiting the area in an orderly fashion.",

    "No signs of conflict, theft, or medical emergency.",

    "Scene is clear — standard monitoring conditions.",

    "Typical crowd movement with no anomalies.",

    "Pre-incident scene — normal activity, no security event yet.",

]


def _pick_diverse_summary(

    incident_type: str,

    description: str,

    video_id: str,

    sample_idx: int,

    rng: random.Random,

) -> str:
    """Return a diverse summary for the assistant response.

    For classes with a limited description pool (fainting, normal), randomly
    select from a large template bank so the model sees varied phrasings.
    When a real, unique description exists (e.g. normal videos with captions),
    keep it most of the time but occasionally swap in a template.
    """

    pool: list[str] | None = None

    if incident_type == "fainting":

        pool = _FAINTING_SUMMARIES

    elif incident_type == "normal":

        pool = _NORMAL_SUMMARIES

    if pool is None:

        return _safe_description(description, incident_type)

    desc = _safe_description(description, incident_type)

    is_generic = desc.startswith(incident_type) and "activity detected" in desc

    if is_generic:

        idx = _stable_hash(f"{video_id}_{sample_idx}") % len(pool)

        return pool[idx]

    if rng.random() < 0.40:

        idx = _stable_hash(f"{video_id}_{sample_idx}_swap") % len(pool)

        return pool[idx]

    return desc


def _stable_hash(s: str) -> int:
    """Reproducible integer hash (PYTHONHASHSEED-independent)."""

    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)


def _is_augmented(frame_path: str) -> bool:
    """Detect augmented frames by filename pattern (from augment.py).

    augment.py writes:  {stem}_aug_{uid}.jpg
    The source column stays "ucf_uca"/"urfd" so we cannot use it.
    """

    return "_aug_" in Path(frame_path).name


def _select_evenly_spaced(items: list, n: int) -> list:
    """Return up to *n* evenly-spaced items preserving temporal order."""

    if not items or n <= 0:

        return []

    if len(items) <= n:

        return list(items)

    if n == 1:

        return [items[len(items) // 2]]

    indices = [round(i * (len(items) - 1) / (n - 1)) for i in range(n)]

    seen: set[int] = set()

    out: list = []

    for idx in indices:

        if idx not in seen:

            out.append(items[idx])

            seen.add(idx)

    return out


def _safe_description(desc, incident_type: str) -> str:
    """Guard against NaN / None / empty description."""

    if desc is None or (isinstance(desc, float) and pd.isna(desc)):

        return f"{incident_type} activity detected in surveillance footage."

    s = str(desc).strip()

    return s if s else f"{incident_type} activity detected in surveillance footage."


def _sample_fingerprint(sample: dict) -> str:
    """Fingerprint by ordered image paths (for dedup)."""

    images: list[str] = []

    for msg in sample.get("messages", []):

        if msg.get("role") != "user":

            continue

        content = msg.get("content", [])

        if isinstance(content, list):

            for part in content:

                if isinstance(part, dict) and part.get("type") == "image":

                    images.append(part["image"])

    return "|".join(images)


def make_sample(

    video_id: str,

    incident_type: str,

    frame_paths: list[str],

    description: str,

    rng: random.Random,

    window_start_s: float = 0.0,

) -> dict:
    """Build one JSONL sample in Gemma 3n chat format.

    Format matches what finetune_gemma3n_unsloth.py's _to_unsloth_messages()
    expects:
      - system:    plain string  → wrapped to [{"type":"text","text":...}]
      - user:      list of dicts → image paths resolved to PIL at train time
      - assistant: plain JSON string → wrapped to [{"type":"text","text":...}]
    """

    n = len(frame_paths)

    template_idx = _stable_hash(
        f"{video_id}_{window_start_s}") % len(USER_TEMPLATES)

    fr = _frames_ref(n)

    frs = _frames_ref_short(n)

    user_text = USER_TEMPLATES[template_idx].format(

        n=n, frames_ref=fr, frames_ref_short=frs,

        frames_ref_cap=fr[0].upper() + fr[1:],

    )

    user_content: list[dict] = [{"type": "text", "text": user_text}]

    for fp in frame_paths:

        user_content.append(
            {"type": "image", "image": str(Path(fp).resolve())})

    lo, hi = CONFIDENCE_RANGES.get(incident_type, (0.40, 0.98))

    frame_boost = min(n / 10.0, 1.0)

    midpoint = lo + (hi - lo) * (0.2 + 0.6 * frame_boost)

    jitter = rng.uniform(-(midpoint - lo), hi - midpoint)

    confidence = round(max(lo, min(hi, midpoint + jitter)), 2)

    desc = _pick_diverse_summary(incident_type, description, video_id, n, rng)

    assistant_response = json.dumps({

        "incident_type":      incident_type,

        "confidence":         confidence,

        "timestamp_seconds":  window_start_s,

        "evidence":           desc,

        "recommended_action": RECOMMENDED_ACTIONS.get(incident_type, "investigate"),

    }, ensure_ascii=False)

    return {

        "messages": [

            {"role": "system",    "content": SYSTEM_PROMPT},

            {"role": "user",      "content": user_content},

            {"role": "assistant", "content": assistant_response},

        ]

    }


def _oversample_minority_samples(

    samples: list[dict],

    window_frame_pools: dict[str, list[str]],

    classes: list[str],

    target_per_class: int,

    max_frames: int,

    rng: random.Random,

) -> list[dict]:
    """Duplicate minority-class samples with different frame subsets.

    For classes with fewer than *target_per_class* samples, create
    additional samples from the same windows using different frame
    combinations drawn from the full augmented pool.  This lets the
    model see the same incident from slightly different visual angles
    (augmented brightness/flip/blur).

    Only applied to training data.
    """

    by_class: dict[str, list[dict]] = {cls: [] for cls in classes}

    for s in samples:

        try:

            cls = json.loads(s["messages"][2]["content"])["incident_type"]

        except (json.JSONDecodeError, KeyError, IndexError):

            cls = "unknown"

        if cls in by_class:

            by_class[cls].append(s)

    new_samples: list[dict] = []

    for cls in classes:

        cls_samples = by_class.get(cls, [])

        n = len(cls_samples)

        if n == 0 or n >= target_per_class:

            continue

        needed = target_per_class - n

        added = 0

        for i in range(needed):

            src = cls_samples[i % n]

            fp = _sample_fingerprint(src)

            pool = window_frame_pools.get(fp, [])

            if len(pool) <= max_frames:

                new_samples.append(src)

                added += 1

                continue

            unique_pool = list(dict.fromkeys(pool))

            subset = rng.sample(unique_pool, min(max_frames, len(unique_pool)))

            subset.sort()

            vid_id = f"{fp}_oversample_{i}"

            incident_type = cls

            try:

                orig_resp = json.loads(src["messages"][2]["content"])

                desc = orig_resp.get("evidence", orig_resp.get("summary", ""))

            except (json.JSONDecodeError, KeyError):

                desc = ""

            new_sample = make_sample(vid_id, incident_type, subset, desc, rng)

            new_samples.append(new_sample)

            added += 1

        print(f"    Oversampled {cls}: {n} → {n + added} samples "

              f"(+{added} from frame-subset variants)")

    return samples + new_samples


def build_split(

    df: pd.DataFrame,

    split: str,

    out_path: Path,

    seed: int = DEFAULT_SEED,

    max_frames: int = DEFAULT_MAX_FRAMES,

    oversample_target: int = 0,

) -> tuple[list[dict], dict]:
    """Build JSONL samples for one split.

    If *oversample_target* > 0 and split == 'train', minority classes
    with fewer samples will be oversampled to this target using
    frame-subset variants.

    Returns (samples_list, stats_dict) for metadata logging.
    """

    split_df = df[df["split"] == split].copy()

    aug_removed = 0

    if split in ["val", "test"]:

        mask = split_df["frame_path"].apply(_is_augmented)

        aug_removed = int(mask.sum())

        split_df = split_df[~mask]

        if aug_removed > 0:

            print(f"    Removed {aug_removed} augmented frames from {split}")

    if len(split_df) == 0:

        print(f"  ⚠ {split}: no frames — writing empty JSONL")

        out_path.parent.mkdir(parents=True, exist_ok=True)

        out_path.write_text("")

        return [], {"samples": 0, "augmented_removed": aug_removed, "per_class": {}}

    vid_col = "video_id" if "video_id" in split_df.columns else "video_name"

    group_keys = [vid_col, "window_start_s", "window_end_s", "incident_type"]

    rng = random.Random(seed)

    samples: list[dict] = []

    window_frame_pools: dict[str, list[str]] = {}

    skipped_missing = 0

    skipped_empty = 0

    for group_id, group in split_df.groupby(group_keys):

        vid, window_start, window_end, incident_type = group_id

        if "timestamp_s" in group.columns:

            group = group.sort_values("timestamp_s")

        elif "frame_idx" in group.columns:

            group = group.sort_values("frame_idx")

        else:

            group = group.sort_values("frame_path")

        frame_paths = group["frame_path"].tolist()

        valid_paths: list[str] = []

        seen_paths: set[str] = set()

        for fp in frame_paths:

            if not Path(fp).exists():

                skipped_missing += 1

            elif fp not in seen_paths:

                valid_paths.append(fp)

                seen_paths.add(fp)

        if len(valid_paths) < MIN_FRAMES_PER_SAMPLE:

            skipped_empty += 1

            continue

        selected = valid_paths

        if len(valid_paths) > max_frames:

            selected = _select_evenly_spaced(valid_paths, max_frames)

        selected = list(dict.fromkeys(selected))

        desc = group["description"].iloc[0]

        openai_key = f"{vid}_{window_start}"

        if openai_key in _OPENAI_SUMMARIES:

            desc = _OPENAI_SUMMARIES[openai_key]

        sample = make_sample(str(vid), incident_type,
                             selected, desc, rng, window_start_s=window_start)

        samples.append(sample)

        fp_key = _sample_fingerprint(sample)

        window_frame_pools[fp_key] = valid_paths

    if skipped_missing > 0:

        print(f"    ⚠ {split}: {skipped_missing} frame files not found on disk")

    if skipped_empty > 0:

        print(
            f"    ⚠ {split}: {skipped_empty} windows skipped (no valid frames)")

    seen: set[str] = set()

    unique: list[dict] = []

    dupes = 0

    for s in samples:

        fp = _sample_fingerprint(s)

        if fp not in seen:

            seen.add(fp)

            unique.append(s)

        else:

            dupes += 1

    if dupes > 0:

        print(f"    Removed {dupes} duplicate samples from {split}")

    samples = unique

    oversampled = 0

    if split == "train" and oversample_target > 0:

        before = len(samples)

        samples = _oversample_minority_samples(

            samples, window_frame_pools, INSTAMIND_CLASSES,

            target_per_class=oversample_target,

            max_frames=max_frames,

            rng=rng,

        )

        oversampled = len(samples) - before

    rng.shuffle(samples)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:

        for s in samples:

            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    errors = _validate_jsonl(out_path)

    if errors > 0:

        print(f"     {split}.jsonl: {errors} validation errors")

    else:

        print(f"     {split}.jsonl validated")

    dup_image_samples = 0

    for s in samples:

        imgs = [

            p["image"] for m in s.get("messages", [])

            if m.get("role") == "user"

            for p in (m.get("content") or [])

            if isinstance(p, dict) and p.get("type") == "image"

        ]

        if len(imgs) != len(set(imgs)):

            dup_image_samples += 1

    if dup_image_samples > 0:

        print(
            f"     {split}: {dup_image_samples} samples have duplicate image paths")

    else:

        print(f"     {split}: no within-sample duplicate images")

    class_counts: dict[str, int] = {}

    for s in samples:

        cls = json.loads(s["messages"][2]["content"])["incident_type"]

        class_counts[cls] = class_counts.get(cls, 0) + 1

    print(f"\n  {split}.jsonl: {len(samples)} samples")

    for cls in INSTAMIND_CLASSES:

        count = class_counts.get(cls, 0)

        tag = "  ⚠ MISSING" if count == 0 and split == "train" else ""

        print(f"    {cls:<15}: {count:4d}{tag}")

    return samples, {

        "samples": len(samples),

        "augmented_removed": aug_removed,

        "duplicates_removed": dupes,

        "oversampled": oversampled,

        "frames_missing": skipped_missing,

        "windows_skipped": skipped_empty,

        "validation_errors": errors,

        "per_class": class_counts,

    }


REQUIRED_RESPONSE_KEYS = {"incident_type", "confidence",
                          "timestamp_seconds", "evidence", "recommended_action"}

KNOWN_CLASSES = frozenset(INSTAMIND_CLASSES)


def _validate_jsonl(path: Path) -> int:
    """Post-write integrity check.

    Validates:
      - Each line is valid JSON
      - messages array has system / user / assistant roles
      - Assistant content is valid JSON with all required keys
      - incident_type is a known class
      - User message has at least one text part and one image part
    """

    errors = 0

    text = path.read_text(encoding="utf-8").strip()

    if not text:

        return 0

    for i, line in enumerate(text.splitlines(), 1):

        try:

            sample = json.loads(line)

        except json.JSONDecodeError:

            errors += 1

            continue

        msgs = sample.get("messages")

        if not msgs or not isinstance(msgs, list) or len(msgs) < 3:

            errors += 1

            continue

        roles = [m.get("role") for m in msgs]

        if roles != ["system", "user", "assistant"]:

            errors += 1

            continue

        try:

            resp = json.loads(msgs[2]["content"])

            missing = REQUIRED_RESPONSE_KEYS - set(resp.keys())

            if missing:

                errors += 1

            elif resp["incident_type"] not in KNOWN_CLASSES:

                errors += 1

        except (json.JSONDecodeError, KeyError, TypeError):

            errors += 1

        content = msgs[1].get("content", [])

        if not isinstance(content, list):

            errors += 1

            continue

        has_text = any(

            isinstance(p, dict) and p.get(
                "type") == "text" and p.get("text", "").strip()

            for p in content

        )

        has_image = any(

            isinstance(p, dict) and p.get("type") == "image"

            for p in content

        )

        if not has_text or not has_image:

            errors += 1

    return errors


def main():

    parser = argparse.ArgumentParser(
        description="Phase 7: Build JSONL annotations")

    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,

                        help="Random seed for reproducibility")

    parser.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES,

                        help="Max frames per sample (evenly-spaced selection)")

    parser.add_argument("--oversample-target", type=int, default=0,

                        help="Target samples/class for minority oversampling. "

                             "0 = auto (median of class counts).")

    args = parser.parse_args()

    print("── Phase 7: Build JSONL annotations ──")

    if not INPUT.exists():

        print(f"  ERROR: {INPUT} not found. Run rebalance.py first.")

        sys.exit(1)

    df = pd.read_parquet(INPUT)

    if "video_id" not in df.columns:

        df["video_id"] = df["source"].str.cat(df["video_name"], sep="/")

        print("  ⚠ video_id derived from source/video_name")

    print(f"  Input: {len(df)} frames, {df['video_id'].nunique()} videos, "

          f"{df['incident_type'].nunique()} classes")

    print(f"  Max frames/sample: {args.max_frames}")

    print(f"  Seed: {args.seed}")

    vid_col = "video_id" if "video_id" in df.columns else "video_name"

    vids_per_split = {

        s: set(df.loc[df["split"] == s, vid_col])

        for s in ["train", "val", "test"]

    }

    leaks = []

    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:

        overlap = vids_per_split[a] & vids_per_split[b]

        if overlap:

            leaks.append(f"{a}∩{b}: {sorted(overlap)[:5]}")

    if leaks:

        print(f"   VIDEO-LEVEL LEAKAGE DETECTED:")

        for lk in leaks:

            print(f"    {lk}")

        sys.exit(1)

    print(f"   No video-level leakage (checked via {vid_col})")

    oversample_target = args.oversample_target

    all_stats: dict[str, dict] = {}

    total_samples = 0

    for split in ["train", "val", "test"]:

        out_path = ANNOTATIONS / f"{split}.jsonl"

        ot = oversample_target if split == "train" else 0

        samples, stats = build_split(

            df, split, out_path, seed=args.seed, max_frames=args.max_frames,

            oversample_target=ot,

        )

        if split == "train" and oversample_target == 0 and stats["per_class"]:

            counts = list(stats["per_class"].values())

            counts.sort()

            median = counts[len(counts) // 2] if counts else 0

            if median > 0:

                oversample_target = median

                print(f"\n    Auto oversample target: {oversample_target} "

                      f"(median of class counts)")

                samples, stats = build_split(

                    df, split, out_path, seed=args.seed,

                    max_frames=args.max_frames,

                    oversample_target=oversample_target,

                )

        all_stats[split] = stats

        total_samples += len(samples)

    meta = {

        "seed": args.seed,

        "max_frames_per_sample": args.max_frames,

        "input_manifest": str(INPUT),

        "total_samples": total_samples,

        "splits": all_stats,

        "classes": INSTAMIND_CLASSES,

        "n_user_templates": len(USER_TEMPLATES),

        "system_prompt_length": len(SYSTEM_PROMPT),

    }

    meta_path = ANNOTATIONS / "build_meta.json"

    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    images_by_split: dict[str, set[str]] = {}

    for split_name in ["train", "val", "test"]:

        jsonl_path = ANNOTATIONS / f"{split_name}.jsonl"

        paths: set[str] = set()

        if jsonl_path.exists():

            for line in jsonl_path.read_text(encoding="utf-8").splitlines():

                if not line.strip():

                    continue

                sample = json.loads(line)

                for msg in sample.get("messages", []):

                    if msg.get("role") != "user":

                        continue

                    for part in (msg.get("content") or []):

                        if isinstance(part, dict) and part.get("type") == "image":

                            paths.add(part["image"])

        images_by_split[split_name] = paths

    frame_leaks = []

    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:

        shared = images_by_split[a] & images_by_split[b]

        if shared:

            frame_leaks.append(f"{a}∩{b}: {len(shared)} shared frames")

    if frame_leaks:

        print(f"\n  ✗ FRAME-LEVEL LEAKAGE IN JSONL:")

        for fl in frame_leaks:

            print(f"    {fl}")

        sys.exit(1)

    print(f"  No frame-level leakage across JSONL splits")

    print(f"\n  Total: {total_samples} samples across all splits")

    print(f"  Metadata → {meta_path}")

    print(f"\nAnnotations written to {ANNOTATIONS}/")


if __name__ == "__main__":

    main()
