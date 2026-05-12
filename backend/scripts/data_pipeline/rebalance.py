"""
Phase 5 — Production-grade: Verify → Deduplicate → Balance (v3)

Orchestrator — all logic lives in modular lib/ files:
  lib/config.py    — externalized config with CLI override
  lib/dedup.py     — two-stage dhash + skimage SSIM, split-aware
  lib/augment.py   — augmentation pipeline, collision-free filenames
  lib/balance.py   — dynamic cap, video-diverse downsampling
  lib/validate.py  — leakage checks, adaptive thresholds, JSON summary

Pipeline stages:
  1. VERIFY   — Video-level split integrity (leakage = inflated metrics)
  2. DEDUP    — dhash fast-filter + SSIM fine-filter (train)
                Exact-duplicate removal only (val/test)
                Sliding window + temporal gap enforcement
  3. BALANCE  — Dynamic cap (p75 / max-aware) + augmented oversampling
  4. VALIDATE — Post-pipeline assertions + JSON summary artifact

Input:  manifests/frames_extracted.parquet
Output: manifests/frames_final.parquet
        manifests/pipeline_summary.json

Usage:
  python rebalance.py                          # default config
  python rebalance.py --cap-percentile 80      # CLI override
  python rebalance.py --config my_config.json  # file override
"""


from lib.validate import (

    verify_no_leakage, validate_output, save_summary_json, check_identity_collisions

)
from lib.balance import balance_training_v3
from lib.dedup import enhanced_deduplicate
from lib.config import PipelineConfig
from lib.classes import INSTAMIND_CLASSES
from lib.paths import MANIFESTS
import sys

import pandas as pd

from datetime import datetime

from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))


INPUT = MANIFESTS / "frames_extracted.parquet"

OUTPUT = MANIFESTS / "frames_final.parquet"


def main():

    cfg = PipelineConfig.from_cli()

    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = MANIFESTS / f"run_{run_ts}"

    run_dir.mkdir(parents=True, exist_ok=True)

    print("══ Phase 5: Verify → Deduplicate → Balance (v3) ══\n")

    print(f"  Config fingerprint: {cfg.fingerprint()}")

    print(f"  Config: seed={cfg.random_seed}, "

          f"hamming_train={cfg.hamming_thresh_train}, "

          f"hamming_valtest={cfg.hamming_thresh_valtest}, "

          f"ssim={cfg.ssim_thresh}, "

          f"temporal_gap={cfg.min_temporal_gap_s}s")

    print(f"  Run directory: {run_dir}")

    df = pd.read_parquet(INPUT)

    has_timestamps = "timestamp_s" in df.columns

    if "video_id" not in df.columns:

        df["video_id"] = df["source"].str.cat(df["video_name"], sep="/")

        print("  ⚠ video_id derived from source/video_name (re-extract frames for native column)")

    print(f"\n  Input: {len(df)} frames, {df['video_id'].nunique()} unique videos, "

          f"{df['incident_type'].nunique()} classes")

    print(
        f"  Timestamp column: {'present ✓' if has_timestamps else 'absent (will interpolate)'}")

    collisions = check_identity_collisions(
        df, strict=cfg.strict_identity_check)

    if collisions:

        print(f"  ⚠ {len(collisions)} video_name(s) reused across sources:")

        for name, ids in list(collisions.items())[:5]:

            print(f"      {name} → {ids}")

    else:

        print(f"  ✓ No video_name identity collisions")

    print("\n  STEP 1: Video-level split verification (using video_id)")

    split_counts = verify_no_leakage(df)

    for s, n in split_counts.items():

        print(f"    {s:<6}: {n} videos")

    print(f"    ✓ No leakage detected")

    print(f"\n  STEP 2: Enhanced dedup")

    print(f"    Train: dhash sliding-window ({cfg.sliding_window_size}) + "

          f"SSIM ({cfg.ssim_thresh}), temporal gap {cfg.min_temporal_gap_s}s")

    print(
        f"    Val/test: exact-duplicate removal only (hamming={cfg.hamming_thresh_valtest})")

    before_total = len(df)

    before_per_split = df.groupby("split").size()

    before_per_class = df[df["split"] == "train"].groupby(
        "incident_type").size()

    df, dedup_stats = enhanced_deduplicate(df, cfg, manifests_dir=run_dir)

    after_per_split = df.groupby("split").size()

    after_per_class = df[df["split"] == "train"].groupby(
        "incident_type").size()

    raw = dedup_stats["raw_frames"]

    corrupt_n = dedup_stats["corrupt_removed"]

    after_corrupt = dedup_stats["after_corrupt"]

    after_dedup = dedup_stats["after_dedup"]

    dups = dedup_stats["duplicates_removed"]

    print(f"\n    Raw frames:     {raw}")

    if corrupt_n > 0:

        print(f"    Corrupt removed: {corrupt_n} → {after_corrupt} remain")

    print(f"    Dedup removed:   {dups} → {after_dedup} remain")

    print(f"    Total reduction: {raw - after_dedup} "

          f"({(raw - after_dedup) / max(raw, 1) * 100:.1f}%)")

    print(f"\n    Per-split:")

    for s in ["train", "val", "test"]:

        b = before_per_split.get(s, 0)

        a = after_per_split.get(s, 0)

        pct = (b - a) / b * 100 if b > 0 else 0

        print(f"      {s:<6}: {b:5d} → {a:5d} ({b-a:4d} removed, {pct:.1f}%)")

    print(f"\n    Per-class (train) after dedup:")

    for cls in INSTAMIND_CLASSES:

        b = before_per_class.get(cls, 0)

        a = after_per_class.get(cls, 0)

        pct = (b - a) / b * 100 if b > 0 else 0

        print(f"      {cls:<15}: {b:5d} → {a:5d} ({pct:.0f}% removed)")

    print(f"\n  STEP 3: Class balancing (dynamic cap + augmentation oversampling)")

    df, balance_stats = balance_training_v3(df, INSTAMIND_CLASSES, cfg)

    print(f"\n  STEP 4: Output validation")

    validate_output(df, INSTAMIND_CLASSES, cfg)

    print(f"\n  ═══ Final manifest ═══")

    final_train = df[df["split"] == "train"]

    total_vids = set()

    for cls in INSTAMIND_CLASSES:

        cls_df = final_train[final_train["incident_type"] == cls]

        n = len(cls_df)

        v = cls_df["video_id"].nunique()

        total_vids.update(cls_df["video_id"].unique())

        crit = " *" if cls in cfg.critical_classes else ""

        print(f"    {cls:<15}: {n:5d} frames, {v:3d} videos{crit}")

    n_train = len(final_train)

    n_val = (df["split"] == "val").sum()

    n_test = (df["split"] == "test").sum()

    print(f"\n    Totals: {n_train} train | {n_val} val | {n_test} test")

    print(f"    Training videos (by video_id): {len(total_vids)}")

    if cfg.critical_classes:

        print(f"    Safety-critical classes: {', '.join(cfg.critical_classes)} "

              f"(weight boost: {cfg.critical_class_weight_boost}x)")

    print(f"    (* = safety-critical class)")

    df.to_parquet(OUTPUT)

    df.to_parquet(run_dir / "frames_final.parquet")

    print(f"\n  ✅ Saved → {OUTPUT}")

    if cfg.log_json:

        json_path = save_summary_json(

            run_dir, dedup_stats, balance_stats, df, INSTAMIND_CLASSES, cfg

        )

        print(f"  Summary → {json_path}")

    cfg.save(run_dir / "pipeline_config.json")

    cfg.save(MANIFESTS / "pipeline_config.json")

    print(f"  ⚙  Config  → {run_dir / 'pipeline_config.json'}")

    print(f"  📁 Run dir → {run_dir}")


if __name__ == "__main__":

    main()
