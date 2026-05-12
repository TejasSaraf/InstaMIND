

"""
Class balancing module — dynamic cap + video-diverse downsampling.

Strategy:
  - Dynamic cap: min(p75 * 2, max_class * cap_max_ratio, median * 3)
    Adapts to the actual distribution instead of a static multiplier.
  - Floor: augmentation-based oversampling for minority classes.
  - Video diversity: guaranteed ≥1 frame per video, long videos capped.
  - Val/test untouched — honest evaluation requires natural distribution.
"""



from __future__ import annotations

import numpy as np

import pandas as pd



from .config import PipelineConfig

from .augment import oversample_with_augmentation









VID_COL = "video_id"





def _vid_col(df: pd.DataFrame) -> str:

    """Return the best available video identity column."""

    return VID_COL if VID_COL in df.columns else "video_name"





def diverse_downsample(

    group: pd.DataFrame, target: int, rng: np.random.Generator,

    seed: int = 42,

) -> pd.DataFrame:

    """
    Downsample while preserving video diversity.

    Phase 1: Guarantee ≥1 frame per video (by video_id).
    Phase 2: Fill remaining budget with per-video cap (2× fair share)
             to prevent long videos from dominating.
    """

    vcol = _vid_col(group)

    videos = group[vcol].unique()



    if len(videos) >= target:



        chosen = rng.choice(videos, size=target, replace=False)

        idx = [group[group[vcol] == v].sample(1, random_state=seed).index[0]

               for v in chosen]

        return group.loc[idx]





    guaranteed = [

        group[group[vcol] == v].sample(1, random_state=seed).index[0]

        for v in videos

    ]





    remaining = target - len(guaranteed)

    pool = group.drop(guaranteed)



    if remaining > 0 and len(pool) > 0:

        fair_share = max(1, remaining // len(videos)) * 2

        capped_parts = []

        for v in videos:

            v_pool = pool[pool[vcol] == v]

            n_take = min(fair_share, len(v_pool))

            capped_parts.append(v_pool.sample(n=n_take, random_state=seed))

        capped_pool = pd.concat(capped_parts)



        n_extra = min(remaining, len(capped_pool))

        extra = capped_pool.sample(n=n_extra, random_state=seed)

        return pd.concat([group.loc[guaranteed], extra])



    return group.loc[guaranteed]





def compute_dynamic_cap(counts: pd.Series, cfg: PipelineConfig) -> tuple[int, int]:

    """
    Compute dynamic cap and floor from class distribution.

    Cap formula: min(p75 * 2, max * cap_max_ratio, median * 3)
    Floor formula: min(min_floor, median)  — never floor above median.

    Returns (cap, floor).
    """

    nonzero = counts[counts > 0]

    p75 = int(np.percentile(nonzero.values, cfg.cap_percentile))

    median_count = int(nonzero.median())

    max_count = int(nonzero.max())



    cap = int(min(p75 * 2, max_count * cfg.cap_max_ratio, median_count * 3))

    cap = max(cap, median_count)

    floor = min(cfg.min_floor, median_count)



    return cap, floor





def balance_training_v3(

    df: pd.DataFrame,

    classes: list[str],

    cfg: PipelineConfig,

) -> tuple[pd.DataFrame, dict]:

    """
    Dynamic cap + augmentation-based oversampling. Val/test untouched.

    Returns:
      - Balanced DataFrame (train balanced + val/test intact)
      - Stats dict for logging
    """

    vcol = _vid_col(df)

    train = df[df["split"] == "train"].copy()

    val_test = df[df["split"].isin(["val", "test"])]





    sort_cols = [vcol]

    if "timestamp_s" in train.columns:

        sort_cols.append("timestamp_s")

    elif "frame_idx" in train.columns:

        sort_cols.append("frame_idx")

    train = train.sort_values(sort_cols).reset_index(drop=True)



    counts = train["incident_type"].value_counts()

    nonzero = counts[counts > 0]



    if len(nonzero) == 0:

        raise ValueError("No training frames after dedup")



    cap, floor = compute_dynamic_cap(counts, cfg)



    stats = {

        "min": int(nonzero.min()),

        "median": int(nonzero.median()),

        "p75": int(np.percentile(nonzero.values, cfg.cap_percentile)),

        "max": int(nonzero.max()),

        "cap": cap,

        "floor": floor,

        "per_class": {},

    }



    print(f"    Class stats: min={stats['min']}, median={stats['median']}, "

          f"p75={stats['p75']}, max={stats['max']}")

    print(f"    Dynamic cap:  {cap} frames")

    print(f"    Floor:        {floor} frames (augmentation-based)")



    rng = np.random.default_rng(cfg.random_seed)

    parts = []

    total_augmented = 0



    for cls in classes:

        group = train[train["incident_type"] == cls]

        n = len(group)

        n_vids = group[vcol].nunique()



        if n == 0:

            print(f"    {cls:<15}: 0 frames — MISSING CLASS")

            stats["per_class"][cls] = {"before": 0, "after": 0, "action": "missing"}

            continue





        cls_floor = floor

        is_critical = cls in cfg.critical_classes

        if is_critical:

            cls_floor = max(floor, cfg.critical_class_min_floor)



        if n > cap:

            sampled = diverse_downsample(group, cap, rng, cfg.random_seed)

            action = "capped"

            print(f"    {cls:<15}: {n:5d} → {len(sampled):5d} (capped) "

                  f"[{n_vids} videos]")

        elif n < cls_floor:

            sampled, aug_count = oversample_with_augmentation(group, cls_floor, rng, cfg)

            total_augmented += aug_count

            action = "augmented"

            tag = " CRITICAL" if is_critical else ""

            print(f"    {cls:<15}: {n:5d} → {len(sampled):5d} (augmented{tag}, "

                  f"{aug_count} new files) [{n_vids} videos]")

        else:

            sampled = group

            action = "unchanged"

            print(f"    {cls:<15}: {n:5d}         (unchanged) [{n_vids} videos]")





        if is_critical and len(sampled) < cfg.critical_class_min_floor:

            print(f"    ⚠ WARNING: critical class '{cls}' has {len(sampled)} frames "

                  f"(target: {cfg.critical_class_min_floor})")



        stats["per_class"][cls] = {

            "before": n,

            "after": len(sampled),

            "action": action,

            "videos": n_vids,

            "is_critical": is_critical,

        }

        parts.append(sampled)



    stats["total_augmented_files"] = total_augmented

    balanced = pd.concat(parts, ignore_index=True)

    return pd.concat([balanced, val_test], ignore_index=True), stats
