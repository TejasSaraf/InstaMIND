

"""
Output validation and logging module.

Assertions:
  - No video leakage across splits (re-verified post-pipeline)
  - All classes present in training set
  - Adaptive minimum sample threshold per class
  - Val/test non-empty

Logging:
  - Detailed per-stage stats to stdout
  - Optional JSON summary artifact for reproducibility tracking
"""



from __future__ import annotations

import json

import numpy as np

import pandas as pd

from pathlib import Path



from .config import PipelineConfig





def _vid_col(df: pd.DataFrame) -> str:

    """Return the best available video identity column."""

    return "video_id" if "video_id" in df.columns else "video_name"





def verify_no_leakage(df: pd.DataFrame) -> dict:

    """
    Hard check: no video may appear in multiple splits.
    Uses video_id (globally unique) instead of video_name.
    Returns split→video_count dict on success.
    Raises ValueError on leakage.
    """

    vcol = _vid_col(df)

    video_splits = df.groupby(vcol)["split"].nunique()

    leaked = video_splits[video_splits > 1]

    if len(leaked) > 0:

        raise ValueError(

            f"SPLIT LEAKAGE — {len(leaked)} videos in multiple splits "

            f"(identity column: {vcol}):\n"

            + "\n".join(f"  {v}" for v in leaked.index[:10])

        )



    splits = {s: set(df[df["split"] == s][vcol]) for s in ["train", "val", "test"]}

    assert not (splits["train"] & splits["val"]),  "train ∩ val overlap"

    assert not (splits["train"] & splits["test"]), "train ∩ test overlap"

    assert not (splits["val"]   & splits["test"]), "val ∩ test overlap"



    return {s: len(v) for s, v in splits.items()}





def check_identity_collisions(

    df: pd.DataFrame, strict: bool = False

) -> dict:

    """
    Scan for video_name values that map to more than one video_id.
    This indicates the same human-readable name is reused across sources.

    Returns dict of {video_name: [video_id1, video_id2, ...]} for collisions.
    Raises ValueError if strict=True and collisions found.
    """

    if "video_id" not in df.columns:

        return {}



    name_to_ids = df.groupby("video_name")["video_id"].apply(

        lambda x: sorted(x.unique().tolist())

    )

    collisions = {name: ids for name, ids in name_to_ids.items() if len(ids) > 1}



    if collisions and strict:

        raise ValueError(

            f"IDENTITY COLLISION — {len(collisions)} video_name values map to "

            f"multiple video_ids (strict mode).\n"

            + "\n".join(f"  {n}: {ids}" for n, ids in list(collisions.items())[:10])

        )



    return collisions





def validate_output(

    df: pd.DataFrame,

    classes: list[str],

    cfg: PipelineConfig,

) -> None:

    """
    Post-pipeline assertions. Fail loudly if data is broken.
    Uses adaptive minimum threshold: min(50, 0.1 * median_class_size).
    """



    split_counts = verify_no_leakage(df)

    print(f"    ✓ No leakage (re-verified): {split_counts}")



    train = df[df["split"] == "train"]





    for cls in classes:

        n = (train["incident_type"] == cls).sum()

        assert n > 0, f"Training class '{cls}' has 0 frames after pipeline"





    class_sizes = train.groupby("incident_type").size()

    median_size = int(class_sizes.median())

    min_per_class = int(class_sizes.min())



    if cfg.adaptive_min_enabled:

        required = int(min(cfg.min_samples_per_class, 0.1 * median_size))

        required = max(required, 10)

    else:

        required = cfg.min_samples_per_class



    assert min_per_class >= required, (

        f"Minimum training class has {min_per_class} frames — "

        f"need at least {required} (adaptive threshold)"

    )





    for s in ["val", "test"]:

        n = (df["split"] == s).sum()

        assert n > 0, f"Split '{s}' has 0 frames"





    for cls in cfg.critical_classes:

        cls_n = (train["incident_type"] == cls).sum()

        if cls_n < cfg.critical_class_min_floor:

            print(f"    ⚠ Critical class '{cls}': {cls_n} frames "

                  f"(target: {cfg.critical_class_min_floor})")



    print(f"    ✓ All {len(classes)} classes present (min={min_per_class}, required≥{required})")

    print(f"    ✓ Val/test splits intact")





def save_summary_json(

    output_dir: Path,

    dedup_stats: dict,

    balance_stats: dict,

    final_df: pd.DataFrame,

    classes: list[str],

    cfg: PipelineConfig,

) -> Path:

    """
    Save pipeline summary as JSON artifact for reproducibility.
    """

    vcol = _vid_col(final_df)

    train = final_df[final_df["split"] == "train"]





    class_counts = train.groupby("incident_type").size()

    total_train = class_counts.sum()

    n_classes = len(class_counts)

    class_weights = {}

    for cls in classes:

        n = class_counts.get(cls, 1)

        w = round(total_train / (n_classes * n), 4)

        if cls in cfg.critical_classes:

            w = round(w * cfg.critical_class_weight_boost, 4)

        class_weights[cls] = w



    final_per_class = {}

    for cls in classes:

        cls_df = train[train["incident_type"] == cls]

        final_per_class[cls] = {

            "frames": len(cls_df),

            "videos": int(cls_df[vcol].nunique()),

        }



    raw = dedup_stats.get("raw_frames", dedup_stats.get("before", 0))

    after_corrupt = dedup_stats.get("after_corrupt", raw - dedup_stats.get("corrupt_removed", 0))

    after_dedup = dedup_stats.get("after_dedup", dedup_stats.get("after", 0))

    dups_removed = dedup_stats.get("duplicates_removed", dedup_stats.get("removed", 0))

    total_removed = raw - after_dedup



    summary = {

        "config": cfg.to_dict(),

        "config_fingerprint": cfg.fingerprint(),

        "dedup": {

            "raw_frames": raw,

            "corrupt_removed": dedup_stats.get("corrupt_removed", 0),

            "after_corrupt": after_corrupt,

            "after_dedup": after_dedup,

            "duplicates_removed": dups_removed,

            "total_removed": total_removed,

            "pct_removed": round(

                total_removed / max(raw, 1) * 100, 2

            ),

        },

        "balance": {

            "cap": balance_stats.get("cap", 0),

            "floor": balance_stats.get("floor", 0),

            "total_augmented_files": balance_stats.get("total_augmented_files", 0),

            "per_class": balance_stats.get("per_class", {}),

        },

        "final": {

            "train_frames": int((final_df["split"] == "train").sum()),

            "val_frames": int((final_df["split"] == "val").sum()),

            "test_frames": int((final_df["split"] == "test").sum()),

            "train_videos": int(train[vcol].nunique()),

            "per_class": final_per_class,

            "identity_column": vcol,

        },

        "class_weights": class_weights,

        "identity": {

            "column": vcol,

            "collisions": check_identity_collisions(final_df),

        },

    }



    path = output_dir / "pipeline_summary.json"

    path.write_text(json.dumps(summary, indent=2))

    return path
