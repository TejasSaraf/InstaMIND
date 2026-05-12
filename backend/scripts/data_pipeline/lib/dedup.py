

"""
Deduplication module — two-stage perceptual dedup with split-awareness.

Stage 1 (fast): dhash hamming distance against sliding window of N kept hashes.
Stage 2 (fine): SSIM via skimage against sliding window (train only).
Temporal gap:   Enforce minimum seconds between kept frames.

Val/test: exact-duplicate removal only (hamming=0) + corrupt frame removal.
          No SSIM filtering — preserves real-world eval distribution.
"""



from __future__ import annotations

import numpy as np

import pandas as pd

import cv2

from PIL import Image

from pathlib import Path

from collections import Counter







try:

    from skimage.metrics import structural_similarity as _skimage_ssim

    _HAS_SKIMAGE = True

except ImportError:

    _HAS_SKIMAGE = False

    import warnings

    warnings.warn(

        "scikit-image not installed — falling back to cv2-based SSIM approximation. "

        "Install with: pip install scikit-image",

        stacklevel=2,

    )



from .config import PipelineConfig









def compute_dhash(image_path: str, hash_size: int = 16) -> np.ndarray | None:

    """
    Difference hash — captures horizontal gradient between adjacent pixels.
    Returns flat boolean array of hash_size² bits, or None if unreadable.
    O(1) per frame.
    """

    try:

        img = Image.open(image_path).convert("L")

        img = img.resize((hash_size + 1, hash_size), Image.LANCZOS)

        px = np.array(img, dtype=np.int16)

        return (px[:, 1:] > px[:, :-1]).flatten()

    except Exception:

        return None





def hamming_distance(h1: np.ndarray, h2: np.ndarray) -> int:

    return int(np.sum(h1 != h2))









def compute_structural_similarity(

    path_a: str, path_b: str, size: int = 64

) -> float:

    """
    Structural Similarity Index between two frames.

    Uses skimage.metrics.structural_similarity (validated, peer-reviewed)
    with fallback to manual SSIM formula if skimage is not installed.

    Both paths are loaded, downscaled to `size×size` grayscale for speed.
    Returns SSIM in [0, 1]. Higher = more similar.
    """

    try:

        a = cv2.imread(path_a, cv2.IMREAD_GRAYSCALE)

        b = cv2.imread(path_b, cv2.IMREAD_GRAYSCALE)

        if a is None or b is None:

            return 0.0

        a = cv2.resize(a, (size, size))

        b = cv2.resize(b, (size, size))



        if _HAS_SKIMAGE:

            return float(_skimage_ssim(a, b, data_range=255))

        else:



            a = a.astype(np.float64)

            b = b.astype(np.float64)

            C1 = (0.01 * 255) ** 2

            C2 = (0.03 * 255) ** 2

            mu_a, mu_b = a.mean(), b.mean()

            sig_a_sq, sig_b_sq = a.var(), b.var()

            sig_ab = ((a - mu_a) * (b - mu_b)).mean()

            num = (2 * mu_a * mu_b + C1) * (2 * sig_ab + C2)

            den = (mu_a ** 2 + mu_b ** 2 + C1) * (sig_a_sq + sig_b_sq + C2)

            return float(num / den)

    except Exception:

        return 0.0









def find_corrupt_frames(

    df: pd.DataFrame, log_path: Path | None = None

) -> set[str]:

    """
    Identify frames that cannot be read by PIL.
    Returns set of frame_path strings that should be removed.
    Optionally logs corrupt paths + per-class counts to a text file.
    """

    corrupt = set()

    corrupt_classes: Counter[str] = Counter()



    for _, row in df.iterrows():

        fp = row["frame_path"]

        try:

            img = Image.open(fp)

            img.verify()

        except Exception:

            corrupt.add(fp)

            corrupt_classes[row.get("incident_type", "unknown")] += 1



    if corrupt and log_path is not None:

        lines = [f"# Corrupt frames detected: {len(corrupt)}"]

        lines.append(f"# Per-class: {dict(corrupt_classes)}")

        lines.extend(sorted(corrupt))

        log_path.write_text("\n".join(lines))



    return corrupt









def deduplicate_clip_train(

    frame_paths: list[str],

    timestamps: list[float],

    cfg: PipelineConfig,

) -> list[str]:

    """
    Full two-stage dedup for training clips.

    1. Temporal gap: skip frames < min_temporal_gap_s from last kept
    2. dhash sliding window: skip if hamming ≤ threshold against ANY recent hash
    3. SSIM fine-filter: skip if SSIM > threshold against ANY recent kept frame

    This catches:
      - Consecutive near-duplicates (temporal gap)
      - Non-consecutive visual duplicates (sliding window dhash)
      - Subtle duplicates with similar gradients (SSIM)
    """

    if len(frame_paths) <= 1:

        return frame_paths





    entries = []

    for fp, ts in zip(frame_paths, timestamps):

        h = compute_dhash(fp, cfg.dhash_size)

        if h is not None:

            entries.append((fp, h, ts))



    if not entries:

        return []





    kept_paths = [entries[0][0]]

    kept_hashes = [entries[0][1]]

    last_ts = entries[0][2]



    for fp, h, ts in entries[1:]:



        if (ts - last_ts) < cfg.min_temporal_gap_s:

            continue





        window_hashes = kept_hashes[-cfg.sliding_window_size:]

        if any(hamming_distance(h, kh) <= cfg.hamming_thresh_train

               for kh in window_hashes):

            continue





        window_paths = kept_paths[-cfg.sliding_window_size:]

        if any(compute_structural_similarity(fp, kp) > cfg.ssim_thresh

               for kp in window_paths):

            continue



        kept_paths.append(fp)

        kept_hashes.append(h)

        last_ts = ts



    return kept_paths





def deduplicate_clip_valtest(

    frame_paths: list[str],

    cfg: PipelineConfig,

) -> list[str]:

    """
    Minimal dedup for val/test: remove ONLY exact hash duplicates.
    No SSIM, no temporal gap — preserve real-world distribution for honest eval.
    """

    if len(frame_paths) <= 1:

        return frame_paths



    seen_hashes: list[np.ndarray] = []

    kept = []



    for fp in frame_paths:

        h = compute_dhash(fp, cfg.dhash_size)

        if h is None:

            continue





        if cfg.hamming_thresh_valtest == 0:

            is_exact_dup = any(hamming_distance(h, sh) == 0 for sh in seen_hashes)

        else:

            is_exact_dup = any(hamming_distance(h, sh) <= cfg.hamming_thresh_valtest

                               for sh in seen_hashes)



        if not is_exact_dup:

            kept.append(fp)

            seen_hashes.append(h)



    return kept









def enhanced_deduplicate(

    df: pd.DataFrame, cfg: PipelineConfig,

    manifests_dir: Path | None = None,

) -> tuple[pd.DataFrame, dict]:

    """
    Split-aware deduplication across entire dataset.

    Guarantees:
      - Deterministic ordering before any processing
      - Explicit temporal sort within each clip before dedup
      - Granular stats for transparent reporting

    Returns:
      - Deduplicated DataFrame
      - Stats dict with raw_frames, after_corrupt, after_dedup, duplicates_removed
    """

    raw_total = len(df)

    has_timestamps = "timestamp_s" in df.columns

    has_frame_idx = "frame_idx" in df.columns





    if "video_id" not in df.columns:

        df = df.copy()

        df["video_id"] = df["source"].str.cat(df["video_name"], sep="/")





    group_cols = ["video_id", "window_start_s", "window_end_s"]





    sort_cols = ["video_id", "window_start_s"]

    if has_timestamps:

        sort_cols.append("timestamp_s")

    elif has_frame_idx:

        sort_cols.append("frame_idx")

    df = df.sort_values(sort_cols).reset_index(drop=True)





    print("    Scanning for corrupt frames...")

    log_path = manifests_dir / "corrupt_frames.txt" if manifests_dir else None

    corrupt = find_corrupt_frames(df, log_path=log_path)

    if corrupt:

        print(f"    ⚠ Found {len(corrupt)} corrupt frames — removing")

        if log_path:

            print(f"    ⚠ Logged to {log_path}")

        df = df[~df["frame_path"].isin(corrupt)].copy()

    after_corrupt_total = len(df)



    total = df.groupby(group_cols).ngroups

    kept_paths = set()

    processed = 0



    for (vid, ws, we), grp in df.groupby(group_cols):

        split = grp["split"].iloc[0]





        if has_timestamps:

            grp = grp.sort_values("timestamp_s")

        elif has_frame_idx:

            grp = grp.sort_values("frame_idx")





        clip_paths = grp["frame_path"].tolist()



        if split == "train":

            if has_timestamps:

                timestamps = grp["timestamp_s"].tolist()

            else:

                n = len(clip_paths)

                timestamps = list(np.linspace(ws, we, n))



            kept = deduplicate_clip_train(clip_paths, timestamps, cfg)

        else:

            kept = deduplicate_clip_valtest(clip_paths, cfg)



        kept_paths.update(kept)

        processed += 1



        if processed % 500 == 0 or processed == total:

            print(f"    {processed}/{total} clips — {len(kept_paths)} frames kept")



    result = df[df["frame_path"].isin(kept_paths)].copy()





    stats = {

        "raw_frames": raw_total,

        "corrupt_removed": len(corrupt),

        "after_corrupt": after_corrupt_total,

        "after_dedup": len(result),

        "duplicates_removed": after_corrupt_total - len(result),



        "before": after_corrupt_total,

        "after": len(result),

        "removed": after_corrupt_total - len(result),

    }



    return result, stats
