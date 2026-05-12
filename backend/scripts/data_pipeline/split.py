"""
Phase 3: Assign train/val/test splits at VIDEO level.

Critical: All windows from the same video go to the same split.
This prevents the model from memorizing camera angles.

Ignores UCA's original_split — we re-split to get correct ratios
for our specific 249 videos and 6 classes.

Stratified by class: each split gets proportional representation
of every instaMIND class.

Input:  manifests/clips_raw.parquet
Output: manifests/clips_split.parquet
        manifests/train_videos.txt
        manifests/val_videos.txt
        manifests/test_videos.txt
"""


from lib.classes import INSTAMIND_CLASSES
from lib.paths import MANIFESTS
import sys

import pandas as pd

import numpy as np

from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))


INPUT = MANIFESTS / "clips_raw.parquet"

OUTPUT = MANIFESTS / "clips_split.parquet"


TRAIN_RATIO = 0.70

VAL_RATIO = 0.15

TEST_RATIO = 0.15

RANDOM_SEED = 42


def stratified_video_split(df: pd.DataFrame) -> dict[str, str]:
    """
    Returns dict of {video_name → split_name}.
    Stratified so each class is proportionally represented.
    """

    video_class = (

        df.groupby("video_name")["incident_type"]

        .agg(lambda x: x.value_counts().index[0])

        .reset_index()

        .rename(columns={"incident_type": "dominant_class"})

    )

    rng = np.random.default_rng(RANDOM_SEED)

    assignments = {}

    print(f"\n  {'Class':<15} {'Total':>6} {'Train':>6} {'Val':>5} {'Test':>5}")

    print("  " + "─" * 44)

    for cls in INSTAMIND_CLASSES:

        videos = video_class[

            video_class["dominant_class"] == cls

        ]["video_name"].tolist()

        if not videos:

            print(f"  {cls:<15} {'0':>6}  (no videos)")

            continue

        rng.shuffle(videos)

        n = len(videos)

        n_train = max(1, int(n * TRAIN_RATIO))

        n_val = max(1, int(n * VAL_RATIO))

        n_test = n - n_train - n_val

        if n < 3:

            n_train, n_val, n_test = n, 0, 0

        elif n_test < 1:

            n_train -= 1

            n_test = 1

        for v in videos[:n_train]:

            assignments[v] = "train"

        for v in videos[n_train:n_train + n_val]:

            assignments[v] = "val"

        for v in videos[n_train + n_val:]:

            assignments[v] = "test"

        print(f"  {cls:<15} {n:>6} {n_train:>6} {n_val:>5} {n_test:>5}")

    return assignments


def main():

    print("── Phase 3: Video-level splits ──")

    df = pd.read_parquet(INPUT)

    print(
        f"  Input: {len(df)} windows across {df['video_name'].nunique()} videos")

    assignments = stratified_video_split(df)

    df["split"] = df["video_name"].map(assignments).fillna("unassigned")

    unassigned = df[df["split"] == "unassigned"]["video_name"].unique()

    if len(unassigned) > 0:

        raise ValueError(

            f"UNASSIGNED VIDEOS: {len(unassigned)} videos have no split. "

            f"First 5: {list(unassigned[:5])}"

        )

    train_v = set(df[df["split"] == "train"]["video_name"])

    val_v = set(df[df["split"] == "val"]["video_name"])

    test_v = set(df[df["split"] == "test"]["video_name"])

    overlaps = []

    if train_v & val_v:

        overlaps.append(f"train∩val = {len(train_v & val_v)} videos")

    if train_v & test_v:

        overlaps.append(f"train∩test = {len(train_v & test_v)} videos")

    if val_v & test_v:

        overlaps.append(f"val∩test = {len(val_v & test_v)} videos")

    if overlaps:

        raise ValueError(f"SPLIT CONTAMINATION: {overlaps}")

    splits_per_video = df.groupby("video_name")["split"].nunique()

    multi_split = splits_per_video[splits_per_video > 1]

    if len(multi_split) > 0:

        raise ValueError(

            f"INTRA-VIDEO LEAKAGE: {len(multi_split)} videos have windows in "

            f"multiple splits: {list(multi_split.index[:5])}"

        )

    print(
        f"\n  ✓ No video overlap across splits ({len(train_v)}+{len(val_v)}+{len(test_v)} videos)")

    (MANIFESTS / "train_videos.txt").write_text("\n".join(sorted(train_v)))

    (MANIFESTS / "val_videos.txt").write_text("\n".join(sorted(val_v)))

    (MANIFESTS / "test_videos.txt").write_text("\n".join(sorted(test_v)))

    print(f"\n  Split window distribution:")

    print(f"  {'Split':<8} {'Windows':>8} {'Videos':>8}")

    print("  " + "─" * 28)

    for split in ["train", "val", "test"]:

        sdf = df[df["split"] == split]

        print(f"  {split:<8} {len(sdf):>8} {sdf['video_name'].nunique():>8}")

    df.to_parquet(OUTPUT)

    print(f"\n✅ Saved → {OUTPUT}")


if __name__ == "__main__":

    main()
