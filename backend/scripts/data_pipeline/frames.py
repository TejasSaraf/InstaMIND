

"""
Phase 4: Extract frames from temporal windows.

For UCF/UCA windows: sample N frames uniformly across [start_s, end_s].
For URFD fall/adl: sample N frames across full video (pre-trimmed).

KEY CHANGE v2:
  URFD fall videos come in two formats:
    cam0/cam1: split-screen (640×240) — left=depth, right=color
    cam2:      full-frame RGB (1920×1080) — no crop needed

  extract_color_half() crops to the right half BEFORE resize/crop
  for URFD cam0/cam1 only.  cam2 videos are passed through as-is.

Frame budget per window:
  UCF/UCA:
    < 3s:   3 frames
    3-8s:   4 frames
    8-20s:  6 frames
    > 20s:  8 frames

  URFD (fall/adl):
    All durations: ~2 frames/sec, minimum 6, maximum 10
    Rationale: fall videos are short (1-7s) and every frame is informative.
    With 60+ fall videos we can afford more frames without oversampling.

Output: data/processed/frames/{split}/{incident_type}/{prefix}_f{n}.jpg
Manifest: data/processed/manifests/frames_extracted.parquet

Resume support: skips frames that already exist on disk.
To re-extract fall frames after this update:
  rm -rf data/processed/frames/*/fainting/
  rm -rf data/processed/frames/*/normal/  (if adl was included)
  python frames.py
"""


from lib.paths import MANIFESTS, FRAMES_ROOT
import sys

import cv2

import pandas as pd

import numpy as np

from PIL import Image

from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))


INPUT = MANIFESTS / "clips_split.parquet"

OUTPUT = MANIFESTS / "frames_extracted.parquet"


TARGET_SIZE = 512

JPEG_QUALITY = 92

MIN_FRAMES = 3


def frames_per_window(duration: float, source: str = "ucf_uca") -> int:
    """
    How many frames to extract from a window of the given duration.

    URFD videos get a higher per-second budget because:
      1. They are short (1-7s) — standard budget gives only 3 frames
      2. Falls happen fast — temporal detail within the window matters
      3. With 60+ fall videos we no longer need to oversample as aggressively
         so extracting more frames naturally reduces the oversampling pressure
    """

    if source == "urfd":

        return int(max(6, min(10, duration * 2)))

    if duration < 3.0:
        return 3

    if duration < 8.0:
        return 4

    if duration < 20.0:
        return 6

    return 8


def extract_color_half(frame_bgr: np.ndarray) -> np.ndarray:
    """
    URFD split-screen handler (cam0/cam1 only).

    URFD cam0/cam1 videos have the format:
      [ GRAYSCALE DEPTH IMAGE | COLOR RGB IMAGE ]
      ← left half (depth) →    ← right half (color) →

    This function crops to the right half, discarding the depth channel.
    NOT called for cam2 videos which are full-frame RGB.
    """

    h, w = frame_bgr.shape[:2]

    right_half = frame_bgr[:, w // 2:, :]

    return right_half


def _is_cam2(video_name: str) -> bool:
    """Return True if the video is a URFD cam2 (full-frame, no crop needed)."""

    return "cam2" in video_name.lower()


def resize_center_crop(frame_bgr: np.ndarray, size: int) -> Image.Image:
    """
    Resize so the shorter side = size, then center-crop to size×size.
    Input is BGR numpy array, output is PIL RGB Image.
    """

    h, w = frame_bgr.shape[:2]

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    pil = Image.fromarray(frame_rgb)

    if w < h:

        new_w, new_h = size, int(h * size / w)

    else:

        new_w, new_h = int(w * size / h), size

    pil = pil.resize((new_w, new_h), Image.LANCZOS)

    left = (new_w - size) // 2

    top = (new_h - size) // 2

    return pil.crop((left, top, left + size, top + size))


def extract_clip_frames(row: dict) -> list[dict]:
    """
    Extract frames for a single clip row.
    Returns a list of frame-level dicts (one per saved frame).
    """

    video_path = Path(row["video_path"])

    if not video_path.exists():

        return []

    source = row.get("source", "ucf_uca")

    is_urfd = (source == "urfd")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():

        return []

    fps = cap.get(cv2.CAP_PROP_FPS)

    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    if fps <= 0 or total_f <= 0:

        return []

    start_f = int(row["window_start_s"] * fps)

    end_f = min(int(row["window_end_s"] * fps), total_f - 1)

    if (end_f - start_f + 1) < MIN_FRAMES:

        return []

    n = frames_per_window(row["window_duration"], source)

    frame_indices = np.linspace(start_f, end_f, n).astype(int)

    frame_indices = np.unique(frame_indices)

    out_dir = FRAMES_ROOT / row["split"] / row["incident_type"]

    out_dir.mkdir(parents=True, exist_ok=True)

    safe_name = row["video_name"].replace("/", "_").replace(" ", "_")

    prefix = f"{safe_name}_t{row['window_start_s']:.1f}-{row['window_end_s']:.1f}"

    cap = cv2.VideoCapture(str(video_path))

    results = []

    for i, fidx in enumerate(frame_indices):

        out_path = out_dir / f"{prefix}_f{i:02d}.jpg"

        ts = float(fidx) / fps

        if out_path.exists():

            results.append((str(out_path), int(fidx), ts))

            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fidx))

        ret, frame = cap.read()

        if not ret:

            continue

        try:

            if is_urfd and not _is_cam2(row.get("video_name", "")):

                frame = extract_color_half(frame)

            pil = resize_center_crop(frame, TARGET_SIZE)

            pil.save(str(out_path), "JPEG", quality=JPEG_QUALITY)

            results.append((str(out_path), int(fidx), ts))

        except Exception:

            continue

    cap.release()

    vid_id = f"{source}/{row['video_name']}"

    return [

        {

            "frame_path":     fp,

            "frame_idx":      fidx,

            "timestamp_s":    ts,

            "video_id":       vid_id,

            "video_name":     row["video_name"],

            "source":         source,

            "incident_type":  row["incident_type"],

            "split":          row["split"],

            "window_start_s": row["window_start_s"],

            "window_end_s":   row["window_end_s"],

            "description":    row["description"],

        }

        for fp, fidx, ts in results

    ]


def main():

    print("── Phase 4: Frame extraction ──")

    print("  URFD cam0/cam1: right-half color crop ENABLED")

    print("  URFD cam2:      full-frame (no crop)")

    print("  URFD frame budget: ~2 frames/sec (min 6, max 10)")

    print()

    df = pd.read_parquet(INPUT)

    df = df[df["split"].isin(["train", "val", "test"])].copy()

    ucf_clips = (df["source"] == "ucf_uca").sum()

    urfd_clips = (df["source"] == "urfd").sum()

    print(
        f"  Input clips: {len(df)} total  ({ucf_clips} UCF/UCA + {urfd_clips} URFD)")

    print(f"  Target resolution: {TARGET_SIZE}×{TARGET_SIZE}")

    print()

    rows_as_dicts = df.to_dict("records")

    all_frame_rows = []

    done = 0

    urfd_frame_count = 0

    for row in rows_as_dicts:

        frames = extract_clip_frames(row)

        all_frame_rows.extend(frames)

        if row.get("source") == "urfd":

            urfd_frame_count += len(frames)

        done += 1

        if done % 50 == 0 or done == len(rows_as_dicts):

            print(f"  {done}/{len(rows_as_dicts)} clips — "

                  f"{len(all_frame_rows)} frames "

                  f"({urfd_frame_count} URFD)")

    if not all_frame_rows:

        print("❌ No frames extracted — check video paths and splits")

        return

    frames_df = pd.DataFrame(all_frame_rows)

    frames_df.to_parquet(OUTPUT)

    print(f"\n── Frame extraction complete ──")

    print(f"  Total frames: {len(frames_df)}")

    print(f"\n  Per-split, per-class breakdown:")

    pt = frames_df.groupby(["split", "incident_type"]
                           ).size().unstack(fill_value=0)

    print(pt.to_string())

    fall_train = frames_df[

        (frames_df["incident_type"] == "fainting") &

        (frames_df["split"] == "train")

    ]

    print(f"\n  fainting training frames: {len(fall_train)} "

          f"from {fall_train['video_name'].nunique()} unique videos")

    print(f"\n✅ Saved → {OUTPUT}")

    print()

    print("  NOTE: URFD frames now contain only the color (right) half.")

    print("  Existing fall frames extracted before this update need re-extraction:")

    print("    rm -rf data/processed/frames/*/fainting/")

    print("    python frames.py  (resume support skips non-fall classes)")


if __name__ == "__main__":

    main()
