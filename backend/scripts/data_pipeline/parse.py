

"""
Phase 2: Parse UCA JSON annotations + URFD fall data into a normalized parquet manifest.

Inputs:
  - data/raw/ucf_crimes/{class}/*.mp4         (verified in Phase 1)
  - data/raw/uca_annotations/json/UCFCrime_{Train,Val,Test}.json
  - data/raw/ucf_crimes/Fall/fall/*.mp4        → fainting class
  - data/raw/ucf_crimes/Fall/adl/*.mp4         → normal class (ADL)

Output:
  - data/processed/manifests/clips_raw.parquet

Each row = one temporal window from one video with columns:
  video_name, source, ucf_class, incident_type, video_path,
  video_duration, window_start_s, window_end_s, window_duration,
  description, original_split

Changes from v1:
  - URFD now found at UCF_ROOT/Fall (not separate urfd/ dir)
  - Hard negative cap per source class (200 windows max) prevents
    normal class from being dominated by pre-shoplifting scenes
  - Updated descriptions for fainting to be more detailed
"""


from lib.classes import UCF_TO_INSTAMIND, urfd_class_of
from lib.paths import UCF_ROOT, UCA_ROOT, URFD_ROOT, MANIFESTS
import json

import sys

import pandas as pd

import cv2

from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))


OUTPUT = MANIFESTS / "clips_raw.parquet"


MAX_HARD_NEG_PER_CLASS = 15


def get_video_duration(video_path: Path) -> float:
    """Return duration in seconds. Returns 0.0 if video can't be opened."""

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():

        return 0.0

    fps = cap.get(cv2.CAP_PROP_FPS)

    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    cap.release()

    if fps <= 0:

        return 0.0

    return frames / fps


def build_disk_index() -> dict[str, Path]:
    """
    Build {video_stem → full_path} for all UCF-Crime videos on disk.
    Excludes the Fall/ subdirectory — those are handled separately by parse_urfd().
    """

    index = {}

    for cls_dir in UCF_ROOT.iterdir():

        if not cls_dir.is_dir():

            continue

        if cls_dir.name == "Fall":

            continue

        for video in cls_dir.glob("*.mp4"):

            index[video.stem] = video

    return index


def parse_uca_split(

    split_file: str,

    disk_index: dict[str, Path],

) -> list[dict]:
    """
    Parse one UCA JSON split file into clip rows.
    Only includes videos that exist on disk AND map to a known instaMIND class.
    """

    json_path = UCA_ROOT / "json" / split_file

    if not json_path.exists():

        print(f"  ⚠️  {split_file} not found — skipping")

        return []

    with open(json_path) as f:

        data = json.load(f)

    rows = []

    skipped_missing = 0

    skipped_excluded = 0

    skipped_degenerate = 0

    for video_stem, ann in data.items():

        video_path = disk_index.get(video_stem)

        if video_path is None:

            skipped_missing += 1

            continue

        ucf_class = video_path.parent.name

        instamind_class = UCF_TO_INSTAMIND.get(ucf_class)

        if instamind_class is None:

            skipped_excluded += 1

            continue

        duration = ann.get("duration", 0.0)

        timestamps = ann.get("timestamps", [])

        sentences = ann.get("sentences", [])

        for (start, end), sentence in zip(timestamps, sentences):

            window_dur = end - start

            if window_dur < 0.5:

                skipped_degenerate += 1

                continue

            if window_dur > 120.0:

                end = start + 60.0

                window_dur = 60.0

            rows.append({

                "video_name":      video_stem,

                "source":          "ucf_uca",

                "ucf_class":       ucf_class,

                "incident_type":   instamind_class,

                "video_path":      str(video_path),

                "video_duration":  duration if duration > 0 else get_video_duration(video_path),

                "window_start_s":  round(start, 2),

                "window_end_s":    round(end, 2),

                "window_duration": round(window_dur, 2),

                "description":     sentence.strip(),

                "original_split":  split_file.replace("UCFCrime_", "").replace(".json", "").lower(),

            })

    split_label = split_file.replace("UCFCrime_", "").replace(".json", "")

    print(f"  {split_label}: {len(rows)} windows "

          f"({skipped_missing} missing video, "

          f"{skipped_excluded} excluded class, "

          f"{skipped_degenerate} degenerate window)")

    return rows


FALL_DESCRIPTIONS = [

    "Person has fallen to the ground and is motionless.",

    "Individual has collapsed and is lying on the floor.",

    "Fall event detected — person is on the ground and not moving.",

    "Subject has fallen and may require medical assistance.",

    "Person collapsed suddenly and is lying still on the ground.",

]


ADL_DESCRIPTIONS = [

    "Normal daily activity — no security incident present.",

    "Person performing routine activity, no incident detected.",

    "Normal behaviour observed — no security event.",

    "Individual conducting regular daily activity.",

    "No anomalous activity present in this sequence.",

]


def parse_urfd() -> list[dict]:
    """
    Parse fall/adl videos from URFD_ROOT (= UCF_ROOT / 'Fall').

    Structure:
      data/raw/ucf_crimes/Fall/fall/*.mp4  → fainting
      data/raw/ucf_crimes/Fall/adl/*.mp4   → normal

    Video formats:
      cam0/cam1: split-screen (640×240, left=depth, right=color)
                 → frames.py crops to right half automatically
      cam2:      full-frame RGB (1920×1080) — no crop needed
    Each video is pre-trimmed — the whole video = the event window.

    With 60+ fall videos, this now produces sufficient fainting samples
    to avoid extreme oversampling in the balancing phase.
    """

    if not URFD_ROOT.exists():

        print(f"   URFD root not found: {URFD_ROOT}")

        return []

    rows = []

    for folder, mapped_class, descriptions in [

        ("fall", "fainting", FALL_DESCRIPTIONS),

        ("adl",  "normal",     ADL_DESCRIPTIONS),

    ]:

        folder_dir = URFD_ROOT / folder

        if not folder_dir.exists():

            print(f"  ⚠️  {URFD_ROOT.name}/{folder} not found")

            continue

        videos = list(folder_dir.glob("*.mp4")) + \
            list(folder_dir.glob("*.avi"))

        for i, video in enumerate(videos):

            duration = get_video_duration(video)

            if duration < 0.5:

                continue

            description = descriptions[i % len(descriptions)]

            rows.append({

                "video_name":      video.stem,

                "source":          "urfd",

                "ucf_class":       f"urfd_{folder}",

                "incident_type":   mapped_class,

                "video_path":      str(video),

                "video_duration":  round(duration, 2),

                "window_start_s":  0.0,

                "window_end_s":    round(duration, 2),

                "window_duration": round(duration, 2),

                "description":     description,

                "original_split":  "unassigned",

            })

        print(f"  URFD/{folder}: {len(videos)} videos → {mapped_class}")

    return rows


PRE_INCIDENT_DESCRIPTIONS = [

    "Pre-incident scene — normal activity, no security event yet.",

    "Routine activity prior to the incident, scene is clear.",

    "Normal conditions observed before any security event.",

    "Pre-event footage showing standard patron and staff behavior.",

    "Scene is calm with regular movement, no signs of an incident.",

    "Prior to the event, all visible activity appears normal.",

    "Normal operations in the area before the incident occurs.",

    "Pre-incident monitoring shows no suspicious behavior.",

]


def extract_normal_windows_from_incident_videos(

    incident_rows: list[dict],

) -> list[dict]:
    """
    For each incident video, extract a 5-second pre-incident window labeled 'normal'.

    Why this matters:
    - The model needs to see the same camera/scene labeled 'normal' to learn
      that 'normal' = absence of incident, not a different location
    - Without this, model learns 'store background = shoplifting' instead of
      'person concealing item = shoplifting'

    v2 change: Cap hard negatives to MAX_HARD_NEG_PER_CLASS per source incident
    class to prevent the normal class from being dominated by pre-shoplifting
    scenes (which was causing the 'default to shoplifting' failure mode).
    """

    normal_rows = []

    seen = set()

    source_counts: dict[str, int] = {}

    pre_incident_idx = 0

    for row in incident_rows:

        if row["incident_type"] == "normal":

            continue

        ucf_class = row["ucf_class"]

        if source_counts.get(ucf_class, 0) >= MAX_HARD_NEG_PER_CLASS:

            continue

        pre_end = row["window_start_s"]

        pre_start = max(0.0, pre_end - 5.0)

        if pre_end - pre_start < 1.0:

            continue

        key = (row["video_name"], round(pre_start, 1))

        if key in seen:

            continue

        seen.add(key)

        source_counts[ucf_class] = source_counts.get(ucf_class, 0) + 1

        description = PRE_INCIDENT_DESCRIPTIONS[pre_incident_idx % len(
            PRE_INCIDENT_DESCRIPTIONS)]

        pre_incident_idx += 1

        normal_rows.append({

            **row,

            "incident_type":   "normal",

            "window_start_s":  round(pre_start, 2),

            "window_end_s":    round(pre_end, 2),

            "window_duration": round(pre_end - pre_start, 2),

            "description":     description,

            "original_split":  row["original_split"],

        })

    print(
        f"  Hard negatives (pre-incident normal windows): {len(normal_rows)}")

    print(f"  Hard neg per source class:")

    for cls, count in sorted(source_counts.items()):

        print(f"    {cls:<20}: {count}")

    return normal_rows


def main():

    print("── Phase 2: Parsing annotations ──")

    disk_index = build_disk_index()

    print(f"  UCF videos on disk (excl. Fall/): {len(disk_index)}")

    ucf_rows = []

    for split_file in ["UCFCrime_Train.json", "UCFCrime_Val.json", "UCFCrime_Test.json"]:

        ucf_rows.extend(parse_uca_split(split_file, disk_index))

    fall_rows = parse_urfd()

    normal_rows = extract_normal_windows_from_incident_videos(ucf_rows)

    all_rows = ucf_rows + fall_rows + normal_rows

    df = pd.DataFrame(all_rows)

    print(f"\n── Raw clip manifest ──")

    print(f"  Total windows:  {len(df)}")

    print(f"  Unique videos:  {df['video_name'].nunique()}")

    print(f"\n  Class distribution:")

    counts = df["incident_type"].value_counts()

    for cls, count in counts.items():

        print(f"    {cls:15s}: {count:4d} windows")

    df.to_parquet(OUTPUT)

    print(f"\n✅ Saved → {OUTPUT}")


if __name__ == "__main__":

    main()
