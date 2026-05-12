"""
Phase 1: Audit raw data before running the pipeline.

Checks:
 - UCF-Crime class directories exist and contain openable mp4s
 - UCA JSON annotation files reference videos that exist on disk
 - URFD fall/adl directories exist and contain videos
 - Reports split-screen format awareness for URFD videos
"""


from pathlib import Path

import cv2

import json

import sys


sys.path.insert(0, str(Path(__file__).resolve().parent))


PROJECT_ROOT = Path(__file__).resolve().parents[3]

UCF_ROOT = PROJECT_ROOT / "data" / "raw" / "ucf_crimes"

UCA_ROOT = PROJECT_ROOT / "data" / "raw" / "uca_annotations"

URFD_ROOT = UCF_ROOT / "Fall"


YOUR_CLASSES = {

    "Fighting":      "fighting",

    "Robbery":       "robbery",

    "Shooting":      "shooting",

    "Shoplifting":   "shoplifting",

    "Normal_Videos": "normal",

}


def verify_videos() -> list[str]:

    issues = []

    total = 0

    print("── Video audit ──")

    for ucf_name, instamind_name in YOUR_CLASSES.items():

        cls_dir = UCF_ROOT / ucf_name

        if not cls_dir.exists():

            issues.append(f"Missing class dir: {ucf_name}")

            continue

        videos = list(cls_dir.glob("*.mp4"))

        if not videos:

            issues.append(f"No mp4s in: {ucf_name}")

            continue

        bad = []

        for v in videos[:3]:

            cap = cv2.VideoCapture(str(v))

            ok = cap.isOpened()

            cap.release()

            if not ok:

                bad.append(v.name)

        if bad:

            issues.append(f"Unreadable videos in {ucf_name}: {bad}")

        print(f"  {ucf_name}: {len(videos)} videos OK")

        total += len(videos)

    print(f"  Total UCF videos on disk: {total}")

    return issues


def verify_uca_annotations() -> list[str]:

    issues = []

    print("── Annotation audit (filtered to your videos) ──")

    on_disk = set()

    for ucf_name in YOUR_CLASSES:

        cls_dir = UCF_ROOT / ucf_name

        if cls_dir.exists():

            for v in cls_dir.glob("*.mp4"):

                on_disk.add(v.stem)

    for split_file in ["UCFCrime_Train.json", "UCFCrime_Val.json", "UCFCrime_Test.json"]:

        json_path = UCA_ROOT / "json" / split_file

        if not json_path.exists():

            issues.append(f"Missing annotation file: {split_file}")

            continue

        with open(json_path) as f:

            data = json.load(f)

        matched = {k: v for k, v in data.items() if k in on_disk}

        total_windows = sum(len(v.get("timestamps", []))
                            for v in matched.values())

        print(f"  {split_file}:")

        print(f"    Total entries in JSON:      {len(data)}")

        print(f"    Entries matching your disk: {len(matched)}")

        print(f"    Temporal windows available: {total_windows}")

        if len(matched) == 0:

            issues.append(
                f"{split_file}: 0 entries match — check video naming convention")

        for video_name, ann in list(matched.items())[:5]:

            if "timestamps" not in ann or "sentences" not in ann:

                issues.append(f"Malformed annotation: {video_name}")

            if len(ann["timestamps"]) != len(ann["sentences"]):

                issues.append(

                    f"Timestamp/sentence mismatch in {video_name}: "

                    f"{len(ann['timestamps'])} vs {len(ann['sentences'])}"

                )

    return issues


def verify_fall_urfd() -> list[str]:
    """
    Audit URFD fall/adl data.
    Also checks that videos are split-screen format by inspecting one frame.
    """

    issues = []

    print("── Fall / URFD audit ──")

    if not URFD_ROOT.exists():

        issues.append(f"URFD root missing: {URFD_ROOT}")

        print(f"  {URFD_ROOT} not found")

        return issues

    for folder, mapped_class in [("fall", "fainting"), ("adl", "normal")]:

        folder_dir = URFD_ROOT / folder

        if not folder_dir.exists():

            issues.append(f"Missing URFD subdir: {folder_dir}")

            print(f"  {folder}/ not found")

            continue

        videos = list(folder_dir.glob("*.mp4")) + \
            list(folder_dir.glob("*.avi"))

        if not videos:

            issues.append(f"No videos in {folder_dir}")

            print(f"  {folder}/: 0 videos ❌")

            continue

        cap = cv2.VideoCapture(str(videos[0]))

        split_screen_detected = False

        if cap.isOpened():

            ret, frame = cap.read()

            if ret:

                h, w = frame.shape[:2]

                left_quarter = frame[:, :w // 4, :]

                right_quarter = frame[:, 3 * w // 4:, :]

                left_brightness = left_quarter.mean()

                right_brightness = right_quarter.mean()

                split_screen_detected = (
                    right_brightness > left_brightness * 2)

        cap.release()

        format_note = (
            "⚠️  split-screen detected (left=black, right=color — will crop right half)"
            if split_screen_detected
            else "single-channel format (no crop needed)"
        )

        print(f"  {folder}/: {len(videos)} videos → {mapped_class}")

        print(f"    Format: {format_note}")

    return issues


def main():

    print(f"Project root: {PROJECT_ROOT}")

    print(f"UCF root:     {UCF_ROOT}  (exists={UCF_ROOT.exists()})")

    print(f"UCA root:     {UCA_ROOT}  (exists={UCA_ROOT.exists()})")

    print(f"URFD root:    {URFD_ROOT}  (exists={URFD_ROOT.exists()})")

    print()

    issues = []

    issues += verify_videos()

    print()

    issues += verify_uca_annotations()

    print()

    issues += verify_fall_urfd()

    print()

    if issues:

        print(f"{len(issues)} real issues:")

        for i in issues:

            print(f"  - {i}")

    else:

        print("Dataset ready for Phase 2 (parse.py)")

        print()

        print(
            "Note: Fall videos will have right-half crop applied during frame extraction.")

        print("      If you already ran frames.py, delete existing fall frames and re-run:")

        print("      rm -rf data/processed/frames/*/fainting/")

        print("      python frames.py")


if __name__ == "__main__":

    main()
