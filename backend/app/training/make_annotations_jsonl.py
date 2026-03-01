import argparse
import csv
import json
from pathlib import Path
import math

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pose-root", required=True, help="Root containing PoseLift pose JSON files")
    p.add_argument("--label-map", required=True, help="CSV: filename,label")
    p.add_argument("--output", required=True, help="annotations.jsonl output")
    return p.parse_args()

def load_label_map(path: Path):
    m = {}
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            m[row["filename"].strip()] = row["label"].strip()
    return m

def reshape_keypoints(flat):
    # Converts [x1,y1,s1,x2,y2,s2,...] -> [[x,y,z], ...] where z uses score.
    out = []
    vals = [float(x) for x in flat]
    for i in range(0, len(vals), 3):
        x = vals[i]
        y = vals[i + 1] if i + 1 < len(vals) else 0.0
        s = vals[i + 2] if i + 2 < len(vals) else 0.0
        z = s
        out.append([x, y, z])
    # normalize to 17 joints
    if len(out) < 17:
        out += [[0.0, 0.0, 0.0]] * (17 - len(out))
    return out[:17]

def frame_motion(curr, prev):
    if prev is None:
        return 0.0
    total = 0.0
    n = min(len(curr), len(prev))
    for i in range(n):
        dx = curr[i][0] - prev[i][0]
        dy = curr[i][1] - prev[i][1]
        total += math.sqrt(dx * dx + dy * dy)
    return total / max(1, n)

def main():
    args = parse_args()
    pose_root = Path(args.pose_root)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    label_map = load_label_map(Path(args.label_map))

    files = sorted(pose_root.rglob("*_alphapose_tracked_person.json"))
    written = 0
    with out_path.open("w", encoding="utf-8") as out:
        for fp in files:
            name = fp.name
            if name not in label_map:
                continue

            data = json.loads(fp.read_text(encoding="utf-8"))
            # choose first person track with most frames
            best_track = None
            best_len = -1
            for person_id, frames in data.items():
                if isinstance(frames, dict) and len(frames) > best_len:
                    best_track = frames
                    best_len = len(frames)
            if not best_track:
                continue

            frame_items = sorted(best_track.items(), key=lambda kv: kv[0])
            frames_out = []
            prev_kp = None
            for _, frame_obj in frame_items:
                kp_flat = frame_obj.get("keypoints", [])
                kp = reshape_keypoints(kp_flat)
                motion = frame_motion(kp, prev_kp)
                prev_kp = kp
                frames_out.append({
                    "keypoints": kp,
                    "motion": float(motion),
                    "audio_distress": 0.0
                })

            if len(frames_out) < 32:
                continue

            rec = {
                "label": label_map[name],
                "frames": frames_out
            }
            out.write(json.dumps(rec) + "\n")
            written += 1

    print(f"Wrote {written} records -> {out_path}")

if __name__ == "__main__":
    main()
