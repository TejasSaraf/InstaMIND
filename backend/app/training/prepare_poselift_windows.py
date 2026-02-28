import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build PoseLift training windows.")
    parser.add_argument("--input", required=True, help="Path to PoseLift JSONL input.")
    parser.add_argument("--output", required=True, help="Output .npz dataset path.")
    parser.add_argument("--window", type=int, default=32, help="Frames per sample window.")
    parser.add_argument("--stride", type=int, default=8, help="Stride for sliding windows.")
    return parser.parse_args()


def _to_feature_frame(frame: dict) -> list[float]:
    keypoints = frame.get("keypoints", [])
    flat = []
    for kp in keypoints[:17]:
        if isinstance(kp, list) and len(kp) >= 3:
            flat.extend([float(kp[0]), float(kp[1]), float(kp[2])])
        else:
            flat.extend([0.0, 0.0, 0.0])
    while len(flat) < 17 * 3:
        flat.extend([0.0, 0.0, 0.0])

    motion = float(frame.get("motion", 0.0))
    audio_distress = float(frame.get("audio_distress", 0.0))
    return flat + [motion, audio_distress]


def main() -> None:
    args = parse_args()
    inp = Path(args.input)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    label_to_idx: dict[str, int] = {}
    X = []
    y = []

    for line in inp.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        label = str(row.get("label", "none"))
        frames = row.get("frames", [])
        if len(frames) < args.window:
            continue

        if label not in label_to_idx:
            label_to_idx[label] = len(label_to_idx)

        feat_frames = [_to_feature_frame(f) for f in frames]
        for start in range(0, len(feat_frames) - args.window + 1, args.stride):
            window = feat_frames[start : start + args.window]
            X.append(window)
            y.append(label_to_idx[label])

    X_np = np.array(X, dtype=np.float32)
    y_np = np.array(y, dtype=np.int32)
    labels = [k for k, _ in sorted(label_to_idx.items(), key=lambda kv: kv[1])]

    np.savez_compressed(out, X=X_np, y=y_np, labels=np.array(labels))
    print(f"Saved {len(X_np)} windows to {out}")
    print(f"Labels: {labels}")


if __name__ == "__main__":
    main()
