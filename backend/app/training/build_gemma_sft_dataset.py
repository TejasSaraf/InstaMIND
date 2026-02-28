import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Gemma SFT JSONL from pose windows.")
    parser.add_argument("--dataset", required=True, help="NPZ dataset path from prepare_poselift_windows.py")
    parser.add_argument("--output", required=True, help="Output JSONL path for instruction tuning")
    return parser.parse_args()


def _incident_from_label(label: str) -> tuple[str, str]:
    mapping = {
        "fainting": ("Person posture collapsed to near-horizontal rapidly.", "Dispatch nearby responder immediately."),
        "choking": ("Breathing/distress pattern and abrupt upper-body instability observed.", "Issue emergency alert and request human confirmation."),
        "violent_activity": ("High-variance motion and conflict-like pose transitions detected.", "Trigger high-priority security escalation."),
        "suspicious_activity": ("Sustained movement anomaly detected around restricted area.", "Notify guard and keep tracking."),
        "none": ("No severe event pattern found in this window.", "Continue monitoring."),
    }
    return mapping.get(label, mapping["none"])


def main() -> None:
    args = parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(args.dataset, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    labels = [str(x) for x in data["labels"].tolist()]

    lines = []
    for idx in range(len(X)):
        label = labels[int(y[idx])]
        sample = X[idx]

        horizontal_proxy = float(np.mean(sample[:, 0]))
        motion_proxy = float(np.mean(sample[:, -2]))
        audio_proxy = float(np.mean(sample[:, -1]))
        evidence, action = _incident_from_label(label)

        prompt = (
            "You are InstaMIND incident analyst."
            "Classify the event and return strict JSON list with one item: "
            "[{incident_type, confidence, timestamp_seconds, evidence, recommended_action}].\n"
            f"Features: horizontal_proxy={horizontal_proxy:.4f}, motion_proxy={motion_proxy:.4f}, "
            f"audio_distress={audio_proxy:.4f}"
        )
        target = [
            {
                "incident_type": label,
                "confidence": 0.9 if label != "none" else 0.8,
                "timestamp_seconds": 1.0,
                "evidence": evidence,
                "recommended_action": action,
            }
        ]
        lines.append({"prompt": prompt, "response": json.dumps(target)})

    with out.open("w", encoding="utf-8") as f:
        for row in lines:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(lines)} SFT samples to {out}")


if __name__ == "__main__":
    main()
