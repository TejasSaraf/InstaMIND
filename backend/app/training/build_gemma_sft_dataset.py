"""
Build Gemma SFT JSONL for LoRA fine-tuning.
Prompt/summary format matches gemma_agent._gemini_primary_classify and _build_multimodal_summary
so the model learns: shoplifting activity -> shoplifting, fainting (collapse) -> fainting, else -> none.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np

# Match inference prompt exactly (gemma_agent._gemini_primary_classify)
CLASSIFIER_RULES = (
    "You are the primary security incident classifier. Use ONLY the multimodal summary below.\n\n"
    "RULES:\n"
    "- Choose the SINGLE most likely incident type. Do NOT default to fainting.\n"
    "- Fainting = person collapsed (horizontal body, very low movement, medical emergency).\n"
    "- Shoplifting = concealment, item handling, retail context, person leaving without paying.\n"
    "- Use 'suspicious_activity' or 'none' for unclear or normal activity.\n"
    "- Return ONLY a JSON array with exactly one object: {\"incident_type\", \"confidence\", \"timestamp_seconds\", \"evidence\", \"recommended_action\"}.\n"
    "Allowed incident_type: fainting, choking, violent_activity, shoplifting, suspicious_activity, intrusion, none.\n\n"
)

# Evidence/action text per label (for response target)
def _incident_from_label(label: str) -> tuple[str, str]:
    mapping = {
        "fainting": (
            "Person posture collapsed to near-horizontal; very low movement; possible medical emergency.",
            "Dispatch nearby responder immediately.",
        ),
        "choking": (
            "Breathing/distress pattern and abrupt upper-body instability observed.",
            "Issue emergency alert and request human confirmation.",
        ),
        "violent_activity": (
            "High-variance motion and conflict-like pose transitions detected.",
            "Trigger high-priority security escalation.",
        ),
        "shoplifting": (
            "Object interaction followed by concealment and exit-like movement pattern detected.",
            "Notify security team and retain timestamped evidence clip.",
        ),
        "suspicious_activity": (
            "Sustained movement anomaly detected around restricted area.",
            "Notify guard and keep tracking.",
        ),
        "intrusion": (
            "Unauthorized entry or perimeter breach detected.",
            "Notify security and verify identity.",
        ),
        "none": (
            "No severe event pattern found in this window; normal or ambiguous activity.",
            "Continue monitoring.",
        ),
    }
    return mapping.get(label, mapping["none"])


def _horizontal_posture_score_from_window(sample: np.ndarray) -> float:
    """Compute horizontal posture proxy from keypoints (1 = lying/collapsed)."""
    # Keypoints: 5 L shoulder (16,17), 6 R shoulder (19,20), 11 L hip (34,35), 12 R hip (37,38) -> y at 16,19,34,37
    try:
        sh_y = (sample[:, 16] + sample[:, 19]) / 2
        hip_y = (sample[:, 34] + sample[:, 37]) / 2
        vertical_span = np.abs(sh_y - hip_y)
        mean_span = float(np.mean(vertical_span))
        # Standing: large span; lying: small span. Score 1 = lying.
        score = 1.0 - min(1.0, mean_span / 150.0)
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.0


def _area_change_mean_from_window(sample: np.ndarray) -> float:
    """Rough area change from keypoint bbox frame-to-frame."""
    try:
        xs = []
        ys = []
        for row in sample:
            x_vals = [row[i] for i in range(0, 51, 3)]
            y_vals = [row[i] for i in range(1, 51, 3)]
            xs.append(max(x_vals) - min(x_vals) if x_vals else 0)
            ys.append(max(y_vals) - min(y_vals) if y_vals else 0)
        areas = [xs[i] * ys[i] for i in range(len(xs))]
        if len(areas) < 2:
            return 0.0
        return float(np.mean(np.abs(np.diff(areas))))
    except Exception:
        return 0.0


def build_summary_lines(
    motion_mean: float,
    motion_std: float,
    horizontal: float,
    area_change: float,
    audio_distress: float,
    fps: float = 30.0,
    duration_sec: float = 1.0,
    brightness: float = 100.0,
    fast_probs: dict | None = None,
) -> str:
    """Same format as gemma_agent._build_multimodal_summary."""
    lines = [
        f"Video: fps={fps:.1f}, duration_sec={duration_sec:.1f}, motion_mean={motion_mean:.2f}, motion_std={motion_std:.2f}, brightness_mean={brightness:.1f}.",
        f"Body pose (TensorFlow): horizontal_posture_score={horizontal:.2f} (1=lying/collapsed), area_change_mean={area_change:.0f}.",
        f"Audio: distress_score={audio_distress:.2f} (high=distress/coughing).",
    ]
    if fast_probs:
        lines.append(f"Fast detector probs: {json.dumps(fast_probs, indent=0)}.")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Gemma SFT JSONL from pose windows (format matches inference)."
    )
    parser.add_argument("--dataset", required=True, help="NPZ dataset path from prepare_poselift_windows.py")
    parser.add_argument("--output", required=True, help="Output JSONL path for LoRA instruction tuning")
    parser.add_argument(
        "--synthetic-none",
        type=int,
        default=0,
        help="Add N synthetic 'none' examples (normal activity) to reduce false fainting",
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Oversample shoplifting and fainting so each has at least as many as none",
    )
    parser.add_argument(
        "--synthetic-shoplifting",
        type=int,
        default=0,
        help="Add N synthetic 'shoplifting' examples (upright, moderate motion, item-handling proxy)",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    data = np.load(args.dataset, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    labels_list = [str(x) for x in data["labels"].tolist()]

    lines: list[dict] = []
    for idx in range(len(X)):
        label = labels_list[int(y[idx])]
        sample = X[idx]

        motion_mean = float(np.mean(sample[:, -2]))
        motion_std = float(np.std(sample[:, -2]))
        horizontal = _horizontal_posture_score_from_window(sample)
        area_change = _area_change_mean_from_window(sample)
        audio_distress = float(np.mean(sample[:, -1]))

        summary_text = build_summary_lines(
            motion_mean=motion_mean,
            motion_std=motion_std,
            horizontal=horizontal,
            area_change=area_change,
            audio_distress=audio_distress,
            duration_sec=sample.shape[0] / 30.0,
        )
        prompt = CLASSIFIER_RULES + f"MULTIMODAL SUMMARY:\n{summary_text}\n\nJSON array:"

        evidence, action = _incident_from_label(label)
        target = [
            {
                "incident_type": label,
                "confidence": 0.92 if label not in ("none", "suspicious_activity") else 0.85,
                "timestamp_seconds": 1.0,
                "evidence": evidence,
                "recommended_action": action,
            }
        ]
        lines.append({"prompt": prompt, "response": json.dumps(target), "label": label})

    # Synthetic "none" examples: normal activity (upright, some motion, low distress)
    for _ in range(args.synthetic_none):
        motion_mean = float(np.random.uniform(5, 18))
        motion_std = float(np.random.uniform(1, 6))
        horizontal = float(np.random.uniform(0.05, 0.35))
        area_change = float(np.random.uniform(0, 80))
        audio_distress = float(np.random.uniform(0, 0.25))
        summary_text = build_summary_lines(
            motion_mean=motion_mean,
            motion_std=motion_std,
            horizontal=horizontal,
            area_change=area_change,
            audio_distress=audio_distress,
        )
        prompt = CLASSIFIER_RULES + f"MULTIMODAL SUMMARY:\n{summary_text}\n\nJSON array:"
        evidence, action = _incident_from_label("none")
        target = [
            {
                "incident_type": "none",
                "confidence": 0.88,
                "timestamp_seconds": 1.0,
                "evidence": evidence,
                "recommended_action": action,
            }
        ]
        lines.append({"prompt": prompt, "response": json.dumps(target), "label": "none"})

    # Synthetic "shoplifting" examples: upright person, moderate motion (handling/concealment proxy), low distress
    for _ in range(args.synthetic_shoplifting):
        motion_mean = float(np.random.uniform(8, 22))
        motion_std = float(np.random.uniform(2, 7))
        horizontal = float(np.random.uniform(0.1, 0.45))
        area_change = float(np.random.uniform(20, 120))
        audio_distress = float(np.random.uniform(0, 0.35))
        summary_text = build_summary_lines(
            motion_mean=motion_mean,
            motion_std=motion_std,
            horizontal=horizontal,
            area_change=area_change,
            audio_distress=audio_distress,
        )
        prompt = CLASSIFIER_RULES + f"MULTIMODAL SUMMARY:\n{summary_text}\n\nJSON array:"
        evidence, action = _incident_from_label("shoplifting")
        target = [
            {
                "incident_type": "shoplifting",
                "confidence": 0.88,
                "timestamp_seconds": 1.0,
                "evidence": evidence,
                "recommended_action": action,
            }
        ]
        lines.append({"prompt": prompt, "response": json.dumps(target), "label": "shoplifting"})

    if args.balance:
        by_label: dict[str, list[dict]] = {}
        for row in lines:
            by_label.setdefault(row["label"], []).append(row)
        max_count = max(len(v) for v in by_label.values()) if by_label else 0
        balanced = []
        for label, rows in by_label.items():
            balanced.extend(rows)
            shortfall = max_count - len(rows)
            if shortfall > 0 and rows:
                balanced.extend(random.choices(rows, k=shortfall))
        random.shuffle(balanced)
        lines = balanced

    with out.open("w", encoding="utf-8") as f:
        for row in lines:
            f.write(json.dumps({"prompt": row["prompt"], "response": row["response"]}) + "\n")

    counts: dict[str, int] = {}
    for row in lines:
        lbl = row.get("label", "?")
        counts[lbl] = counts.get(lbl, 0) + 1
    print(f"Wrote {len(lines)} SFT samples to {out}")
    print(f"Class counts: {counts}")


if __name__ == "__main__":
    main()
