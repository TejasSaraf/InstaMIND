from pathlib import Path

import cv2
import numpy as np


def _safe_float(value: float) -> float:
    if np.isnan(value) or np.isinf(value):
        return 0.0
    return float(value)


class VideoSignalExtractor:
    def __init__(self, max_sample_frames: int = 48) -> None:
        self.max_sample_frames = max_sample_frames

    def extract(self, video_path: Path) -> dict:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_seconds = (total_frames / fps) if fps > 0 else 0.0

        frame_stride = max(1, total_frames // self.max_sample_frames) if total_frames else 1
        sampled_frames = []
        frame_idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % frame_stride == 0:
                sampled_frames.append(frame)
            frame_idx += 1

        cap.release()

        grayscale_means = []
        motion_scores = []
        prev_gray = None
        for frame in sampled_frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            grayscale_means.append(_safe_float(float(gray.mean())))
            if prev_gray is None:
                motion_scores.append(0.0)
            else:
                diff = cv2.absdiff(gray, prev_gray)
                motion_scores.append(_safe_float(float(diff.mean())))
            prev_gray = gray

        return {
            "fps": _safe_float(fps),
            "total_frames": total_frames,
            "duration_seconds": _safe_float(duration_seconds),
            "sample_count": len(sampled_frames),
            "brightness_mean": _safe_float(float(np.mean(grayscale_means))) if grayscale_means else 0.0,
            "motion_mean": _safe_float(float(np.mean(motion_scores))) if motion_scores else 0.0,
            "motion_std": _safe_float(float(np.std(motion_scores))) if motion_scores else 0.0,
            "motion_series": [round(x, 4) for x in motion_scores[:120]],
        }
