import time
from pathlib import Path

import cv2
import numpy as np

from app.config import settings


class FrameStreamAnalyzer:
    """
    Frame-by-frame analyzer with latency-aware adaptive processing.
    Keeps per-frame compute lightweight to stay within sub-100ms targets.
    """

    def __init__(self, max_frames: int = 600) -> None:
        self.max_frames = max_frames

    def analyze(self, video_path: Path) -> dict:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_seconds = (total_frames / fps) if fps > 0 else 0.0

        motion_scores: list[float] = []
        brightness_scores: list[float] = []
        horizontal_scores: list[float] = []
        area_changes: list[float] = []
        frame_latencies_ms: list[float] = []

        prev_gray = None
        prev_area = 0.0

        downscale = 0.5
        skip_stride = 1
        frame_idx = 0
        processed_count = 0
        target_ms = float(settings.emergency_latency_target_ms)

        while processed_count < self.max_frames:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_idx % skip_stride != 0:
                frame_idx += 1
                continue

            start = time.perf_counter()
            small = cv2.resize(frame, None, fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (3, 3), 0)

            brightness_scores.append(float(gray.mean()))
            if prev_gray is None:
                motion_scores.append(0.0)
            else:
                diff = cv2.absdiff(gray, prev_gray)
                motion_scores.append(float(diff.mean()))

            _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / max(float(h), 1.0)
                area = float(w * h)
            else:
                aspect_ratio = 0.0
                area = 0.0

            horizontal_prob = float(1.0 / (1.0 + np.exp(-(aspect_ratio - 1.4) * 3.0)))
            horizontal_scores.append(horizontal_prob)
            area_changes.append(abs(area - prev_area))

            prev_area = area
            prev_gray = gray
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            frame_latencies_ms.append(elapsed_ms)

            if elapsed_ms > target_ms:
                downscale = max(0.25, downscale - 0.1)
                skip_stride = min(4, skip_stride + 1)

            processed_count += 1
            frame_idx += 1

        cap.release()

        if not frame_latencies_ms:
            raise ValueError("No frames processed from uploaded video.")

        latency = np.array(frame_latencies_ms, dtype=np.float32)
        motion = np.array(motion_scores, dtype=np.float32)
        bright = np.array(brightness_scores, dtype=np.float32)
        horizontal = np.array(horizontal_scores, dtype=np.float32)
        area_delta = np.array(area_changes, dtype=np.float32)

        latency_summary = {
            "target_ms": int(target_ms),
            "frame_count_processed": int(len(frame_latencies_ms)),
            "p50_ms": float(np.percentile(latency, 50)),
            "p95_ms": float(np.percentile(latency, 95)),
            "max_ms": float(np.max(latency)),
            "violations": int(np.sum(latency > target_ms)),
            "met_target": bool(np.max(latency) <= target_ms),
            "downscale_final": downscale,
            "skip_stride_final": skip_stride,
        }

        video_signals = {
            "fps": float(fps),
            "total_frames": total_frames,
            "duration_seconds": float(duration_seconds),
            "sample_count": int(len(frame_latencies_ms)),
            "brightness_mean": float(np.mean(bright)) if bright.size else 0.0,
            "motion_mean": float(np.mean(motion)) if motion.size else 0.0,
            "motion_std": float(np.std(motion)) if motion.size else 0.0,
            "motion_series": [round(float(x), 4) for x in motion[:120].tolist()],
        }
        pose_signals = {
            "pose_sample_count": int(len(horizontal_scores)),
            "horizontal_posture_score": float(np.mean(horizontal)) if horizontal.size else 0.0,
            "area_change_mean": float(np.mean(area_delta)) if area_delta.size else 0.0,
            "horizontal_series": [round(float(x), 4) for x in horizontal[:120].tolist()],
        }
        audio_signals = {
            "distress_score": min(1.0, float(video_signals["motion_std"]) / 25.0),
            "audio_pipeline_status": "placeholder_from_video_proxy",
        }

        return {
            "video": video_signals,
            "pose": pose_signals,
            "audio": audio_signals,
            "latency": latency_summary,
        }
