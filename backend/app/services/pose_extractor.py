from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


class TensorflowPoseSignalExtractor:
    """
    Lightweight body-position proxy signals for rapid hackathon iteration.
    This module uses TensorFlow tensor ops for posture-derived metrics that
    can feed an agentic incident detection loop.
    """

    def __init__(self, max_sample_frames: int = 48) -> None:
        self.max_sample_frames = max_sample_frames

    def extract(self, video_path: Path) -> dict:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_stride = max(1, total_frames // self.max_sample_frames) if total_frames else 1

        aspect_ratios = []
        areas = []
        idx = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % frame_stride != 0:
                idx += 1
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                cnt = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratios.append(float(w) / max(float(h), 1.0))
                areas.append(float(w * h))
            else:
                aspect_ratios.append(0.0)
                areas.append(0.0)

            idx += 1

        cap.release()

        aspect_tensor = tf.convert_to_tensor(aspect_ratios if aspect_ratios else [0.0], dtype=tf.float32)
        area_tensor = tf.convert_to_tensor(areas if areas else [0.0], dtype=tf.float32)

        horizontal_probability = tf.math.sigmoid((aspect_tensor - 1.4) * 3.0)
        movement_proxy = tf.abs(area_tensor[1:] - area_tensor[:-1]) if tf.size(area_tensor) > 1 else tf.constant([0.0])

        return {
            "pose_sample_count": int(tf.size(aspect_tensor).numpy()),
            "aspect_ratio_mean": float(tf.reduce_mean(aspect_tensor).numpy()),
            "horizontal_posture_score": float(tf.reduce_mean(horizontal_probability).numpy()),
            "area_change_mean": float(tf.reduce_mean(movement_proxy).numpy()),
            "horizontal_series": [round(float(x), 4) for x in horizontal_probability.numpy().tolist()[:120]],
        }
