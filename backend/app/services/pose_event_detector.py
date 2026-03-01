import json
from pathlib import Path

import numpy as np
import tensorflow as tf


class PoseEventDetector:
    """
    Fast-path event detector trained on pose windows.
    Returns class probabilities for low-latency emergency trigger decisions.
    """

    def __init__(self, model_path: str, label_path: str, window_size: int = 32) -> None:
        self.window_size = window_size
        self.model = None
        self.labels: list[str] = []

        model_file = Path(model_path)
        labels_file = Path(label_path)
        print(f"[PoseEventDetector] model_path={model_file.resolve()} exists={model_file.exists()}")
        print(f"[PoseEventDetector] label_path={labels_file.resolve()} exists={labels_file.exists()}")
        if model_file.exists() and labels_file.exists():
            try:
                self.model = tf.keras.models.load_model(model_file)
                self.labels = json.loads(labels_file.read_text(encoding="utf-8"))
                print(f"[PoseEventDetector] Loaded OK — labels={self.labels}")
            except Exception as e:
                print(f"[PoseEventDetector] FAILED to load: {e}")
                self.model = None

    def available(self) -> bool:
        return self.model is not None and bool(self.labels)

    def predict(self, signals: dict) -> dict:
        if not self.available():
            return {"available": False, "event_probs": {}}

        window = self._build_window(signals)
        preds = self.model.predict(window, verbose=0)
        probs = preds[0].tolist()
        event_probs = {self.labels[i]: float(probs[i]) for i in range(len(self.labels))}
        top_event = max(event_probs, key=event_probs.get)
        return {
            "available": True,
            "event_probs": event_probs,
            "top_event": top_event,
            "top_confidence": event_probs[top_event],
        }

    def _build_window(self, signals: dict) -> np.ndarray:
        """
        Build (1, window_size, 53) to match training: 17*3 keypoints + motion + audio.
        At runtime we have no keypoints; we use horizontal_series, motion_series, distress
        in the same slots as training (keypoints 0..50 zeroed, motion at 51, audio at 52).
        """
        pose = signals.get("pose", {})
        video = signals.get("video", {})
        audio = signals.get("audio", {})

        horizontal_series = np.array(pose.get("horizontal_series", []), dtype=np.float32)
        motion_series = np.array(video.get("motion_series", []), dtype=np.float32)
        distress_score = float(audio.get("distress_score", 0.0))

        # Training shape: (batch, 32, 53) with 0..50 = keypoints, 51 = motion, 52 = audio
        features = np.zeros((self.window_size, 53), dtype=np.float32)
        for i in range(self.window_size):
            hs = horizontal_series[i] if i < len(horizontal_series) else 0.0
            ms = motion_series[i] if i < len(motion_series) else 0.0
            features[i, 0] = hs
            features[i, 51] = ms
            features[i, 52] = distress_score
        return np.expand_dims(features, axis=0)
