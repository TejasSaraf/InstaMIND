"""
On-device human detection using OpenCV DNN + YOLOv4-tiny.
Mirrors the logic in cpp_extensions/human_detector.cpp but runs as a Python
service callable from FastAPI endpoints.

Expects yolov4-tiny.cfg and yolov4-tiny.weights in backend/models/.
"""



import base64

import logging

import time

from pathlib import Path

from typing import Optional



import cv2

import numpy as np



logger = logging.getLogger(__name__)



BACKEND_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = BACKEND_ROOT / "models"



NET_INPUT_SIZE = 416

CONF_THRESHOLD = 0.45

NMS_THRESHOLD = 0.40

PERSON_CLASS = 0





class HumanDetector:

    """Singleton-style detector. Loads YOLOv4-tiny once, reuses for every frame."""



    def __init__(

        self,

        cfg_path: Optional[str] = None,

        weights_path: Optional[str] = None,

        conf_threshold: float = CONF_THRESHOLD,

        nms_threshold: float = NMS_THRESHOLD,

    ):

        self.conf_threshold = conf_threshold

        self.nms_threshold = nms_threshold

        self._net: Optional[cv2.dnn.Net] = None



        self._cfg_path = cfg_path or str(MODELS_DIR / "yolov4-tiny.cfg")

        self._weights_path = weights_path or str(

            MODELS_DIR / "yolov4-tiny.weights")



    def _load_net(self) -> cv2.dnn.Net:

        if self._net is not None:

            return self._net



        cfg = Path(self._cfg_path)

        weights = Path(self._weights_path)



        if not cfg.exists() or not weights.exists():

            raise FileNotFoundError(

                f"YOLO model files not found.\n"

                f"  Expected cfg:     {cfg}\n"

                f"  Expected weights: {weights}\n"

                f"Download with:\n"

                f"  wget https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights -P {MODELS_DIR}\n"

                f"  wget https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg -P {MODELS_DIR}"

            )



        logger.info("Loading YOLOv4-tiny from %s", weights)

        net = cv2.dnn.readNetFromDarknet(str(cfg), str(weights))

        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)

        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self._net = net

        logger.info("YOLOv4-tiny loaded successfully")

        return net



    def available(self) -> bool:

        cfg = Path(self._cfg_path)

        weights = Path(self._weights_path)

        return cfg.exists() and weights.exists()



    def detect_frame_b64(self, frame_b64: str) -> dict:

        """Accept base64 JPEG, return list of person bounding boxes.

        Returns:
            {
                "detections": [
                    {"x": int, "y": int, "w": int, "h": int, "confidence": float},
                    ...
                ],
                "person_count": int,
                "inference_ms": float,
            }
        """

        net = self._load_net()



        img_bytes = base64.b64decode(frame_b64)

        arr = np.frombuffer(img_bytes, dtype=np.uint8)

        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        if frame is None:

            return {"detections": [], "person_count": 0, "inference_ms": 0}



        fh, fw = frame.shape[:2]



        t0 = time.perf_counter()



        blob = cv2.dnn.blobFromImage(

            frame, 1.0 / 255.0,

            (NET_INPUT_SIZE, NET_INPUT_SIZE),

            swapRB=True, crop=False,

        )

        net.setInput(blob)



        out_names = net.getUnconnectedOutLayersNames()

        outs = net.forward(out_names)



        boxes = []

        confidences = []



        for out in outs:

            for detection in out:

                scores = detection[5:]

                obj_conf = detection[4]

                if obj_conf < self.conf_threshold:

                    continue

                person_score = float(scores[PERSON_CLASS] * obj_conf)

                if person_score < self.conf_threshold:

                    continue



                cx = int(detection[0] * fw)

                cy = int(detection[1] * fh)

                w = int(detection[2] * fw)

                h = int(detection[3] * fh)

                x = cx - w // 2

                y = cy - h // 2



                boxes.append([x, y, w, h])

                confidences.append(person_score)



        indices = cv2.dnn.NMSBoxes(

            boxes, confidences, self.conf_threshold, self.nms_threshold)



        t1 = time.perf_counter()

        inference_ms = round((t1 - t0) * 1000, 1)



        detections = []

        if len(indices) > 0:

            for i in indices.flatten():

                bx, by, bw, bh = boxes[i]

                detections.append({

                    "x": max(bx, 0),

                    "y": max(by, 0),

                    "w": min(bw, fw - max(bx, 0)),

                    "h": min(bh, fh - max(by, 0)),

                    "confidence": round(confidences[i], 3),

                })



        return {

            "detections": detections,

            "person_count": len(detections),

            "inference_ms": inference_ms,

        }
