import base64

import tempfile

import time

from pathlib import Path



import cv2

import numpy as np

from fastapi import WebSocket, WebSocketDisconnect





class RealtimeStreamHandler:

    def __init__(

        self,

        fps_target: int = 2,

        max_dimension: int = 640,

        scene_change_threshold: float = 30.0,

        keyframes_only_save: bool = True,

        emit_interval_s: float = 2.0,

    ):

        self.fps_target = fps_target

        self.max_dimension = max_dimension

        self.scene_change_threshold = scene_change_threshold

        self.keyframes_only_save = keyframes_only_save

        self.emit_interval_s = emit_interval_s



        self.frame_buffer = []

        self.last_emit_time = time.time()



    def perceptual_hash(self, gray_image: np.ndarray) -> str:

        """Lightweight perceptual hash (Average Hash - aHash)."""

        resized = cv2.resize(gray_image, (8, 8), interpolation=cv2.INTER_AREA)

        mean_val = np.mean(resized)

        hash_binary = (resized > mean_val).flatten()



        hash_int = 0

        for bit in hash_binary:

            hash_int = (hash_int << 1) | int(bit)

        return f"{hash_int:016x}"



    def downscale_if_needed(self, frame: np.ndarray) -> np.ndarray:

        h, w = frame.shape[:2]

        if max(h, w) > self.max_dimension:

            scale = self.max_dimension / float(max(h, w))

            return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        return frame



    async def handle_websocket(self, websocket: WebSocket):

        await websocket.accept()



        session_id = str(int(time.time()))

        session_dir = Path(tempfile.gettempdir()) / f"instamind_ws_{session_id}"

        session_dir.mkdir(parents=True, exist_ok=True)



        video_path = session_dir / "stream.webm"

        file_handle = open(video_path, "wb")



        prev_gray = None

        frame_count = 0



        try:

            while True:



                data = await websocket.receive_bytes()

                file_handle.write(data)

                file_handle.flush()



                chunk_path = session_dir / f"chunk_{frame_count}.webm"

                with open(chunk_path, "wb") as f:

                    f.write(data)



                cap = cv2.VideoCapture(str(chunk_path))

                if not cap.isOpened():

                    chunk_path.unlink(missing_ok=True)

                    continue



                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

                skip_frames = max(1, int(fps / self.fps_target))



                local_idx = 0

                while cap.isOpened():

                    ok, frame = cap.read()

                    if not ok:

                        break



                    if local_idx % skip_frames != 0:

                        local_idx += 1

                        continue



                    small_frame = self.downscale_if_needed(frame)

                    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)



                    is_keyframe = False

                    if prev_gray is None:

                        is_keyframe = True

                    else:

                        if prev_gray.shape == gray.shape:

                            diff = cv2.absdiff(gray, prev_gray)

                            if np.mean(diff) > self.scene_change_threshold:

                                is_keyframe = True

                        else:

                            is_keyframe = True



                    prev_gray = gray



                    p_hash = self.perceptual_hash(gray)



                    if is_keyframe or not self.keyframes_only_save:

                        frame_filename = session_dir / f"frame_{frame_count}_{p_hash}.jpg"

                        cv2.imwrite(str(frame_filename), small_frame, [

                                    cv2.IMWRITE_JPEG_QUALITY, 85])



                        _, buf = cv2.imencode(".jpg", small_frame, [

                                              cv2.IMWRITE_JPEG_QUALITY, 80])

                        b64 = base64.b64encode(buf.tobytes()).decode("ascii")



                        self.frame_buffer.append({

                            "timestamp": time.time(),

                            "hash": p_hash,

                            "is_keyframe": is_keyframe,

                            "image_b64": b64

                        })



                    local_idx += 1

                    frame_count += 1



                cap.release()

                chunk_path.unlink(missing_ok=True)



                current_time = time.time()

                if current_time - self.last_emit_time >= self.emit_interval_s:

                    if self.frame_buffer:

                        await websocket.send_json({

                            "type": "frames",

                            "count": len(self.frame_buffer),

                            "data": self.frame_buffer

                        })

                        self.frame_buffer = []

                    self.last_emit_time = current_time



        except WebSocketDisconnect:

            print("Client disconnected from WebSocket.")

        except Exception as e:

            print(f"WebSocket error: {e}")

        finally:

            file_handle.close()
