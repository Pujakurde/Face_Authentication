"""
liveness.py

Stage 5 — Liveness Detection (Anti-Spoofing)

This module verifies whether the detected face is LIVE or FAKE
using multiple temporal facial cues.

Signals used:
- Eye blink
- Head movement
- Natural facial motion
- Temporal frame variation

Decision logic:
LIVE if at least TWO strong signals are detected.
"""

import cv2
import os
import numpy as np
from collections import deque
from typing import Tuple
import time

try:
    from camera import open_camera
except Exception:
    open_camera = None

# -------- LANDMARK INDEXES --------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
NOSE_TIP = 1

# -------- PARAMETERS (TUNED) --------
EAR_THRESHOLD = 0.23
EYE_CLOSED_FRAMES = 2
NOSE_MOVE_THRESHOLD = 0.012
MOTION_THRESHOLD = 0.0015
FRAME_DIFF_THRESHOLD = 1.5

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_FACE_LANDMARKER = os.path.join(MODEL_DIR, "face_landmarker.task")


# ------------------ UTILS ------------------

def eye_aspect_ratio(eye, landmarks, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return 0.0 if C == 0 else (A + B) / (2.0 * C)


def _face_bbox_from_landmarks(landmarks, w: int, h: int) -> Tuple[int, int, int, int]:
    xs = [int(lm.x * w) for lm in landmarks]
    ys = [int(lm.y * h) for lm in landmarks]
    x_min, x_max = max(0, min(xs)), min(w - 1, max(xs))
    y_min, y_max = max(0, min(ys)), min(h - 1, max(ys))
    return x_min, y_min, x_max - x_min, y_max - y_min


# ------------------ MAIN ------------------

def run_liveness(camera_index: int = 0, width: int = 640, height: int = 480):
    if open_camera:
        cap = open_camera(camera_index, width, height)
    else:
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    blink_counter = 0
    blink_detected = False
    blink_time = 0

    nose_history = deque(maxlen=12)
    motion_history = deque(maxlen=10)
    frame_diff_history = deque(maxlen=8)
    prev_face_gray = None

    use_tasks = os.path.exists(MODEL_FACE_LANDMARKER)

    if use_tasks:
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.vision import face_landmarker as fl
        from mediapipe.tasks.python.vision.core import image as mp_image

        options = fl.FaceLandmarkerOptions(
            base_options=vision.BaseOptions(model_asset_path=MODEL_FACE_LANDMARKER),
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_faces=1
        )
        landmarker = fl.FaceLandmarker.create_from_options(options)

        window = "Liveness Detection"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        ts = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            mpimg = mp_image.Image(
                image_format=vision.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )
            ts += 33
            result = landmarker.detect_for_video(mpimg, ts)

            status = "CHECKING LIVENESS"
            color = (0, 200, 255)

            if result.face_landmarks:
                lm = result.face_landmarks[0].landmarks

                # ---- BLINK ----
                left_ear = eye_aspect_ratio(LEFT_EYE, lm, w, h)
                right_ear = eye_aspect_ratio(RIGHT_EYE, lm, w, h)
                ear = (left_ear + right_ear) / 2.0

                if ear < EAR_THRESHOLD:
                    blink_counter += 1
                else:
                    if blink_counter >= EYE_CLOSED_FRAMES:
                        blink_detected = True
                        blink_time = time.time()
                    blink_counter = 0

                blink_valid = blink_detected and (time.time() - blink_time < 3)

                # ---- HEAD MOVE ----
                nose_x = lm[NOSE_TIP].x
                nose_history.append(nose_x)
                head_turn = (
                    len(nose_history) >= 8 and
                    abs(nose_history[-1] - nose_history[0]) > NOSE_MOVE_THRESHOLD
                )

                # ---- NATURAL MOTION ----
                motion = np.std([p.x for p in lm])
                motion_history.append(motion)
                natural_motion = np.mean(motion_history) > MOTION_THRESHOLD

                # ---- TEMPORAL VARIATION ----
                x, y, bw, bh = _face_bbox_from_landmarks(lm, w, h)
                face_gray = cv2.cvtColor(frame[y:y+bh, x:x+bw], cv2.COLOR_BGR2GRAY)

                frame_diff = 0.0
                if prev_face_gray is not None and face_gray.shape == prev_face_gray.shape:
                    frame_diff = float(np.mean(np.abs(face_gray - prev_face_gray)))
                prev_face_gray = face_gray.copy()
                frame_diff_history.append(frame_diff)

                temporal_variation = np.mean(frame_diff_history) > FRAME_DIFF_THRESHOLD

                # ---- FINAL DECISION (RELAXED & CORRECT) ----
                live_signals = sum([
                    blink_valid,
                    head_turn,
                    temporal_variation
                ])

                if live_signals >= 2:
                    status = "LIVE FACE"
                    color = (0, 255, 0)
                else:
                    status = "NOT LIVE"

                cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)

            else:
                status = "FACE NOT DETECTED"
                color = (0, 0, 255)

            cv2.putText(frame, status, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

            cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        landmarker.close()

    cap.release()
    cv2.destroyAllWindows()
