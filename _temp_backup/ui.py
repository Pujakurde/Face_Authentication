"""
ui.py

Stage 8 — Interactive Face-ID–style UI using OpenCV overlays.

Features:
- Live camera feed with a centered circular face guide
- Real-time statuses: FACE NOT DETECTED, MULTIPLE FACES, FAKE FACE,
  AUTHENTICATING..., ACCESS GRANTED, ACCESS DENIED
- Color-coded feedback (red/yellow/green)
- Smooth gating: require several consecutive frames for ACCESS GRANTED
- Keyboard shortcuts: 'r' to run registration (calls register_face.main()), 'q' to quit

This file keeps everything offline and uses the existing MediaPipe-based
landmark logic implemented in earlier stages.
"""

import os
import time
import cv2
import numpy as np

MATCH_THRESHOLD = 0.6
GRANT_CONSECUTIVE_FRAMES = 5

try:
    from camera import open_camera
except Exception:
    open_camera = None


def draw_face_guide(frame, center=None, radius=140, color=(200, 200, 200), thickness=2):
    h, w = frame.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    cv2.circle(frame, center, radius, color, thickness, lineType=cv2.LINE_AA)


def run_ui(camera_index: int = 0, width: int = 640, height: int = 480):
    # prefer Tasks model if present
    MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
    MODEL_FACE_LANDMARKER = os.path.join(MODEL_DIR, "face_landmarker.task")
    use_tasks = os.path.exists(MODEL_FACE_LANDMARKER)

    face_processor = None
    legacy_mesh = None

    if use_tasks:
        try:
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.vision import face_landmarker as fl
            from mediapipe.tasks.python.vision.core import image as mp_image

            base_options = vision.BaseOptions(model_asset_path=MODEL_FACE_LANDMARKER)
            options = fl.FaceLandmarkerOptions(base_options=base_options,
                                               running_mode=vision.RunningMode.LIVE_STREAM,
                                               num_faces=2)
            face_processor = fl.FaceLandmarker.create_from_options(options)
            print("INFO: UI using Tasks FaceLandmarker")
        except Exception as e:
            print("WARN: Tasks FaceLandmarker failed, falling back:", e)
            use_tasks = False

    if not use_tasks:
        try:
            import mediapipe as mp
            mp_face_mesh = mp.solutions.face_mesh
            legacy_mesh = mp_face_mesh.FaceMesh(
                max_num_faces=2,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except Exception as e:
            print("ERROR: No available face landmarker for UI:", e)

    if open_camera:
        cap = open_camera(camera_index, width, height)
    else:
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return

    registered = None
    if os.path.exists("registered_face.npy"):
        try:
            registered = np.load("registered_face.npy")
        except Exception:
            registered = None

    with mp_face_mesh.FaceMesh(
        max_num_faces=2,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        window = "Face-ID UI - Press 'r' to register, 'q' to quit"
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)

        # temporal buffers
        blink_counter = 0
        blink_detected = False
        nose_history = []
        motion_history = []
        prev_face_gray = None
        frame_diff_history = []

        grant_counter = 0

        ts = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            landmarks = None
            face_count = 0

            # draw guide
            draw_face_guide(frame)

            if use_tasks and face_processor is not None:
                try:
                    from mediapipe.tasks.python.vision import image as mp_image
                    from mediapipe.tasks.python import vision
                    mpimg = mp_image.Image(image_format=vision.ImageFormat.SRGB,
                                          data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    ts += 33
                    res = face_processor.detect_for_video(mpimg, ts)
                    if res and res.face_landmarks:
                        face_count = len(res.face_landmarks)
                        if face_count == 1:
                            landmarks = res.face_landmarks[0].landmarks
                except Exception as e:
                    print("UI (Tasks) error:", e)
            elif legacy_mesh is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = legacy_mesh.process(rgb)
                if getattr(results, 'multi_face_landmarks', None):
                    face_count = len(results.multi_face_landmarks)
                    if face_count == 1:
                        landmarks = results.multi_face_landmarks[0].landmark

            status = "FACE NOT DETECTED"
            color = (0, 0, 255)

            if face_count == 0:
                status = "FACE NOT DETECTED"
                color = (0, 0, 255)
                grant_counter = 0
            elif face_count > 1:
                status = "MULTIPLE FACES"
                color = (0, 200, 255)
                grant_counter = 0
            else:
                lm = landmarks

                # EAR blink heuristic (use same indices as other modules)
                LEFT_EYE = [33, 160, 158, 133, 153, 144]
                RIGHT_EYE = [362, 385, 387, 263, 373, 380]
                def ear(eye):
                    pts = [(int(lm[i].x * w), int(lm[i].y * h)) for i in eye]
                    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
                    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
                    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
                    return (A + B) / (2.0 * C) if C != 0 else 0.0

                left_ear = ear(LEFT_EYE)
                right_ear = ear(RIGHT_EYE)
                ear_mean = (left_ear + right_ear) / 2.0

                EAR_THRESHOLD = 0.25
                EYE_CLOSED_FRAMES = 3
                if ear_mean < EAR_THRESHOLD:
                    blink_counter += 1
                else:
                    if blink_counter >= EYE_CLOSED_FRAMES:
                        blink_detected = True
                    blink_counter = 0

                # nose movement
                nose_x = lm[1].x
                nose_history.append(nose_x)
                if len(nose_history) > 15:
                    nose_history.pop(0)
                head_turn = False
                if len(nose_history) >= 10 and abs(nose_history[-1] - nose_history[0]) > 0.015:
                    head_turn = True

                # natural motion (landmark variance)
                motion = float(np.std([p.x for p in lm]))
                motion_history.append(motion)
                if len(motion_history) > 10:
                    motion_history.pop(0)
                natural_motion = np.mean(motion_history) > 0.002

                # temporal frame diff
                x_coords = [int(p.x * w) for p in lm]
                y_coords = [int(p.y * h) for p in lm]
                x0, x1 = max(0, min(x_coords)), min(w - 1, max(x_coords))
                y0, y1 = max(0, min(y_coords)), min(h - 1, max(y_coords))
                pad = int(max(8, 0.05 * max(x1 - x0, y1 - y0)))
                xa, ya = max(0, x0 - pad), max(0, y0 - pad)
                xb, yb = min(w, x1 + pad), min(h, y1 + pad)

                face_gray = None
                if xb > xa and yb > ya:
                    face_gray = cv2.cvtColor(frame[ya:yb, xa:xb], cv2.COLOR_BGR2GRAY)

                frame_diff = 0.0
                if face_gray is not None and prev_face_gray is not None and face_gray.size == prev_face_gray.size:
                    frame_diff = float(np.mean(np.abs(face_gray.astype(np.float32) - prev_face_gray.astype(np.float32))))
                prev_face_gray = face_gray.copy() if face_gray is not None else None

                frame_diff_history.append(frame_diff)
                if len(frame_diff_history) > 8:
                    frame_diff_history.pop(0)
                temporal_variation = np.mean(frame_diff_history) > 2.0

                # embedding and match
                current_emb = np.array([p.x for p in lm] + [p.y for p in lm] + [getattr(p, 'z', 0.0) for p in lm], dtype=np.float32)
                match = False
                dist = None
                if registered is not None and current_emb.shape == registered.shape:
                    dist = float(np.linalg.norm(registered - current_emb))
                    match = dist < MATCH_THRESHOLD

                # decide UI status
                liveness_ok = blink_detected and head_turn and natural_motion and temporal_variation

                if not liveness_ok:
                    status = "FAKE FACE"
                    color = (0, 0, 255)
                    grant_counter = 0
                else:
                    if registered is None:
                        status = "AUTHENTICATING... (no registered face)"
                        color = (0, 200, 255)
                        grant_counter = 0
                    else:
                        status = "AUTHENTICATING..."
                        color = (0, 200, 255)
                        if match:
                            grant_counter += 1
                        else:
                            grant_counter = 0

                        if grant_counter >= GRANT_CONSECUTIVE_FRAMES:
                            status = "ACCESS GRANTED"
                            color = (0, 255, 0)
                        else:
                            # still authenticating
                            status = f"AUTHENTICATING... (dist={dist:.3f})" if dist is not None else "AUTHENTICATING..."
                            color = (0, 200, 255)

                # draw bbox
                if xb > xa and yb > ya:
                    cv2.rectangle(frame, (xa, ya), (xb, yb), color, 2)

            # overlay status
            cv2.putText(frame, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(frame, "Press 'r' to register, 'q' to quit", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

            cv2.imshow(window, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('r'):
                # run registration flow from register_face.py if available
                try:
                    import face_id_sys
                    cv2.destroyAllWindows()
                    print("Launching registration...")
                    face_id_sys.main()
                    # reload registered embedding after registration
                    if os.path.exists("registered_face.npy"):
                        registered = np.load("registered_face.npy")
                    # re-create window after registration
                    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
                except Exception as e:
                    print("Registration failed or module not found:", e)

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_ui()
