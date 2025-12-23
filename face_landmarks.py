"""
face_landmarks.py

Stage 3 — Advanced Face Landmarks module.

This version prefers the MediaPipe Tasks `FaceLandmarker` model if found
in `models/face_landmarker.task`. If the model is missing or the Tasks API
is unavailable, it falls back to the legacy `mp.solutions.face_mesh` API.

Exports:
- `extract_landmark_array(landmarks)` — accepts either Tasks or Solutions landmarks
- `run_face_landmarks()` — live visualizer
"""

import os
import cv2
import numpy as np

try:
    from camera import open_camera
except Exception:
    open_camera = None

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_FACE_LANDMARKER = os.path.join(MODEL_DIR, "face_landmarker.task")


def _lm_to_xyztuple(lm):
    # tasks-landmark has .x/.y/.z, solutions-landmark same — keep generic
    return (float(lm.x), float(lm.y), float(getattr(lm, 'z', 0.0)))


def extract_landmark_array(landmarks):
    """Convert landmarks to flat numpy array [x0,y0,z0,...].

    Accepts either a Tasks-style landmark list or a Solutions-style `landmark` sequence.
    """
    arr = []
    for lm in landmarks:
        x, y, z = _lm_to_xyztuple(lm)
        arr.extend([x, y, z])
    return np.array(arr, dtype=np.float32)


def run_face_landmarks(camera_index: int = 0,
                       width: int = 640,
                       height: int = 480,
                       min_detection_confidence: float = 0.5,
                       min_tracking_confidence: float = 0.5,
                       show_landmark_indices: bool = False):
    """Run a live face mesh visualizer using Tasks API if available."""

    # open camera
    if open_camera:
        cap = open_camera(camera_index, width, height)
    else:
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    use_tasks = os.path.exists(MODEL_FACE_LANDMARKER)

    if use_tasks:
        try:
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.vision import face_landmarker as fl
            from mediapipe.tasks.python.vision.core import image as mp_image

            base_options = vision.BaseOptions(model_asset_path=MODEL_FACE_LANDMARKER)
            options = fl.FaceLandmarkerOptions(base_options=base_options,
                                               running_mode=vision.RunningMode.LIVE_STREAM,
                                               num_faces=1)
            landmarker = fl.FaceLandmarker.create_from_options(options)

            window = "Face Landmarks - Press 'q' to quit"
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
            ts = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                h, w, _ = frame.shape
                mpimg = mp_image.Image(image_format=vision.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                ts += 33
                result = landmarker.detect_for_video(mpimg, ts)

                if result.face_landmarks:
                    face_landmarks = result.face_landmarks[0]

                    # draw simple dots for landmarks
                    for idx, lm in enumerate(face_landmarks.landmarks):
                        x = int(lm.x * w)
                        y = int(lm.y * h)
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                        if show_landmark_indices:
                            cv2.putText(frame, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)
                else:
                    cv2.putText(frame, "No face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                cv2.imshow(window, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            landmarker.close()
            cap.release()
            cv2.destroyAllWindows()
            return
        except Exception as e:
            print("Tasks API FaceLandmarker failed, falling back:", e)

    # Fallback to legacy solutions API
    try:
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles

        with mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=min_detection_confidence,
                                  min_tracking_confidence=min_tracking_confidence) as face_mesh:

            window = "Face Landmarks - Press 'q' to quit"
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                h, w, _ = frame.shape
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(rgb)

                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    mp_drawing.draw_landmarks(image=frame,
                                              landmark_list=face_landmarks,
                                              connections=mp_face_mesh.FACEMESH_TESSELATION,
                                              landmark_drawing_spec=None,
                                              connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                else:
                    cv2.putText(frame, "No face detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                cv2.imshow(window, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
            return
    except Exception as e:
        print("Failed to run legacy FaceMesh:", e)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_face_landmarks()

