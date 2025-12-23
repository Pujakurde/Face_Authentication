"""
face_detection.py

Stage 2: Face Detection module.

Responsibilities:
- Detect faces in a frame
- Return number of faces
- Return bounding box of detected face(s)

Uses:
- MediaPipe Face Detection (6 landmarks)

Does NOT:
- Identify the person
- Perform liveness detection
- Perform authentication
"""

import cv2
import mediapipe as mp
from typing import List, Tuple

# MediaPipe face detection
mp_face_detection = mp.solutions.face_detection


def detect_faces(frame, min_detection_confidence: float = 0.5):
    """
    Detect faces in a frame.

    Args:
        frame (np.ndarray): BGR image
        min_detection_confidence (float): confidence threshold

    Returns:
        faces (list): list of bounding boxes [(x, y, w, h)]
    """
    h, w, _ = frame.shape
    faces: List[Tuple[int, int, int, int]] = []

    with mp_face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=min_detection_confidence
    ) as detector:

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)

        if results.detections:
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)
                faces.append((x, y, bw, bh))

    return faces


def draw_faces(frame, faces):
    """
    Draw bounding boxes on frame.

    Args:
        frame (np.ndarray)
        faces (list): bounding boxes
    """
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)


def run_face_detection_test():
    """
    Standalone test for face detection.
    Press 'q' to quit.
    """
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return

    cv2.namedWindow("Face Detection", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = detect_faces(frame)
        draw_faces(frame, faces)

        cv2.putText(
            frame,
            f"Faces detected: {len(faces)}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_face_detection_test()
