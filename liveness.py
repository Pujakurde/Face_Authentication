import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# -------- LANDMARK INDEXES --------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
NOSE_TIP = 1

# -------- PARAMETERS --------
EAR_THRESHOLD = 0.25
EYE_CLOSED_FRAMES = 3
NOSE_MOVE_THRESHOLD = 0.015
MOTION_THRESHOLD = 0.002

# -------- UTIL FUNCTIONS --------
def eye_aspect_ratio(eye, landmarks, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C)

# -------- MAIN --------
def main():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)

    blink_counter = 0
    blink_detected = False
    nose_history = deque(maxlen=15)
    motion_history = deque(maxlen=10)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        status = "FAKE / NOT LIVE"

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            # -------- BLINK CHECK --------
            left_ear = eye_aspect_ratio(LEFT_EYE, lm, w, h)
            right_ear = eye_aspect_ratio(RIGHT_EYE, lm, w, h)
            ear = (left_ear + right_ear) / 2.0

            if ear < EAR_THRESHOLD:
                blink_counter += 1
            else:
                if blink_counter >= EYE_CLOSED_FRAMES:
                    blink_detected = True
                blink_counter = 0

            # -------- HEAD TURN CHECK --------
            nose_x = lm[NOSE_TIP].x
            nose_history.append(nose_x)

            head_turn = False
            if len(nose_history) >= 10:
                if abs(nose_history[-1] - nose_history[0]) > NOSE_MOVE_THRESHOLD:
                    head_turn = True

            # -------- NATURAL MOVEMENT CHECK --------
            motion = np.std([p.x for p in lm])
            motion_history.append(motion)
            natural_motion = np.std(motion_history) > MOTION_THRESHOLD

            # -------- FINAL DECISION --------
            if blink_detected and head_turn and natural_motion:
                status = "LIVE FACE"

        cv2.putText(frame, status, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0) if status == "LIVE FACE" else (0, 0, 255), 2)

        cv2.imshow("Liveness Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
