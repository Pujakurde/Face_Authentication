import cv2
import mediapipe as mp
import numpy as np
import os
from collections import deque

# ---------------- FILES ----------------
REGISTERED_FACE = "registered_face.npy"

# ---------------- LANDMARK INDEXES ----------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
NOSE_TIP = 1

# ---------------- THRESHOLDS ----------------
EAR_THRESHOLD = 0.25
EYE_CLOSED_FRAMES = 3
NOSE_MOVE_THRESHOLD = 0.01
MOTION_THRESHOLD = 0.0005
MATCH_THRESHOLD = 0.6

# ---------------- UTIL FUNCTIONS ----------------
def eye_aspect_ratio(eye, landmarks, w, h):
    pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye]
    A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (A + B) / (2.0 * C)

def extract_embedding(landmarks):
    emb = []
    for lm in landmarks:
        emb.extend([lm.x, lm.y, lm.z])
    return np.array(emb)

# ---------------- MAIN ----------------
def main():
    if not os.path.exists(REGISTERED_FACE):
        print("ERROR: No registered face found")
        return

    registered_embedding = np.load(REGISTERED_FACE)

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
    head_turn_detected = False

    nose_history = deque(maxlen=15)
    motion_history = deque(maxlen=10)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        status = "ACCESS DENIED"
        color = (0, 0, 255)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # -------- BLINK CHECK --------
            left_ear = eye_aspect_ratio(LEFT_EYE, landmarks, w, h)
            right_ear = eye_aspect_ratio(RIGHT_EYE, landmarks, w, h)
            ear = (left_ear + right_ear) / 2.0

            if ear < EAR_THRESHOLD:
                blink_counter += 1
            else:
                if blink_counter >= EYE_CLOSED_FRAMES:
                    blink_detected = True
                blink_counter = 0

            # -------- HEAD TURN CHECK --------
            nose_x = landmarks[NOSE_TIP].x
            nose_history.append(nose_x)

            if len(nose_history) >= 10:
                if abs(nose_history[-1] - nose_history[0]) > NOSE_MOVE_THRESHOLD:
                    head_turn_detected = True

            # -------- NATURAL MOTION --------
            motion = np.std([p.x for p in landmarks])
            motion_history.append(motion)
            natural_motion = np.mean(motion_history) > MOTION_THRESHOLD

            # -------- FACE MATCH --------
            current_embedding = extract_embedding(landmarks)
            distance = np.linalg.norm(registered_embedding - current_embedding)
            face_matched = distance < MATCH_THRESHOLD

            # -------- FINAL DECISION --------
            if blink_detected and head_turn_detected and natural_motion and face_matched:
                status = "ACCESS GRANTED"
                color = (0, 255, 0)

            # -------- DEBUG INFO --------
            cv2.putText(frame, f"Blink: {blink_detected}", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.putText(frame, f"HeadTurn: {head_turn_detected}", (30, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.putText(frame, f"Match: {face_matched}", (30, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        cv2.putText(frame, status, (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        cv2.imshow("Face Authentication", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

