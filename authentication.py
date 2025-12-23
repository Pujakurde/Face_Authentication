# authenticate.py
"""
Stage 6: Face Authentication with Liveness

- Uses face_detection.py
- Uses MediaPipe Face Mesh
- SAME embedding logic as Stage-4 registration
- Blink-based liveness detection
- Access Granted / Denied
"""

import cv2
import mediapipe as mp
import numpy as np
import os
from face_detection import detect_faces

# ---------------- CONFIG ----------------
BASE_DIR = "face_embeddings"
MATCH_THRESHOLD = 0.55

# ---------------- MEDIAPIPE ----------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)

# ---------------- LIVENESS (BLINK) ----------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESHOLD = 0.21
BLINK_FRAMES = 2


class BlinkDetector:
    def __init__(self):
        self.counter = 0
        self.blinked = False

    def _ear(self, landmarks, idx):
        p = [landmarks[i] for i in idx]
        A = np.linalg.norm([p[1].x - p[5].x, p[1].y - p[5].y])
        B = np.linalg.norm([p[2].x - p[4].x, p[2].y - p[4].y])
        C = np.linalg.norm([p[0].x - p[3].x, p[0].y - p[3].y])
        return (A + B) / (2.0 * C)

    def update(self, landmarks):
        ear = (
            self._ear(landmarks, LEFT_EYE) +
            self._ear(landmarks, RIGHT_EYE)
        ) / 2

        if ear < EAR_THRESHOLD:
            self.counter += 1
        else:
            if self.counter >= BLINK_FRAMES:
                self.blinked = True
            self.counter = 0

        return self.blinked


blink_detector = BlinkDetector()

# ---------------- LOAD REGISTERED EMBEDDINGS ----------------
def load_registered_faces():
    users = {}

    for user in os.listdir(BASE_DIR):
        user_dir = os.path.join(BASE_DIR, user)
        if not os.path.isdir(user_dir):
            continue

        embeddings = []
        for file in os.listdir(user_dir):
            if file.endswith(".npy"):
                emb = np.load(os.path.join(user_dir, file))
                embeddings.append(emb)

        if embeddings:
            users[user] = embeddings

    return users


known_faces = load_registered_faces()

# ---------------- EMBEDDING (SAME AS REGISTRATION) ----------------
def extract_embedding(landmarks):
    emb = []
    for lm in landmarks:
        emb.extend([lm.x, lm.y, lm.z])

    emb = np.array(emb, dtype=np.float32)
    norm = np.linalg.norm(emb)
    if norm > 0:
        emb /= norm

    return emb


def l2_distance(a, b):
    return np.linalg.norm(a - b)


print("\n[INFO] Authentication Started")
print("[INFO] Blink to authenticate | Press Q to quit\n")

# ---------------- MAIN LOOP ----------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    faces = detect_faces(frame)

    # ---------- FACE COUNT CHECK ----------
    if len(faces) != 1:
        msg = "NO FACE" if len(faces) == 0 else "MULTIPLE FACES"
        cv2.putText(frame, msg, (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Authentication", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    # ---------- LANDMARKS ----------
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        continue

    landmarks = results.multi_face_landmarks[0].landmark

    # ---------- LIVENESS ----------
    live_ok = blink_detector.update(landmarks)

    status = "Blink to authenticate"
    color = (0, 255, 255)

    if live_ok:
        query_emb = extract_embedding(landmarks)

        best_user = None
        best_dist = float("inf")

        for user, refs in known_faces.items():
            for ref in refs:
                dist = l2_distance(ref, query_emb)
                if dist < best_dist:
                    best_dist = dist
                    best_user = user

        if best_dist < MATCH_THRESHOLD:
            status = f"ACCESS GRANTED: {best_user}"
            color = (0, 255, 0)
        else:
            status = "ACCESS DENIED"
            color = (0, 0, 255)

    cv2.putText(frame, status, (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Authentication", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
