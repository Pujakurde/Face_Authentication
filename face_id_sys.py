# face_id_sys.py
"""
Face ID System (FINAL SECURE VERSION)

Features:
- Face Registration
- Face Authentication
- Continuous liveness (blink-based)
- Real head rotation (not frame movement)
- Prevents blink-then-photo attack
"""

import cv2
import mediapipe as mp
import numpy as np
import os
from face_detection import detect_faces

# CONFIG 
BASE_DIR = "face_embeddings"
MATCH_THRESHOLD = 0.85
POSES = ["front", "left", "right", "up"]

EAR_THRESHOLD = 0.21
BLINK_FRAMES = 2
LIVENESS_VALID_FRAMES = 15   # ~0.5 sec window

os.makedirs(BASE_DIR, exist_ok=True)

#  MEDIAPIPE 
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# BLINK + CONTINUOUS LIVENESS 
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


class LivenessDetector:
    def __init__(self):
        self.eye_counter = 0
        self.live_frames = 0

    def _ear(self, lm, idx):
        p = [lm[i] for i in idx]
        A = np.linalg.norm([p[1].x - p[5].x, p[1].y - p[5].y])
        B = np.linalg.norm([p[2].x - p[4].x, p[2].y - p[4].y])
        C = np.linalg.norm([p[0].x - p[3].x, p[0].y - p[3].y])
        return (A + B) / (2.0 * C)

    def update(self, landmarks):
        ear = (self._ear(landmarks, LEFT_EYE) +
               self._ear(landmarks, RIGHT_EYE)) / 2

        if ear < EAR_THRESHOLD:
            self.eye_counter += 1
        else:
            if self.eye_counter >= BLINK_FRAMES:
                self.live_frames = LIVENESS_VALID_FRAMES
            self.eye_counter = 0

        if self.live_frames > 0:
            self.live_frames -= 1
            return True

        return False


# REAL HEAD POSE 
def detect_head_pose(landmarks):
    nose = landmarks[1]
    left_eye = landmarks[33]
    right_eye = landmarks[263]

    nose_left = abs(nose.x - left_eye.x)
    nose_right = abs(right_eye.x - nose.x)

    eye_avg_y = (left_eye.y + right_eye.y) / 2
    pitch = eye_avg_y - nose.y

    if nose_left - nose_right > 0.03:
        return "right"
    if nose_right - nose_left > 0.03:
        return "left"
    if pitch > 0.03:
        return "up"

    return "front"


#EMBEDDING
def extract_embedding(landmarks):
    emb = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    emb = emb.flatten()
    emb /= np.linalg.norm(emb) + 1e-6
    return emb


# LOAD USERS
def load_users():
    users = {}
    for user in os.listdir(BASE_DIR):
        user_dir = os.path.join(BASE_DIR, user)
        if not os.path.isdir(user_dir):
            continue

        vecs = []
        for f in os.listdir(user_dir):
            if f.endswith(".npy"):
                v = np.load(os.path.join(user_dir, f))
                v /= np.linalg.norm(v) + 1e-6
                vecs.append(v)

        if vecs:
            users[user] = np.mean(vecs, axis=0)

    return users


# MODE SELECT 
mode = input("Select mode (register / authenticate): ").strip().lower()
cap = cv2.VideoCapture(0)
liveness = LivenessDetector()

if mode == "register":
    username = input("Enter username: ").strip()
    user_dir = os.path.join(BASE_DIR, username)
    os.makedirs(user_dir, exist_ok=True)
    step = 0
    print("\n[INFO] Registration started")
    print("[INFO] Blink and turn head as instructed\n")

elif mode == "authenticate":
    known_users = load_users()
    print("\n[INFO] Authentication started")
    print("[INFO] Blink and stay live to authenticate\n")

else:
    print("Invalid mode")
    exit()

# MAIN LOOP 
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    faces = detect_faces(frame)

    if len(faces) != 1:
        cv2.putText(frame, "ONE FACE ONLY", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Face ID System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        continue

    landmarks = res.multi_face_landmarks[0].landmark
    is_live_now = liveness.update(landmarks)
    pose = detect_head_pose(landmarks)

    status = "Blink to prove liveness"
    color = (0, 255, 255)

    # REGISTER 
    if mode == "register":
        required_pose = POSES[step]
        status = f"Turn head: {required_pose}"

        if is_live_now and pose == required_pose:
            emb = extract_embedding(landmarks)
            np.save(os.path.join(user_dir, f"{required_pose}.npy"), emb)
            print(f"[REGISTERED] {required_pose}")
            step += 1
            liveness = LivenessDetector()

            if step == len(POSES):
                print("\n[INFO] Registration completed")
                break

    # AUTHENTICATE 
    else:
        if is_live_now:
            query = extract_embedding(landmarks)
            best_user, best_dist = None, 999

            for user, ref in known_users.items():
                d = np.linalg.norm(ref - query)
                if d < best_dist:
                    best_dist, best_user = d, user

            if best_dist < MATCH_THRESHOLD:
                status = f"ACCESS GRANTED: {best_user}"
                color = (0, 255, 0)
            else:
                status = "ACCESS DENIED"
                color = (0, 0, 255)

        else:
            status = "LIVENESS LOST"
            color = (0, 0, 255)

    cv2.putText(frame, status, (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.putText(frame, f"Pose: {pose}", (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Face ID System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()