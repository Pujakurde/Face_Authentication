"""
Face ID System — FINAL iPhone-Like Version (Webcam Safe)

Features:
- Face detection (1 face only)
- Multi-pose registration
- Landmark-based face embedding (fixed 468)
- Blink OR mouth-movement liveness
- Continuous liveness window
- Natural head movement detection
- Attention check (front pose)
- Stable multi-frame authentication
- Different exit messages for GRANTED vs DENIED
- Authentication logging to auth_log.txt
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time
from datetime import datetime
from face_detection import detect_faces

# ================= CONFIG =================
BASE_DIR = "face_embeddings"
LOG_FILE = "auth_log.txt"

MATCH_THRESHOLD = 0.95
STABLE_FRAMES_REQUIRED = 8
AUTH_TIME_WINDOW = 1.0

POSES = ["front", "left", "right", "up"]

EAR_THRESHOLD = 0.27
LIVENESS_WINDOW = 10

MOVEMENT_THRESHOLD = 0.001
MOVEMENT_WINDOW = 15

FACE_LANDMARK_COUNT = 468

# Mouth fallback
MOUTH_TOP = 13
MOUTH_BOTTOM = 14
MOUTH_THRESHOLD = 0.03

os.makedirs(BASE_DIR, exist_ok=True)

# ================= MEDIAPIPE =================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# ================= LIVENESS =================
class LivenessDetector:
    def __init__(self):
        self.live_frames = 0
        self.nose_history = []
        self.is_moving = False

    def _ear(self, lm, idx):
        p = [lm[i] for i in idx]
        A = np.linalg.norm([p[1].x - p[5].x, p[1].y - p[5].y])
        B = np.linalg.norm([p[2].x - p[4].x, p[2].y - p[4].y])
        C = np.linalg.norm([p[0].x - p[3].x, p[0].y - p[3].y])
        return (A + B) / (2.0 * C + 1e-6)

    def update(self, landmarks):
        # ---- Natural movement ----
        nose = landmarks[1]
        self.nose_history.append([nose.x, nose.y])
        if len(self.nose_history) > MOVEMENT_WINDOW:
            self.nose_history.pop(0)

        if len(self.nose_history) == MOVEMENT_WINDOW:
            std = np.std(self.nose_history, axis=0).mean()
            self.is_moving = std > MOVEMENT_THRESHOLD

        # ---- Blink ----
        ear = (self._ear(landmarks, LEFT_EYE) +
               self._ear(landmarks, RIGHT_EYE)) * 0.5

        # ---- Mouth open ----
        mouth_open = abs(
            landmarks[MOUTH_TOP].y - landmarks[MOUTH_BOTTOM].y
        ) > MOUTH_THRESHOLD

        if ear < EAR_THRESHOLD or mouth_open:
            self.live_frames = LIVENESS_WINDOW
            return True

        if self.live_frames > 0:
            self.live_frames -= 1
            return True

        return False

# ================= HEAD POSE =================
def detect_head_pose(landmarks):
    nose = landmarks[1]
    le = landmarks[33]
    re = landmarks[263]

    dx_l = abs(nose.x - le.x)
    dx_r = abs(re.x - nose.x)
    pitch = ((le.y + re.y) * 0.5) - nose.y

    if (dx_l - dx_r) > 0.03:
        return "right"
    if (dx_r - dx_l) > 0.03:
        return "left"
    if pitch > 0.03:
        return "up"
    return "front"

def attention_check(pose):
    return pose == "front"

# ================= EMBEDDING =================
def extract_embedding(landmarks):
    pts = []
    for i in range(FACE_LANDMARK_COUNT):
        lm = landmarks[i]
        pts.extend([lm.x, lm.y, lm.z])

    emb = np.array(pts, dtype=np.float32)
    emb /= np.linalg.norm(emb) + 1e-6
    return emb

# ================= LOAD USERS =================
def load_users():
    users = {}
    for user in os.listdir(BASE_DIR):
        udir = os.path.join(BASE_DIR, user)
        if not os.path.isdir(udir):
            continue

        vecs = []
        for f in os.listdir(udir):
            if f.endswith(".npy"):
                v = np.load(os.path.join(udir, f))
                if v.shape[0] == FACE_LANDMARK_COUNT * 3:
                    vecs.append(v)

        if len(vecs) >= 2:
            users[user] = np.mean(vecs, axis=0)

    return users

# ================= LOGGING =================
def log_auth(result, user, score):
    with open(LOG_FILE, "a") as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{ts} | {result} | user={user} | score={score:.3f}\n")

# ================= MODE =================
mode = input("Select mode (register / authenticate): ").strip().lower()
cap = cv2.VideoCapture(0)

liveness = LivenessDetector()
stable_count = 0

access_granted = False
decision_made = False
matched_user = "Unknown"
best_score = 999.0

if mode == "register":
    username = input("Enter username: ").strip()
    user_dir = os.path.join(BASE_DIR, username)
    os.makedirs(user_dir, exist_ok=True)
    step = 0
    print("[INFO] Registration started")

elif mode == "authenticate":
    users = load_users()
    auth_start = None
    print("[INFO] Authentication started")

else:
    print("Invalid mode")
    exit()

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    faces = detect_faces(frame)

    if len(faces) != 1:
        cv2.putText(frame, "ONE FACE ONLY", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Face ID", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            log_auth("DENIED", "Unknown", 999.0)
            break
        continue

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_mesh.process(rgb)
    if not res.multi_face_landmarks:
        continue

    landmarks = res.multi_face_landmarks[0].landmark
    pose = detect_head_pose(landmarks)
    live = liveness.update(landmarks)
    emb = extract_embedding(landmarks)

    status = ""
    color = (0, 255, 255)

    # ---------- REGISTER ----------
    if mode == "register":
        req = POSES[step]
        status = f"Turn head: {req}"

        if live and pose == req:
            np.save(os.path.join(user_dir, f"{req}.npy"), emb)
            step += 1
            liveness = LivenessDetector()
            time.sleep(0.5)

            if step >= len(POSES):
                print("[INFO] Registration completed")
                break

    # ---------- AUTHENTICATE ----------
    else:
        if decision_made:
            if access_granted:
                status = "ACCESS GRANTED | PRESS E TO EXIT"
                color = (0, 255, 0)
            else:
                status = "ACCESS DENIED | PRESS Q TO QUIT"
                color = (0, 0, 255)

        elif not attention_check(pose):
            status = "LOOK AT CAMERA"
            color = (0, 0, 255)
            stable_count = 0
            auth_start = None

        elif not liveness.is_moving:
            status = "MOVE HEAD NATURALLY"
            color = (0, 0, 255)
            stable_count = 0
            auth_start = None

        elif not live:
            status = "BLINK / MOVE MOUTH"
            color = (0, 0, 255)
            stable_count = 0
            auth_start = None

        else:
            if auth_start is None:
                auth_start = time.time()

            if (time.time() - auth_start) >= AUTH_TIME_WINDOW:
                best_user = None
                best_dist = 999.0

                for u, ref in users.items():
                    d = np.linalg.norm(ref - emb)
                    if d < best_dist:
                        best_dist = d
                        best_user = u

                best_score = best_dist
                matched_user = best_user if best_user else "Unknown"

                if best_dist < MATCH_THRESHOLD:
                    stable_count += 1
                else:
                    stable_count = 0

                if stable_count >= STABLE_FRAMES_REQUIRED:
                    status = f"ACCESS GRANTED: {best_user}"
                    color = (0, 255, 0)
                    access_granted = True
                    decision_made = True
                    log_auth("GRANTED", best_user, best_dist)
                else:
                    status = "VERIFYING..."
                    color = (0, 255, 255)

    cv2.putText(frame, status, (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"Pose: {pose}", (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Face ID", frame)

    key = cv2.waitKey(1) & 0xFF
    if decision_made:
        if access_granted and key == ord('e'):
            print("[INFO] Access granted — exiting Face ID system")
            break
        elif not access_granted and key == ord('q'):
            log_auth("DENIED", matched_user, best_score)
            print("[WARNING] Access denied — system terminated")
            break

cap.release()
cv2.destroyAllWindows()
