"""
face_recognition.py

Stage 6 — Face Recognition Module

Responsibilities:
- Convert facial landmarks to embeddings
- Load registered face embeddings
- Match live embeddings against registered users

Does NOT:
- Open camera
- Draw UI
- Perform liveness detection
"""

import os
import numpy as np
from typing import Optional, Tuple

# ---------------- STORAGE ----------------

FACES_DIR = "faces"
os.makedirs(FACES_DIR, exist_ok=True)

MATCH_THRESHOLD = 0.55


# ---------------- EMBEDDINGS ----------------

def extract_embedding_from_landmarks(landmarks) -> np.ndarray:
    """
    Convert landmarks into a normalized embedding.
    landmarks: iterable with .x .y .z
    """
    emb = []
    for lm in landmarks:
        emb.extend([lm.x, lm.y, getattr(lm, "z", 0.0)])

    emb = np.array(emb, dtype=np.float32)

    norm = np.linalg.norm(emb)
    if norm > 0:
        emb /= norm

    return emb


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


# ---------------- LOAD / SAVE ----------------

def load_known_faces():
    """
    Loads all face embeddings from faces/*.npy
    Each file can contain multiple embeddings (rotation support)
    """
    known = {}

    for file in os.listdir(FACES_DIR):
        if file.endswith(".npy"):
            name = file.replace(".npy", "")
            path = os.path.join(FACES_DIR, file)
            known[name] = np.load(path)

    return known


def save_face(name: str, embedding: np.ndarray):
    """
    Save first embedding for a user
    """
    path = os.path.join(FACES_DIR, f"{name}.npy")
    if os.path.exists(path):
        raise ValueError("Name already exists")

    np.save(path, embedding[np.newaxis, :])


def append_face(name: str, embedding: np.ndarray):
    """
    Append rotation embeddings (left/right/up/down)
    """
    path = os.path.join(FACES_DIR, f"{name}.npy")
    data = np.load(path)
    data = np.vstack([data, embedding])
    np.save(path, data)


# ---------------- RECOGNITION ----------------

def recognize_face(embedding: np.ndarray) -> Tuple[Optional[str], float]:
    """
    Match live embedding against registered faces
    """
    known_faces = load_known_faces()

    best_name = None
    best_score = float("inf")

    for name, stored_embeddings in known_faces.items():
        for ref in stored_embeddings:
            dist = l2_distance(ref, embedding)
            if dist < best_score:
                best_score = dist
                best_name = name

    if best_score < MATCH_THRESHOLD:
        return best_name, best_score

    return None, best_score
