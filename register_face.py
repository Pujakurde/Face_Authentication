import cv2
import mediapipe as mp
import numpy as np
import os

SAVE_PATH = "registered_face.npy"

def extract_embedding(landmarks):
    embedding = []
    for lm in landmarks:
        embedding.extend([lm.x, lm.y, lm.z])
    return np.array(embedding)

def main():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)

    print("INFO: Press 'r' to register face (LIVE face only)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            cv2.putText(frame, "Face detected", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                embedding = extract_embedding(landmarks)
                np.save(SAVE_PATH, embedding)
                print("SUCCESS: Face registered")
                break

        cv2.imshow("Face Registration", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
