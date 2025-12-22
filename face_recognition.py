import cv2
import mediapipe as mp
import numpy as np
import os

REGISTERED_FACE = "registered_face.npy"
MATCH_THRESHOLD = 0.6   # tune if needed

def extract_embedding(landmarks):
    embedding = []
    for lm in landmarks:
        embedding.extend([lm.x, lm.y, lm.z])
    return np.array(embedding)

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

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        status = "NO FACE"

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            current_embedding = extract_embedding(landmarks)

            distance = np.linalg.norm(
                registered_embedding - current_embedding
            )

            if distance < MATCH_THRESHOLD:
                status = "FACE MATCHED"
                color = (0,255,0)
            else:
                status = "FACE NOT MATCHED"
                color = (0,0,255)

            cv2.putText(frame, f"Distance: {distance:.3f}",
                        (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

        else:
            color = (0,0,255)

        cv2.putText(frame, status, (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

