import cv2
import mediapipe as mp

def main():
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    face_detection = mp_face_detection.FaceDetection(
        model_selection=0,
        min_detection_confidence=0.5
    )

    # Open camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR (OpenCV) to RGB (MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces
        results = face_detection.process(rgb_frame)

        # Draw bounding boxes
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(frame, detection)

        # Show output
        cv2.imshow("Face Detection", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
