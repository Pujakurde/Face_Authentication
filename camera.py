import cv2

def main():
    # Open the default camera (0)
    cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    while True:
        # Read one frame from the camera
        ret, frame = cap.read()

        # If frame is not read correctly, exit
        if not ret:
            print("Error: Cannot read frame")
            break

        # Display the frame
        cv2.imshow("Live Camera Feed", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

