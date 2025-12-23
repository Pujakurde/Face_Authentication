"""
camera.py

Webcam helper module for the Face Authentication system.

Responsibilities:
- Open and configure the webcam
- Provide frames to other modules
- Handle safe resource cleanup

This module does NOT perform:
- Face detection
- Recognition
- Liveness checks
"""

import cv2


def open_camera(index: int = 0, width: int = 640, height: int = 480):
    """
    Open the webcam and set preferred resolution.

    Args:
        index (int): Camera index (default = 0)
        width (int): Frame width
        height (int): Frame height

    Returns:
        cv2.VideoCapture: Opened camera object

    Raises:
        RuntimeError: If camera cannot be opened
    """
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        # Fallback without backend
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            raise RuntimeError("ERROR: Could not open webcam")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return cap


def release_camera(cap: cv2.VideoCapture):
    """Safely release camera and destroy OpenCV windows."""
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


def preview_camera():
    """
    Simple camera preview for testing.
    Press 'q' to exit.
    """
    try:
        cap = open_camera()
    except RuntimeError as e:
        print(e)
        return

    cv2.namedWindow("Camera Preview", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("WARNING: Failed to read frame")
            break

        cv2.imshow("Camera Preview", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    release_camera(cap)


if __name__ == "__main__":
    preview_camera()
