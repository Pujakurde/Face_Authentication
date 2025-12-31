# Face Authentication System (Face-ID Style)

This project implements a Face-ID–style face authentication system using Python, OpenCV, and MediaPipe.  
It verifies users through a live webcam feed and incorporates liveness detection and anti-spoofing mechanisms to reduce the risk of unauthorized access using photos, videos, or screen replays.

The system works completely offline, without sending data to external servers, ensuring privacy-friendly biometric authentication.

---

## Key Objectives

- Verify that a real human face is present in front of the camera  
- Detect live facial behavior instead of static images  
- Prevent common spoofing attacks such as:
  - Printed photographs  
  - Mobile screen replays  
  - Static images  

---

## Features

- Real-time face detection  
- High-resolution facial landmark extraction (468 points)  
- Liveness detection using:
  - Eye blinking  
  - Head movement  
  - Natural facial motion across frames  
- Face-ID–style guided interaction  
- Anti-spoofing against photos and static images  
- Fully offline and privacy-friendly  

---

## System Overview

The system processes frames captured from a live webcam.  
Each frame is analyzed to ensure:

- A face is present  
- The face belongs to a live human  
- Facial motion is natural and continuous  

Only after passing these checks does the system allow authentication.

---

## System Architecture

```

Camera
↓
Face Detection
↓
Face Landmarks Extraction
↓
Face Registration
↓
Liveness Detection
↓
Authentication Decision

````

---

## Core Modules

### 1. Camera Module (camera.py)

Handles real-time video input.

Responsibilities:
- Capture live video frames from the webcam  
- Ensure stable camera initialization  
- Release camera resources safely  

Purpose:
- Provide continuous live input to the authentication system  

---

### 2. Face Detection Module (face_detection.py)

Detects whether a human face is present in each frame.

Responsibilities:
- Detect faces in real time  
- Ensure at least one face is visible  
- Handle single-face and multiple-face scenarios  

Technology Used:
- MediaPipe Face Detection  

Limitations:
- Does not identify the person  
- Does not perform liveness checks  
- Does not grant or deny access  

---

### 3. Face Landmarks Module (face_landmarks.py)

Extracts and tracks facial landmarks.

Details:
- Uses MediaPipe Face Mesh  
- Extracts 468 facial landmarks  
- Tracks facial geometry across frames  

Purpose:
- Eye blink detection  
- Head movement estimation  
- Support liveness verification  

---

### 4. Face-ID System (face_id_sys.py)

The main controller of the system.

Responsibilities:
- Integrates camera, face detection, and landmarks  
- Guides the user with Face-ID–style instructions  
- Displays real-time feedback  
- Manages registration and authentication logic  

---

## Liveness Detection (Anti-Spoofing)

Liveness detection ensures the detected face belongs to a real, live human and not a static image.

Why Liveness Detection Is Required:
- Without liveness detection, attackers could bypass authentication using printed photos, mobile screens, or static images  

Liveness Checks Implemented:
- Eye blink detection  
- Head movement detection (left, right, up)  
- Natural facial motion across multiple frames  

The system requires consistent live behavior before granting access.

---

## Spoofing Attacks Prevented

- Printed photographs  
- Static images  
- Screen replay attacks  

---

## Privacy and Security

- No face images are permanently stored  
- No biometric data is sent to the cloud  
- No external APIs are used  
- All processing occurs locally on the user’s system  

---

## Requirements

- Python 3.9 or higher  
- OpenCV  
- MediaPipe  
- NumPy  

---

## Installation

Install required dependencies:

```bash
pip install opencv-python mediapipe numpy
````

---

## How to Run

Run the main Face-ID system script:

```bash
python face_id_sys.py
```

Follow the on-screen instructions for registration and authentication.

---

## Limitations

* Uses a standard RGB webcam
* Very high-quality video replays may still bypass detection
* Depth sensing and infrared cameras are not supported

These limitations are common in software-only face authentication systems.

---

## Educational Purpose

This project is designed for:

* Computer Vision learning
* Understanding face authentication pipelines
* Studying liveness detection techniques

It is not intended for production-level biometric security systems.

---

## Author

Puja Kurde
Data Science Project

---

## License

This project is for educational purposes only.


- Review requirements.txt for correctness
```
