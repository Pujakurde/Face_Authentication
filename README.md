# Face Authentication System

This project implements an iPhone-like face authentication system with liveness detection and anti-spoofing capabilities using Python, OpenCV, and MediaPipe.  
The system authenticates users using a live camera feed and denies access to spoofing attempts such as photos, videos, or screen replays.

---

## Features

- Real-time face detection  
- Facial landmark extraction  
- Liveness detection (eye blink, head movement, natural facial motion)  
- Face registration for authorized users  
- Face recognition using numerical embeddings  
- Access control (Access Granted / Access Denied)  
- Interactive Face-ID–style UI feedback  
- Fully offline system (no cloud or external APIs)

---

## System Overview

The system operates on image frames captured from a live camera feed.  
Each frame is processed independently to detect and analyze human faces.  
Detected face regions are passed through multiple verification stages before making a final authentication decision.

---
## System Architecture

Camera --> Face Detection --> Face Landmarks --> Liveness Detection --> Face Recognition --> Authentication Decision --> UI Feedback


---

## Core Modules

---

### 1. Camera Module (`camera.py`)

- Captures live video frames from the webcam  
- Provides real-time input to the system  
- Handles safe initialization and release of camera resources  

---

### 2. Face Detection Module (`face_detection.py`)

Face detection identifies the presence and position of human faces in an image or video frame.

**Responsibilities**
- Detect faces in each frame  
- Draw bounding boxes around detected faces  
- Support single and multiple face detection  

**Tool Used**
- MediaPipe Face Detection (Google)

**Output**
- Bounding boxes  
- Detection confidence score  
- 6 basic facial landmarks  

**Limitations**
- Does not identify the person  
- Does not perform liveness detection  
- Does not make authentication decisions  

---

### 3. Face Landmarks Module (`face_landmarks.py`)

Face landmarks represent specific points on the human face such as the eyes, nose, mouth, jawline, and facial contours.

**Details**
- Uses MediaPipe Face Mesh  
- Extracts 468 detailed facial landmarks  
- Tracks facial geometry and movement in real time  

**Purpose**
- Eye blink detection  
- Head pose estimation  
- Natural facial motion tracking  
- Support liveness detection and recognition  

---

### 4. Face Registration Module (`register_face.py`)

- Registers an authorized user  
- Requires successful liveness verification  
- Converts facial landmarks into numerical feature embeddings  
- Stores embeddings locally (no raw face images)  

---

### 5. Liveness Detection Module (`liveness.py`)

Liveness detection verifies whether the detected face belongs to a real, live human.

**Why It Is Needed**  
Without liveness detection, attackers could gain access using printed photos, static images, or video replays.

**Liveness Checks**
- Eye blinking  
- Head movement (left / right)  
- Natural facial motion across frames  

**Spoofing Attacks Prevented**
- Printed photos  
- Static images  
- Video or screen replay attacks  

**Scope**
- Verifies live presence only  
- Does not identify the person  

---

### 6. Face Recognition Module (`face_recognition.py`)

- Compares live face embeddings with registered embeddings  
- Uses numerical distance comparison  
- Determines whether the face matches an authorized user  

---

### 7. Authentication Module (`authentication.py`)

This module makes the final access decision.

**Access is granted only if**
- A human face is detected  
- Exactly one face is present  
- Liveness detection passes  
- Face matches registered identity  

If any condition fails, access is denied.

---

### 8. Interactive UI

The system provides Face-ID–style UI feedback:

- Live camera feed  
- Face guide overlay  
- Status messages:
  - FACE NOT DETECTED  
  - MULTIPLE FACES  
  - FAKE FACE  
  - AUTHENTICATING  
  - ACCESS GRANTED  
  - ACCESS DENIED  
- Color-based feedback (red / yellow / green)  

UI is implemented using OpenCV overlays and is extendable to Tkinter.
