# Face Authentication System

This project implements an **iPhone-like Face Authentication system** with **liveness detection and anti-spoofing** using **Python, OpenCV, and MediaPipe**.  
The system verifies users using a **live camera feed** and prevents spoofing attempts such as **printed photos, static images, or screen replays**.

The entire system works **fully offline** without using any cloud services or external APIs.

---

## Features

- Real-time face detection  
- Facial landmark extraction (468 landmarks)  
- Liveness detection (eye blink, head movement, natural facial motion)  
- Face-ID–style guided interaction  
- Anti-spoofing against photos and static images  
- Fully offline processing  

---

## System Overview

The system processes image frames captured from a live webcam.  
Each frame is analyzed independently to detect a face, extract facial landmarks, and verify liveness before providing feedback to the user.

---

## System Architecture

Camera --> Face Detection --> Face Landmarks --> Liveness Detection --> Authentication

---


---

## Core Modules

---

### 1. Camera Module (`camera.py`)

- Captures live video frames from the webcam  
- Provides real-time input to the system  
- Handles safe initialization and release of camera resources  

---

### 2. Face Detection Module (`face_detection.py`)

Face detection identifies the presence and position of human faces in each frame.

**Responsibilities**
- Detect faces in real time  
- Ensure face presence  
- Handle single and multiple face scenarios  

**Tool Used**
- MediaPipe Face Detection  

**Output**
- Face bounding boxes  
- Detection confidence  

**Limitations**
- Does not identify the person  
- Does not perform liveness detection  
- Does not make authentication decisions  

---

### 3. Face Landmarks Module (`face_landmarks.py`)

Face landmarks represent key points on the human face such as eyes, nose, mouth, and jawline.

**Details**
- Uses MediaPipe Face Mesh  
- Extracts 468 detailed facial landmarks  
- Tracks facial geometry and movement in real time  

**Purpose**
- Eye blink detection  
- Head movement estimation  
- Support liveness verification  

---

### 4. Face-ID System Controller (`face_id_sys.py`)

- Main controller of the system  
- Integrates camera, face detection, and landmarks  
- Guides the user with Face-ID–style instructions  
- Displays real-time UI feedback  

---

## Liveness Detection (Anti-Spoofing)

Liveness detection verifies whether the detected face belongs to a **real, live human**.

**Why It Is Needed**  
Without liveness detection, attackers could gain access using photos or static images.

**Liveness Checks**
- Eye blinking  
- Head movement (left / right / up)  
- Natural facial motion across frames  

**Spoofing Attacks Prevented**
- Printed photos  
- Static images  
- Screen replay attempts  

**Scope**
- Verifies live presence only  
- Does not identify the person  

---

## Privacy & Security

- No face images stored  
- No biometric embeddings stored  
- No cloud or external API usage  
- Fully offline processing  
---

## How to Run

1. Install dependencies:
```bash
pip install opencv-python mediapipe numpy
2. Run the script:
```bash
python face_id_sys.py
3. Follow the on-screen Face-ID–style instructions.
