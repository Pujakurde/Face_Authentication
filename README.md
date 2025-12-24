# Face Authentication System (Face-ID Style)

This project implements an **iPhone-like Face Authentication system** using **Python, OpenCV, and MediaPipe**.
It verifies users through a **live camera feed** and includes **liveness detection and anti-spoofing mechanisms** to reduce the risk of unauthorized access using photos, videos, or screen replays.

The system works **completely offline**, without storing biometric data or sending information to external servers.

---

## Key Objectives

* Verify that a **real human face** is present in front of the camera
* Detect **live facial behavior** instead of static images
* Prevent common spoofing attacks such as:

  * Printed photos
  * Mobile screen replays
  * Static images

---

## Features

* Real-time face detection
* High-resolution facial landmark extraction (468 points)
* Liveness detection using:

  * Eye blinking
  * Head movement
  * Natural facial motion
* Face-ID–style guided interaction
* Anti-spoofing against photos and static images
* Fully offline and privacy-friendly

---

## System Overview

The system processes frames captured from a **live webcam**.
Each frame is independently analyzed to ensure that:

1. A face is present
2. The face belongs to a live human
3. Facial motion is natural and continuous

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
```

---

## Core Modules

---

### 1. Camera Module (`camera.py`)

The camera module is responsible for handling video input.

**Responsibilities**

* Capture real-time video frames from the webcam
* Ensure stable camera initialization
* Release camera resources safely

**Purpose**

* Provide live input to the authentication system

---

### 2. Face Detection Module (`face_detection.py`)

This module detects whether a human face is present in each frame.

**Responsibilities**

* Detect faces in real time
* Ensure at least one face is visible
* Handle single-face and multiple-face scenarios

**Technology Used**

* MediaPipe Face Detection

**Output**

* Face bounding boxes
* Detection confidence

**Limitations**

* Does not identify the person
* Does not perform liveness checks
* Does not grant or deny access

---

### 3. Face Landmarks Module (`face_landmarks.py`)

Face landmarks represent key facial points such as eyes, nose, lips, and jawline.

**Details**

* Uses MediaPipe Face Mesh
* Extracts 468 facial landmarks
* Tracks facial geometry across frames

**Purpose**

* Eye blink detection
* Head movement estimation
* Support liveness verification

---

### 4. Face-ID System Controller (`face_id_sys.py`)

This is the main controller of the system.

**Responsibilities**

* Integrates camera, face detection, and landmarks
* Guides the user with Face-ID–style instructions
* Displays real-time feedback on the screen
* Manages registration and authentication logic

---

## Liveness Detection (Anti-Spoofing)

Liveness detection ensures that the detected face belongs to a **real, live human** and not a static image.

### Why Liveness Detection Is Required

Without liveness detection, attackers could bypass authentication using:

* Printed photos
* Mobile screens
* Static images

---

### Liveness Checks Implemented

* **Eye blinking detection**
* **Head movement detection** (left, right, up)
* **Natural facial motion across frames**

The system requires consistent live behavior over multiple frames before allowing access.

---

### Spoofing Attacks Prevented

* Printed photographs
* Static images
* Screen replay attacks

---

### Scope of Liveness Detection

* Confirms **live presence**
* Does **not identify** who the person is
* Works entirely offline

---

## Privacy and Security

* No face images are permanently stored
* No biometric data is sent to the cloud
* No external APIs are used
* All processing happens locally on the user’s system

---

## Requirements

* Python 3.9+
* OpenCV
* MediaPipe
* NumPy

---

## Installation

Install the required dependencies:

```bash
pip install opencv-python mediapipe numpy
```

---

## How to Run

Run the main Face-ID system script:

```bash
python face_id_sys.py
```

Follow the on-screen **Face-ID–style instructions** for registration and authentication.

---

## Limitations

* The system uses a **standard RGB webcam**
* Extremely high-quality video replays may still bypass detection
* Depth sensing and infrared cameras are not supported

These limitations are common in software-only face authentication systems.

---

## Educational Purpose

This project is designed for:

* Computer Vision learning
* Understanding face authentication pipelines
* Studying liveness detection techniques

It is **not intended for production-level biometric security systems**.

---

## Author

**Puja Kurde**
Data Science & Computer Vision Project

---

