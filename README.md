
# Face Authentication System

A **secure, webcam-based Face Authentication system** that verifies user identity using **facial landmarks, liveness detection, and natural head movement**, inspired by **iPhone Face ID–like behavior**, without using depth sensors or external APIs.

---

## Project Overview

This project implements a **Face Authentication system** designed to:

* Authenticate **only a live, registered human face**
* Prevent access using **photos, printed images, or static replays**
* Allow exit or continuation **only after a final authentication decision**

The system is developed using **Python, OpenCV, and MediaPipe**, and is suitable for **academic projects and real-time webcam environments**.

---

##  Key Features

* **Single Face Enforcement**
  Ensures only one face is present in the frame.

* **Landmark-Based Face Representation**
  Uses **468 facial landmarks** from MediaPipe FaceMesh.

* **Liveness Detection**

  * Eye blink detection
  * Mouth movement detection
  * Continuous liveness window

* **Natural Head Movement Detection**

  * Detects real, subtle head motion
  * Prevents static photo spoofing

* **Attention Check**

  * Requires the user to look **directly at the camera**

* **Stable Multi-Frame Authentication**

  * Authentication decision based on multiple consecutive frames

* **Controlled Exit Logic**

  * **Access Granted → Press `E` to exit**
  * **Access Denied → Press `Q` to quit**

* **Authentication Logging**

  * Logs results (`GRANTED / DENIED`) with timestamp into `auth_log.txt`

---

## Technologies Used

* **Python 3.10**
* **OpenCV**
* **MediaPipe (FaceMesh)**
* **NumPy**

> No external APIs, cloud services, or deep learning models are used.

---

## Project Structure

```
Face_Authentication/
│
├── camera.py
├── face_detection.py
├── face_id_sys.py        # Main authentication logic
├── face_landmarks.py
├── face_recognition.py
├── README.md
├── requirements.txt
└── .gitignore
```

---

## How to Run the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the system

```bash
python face_id_sys.py
```

### 3. Select mode

```text
register       → Register a new user
authenticate   → Authenticate an existing user
```

---

## Authentication Flow

1. User faces the camera
2. System checks:

   * One face only
   * Attention (front pose)
   * Natural head movement
   * Blink or mouth movement
3. Face embedding is matched with registered users
4. Final decision:

   * **Access Granted** → User can exit & continue system
   * **Access Denied** → System terminates

---

## Security Considerations

### Prevented Attacks

* Printed photo attacks
* Static image replay
* Simple video replays

### Known Limitation

* **Live video-call (VC) attacks** may succeed under controlled cooperation
  (This limitation exists in most webcam-only systems and requires hardware depth sensors to fully prevent.)

---

## Academic Note

This project is designed for:

* **Academic demonstrations**
* **Real-time webcam authentication**
* **Understanding biometric security concepts**

It does **not claim hardware-level security** like iPhone Face ID.

---

## Author

**Puja Kurde**
B.Tech – Data Science
Face Authentication Project

---

## License

This project is for **educational purposes only**.
