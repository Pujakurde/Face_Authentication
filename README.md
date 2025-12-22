# Face Authentication System (face_auth_recog)

A face authentication system that processes visual input from a camera to detect and analyze human faces for access control.

---

## System Overview

The system operates on image frames captured from a live camera feed.  
Each frame is processed independently to detect the presence and location of human faces.  
Detected face regions are passed to subsequent modules for further analysis.

---

## Core Modules

### 1. Face Detection Module

Face detection is responsible for identifying the presence and position of human faces in an image or video frame.

#### Frame Processing
A video stream consists of sequential frames.  
Each frame is analyzed independently.

#### Face Localization
When a frame contains one or more faces, the system assigns a separate bounding box to each detected face.

This module determines **where a face is located**, not **who the person is**.

#### Tool Used
- MediaPipe Face Detection (Google)

#### Input
- Live camera feed  
- Image frames extracted from video

#### Output
- Bounding boxes for detected faces  
- Detection confidence score  
- Basic facial keypoints

#### Characteristics
- Real-time face detection  
- Supports single and multiple faces  
- Resolution independent  
- Lightweight and efficient

#### Limitations
- Does not identify or recognize a person  
- Does not perform liveness detection  
- Does not grant or deny access independently  

These aspects are handled in later modules.

---

### 2. Face Landmarks Module

Face landmarks are specific points on a human face that represent important facial features such as the eyes, nose, mouth, jawline, and facial contours.

#### Landmark Representation
Unlike face detection, which only provides a bounding box, face landmarks describe the detailed geometry and movement of the face.  
Advanced landmark models detect hundreds of points across the face, enabling precise facial analysis.

#### Purpose of Face Landmarks
Face landmarks are used to:
- Track facial movements
- Detect eye blinking
- Estimate head pose
- Support liveness detection
- Help prevent photo and video spoofing attacks

#### Scope
Face landmarks do not identify a person.  
They describe facial structure and motion, which are essential for secure face authentication systems.

---

### 3. Liveness Detection Module

Liveness detection is used to verify whether the face presented to the camera belongs to a real, live human and not a fake input such as a printed photo, image, or video replay.

#### Importance of Liveness Detection
In face authentication systems, liveness detection is a critical security component.  
Without liveness checks, unauthorized users could gain access by showing photos or videos of a registered person.

#### Liveness Analysis
The liveness detection module analyzes natural facial behaviors and movements, including:
- Eye blinking
- Head movement
- Natural facial motion
- Depth and positional changes of facial features

#### Use of Face Landmarks
These behaviors are captured using facial landmarks, which provide detailed information about facial structure and movement in real time.

#### Spoofing Prevention
- Static images lack natural motion  
- Replayed videos show repetitive or unnatural movement patterns  

By detecting these differences, the system can distinguish a real human face from fake inputs.

#### Scope
Liveness detection does not identify a person’s identity.  
It only verifies that the detected face is real and live.  
Identity verification is handled in later stages of the system.

---

## References

- MediaPipe Face Detection Documentation  
  https://ai.google.dev/edge/mediapipe/solutions/vision/face_detector

### 4. Face Recognition and Authentication Module

Face recognition is responsible for identifying a person by comparing their facial features with previously registered data.

Instead of storing raw face images, the system represents each face using a numerical feature representation. This representation captures unique facial characteristics and allows efficient comparison between faces.

The authentication logic combines the results of multiple modules to make a final access decision. Access is granted only when all required conditions are satisfied.

#### Authentication Decision Criteria
Access is granted only if:
- A human face is detected
- Only one face is present in the frame
- The face passes liveness detection
- The face matches a registered identity

If any of the above conditions fail, access is denied.

#### Scope
Face recognition identifies who the person is, while authentication logic determines whether access should be allowed or denied. This module does not perform face detection or liveness detection independently.

