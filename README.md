# 🧘 Real-Time Yoga Pose Evaluator via SVD & Procrustes Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-red.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Pose%20Estimation-00BCD4.svg)](https://developers.google.com/mediapipe)

> **A real-time computer vision application that evaluates human pose accuracy against ground-truth templates. Developed as an applied exploration of Numerical Methods, this project implements Jacobi Singular Value Decomposition (SVD) to solve the Orthogonal Procrustes problem for 3D pose registration.**

## 📌 Overview

Evaluating a user's pose against an instructor's template requires mathematically aligning two sets of 3D spatial coordinates, regardless of the user's position, scale, or camera angle. 

Instead of relying on black-box spatial alignment tools, this project implements a custom **Jacobi SVD algorithm** to calculate the optimal rotation matrix $R$ that minimizes the distance between the user's joints and the ground truth. The system processes a live webcam feed, extracts 12 core structural joints using the MediaPipe Tasks API, and provides real-time biomechanical feedback.

### 🎯 Key Features
* **Custom Numerical Methods:** Implements Jacobi SVD to compute rigid transformations and 3D pose registration.
* **Ground Truth Averaging:** Automatically processes multiple source images (5 distinct instructor photos across 6 stationary yoga poses) to generate a robust, averaged `.npy` structural matrix and `.json` biomechanical angle dictionary.
* **Occlusion Handling:** Utilizes MediaPipe's Heavy Pose Landmarker to estimate hidden or occluded joints during complex poses (e.g., One-Legged King Pigeon).
* **Live UI Validation:** Splits the screen to show the webcam feed alongside a dynamically scaled reference image, providing immediate visual feedback on joint alignment.

---

## 🧮 The Mathematics: Orthogonal Procrustes & SVD

To compare the user's pose matrix $A$ to the ground truth matrix $B$, the system must translate, scale, and rotate $A$ to best fit $B$. 

After centering the matrices at the origin and normalizing their scale, we find the optimal rotation matrix $R$ that minimizes the Frobenius norm:
$$R = \arg\min_{\Omega} || \Omega A - B ||_F$$

This is solved using **Singular Value Decomposition (SVD)** on the covariance matrix $M = B A^T$:
1. Calculate SVD of $M$: $M = U \Sigma V^T$ *(calculated via custom Jacobi eigenvalue algorithm)*
2. The optimal rotation matrix is: $R = U V^T$
3. Apply $R$ to the user's pose to calculate the final alignment error and joint-specific deviations.

---

## 🛠️ System Architecture

### 1. Ground Truth Generation (`src/data_pipeline/`)
* **`yoga_groundtruth_v2.py`**: Batch processes raw instructor images, extracts the 12 core joints (shoulders, elbows, hips, knees, ankles, wrists), filters out low-visibility anomalies, and computes an averaged 3D point cloud and reference angles.
* **`validate_groundtruth.py`**: Generates a 3D matplotlib plot of the resulting `.npy` matrices to visually verify the structural integrity of the averaged ground truth.

### 2. Pose Registration (`src/core/`)
* **`validate_poses.py`**: Executes the Procrustes analysis, comparing the custom Jacobi SVD implementation against NumPy's standard library to ensure numerical stability and accuracy.

### 3. Live Application (`src/app/`)
* **`live_validation_v2.py`**: The main execution script. Captures live webcam frames, extracts the user's 3D pose, scales the selected ground truth image to match the live feed's aspect ratio, and outputs the registration error in real-time.

---

## 🚀 How to Run (Quickstart)

### 1. Environment Setup
Install the required dependencies and download the MediaPipe Tasks model:
```bash
pip install -r requirements.txt
# Ensure pose_landmarker_heavy.task is placed in the models/ directory
```

### 2. Generate Ground Truth Data
Compile the instructor images into reference matrices:
```bash
python src/data_pipeline/yoga_groundtruth_v2.py
```

### 3. Launch the Live Trainer
Start the OpenCV webcam application (ensure your webcam is active):
```bash
python src/app/live_validation_v2.py
```
*Controls: Press `M` to return to the pose selection menu, `Q` to quit.*

---

## 📁 Repository Structure
```text
yoga_pose_evaluator/
├── models/
│   └── pose_landmarker_heavy.task
├── data/
│   ├── raw_images/
│   └── GT_Data/ 
├── src/
│   ├── core/
│   ├── data_pipeline/
│   ├── visualization/
│   └── app/
└── docs/
```

---
### Acknowledgments
*Developed as a capstone project for Numerical Methods.*
