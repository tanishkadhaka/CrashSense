# CrashSense — Real-Time Car Crash Detection from Surveillance Footage

A computer vision pipeline that detects vehicle collisions in surveillance/CCTV video by combining **object detection, multi-object tracking, motion/optical-flow analysis, and 3D CNN-based action classification**.

> This fork focuses on cleaning up the pipeline, packaging it for reproducibility, and documenting the CV/ML techniques involved. Original authorship credited below.

## Overview

Surveillance cameras generate enormous volumes of footage that are impractical to monitor manually. This project processes video streams to automatically flag potential vehicle collisions, so that only relevant clips need human review — a smaller-scale version of the kind of real-time vision/analytics system used in traffic monitoring and industrial safety inspection.

**Pipeline:**
1. **Detection** — YOLOv4 detects vehicles frame by frame.
2. **Tracking** — DeepSORT assigns persistent IDs to each vehicle across frames, so trajectories can be analyzed over time.
3. **Motion features** — Farneback optical flow + Violent Flow (VIF) descriptors capture sudden, high-magnitude motion changes characteristic of a collision.
4. **Classification** — A 3D Convolutional Neural Network (PyTorch, `model_ft.pth`) classifies short clip segments as crash / no-crash.
5. **Alerting (demo)** — A Streamlit app runs the pipeline on an uploaded/streamed video and surfaces flagged clips.

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python |
| Detection | YOLOv4 |
| Tracking | DeepSORT |
| Deep Learning | PyTorch, TensorFlow |
| Classical CV | OpenCV, Farneback Optical Flow |
| Data handling | NumPy, Pandas, Albumentations |
| Demo/App | Streamlit |
| Packaging | setuptools (`CarCrashDetection_3D_Convolution`) |

## Repository Structure

```
.
├── accidentUtils/              # Detection + tracking pipeline (YOLOv4, DeepSORT, VIF, optical flow) + Streamlit app
├── CarCrashDetection_3D_Convolution/  # Packaged 3D-CNN crash classifier (installable module)
├── deep_sort/                  # DeepSORT tracker implementation
├── data_creating_vif/          # Notebooks/scripts for building the VIF training dataset
├── output_results/             # Sample inference outputs
├── Car_Crash_Detection_VIF_3D_V3.ipynb  # End-to-end training/experimentation notebook
├── model_ft.pth                # Trained 3D-CNN weights (hosted externally — see below)
└── requirements.txt
```

## Model Weights

`model_ft.pth` (~127 MB) is not tracked in this repo. Download it from **[add your Google Drive / Hugging Face link here]** and place it in the project root before running inference.

## Setup

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r CarCrashDetection_3D_Convolution/requirements.txt
```

Set the required environment variables (used for the optional email-alert feature in the demo app):
```bash
export SENDER_EMAIL="your-email@example.com"
export SENDER_EMAIL_PASSWORD="your-app-password"
```

## Usage

**Run inference on a sample video:**
```bash
python accidentUtils/object_tracker.py --video crash1.mp4 --output output_results/result.mp4
```

**Launch the Streamlit demo:**
```bash
streamlit run accidentUtils/app.py
```

## Results

Sample outputs are available in `output_results/` (`Test.mp4`, `VIF_infer1.mp4`, `3D_infer1.mp4`), showing detected vehicles, tracked trajectories, and flagged crash segments.

## Limitations & Future Work

- Trained on a relatively small, curated dataset — performance on unconstrained, non-curated surveillance footage (varying lighting, occlusion, camera angle) needs further validation.
- Real-time throughput has not been benchmarked on edge hardware.
- Potential next steps: swap YOLOv4 for a lighter/newer detector, add ONNX/TensorRT export for faster inference, and expand the training set with more diverse crash/no-crash clips.

