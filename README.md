# AI-Based Bone Cancer Detection from X-ray Images

Final Year Project

## Overview
This system uses deep learning to detect bone cancer from X-ray images.

The pipeline consists of:
- EfficientNet-B2 for classification
- Data augmentation for improved generalization
- Grad-CAM for explainability
- React + Electron desktop interface

## Model Performance

Accuracy: 82%

Confusion Matrix:
[[407 81]
 [77 307]]

## Tech Stack
Python
TensorFlow
React
Electron

## How to Run

Train model:
python train.py

Evaluate model:
python evaluate.py

Predict new image:
python predict.py