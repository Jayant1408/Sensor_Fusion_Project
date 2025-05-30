# Project 2.1: 3D Object Detection with LiDAR Data

**Udacity Self-Driving Car Engineer Nanodegree – Sensor Fusion Module**

---

## Project Overview

This project explores 3D object detection using LiDAR point cloud data from the [Waymo Open Dataset](https://waymo.com/open). We implemented and evaluated deep learning-based models, specifically **SFA3D** and **Complex-YOLO**, to identify and classify objects such as vehicles, pedestrians, cyclists, and signs in urban driving environments.

### Core Goals:
- Convert range images into 3D point clouds.
- Preprocess data and visualize scenes using Open3D and OpenCV.
- Run object detection using pre-trained deep learning models.
- Evaluate and visualize performance using BEV maps and camera projections.

---

## Project Workflow

### Introduction

In this project, we developed a complete LiDAR-based object detection pipeline. Starting with raw `.tfrecord` files, we extracted, normalized, and visualized range images, then generated 3D point clouds and BEV (bird’s-eye view) maps. These were then fed into two detection models:

- **SFA3D**: A ResNet-18 + FPN-based 3D detector with keypoint localization.
- **Complex-YOLO**: A YOLO-based architecture incorporating angle regression for 3D boxes.

We benchmarked their performance using COCO-style metrics and visualized both ground-truth and predicted boxes in BEV and image space.

<div align="center">
  <img src="Mid_term_project_figures\Detections-in-BEV-Shown-with-LiDAR-PCL-1_Image_7.png" width="70%" alt="Figure: SFA3D detection results in BEV and camera image space">
</div>

> *Green boxes indicate ground truth; red boxes show model predictions.*

---

## Range Image Analysis & Visibility Study

As part of the midterm write-up requirements, we conducted a thorough inspection of vehicle visibility in the point cloud:

### Vehicle Visibility Examples:
- Captured **10 distinct vehicles** across varying occlusion levels and angles.
- Included both clear full-shape returns and partial detections (e.g., from oblique angles).

### Stable Features Identified:
- **Rear bumpers** and **tail lights** consistently appeared across vehicles with high-intensity returns in LiDAR.
- Other features such as roofs and license plates varied more depending on angle and distance.

### Range Image Findings:
- The **intensity channel** helped isolate strong reflectors like metal surfaces.
- Several frames demonstrated how these intensity cues improved the BEV map's clarity.

Additional figures and discussion are available in the midterm report directory.

---

## Model Inference and Evaluation

We performed inference using the following models:
- **SFA3D** (ResNet18 + FPN)
- **Complex-YOLO** (DarkNet-based)

We compared performance using:
- **Precision / Recall**
- **IoU scores**
- **Qualitative visualizations** (camera view, BEV, and LiDAR PCL)

---

## Project Structure

| File / Folder                   | Description |
|-------------------------------|-------------|
| `loop_over_dataset.py`        | Main script to run preprocessing, detection, and visualization. |
| `student/objdet_detect.py`    | Handles model loading, inference, and bounding box generation. |
| `student/objdet_eval.py`      | Computes detection metrics (IoU, precision, recall). |
| `student/objdet_pcl.py`       | Converts range images to point clouds and generates BEV maps. |
| `misc/objdet_tools.py`        | Utility functions for coordinate transforms and projections. |
| `data/filenames.txt`          | List of input data files from Waymo. |
| `setup.py`                    | Project installation and environment setup. |


## Notes

- Visualizations were key to verifying detection performance and sensor alignment.
- Object features such as tail lights and bumpers were essential for reliable tracking and classification.
- Combining BEV and intensity maps improved both detection accuracy and interpretability.

This project provided valuable insight into the preprocessing, training, and evaluation of real-world 3D object detection systems.