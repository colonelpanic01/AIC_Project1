# Western Engineering Automotive Innovation Challenge
This repository contains a detection and tracking system for Vulnerable Road Users (VRUs) using LiDAR data. The system is optimized for deployment on a NVIDIA Jetson Nano.

## Overview

The system detects and tracks pedestrians, cyclists, and motorcyclists in LiDAR point cloud data. It processes raw LiDAR scans and outputs 3D bounding boxes for each detected VRU.

### Features

- Efficient processing of LiDAR point cloud data
- Support for both rule-based and ML-based detection methods
- Optimized for NVIDIA Jetson Nano deployment
- Visualization tools for result inspection
- Evaluation metrics calculation

## Getting Started

### Prerequisites

- NVIDIA Jetson Nano with JetPack 4.6+
- Python 3.6+
- LiDAR dataset in binary format with 5 columns (X, Y, Z, Intensity, LiDAR Channel)

### Installation

TBD

### Data Format

The system expects the LiDAR data in the following format:

- **Scan data**: Binary files with 5 columns (X, Y, Z, Intensity, LiDAR Channel)
- **Label format**: Text files with each line containing (Label Name, X, Y, Z, W, L, H, yaw)
  - Where X, Y, Z is the center of the 3D bounding box
  - W, L, H are the dimensions of the box
  - yaw is the rotation angle around the Z-axis

Example directory structure:
```
data/
├── scans/
│   ├── 000000.bin
│   ├── 000001.bin
│   └── ...
└── labels/
    ├── 000000.txt
    ├── 000001.txt
    └── ...
```

## Usage

### Detection

To run detection on a dataset:
TBD


### Visualization

To visualize a single scan with its detections:

TBD

### Evaluation

To evaluate the detection performance against ground truth:

TBD

## Technical Details

### Detection Pipeline

1. **Preprocessing**: Filter point cloud based on range and intensity
2. **Clustering**: Group points into clusters using DBSCAN
3. **Feature Extraction**: Compute geometric and statistical features for each cluster
4. **Classification**: Identify VRUs using rule-based and ML-based methods
5. **Bounding Box Generation**: Compute oriented 3D bounding boxes for VRUs

### ML-based Detection

The ML-based approach uses a lightweight neural network that can run efficiently on the Jetson Nano. It processes features extracted from point cloud clusters to classify them into VRU types.

### Rule-based Detection

The rule-based detection uses height, width, and length constraints to identify different VRU types:

- **Pedestrians**: Tall, narrow objects
- **Cyclists**: Taller objects with moderate width and length
- **Motorcycles**: Moderately tall objects with larger width and length

### Performance Optimization

The system includes several optimizations for Jetson Nano deployment:

- Efficient point cloud filtering
- Grid-based downsampling
- Early rejection of non-VRU clusters
- Batch processing

## Results

The system achieves:

TBD

## Acknowledgments
