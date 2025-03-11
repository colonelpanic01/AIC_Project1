# Western Engineering Automotive Innovation Challenge
This repository contains a detection and tracking system for Vulnerable Road Users (VRUs) using LiDAR data. The system is to be deployed on a NVIDIA Jetson Nano, yet to be optimized. 

## Overview

The system detects and tracks pedestrians, cyclists, and motorcyclists in LiDAR point cloud data. It processes raw LiDAR scans and outputs 3D bounding boxes for each detected VRU. It is a comprehensive implementation of a **LiDAR-based VRU (Vulnerable Road Users) detection system**, combining rule-based and machine learning approaches to detect pedestrians, cyclists, and motorcyclists from LiDAR point cloud data. Below is a detailed explanation of how the code works:

### Features

- Efficient processing of LiDAR point cloud data
- Support for both rule-based and ML-based detection methods
- Visualization tools for result inspection
- Evaluation metrics calculation
- Web app to view sequenced data from generated label files 

## Getting Started

### Prerequisites

- NVIDIA Jetson Nano with JetPack 4.6+
- Python 3.6+
- LiDAR dataset in binary format with 5 columns (X, Y, Z, Intensity, LiDAR Channel)

### Installation

Will add later

### Data Format

The system expects the LiDAR data in the following format:

- **Scan data**: Binary files with 5 columns (X, Y, Z, Intensity, LiDAR Channel)
- **Label format**: Generated after processing the scan data. These text files have each line containing (Label Name, X, Y, Z, W, L, H, yaw)
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

Will add later

### Visualization

To visualize scans with its detections:

Will add later

### Evaluation

To evaluate the detection performance against ground truth:

Will add later

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

Will add later

### Performance Optimization

The system includes several optimizations for Jetson Nano deployment:

- Efficient point cloud filtering
- Grid-based downsampling
- Early rejection of non-VRU clusters
- Batch processing

## Results

The system achieves:

Will add later

## Acknowledgments
