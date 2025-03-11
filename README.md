# Western Engineering Automotive Innovation Challenge
This repository contains a detection and tracking system for Vulnerable Road Users (VRUs) using LiDAR data. The system is to be deployed on a NVIDIA Jetson Nano, yet to be optimized it. 
It is comprehensive implementation of a **LiDAR-based VRU (Vulnerable Road Users) detection system**, combining **rule-based** and **machine learning** approaches to detect pedestrians, cyclists, and motorcyclists from LiDAR point cloud data. Below is a detailed explanation of how the code works:


## Overview

The system detects and tracks pedestrians, cyclists, and motorcyclists in LiDAR point cloud data. It processes raw LiDAR scans and outputs 3D bounding boxes for each detected VRU. It is comprehensive implementation of a **LiDAR-based VRU (Vulnerable Road Users) detection system**, combining **rule-based** and **machine learning** approaches to detect pedestrians, cyclists, and motorcyclists from LiDAR point cloud data. Below is a detailed explanation of how the code works:

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

VRU Stats determined from helper_config_rule.py:
* human.pedestrian.adult:
  * Width  - Min: 0.282, Max: 1.505
  * Length - Min: 0.214, Max: 1.674
  * Height - Min: 0.585, Max: 2.744
  * Yaw - Min: -4.712371700070678, Max: 1.570789171031985
* human.pedestrian.construction_worker:
  * Width  - Min: 0.345, Max: 1.971
  * Length - Min: 0.293, Max: 1.521
  * Height - Min: 0.293, Max: 2.573
  * Yaw - Min: -4.712387760134593, Max: 1.5705286304974742
* human.pedestrian.child:
  * Width  - Min: 0.295, Max: 0.93
  * Length - Min: 0.268, Max: 0.995
  * Height - Min: 0.724, Max: 2.0
  * Yaw - Min: -4.710222062114154, Max: 1.5624351131644398
* human.pedestrian.wheelchair:
  * Width  - Min: 0.496, Max: 0.876
  * Length - Min: 0.682, Max: 1.538
  * Height - Min: 1.229, Max: 1.532
  * Yaw - Min: -3.5224930470241946, Max: 1.4828541376923488
* human.pedestrian.personal_mobility:
  * Width  - Min: 0.298, Max: 0.886
  * Length - Min: 0.494, Max: 2.239
  * Height - Min: 0.846, Max: 2.0
  * Yaw - Min: -4.6510599883954304, Max: 1.505014419205605
* human.pedestrian.police_officer:
  * Width  - Min: 0.527, Max: 1.155
  * Length - Min: 0.451, Max: 1.024
  * Height - Min: 1.394, Max: 2.028
  * Yaw - Min: -4.709856889394767, Max: 1.5693674589034963
* human.pedestrian.stroller:
  * Width  - Min: 0.362, Max: 0.87
  * Length - Min: 0.418, Max: 1.753
  * Height - Min: 0.789, Max: 1.888
  * Yaw - Min: -4.691451416352464, Max: 1.5469162803724985
* vehicle.motorcycle:
  * Width  - Min: 0.351, Max: 1.816
  * Length - Min: 0.72, Max: 4.409
  * Height - Min: 0.791, Max: 2.02
  * Yaw - Min: -4.7115172916358325, Max: 1.5695307294768606
* vehicle.bicycle:
  * Width  - Min: 0.233, Max: 1.661
  * Length - Min: 0.454, Max: 3.04
  * Height - Min: 0.349, Max: 2.223
  * Yaw - Min: -4.710947933322235, Max: 1.5702342135720362
* movable_object.pushable_pullable:
  * Width  - Min: 0.193, Max: 3.336
  * Length - Min: 0.223, Max: 7.02
  * Height - Min: 0.53, Max: 3.866
* vehicle.car:
  * Width  - Min: 0.648, Max: 3.875
  * Length - Min: 2.016, Max: 11.519
  * Height - Min: 0.867, Max: 4.49
* vehicle.bus.rigid:
  * Width  - Min: 1.848, Max: 4.611
  * Length - Min: 5.15, Max: 17.918
  * Height - Min: 2.222, Max: 5.304
* vehicle.construction:
  * Width  - Min: 0.682, Max: 7.554
  * Length - Min: 1.222, Max: 17.005
  * Height - Min: 1.131, Max: 7.469
* vehicle.truck:
  * Width  - Min: 1.377, Max: 6.044
  * Length - Min: 2.314, Max: 20.643
  * Height - Min: 1.523, Max: 5.01
* vehicle.trailer:
  * Width  - Min: 0.838, Max: 4.965
  * Length - Min: 0.932, Max: 18.644
  * Height - Min: 0.824, Max: 5.178
* movable_object.trafficcone:
  * Width  - Min: 0.094, Max: 2.037
  * Length - Min: 0.091, Max: 2.012
  * Height - Min: 0.224, Max: 2.0
* movable_object.barrier:
  * Width  - Min: 0.285, Max: 7.677
  * Length - Min: 0.11, Max: 3.022
  * Height - Min: 0.357, Max: 2.044
* movable_object.debris:
  * Width  - Min: 0.23, Max: 11.667
  * Length - Min: 0.162, Max: 7.597
  *  Height - Min: 0.123, Max: 2.613
* static_object.bicycle_rack:
  * Width  - Min: 0.459, Max: 15.774
  * Length - Min: 1.04, Max: 16.168
  * Height - Min: 0.793, Max: 2.508
animal:
  * Width  - Min: 0.193, Max: 0.672
  * Length - Min: 0.401, Max: 1.531
  * Height - Min: 0.291, Max: 0.96
* vehicle.emergency.police:
  * Width  - Min: 1.706, Max: 2.328
  * Length - Min: 4.541, Max: 5.712
  * Height - Min: 1.483, Max: 2.243
* vehicle.bus.bendy:
  * Width  - Min: 2.459, Max: 5.113
  * Length - Min: 6.771, Max: 21.304
  * Height - Min: 2.245, Max: 3.926
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
