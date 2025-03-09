#!/usr/bin/env python
import os
import sys
import numpy as np
import open3d as o3d
import open3d.ml as ml3d
import open3d.ml.torch as ml3d_torch
from open3d.ml.datasets import Custom3D
from open3d.ml.torch.models import PointPillars
from open3d.ml.torch.pipelines import ObjectDetection
import torch
import glob
import time
from pathlib import Path
import argparse

# VRU classes we are interested in detecting
VRU_CLASSES = [
    'human.pedestrian.adult',
    'human.pedestrian.construction_worker',
    'human.pedestrian.child',
    'human.pedestrian.wheelchair',
    'human.pedestrian.personal_mobility',
    'human.pedestrian.police_officer',
    'human.pedestrian.stroller',
    'vehicle.motorcycle',
    'vehicle.bicycle'
]

# Class-specific dimensions constraints based on provided data
CLASS_DIMENSIONS = {
    'human.pedestrian.adult': {'width': (0.282, 1.505), 'length': (0.214, 1.674), 'height': (0.585, 2.744), 'yaw': (-4.712371700070678, 1.570789171031985)},
    'human.pedestrian.construction_worker': {'width': (0.345, 1.971), 'length': (0.293, 1.521), 'height': (0.293, 2.573), 'yaw': (-4.712387760134593, 1.5705286304974742)},
    'human.pedestrian.child': {'width': (0.295, 0.93), 'length': (0.268, 0.995), 'height': (0.724, 2.0), 'yaw': (-4.710222062114154, 1.5624351131644398)},
    'human.pedestrian.wheelchair': {'width': (0.496, 0.876), 'length': (0.682, 1.538), 'height': (1.229, 1.532), 'yaw': (-3.5224930470241946, 1.4828541376923488)},
    'human.pedestrian.personal_mobility': {'width': (0.298, 0.886), 'length': (0.494, 2.239), 'height': (0.846, 2.0), 'yaw': (-4.6510599883954304, 1.505014419205605)},
    'human.pedestrian.police_officer': {'width': (0.527, 1.155), 'length': (0.451, 1.024), 'height': (1.394, 2.028), 'yaw': (-4.709856889394767, 1.5693674589034963)},
    'human.pedestrian.stroller': {'width': (0.362, 0.87), 'length': (0.418, 1.753), 'height': (0.789, 1.888), 'yaw': (-4.691451416352464, 1.5469162803724985)},
    'vehicle.motorcycle': {'width': (0.351, 1.816), 'length': (0.72, 4.409), 'height': (0.791, 2.02), 'yaw': (-4.7115172916358325, 1.5695307294768606)},
    'vehicle.bicycle': {'width': (0.233, 1.661), 'length': (0.454, 3.04), 'height': (0.349, 2.223), 'yaw': (-4.710947933322235, 1.5702342135720362)}
}

def read_binary_lidar(file_path):
    """
    Read binary LiDAR file with X, Y, Z, Intensity, LiDAR Channel columns
    """
    points = np.fromfile(file_path, dtype=np.float32)
    # Reshape to have 5 columns
    points = points.reshape(-1, 5)
    return points

def read_annotation_file(file_path):
    """
    Read annotation file with label name, x, y, z, w, l, h, and yaw
    """
    annotations = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 8:
                label = parts[0]
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                w, l, h = float(parts[4]), float(parts[5]), float(parts[6])
                yaw = float(parts[7])
                annotations.append({
                    'label': label,
                    'center': [x, y, z],
                    'dimensions': [w, l, h],
                    'yaw': yaw
                })
    return annotations

def create_custom_dataset(dataset_path, split='train'):
    """
    Create a custom Open3D-ML dataset from the provided data
    """
    # Define paths
    lidar_dir = os.path.join(dataset_path, 'scans')
    label_dir = os.path.join(dataset_path, 'labels')
    
    # Create the split file
    split_files = {'train': [], 'validation': [], 'test': []}
    
    # Get all binary files
    binary_files = sorted(glob.glob(os.path.join(lidar_dir, '*.bin')))
    
    # Split the dataset
    if split == 'train':
        total_files = len(binary_files)
        train_size = int(0.8 * total_files)
        val_size = int(0.1 * total_files)
        
        # Create splits
        train_files = binary_files[:train_size]
        val_files = binary_files[train_size:train_size + val_size]
        test_files = binary_files[train_size + val_size:]
        
        # Write file paths to split files
        with open(os.path.join(dataset_path, 'train.txt'), 'w') as f:
            for file_path in train_files:
                f.write(os.path.basename(file_path).replace('.bin', '') + '\n')
        
        with open(os.path.join(dataset_path, 'val.txt'), 'w') as f:
            for file_path in val_files:
                f.write(os.path.basename(file_path).replace('.bin', '') + '\n')
                
        with open(os.path.join(dataset_path, 'test.txt'), 'w') as f:
            for file_path in test_files:
                f.write(os.path.basename(file_path).replace('.bin', '') + '\n')
        
        split_files['train'] = os.path.join(dataset_path, 'train.txt')
        split_files['validation'] = os.path.join(dataset_path, 'val.txt')
        split_files['test'] = os.path.join(dataset_path, 'test.txt')
    
    # Create dataset definition for Open3D-ML
    dataset_dict = {
        'name': 'VRUDataset',
        'dataset_path': dataset_path,
        'lidar_dir': 'scans',
        'label_dir': 'labels',
        'prefix': '',
        'suffix': '.bin',
        'label_suffix': '.txt',
        'split_files': split_files,
        'classes': VRU_CLASSES
    }
    
    # Create Custom3D dataset
    dataset = Custom3D(dataset_dict)
    return dataset

class VRUDataProcessing:
    """
    Class for processing VRU data
    """
    def __init__(self):
        # Parameters for data filtering and preprocessing
        self.height_range = (-2.0, 4.0)  # Height range for filtering
        self.distance_range = (0, 60.0)  # Distance range for filtering
        self.intensity_threshold = 0.1  # Intensity threshold
        
    def filter_points(self, points):
        """
        Filter point cloud based on height, distance, and intensity
        """
        # Extract x, y, z, intensity
        x, y, z, intensity = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
        
        # Calculate distance
        distance = np.sqrt(x**2 + y**2)
        
        # Filter by height
        height_mask = (z >= self.height_range[0]) & (z <= self.height_range[1])
        
        # Filter by distance
        distance_mask = (distance >= self.distance_range[0]) & (distance <= self.distance_range[1])
        
        # Filter by intensity
        intensity_mask = intensity >= self.intensity_threshold
        
        # Combined mask
        mask = height_mask & distance_mask & intensity_mask
        
        # Return filtered points
        return points[mask]
    
    def ground_removal(self, points, ground_height_threshold=-1.5, max_slope=0.15):
        """
        Simple ground removal based on height and slope
        """
        if len(points) < 10:  # Not enough points for plane segmentation
            return points
            
        # Sort points by height
        sorted_indices = np.argsort(points[:, 2])
        sorted_points = points[sorted_indices]
        
        # Find ground plane using RANSAC
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(sorted_points[:, :3])
        
        # Estimate plane
        try:
            plane_model, inliers = pcd.segment_plane(distance_threshold=0.1, ransac_n=3, num_iterations=100)
            
            # Create ground mask based on plane model
            a, b, c, d = plane_model
            
            # Ground points typically have normal close to (0, 0, 1)
            slope = np.sqrt(a**2 + b**2) / abs(c) if abs(c) > 1e-6 else float('inf')
            
            # Apply height and slope criteria
            if slope < max_slope:
                # Points with z value close to ground plane are considered ground
                ground_mask = np.abs(np.dot(points[:, :3], [a, b, c]) + d) / np.sqrt(a**2 + b**2 + c**2) < ground_height_threshold
                return points[~ground_mask]
        except:
            # If plane segmentation fails, fall back to simple height-based filtering
            pass
            
        # Fallback: use a simpler height-based filter
        return points[points[:, 2] > ground_height_threshold]
    
    def preprocess(self, points):
        """
        Preprocess point cloud: filter and remove ground
        """
        # Filter points
        filtered_points = self.filter_points(points)
        
        # Safety check for empty point cloud
        if len(filtered_points) < 10:
            return filtered_points
            
        # Remove ground
        non_ground_points = self.ground_removal(filtered_points)
        
        return non_ground_points

class VRUDetector:
    """
    Class for detecting VRUs in LiDAR data
    """
    def __init__(self, model_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model_path = model_path
        self.data_processor = VRUDataProcessing()
        
        # Model configuration
        self.cfg = {
            'name': 'PointPillars',
            'batcher': {
                'batch_size': 1,
                'attr_padding_val': {
                    'point': 0,
                    'bbox': 0,
                    'label': 0,
                }
            },
            'augment': {
                'jitter_x': 0.1,
                'jitter_y': 0.1,
                'jitter_z': 0.1,
                'rotation_range': [-0.17, 0.17],
                'scale_range': [0.95, 1.05]
            },
            'num_classes': len(VRU_CLASSES),
            'use_norm': True,
            'pillar_size': [0.2, 0.2],
            'pillar_point_count': 100,
            'max_pillar_count': 12000,
            'input_height': 432,
            'input_width': 496,
            'voxel_size': [0.16, 0.16, 4],
            'point_cloud_range': [-49.6, -49.6, -3, 49.6, 49.6, 1],
            'grid_size': [496, 432],
            # Additional model-specific parameters
            'npoints': 20000,
            'dynamic_voxelization': True
        }
        
        # Initialize model
        try:
            self.pipeline = self._init_pipeline()
            
            # Load pretrained model if provided
            if model_path and os.path.exists(model_path):
                self.pipeline.load_ckpt(ckpt_path=model_path)
                print(f"Loaded model from {model_path}")
        except Exception as e:
            print(f"Error initializing model: {e}")
            raise
    
    def _init_pipeline(self):
        """
        Initialize PointPillars model and training pipeline
        """
        try:
            # Create model
            model = PointPillars(self.cfg)
            
            # Create pipeline
            pipeline = ObjectDetection(model=model, device=self.device)
            
            return pipeline
        except Exception as e:
            print(f"Error in _init_pipeline: {e}")
            raise
    
    def train(self, dataset_path, epochs=50, lr=0.001):
        """
        Train the model
        """
        # Create dataset
        dataset = create_custom_dataset(dataset_path)
        
        # Training parameters
        train_cfg = {
            'optimizer': {
                'lr': lr,
                'weight_decay': 0.0001,
            },
            'scheduler': {'type': 'StepLR', 'args': {'step_size': 10, 'gamma': 0.1}},
            'epochs': epochs,
            'save_ckpt_freq': 5,
        }
        
        # Set up pipeline for training
        self.pipeline.dataset = dataset
        
        # Train
        print("Starting training...")
        self.pipeline.run_train(train_cfg)
        print("Training complete")
        
        # Save model
        output_path = os.path.join(dataset_path, "trained_model")
        os.makedirs(output_path, exist_ok=True)
        self.pipeline.save_ckpt(os.path.join(output_path, "vru_detection_model.pth"))
        print(f"Model saved to {output_path}")
        
        return os.path.join(output_path, "vru_detection_model.pth")
    
    def detect(self, points):
        """
        Detect VRUs in point cloud
        """
        # Preprocess points
        processed_points = self.data_processor.preprocess(points)
        
        # Check if we have enough points after preprocessing
        if len(processed_points) < 10:
            return [], [], []
        
        # Format for inference
        data = {
            'point': processed_points[:, :3],
            'feat': processed_points[:, 3:5] if processed_points.shape[1] >= 5 else None
        }
        
        # Run inference
        results = self.pipeline.run_inference(data)
        
        # Extract results
        boxes = results['boxes']
        scores = results['scores']
        labels = results['labels']
        
        # Filter detections based on confidence
        confidence_threshold = 0.3
        valid_indices = scores > confidence_threshold
        
        filtered_boxes = boxes[valid_indices]
        filtered_scores = scores[valid_indices]
        filtered_labels = labels[valid_indices]
        
        # Apply class-specific post-processing
        final_boxes = []
        final_scores = []
        final_labels = []
        
        for box, score, label in zip(filtered_boxes, filtered_scores, filtered_labels):
            if label < len(VRU_CLASSES):  # Ensure label is valid
                label_name = VRU_CLASSES[label]
                
                # Get class-specific dimension constraints
                if label_name in CLASS_DIMENSIONS:
                    constraints = CLASS_DIMENSIONS[label_name]
                    
                    # Extract box dimensions
                    center_x, center_y, center_z, width, length, height, yaw = box
                    
                    # Check if dimensions are within the class-specific constraints
                    if (constraints['width'][0] <= width <= constraints['width'][1] and
                        constraints['length'][0] <= length <= constraints['length'][1] and
                        constraints['height'][0] <= height <= constraints['height'][1] and
                        constraints['yaw'][0] <= yaw <= constraints['yaw'][1]):
                        
                        final_boxes.append(box)
                        final_scores.append(score)
                        final_labels.append(label)
        
        return final_boxes, final_scores, final_labels
    
    def detect_and_save(self, lidar_file, output_dir):
        """
        Detect VRUs in a LiDAR file and save results
        """
        try:
            # Read point cloud
            points = read_binary_lidar(lidar_file)
            
            # Detect VRUs
            boxes, scores, labels = self.detect(points)
            
            # Create output file path
            file_name = os.path.basename(lidar_file).replace('.bin', '.txt')
            output_file = os.path.join(output_dir, file_name)
            
            # Save results
            with open(output_file, 'w') as f:
                for box, score, label in zip(boxes, scores, labels):
                    if label < len(VRU_CLASSES):  # Safety check
                        center_x, center_y, center_z, width, length, height, yaw = box
                        class_name = VRU_CLASSES[label]
                        f.write(f"{class_name} {center_x:.6f} {center_y:.6f} {center_z:.6f} {width:.6f} {length:.6f} {height:.6f} {yaw:.6f} {score:.6f}\n")
            
            return output_file
        except Exception as e:
            print(f"Error processing {lidar_file}: {e}")
            # Create empty result file to maintain processing flow
            file_name = os.path.basename(lidar_file).replace('.bin', '.txt')
            output_file = os.path.join(output_dir, file_name)
            with open(output_file, 'w') as f:
                pass
            return output_file
    
    def process_test_data(self, test_dir, output_dir):
        """
        Process all test data and save results
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Get all test files
        test_files = sorted(glob.glob(os.path.join(test_dir, '*.bin')))
        
        # Process each file
        for test_file in test_files:
            print(f"Processing {test_file}...")
            self.detect_and_save(test_file, output_dir)
        
        print(f"Results saved to {output_dir}")

# Simple rule-based fallback detector if ML model fails
class RuleBasedVRUDetector:
    def __init__(self):
        self.height_range = (0.5, 2.5)  # Human height range
        self.width_range = (0.3, 1.5)  # Human width range
        self.length_range = (0.3, 2.0)  # Human length range
        
    def detect(self, points):
        """
        Simple rule-based detector using clustering
        """
        if len(points) < 10:
            return [], [], []
            
        # Convert to Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        
        # DBSCAN clustering
        labels = np.array(pcd.cluster_dbscan(eps=0.5, min_points=10))
        
        if len(np.unique(labels)) <= 1:  # No clusters found
            return [], [], []
            
        # Process each cluster
        boxes = []
        scores = []
        class_labels = []
        
        for label in np.unique(labels):
            if label < 0:  # Skip noise
                continue
                
            # Get points in this cluster
            cluster_indices = np.where(labels == label)[0]
            cluster_points = np.asarray(pcd.points)[cluster_indices]
            
            # Get dimensions
            min_bound = np.min(cluster_points, axis=0)
            max_bound = np.max(cluster_points, axis=0)
            center = (min_bound + max_bound) / 2
            dimensions = max_bound - min_bound
            
            width, length, height = dimensions
            
            # Check if dimensions match VRU
            if (self.height_range[0] <= height <= self.height_range[1] and
                self.width_range[0] <= width <= self.width_range[1] and
                self.length_range[0] <= length <= self.length_range[1]):
                
                # Simple heuristic to identify object type
                if height > 1.5 and width < 0.8:
                    # Likely a pedestrian
                    class_label = 0  # human.pedestrian.adult
                elif length > 1.5 and width < 0.8:
                    # Likely a bicycle
                    class_label = 8  # vehicle.bicycle
                elif length > 1.5 and width > 0.8:
                    # Likely a motorcycle
                    class_label = 7  # vehicle.motorcycle
                else:
                    # Default to pedestrian
                    class_label = 0
                
                # Create box [x, y, z, width, length, height, yaw]
                box = [center[0], center[1], center[2], width, length, height, 0.0]
                
                boxes.append(box)
                scores.append(0.5)  # Fixed confidence score
                class_labels.append(class_label)
        
        return boxes, scores, class_labels

def optimize_for_jetson(model_path, output_path):
    """
    Optimize model for Jetson deployment
    """
    try:
        # Load model
        model = torch.load(model_path, map_location='cpu')
        
        # Convert to TorchScript
        scripted_model = torch.jit.script(model)
        
        # Save optimized model
        scripted_model.save(output_path)
        
        print(f"Optimized model saved to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error optimizing model: {e}")
        return None

def parse_arguments():
    parser = argparse.ArgumentParser(description='VRU Detection')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], required=True, help='Mode: train or test')
    parser.add_argument('--data_dir', type=str, default="data", help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default="results", help='Path to output directory')
    parser.add_argument('--model_path', type=str, default="modelsV2", help='Path to model checkpoint')
    parser.add_argument('--optimize', action='store_true', help='Optimize model for Jetson')
    parser.add_argument('--fallback', action='store_true', help='Use rule-based fallback if ML model fails')
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Set output directory
    output_dir = args.output_dir if args.output_dir else os.path.join(args.data_dir, 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Create detector
        detector = VRUDetector(model_path=args.model_path)
        
        if args.mode == 'train':
            # Train model
            model_path = detector.train(args.data_dir)
            
            # Optimize for Jetson if requested
            if args.optimize:
                optimize_for_jetson(model_path, model_path.replace('.pth', '_optimized.pth'))
        
        elif args.mode == 'test':
            # Process test data
            detector.process_test_data(args.data_dir, output_dir)
            
    except Exception as e:
        print(f"Error in primary detector: {e}")
        
        if args.fallback:
            print("Using rule-based fallback detector...")
            fallback_detector = RuleBasedVRUDetector()
            
            # Get all test files
            test_files = sorted(glob.glob(os.path.join(args.data_dir, '*.bin')))
            
            # Process each file with fallback detector
            for test_file in test_files:
                print(f"Processing {test_file} with fallback detector...")
                
                # Read point cloud
                points = read_binary_lidar(test_file)
                
                # Detect VRUs
                boxes, scores, labels = fallback_detector.detect(points)
                
                # Create output file path
                file_name = os.path.basename(test_file).replace('.bin', '.txt')
                output_file = os.path.join(output_dir, file_name)
                
                # Save results
                with open(output_file, 'w') as f:
                    for box, score, label in zip(boxes, scores, labels):
                        if label < len(VRU_CLASSES):
                            center_x, center_y, center_z, width, length, height, yaw = box
                            class_name = VRU_CLASSES[label]
                            f.write(f"{class_name} {center_x:.6f} {center_y:.6f} {center_z:.6f} {width:.6f} {length:.6f} {height:.6f} {yaw:.6f} {score:.6f}\n")

if __name__ == '__main__':
    main()