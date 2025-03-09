import numpy as np
import os
import glob
import time
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN
import argparse
'''
general VRU dimension details for testing, see the readme for more details :)
PEDESTRIAN ranges for all pedestrians [min, max]:
  total width range is [0.282, 1.971]
  total length range is [0.214, 2.239]
  total height range is []
  adult:
    - width range is [0.282, 1.505]
    - length range is [0.214, 1.674]
    - height range is [0.585, 2.744]
  worker:
    - width range is [0.345, 1.971]
    - length range is [0.293, 1.521]
    - height range is [0.293, 2.573]
  child:
    - width range is [0.295, 0.93]
    - length range is [0.268, 0.995]
    - height range is [0.724, 2.0]
  wheelchair:
    - width range is [0.496, 0.876]
    - length range is [0.682, 1.538]
    - height range is [1.229, 1.532]
  pers mobility:
    - width range is [0.298, 0.886]
    - length range is [0.494, 2.239]
    - height range is [0.846, 2.0]
  officer:
    - width range is [0.527, 1.155]
    - length range is [0.451, 1.024]
    - height range is [1.394, 2.028]
  stroller:
    - width range is [0.362, 0.87]
    - length range is [0.418, 1.753]
    - height range is [0.789, 1.888]

MOTORCYCLE ranges [min, max]:
    - width range is [0.351, 1.816]
    - length range is [0.72, 4.409]
    - height range is [0.791, 2.02]

BICYCLE ranges [min, max]:
    - width range is [0.233, 1.661]
    - length range is [0.454, 3.04]
    - height range is [0.349, 2.223]

all width range [0.233, 1.971]
all length range [0.214, 0.995]
all height range [0.293, 2.744]

'''
# Configuration parameters
# changes post test !!
CONFIG = {
    'voxel_size': [0.05, 0.05, 0.05],  # changes this from [0.1,0.1,0.1,0.1] for finer voxel size so we have better resolution
    'point_cloud_range': [-50, -50, -5, 50, 50, 3],  # range of point cloud to consider
    'max_points_per_voxel': 64,  # maximum number of points in a voxel, increased from 32
    'max_voxels': 40000,  # max number of voxels Doubled from 20000

    'pedestrian_height_range': [0.585, 2.744],  # From human.pedestrian.adult
    'pedestrian_width_range': [0.282, 1.505],
    'pedestrian_length_range': [0.214, 1.674],
    
    'bicycle_height_range': [0.349, 2.223],  # From vehicle.bicycle
    'bicycle_width_range': [0.233, 1.661],
    'bicycle_length_range': [0.454, 3.04],
    
    'motorcycle_height_range': [0.791, 2.02],  # From vehicle.motorcycle
    'motorcycle_width_range': [0.351, 1.816],
    'motorcycle_length_range': [0.72, 4.409],
    
    'min_points_threshold': 15,  # minimum points to consider a cluster, increased from 10
    'cluster_eps': 0.4,  # DBSCAN epislon parameter, reduced from 0.5 for finer clustering
    'cluster_min_samples': 7,  # DBSCAN min_samples parameter, increased from 5
    'ground_height_threshold': -1.5, # threshold for ground points
    'intensity_threshold': 0.08,  # minimum intensity to consider, slightly lower to capture more points
}

class LiDARVRUDetector:
    def __init__(self, config=None):
        self.config = CONFIG if config is None else config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model (if using ML approach)
        self.model = self._build_model()
        
    def _build_model(self):
        """
        lightweight neural network model for VRU detection that can run on Jetson Nano.
        """
        # Assuming input feature size of 64
        model = nn.Sequential(
            # Input layer
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # First hidden layer
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Second hidden layer
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Third hidden layer
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            # Output layer - 3 classes: pedestrian, cyclist, motorcycle
            nn.Linear(64, 3)
        )
        return model.to(self.device)

    def load_model(self, model_path):
        """Load model weights from a file."""
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device)
        try:
            self.model.load_state_dict(checkpoint, strict=False)  # Allow partial loading
            print("Model loaded successfully!")
        except RuntimeError as e:
            print(f"Error loading model: {e}")

    def load_point_cloud(self, bin_file):
        """
        Load point cloud data from binary file
        Format: X, Y, Z, Intensity, LiDAR Channel
        """
        points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 5)
        return points
    
    def preprocess_point_cloud(self, points):
        """
        Preprocess the point cloud data:
        1. Remove ground points
        2. Filter by intensity
        3. Remove points outside the range
        """
        # Extract x, y, z, intensity
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        intensity = points[:, 3]
        
        # Filter by range
        pc_range = self.config['point_cloud_range']
        mask_x = np.logical_and(x >= pc_range[0], x <= pc_range[3])
        mask_y = np.logical_and(y >= pc_range[1], y <= pc_range[4])
        mask_z = np.logical_and(z >= pc_range[2], z <= pc_range[5])
        mask_range = np.logical_and(np.logical_and(mask_x, mask_y), mask_z)
        
        # Filter by intensity
        mask_intensity = intensity > self.config['intensity_threshold']
        
        # Combine masks
        mask = np.logical_and(mask_range, mask_intensity)
        filtered_points = points[mask]
        
        return filtered_points
    
    def extract_clusters(self, points):
        """
        Extract clusters from point cloud using DBSCAN
        """
        # Use only x, y, z for clustering
        xyz = points[:, :3]
        
        # Apply DBSCAN clustering
        db = DBSCAN(
            eps=self.config['cluster_eps'],
            min_samples=self.config['cluster_min_samples'],
            n_jobs=-1
        ).fit(xyz)
        
        labels = db.labels_
        unique_labels = np.unique(labels)
        
        # Collect clusters (excluding noise with label -1)
        clusters = []
        for label in unique_labels:
            if label == -1:
                continue
                
            cluster_mask = labels == label
            if np.sum(cluster_mask) < self.config['min_points_threshold']:
                continue
                
            cluster_points = points[cluster_mask]
            clusters.append(cluster_points)
        
        return clusters
    
    def compute_cluster_features(self, cluster_points):
        """
        Compute features for a cluster of points
        """
        # Geometric features
        xyz = cluster_points[:, :3]
        min_point = np.min(xyz, axis=0)
        max_point = np.max(xyz, axis=0)
        center = np.mean(xyz, axis=0)
        dimensions = max_point - min_point
        
        # Aspect ratios (important for class discrimination)
        width_height_ratio = dimensions[0] / (dimensions[2] + 1e-6)
        length_height_ratio = dimensions[1] / (dimensions[2] + 1e-6)
        width_length_ratio = dimensions[0] / (dimensions[1] + 1e-6)
        
        # Statistical features
        std_dev = np.std(xyz, axis=0)
        intensity_values = cluster_points[:, 3]
        intensity_mean = np.mean(intensity_values)
        intensity_std = np.std(intensity_values)
        intensity_min = np.min(intensity_values)
        intensity_max = np.max(intensity_values)
        
        # Point distribution features
        distance_to_center = np.linalg.norm(xyz - center, axis=1)
        mean_distance = np.mean(distance_to_center)
        std_distance = np.std(distance_to_center)
        max_distance = np.max(distance_to_center)
        
        # Height distribution (useful for pedestrians vs vehicles)
        height_percentiles = np.percentile(xyz[:, 2], [25, 50, 75])
        
        # Density features
        volume = dimensions[0] * dimensions[1] * dimensions[2]
        point_density = len(cluster_points) / (volume + 1e-6)
        
        # Shape features
        eigenvalues = self._compute_eigenvalues(xyz)
        if len(eigenvalues) == 3 and np.sum(eigenvalues) > 0:
            # Normalized eigenvalues for shape description
            normalized_evals = eigenvalues / np.sum(eigenvalues)
            # Linear, planar, and spherical features
            linearity = (normalized_evals[0] - normalized_evals[1]) / (normalized_evals[0] + 1e-6)
            planarity = (normalized_evals[1] - normalized_evals[2]) / (normalized_evals[0] + 1e-6)
            sphericity = normalized_evals[2] / (normalized_evals[0] + 1e-6)
            anisotropy = (normalized_evals[0] - normalized_evals[2]) / (normalized_evals[0] + 1e-6)
        else:
            linearity, planarity, sphericity, anisotropy = 0, 0, 0, 0
        
        # Combine all features
        features = np.concatenate([
            center,  # 3 features
            dimensions,  # 3 features
            [width_height_ratio, length_height_ratio, width_length_ratio],  # 3 features
            std_dev,  # 3 features
            [intensity_mean, intensity_std, intensity_min, intensity_max],  # 4 features
            [mean_distance, std_distance, max_distance],  # 3 features
            height_percentiles,  # 3 features
            [point_density],  # 1 feature
            eigenvalues,  # 3 features
            [linearity, planarity, sphericity, anisotropy],  # 4 features
            [len(cluster_points)]  # 1 feature - number of points
        ])  # Total: 31 features
        
        # Pad to make 64 features if needed
        if len(features) < 64:
            padding = np.zeros(64 - len(features))
            features = np.concatenate([features, padding])
        
        return features

    
    def _compute_eigenvalues(self, points):
        """Compute eigenvalues of the covariance matrix for shape analysis"""
        if len(points) < 3:
            return np.zeros(3)
            
        cov = np.cov(points, rowvar=False)
        eigenvalues, _ = np.linalg.eigh(cov)
        # Sort in descending order
        eigenvalues = np.sort(eigenvalues)[::-1]
        return eigenvalues
    
    def classify_cluster_rule_based(self, cluster_points, features):
        """
        Rule-based classification using precise dimension ranges. clusters to VRU types:
        - pedestrian
        - cyclist
        - motorcycle
        - other/non-vru
        """
        xyz = cluster_points[:, :3]
        min_point = np.min(xyz, axis=0)
        max_point = np.max(xyz, axis=0)
        height = max_point[2] - min_point[2]
        width = max_point[0] - min_point[0]
        length = max_point[1] - min_point[1]
        
        # Compute confidence scores for each class based on dimensions
        pedestrian_score = 0
        bicycle_score = 0
        motorcycle_score = 0
        
        # Check pedestrian dimensions
        if (height >= self.config['pedestrian_height_range'][0] and 
            height <= self.config['pedestrian_height_range'][1] and
            width >= self.config['pedestrian_width_range'][0] and
            width <= self.config['pedestrian_width_range'][1] and
            length >= self.config['pedestrian_length_range'][0] and
            length <= self.config['pedestrian_length_range'][1]):
            
            # Additional pedestrian characteristics
            if len(cluster_points) < 100:  # Pedestrians typically have fewer points
                pedestrian_score += 0.3
                
            # Eigenvalues for pedestrian (typically more vertical)
            eigenvalues = self._compute_eigenvalues(xyz)
            if len(eigenvalues) == 3 and eigenvalues[2] / (eigenvalues[0] + 1e-6) < 0.2:
                pedestrian_score += 0.3
                
            # Height to width ratio for pedestrian
            if height / (width + 1e-6) > 1.5:
                pedestrian_score += 0.4
                
            pedestrian_score = min(1.0, pedestrian_score)
        
        # Check bicycle dimensions
        if (height >= self.config['bicycle_height_range'][0] and 
            height <= self.config['bicycle_height_range'][1] and
            width >= self.config['bicycle_width_range'][0] and
            width <= self.config['bicycle_width_range'][1] and
            length >= self.config['bicycle_length_range'][0] and
            length <= self.config['bicycle_length_range'][1]):
            
            # Bicycles typically have a distinctive length-to-width ratio
            if length / (width + 1e-6) > 1.3:
                bicycle_score += 0.4
                
            # Height distribution for bicycles
            z_values = xyz[:, 2]
            if np.std(z_values) < 0.5:  # Relatively flat height distribution
                bicycle_score += 0.3
                
            # Intensity characteristics
            intensity_values = cluster_points[:, 3]
            if np.mean(intensity_values) > 0.2:  # Metal parts reflect more
                bicycle_score += 0.3
                
            bicycle_score = min(1.0, bicycle_score)
        
        # Check motorcycle dimensions
        if (height >= self.config['motorcycle_height_range'][0] and 
            height <= self.config['motorcycle_height_range'][1] and
            width >= self.config['motorcycle_width_range'][0] and
            width <= self.config['motorcycle_width_range'][1] and
            length >= self.config['motorcycle_length_range'][0] and
            length <= self.config['motorcycle_length_range'][1]):
            
            # Motorcycles typically have more points
            if len(cluster_points) > 80:
                motorcycle_score += 0.3
                
            # Length-to-width ratio for motorcycles
            if length / (width + 1e-6) > 1.8:
                motorcycle_score += 0.4
                
            # Intensity characteristics
            intensity_values = cluster_points[:, 3]
            if np.mean(intensity_values) > 0.25:  # Metal parts reflect more
                motorcycle_score += 0.3
                
            motorcycle_score = min(1.0, motorcycle_score)
        
        # Determine the highest scoring class
        scores = {
            "pedestrian": pedestrian_score,
            "cyclist": bicycle_score,
            "motorcycle": motorcycle_score
        }
        
        best_class = max(scores.items(), key=lambda x: x[1])
        
        # Return best class if score is high enough, otherwise "other"
        if best_class[1] > 0.5:
            return best_class[0], best_class[1]
        else:
            return "other", 0.3

        
    def classify_cluster_ml(self, features):
        """
        ML-based classification of clusters
        """
        self.model.eval()
        # Normalize features
        features = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        # Model prediction
        with torch.no_grad():
            outputs = self.model(features)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            
        # Map prediction to class label
        class_map = {0: "pedestrian", 1: "cyclist", 2: "motorcycle"}
        predicted_class = class_map[predicted.item()]
        
        return predicted_class, confidence.item()
    
    def compute_oriented_bbox(self, cluster_points):
        """
        Compute an oriented 3D bounding box for a cluster
        Returns: center_x, center_y, center_z, width, height, length, yaw
        """
        # Extract x, y, z coordinates
        xyz = cluster_points[:, :3]
        
        # Compute center
        center = np.mean(xyz, axis=0)
        
        # Compute covariance matrix and its eigenvectors for orientation
        if len(xyz) < 3:
            # Not enough points for PCA, return axis-aligned box
            min_point = np.min(xyz, axis=0)
            max_point = np.max(xyz, axis=0)
            dimensions = max_point - min_point
            return [center[0], center[1], center[2], 
                    dimensions[0], dimensions[2], dimensions[1], 0.0]
        
        # PCA for orientation
        cov = np.cov(xyz, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Sort eigenvectors by eigenvalues (in descending order)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # The first eigenvector gives the principal direction
        main_axis = eigenvectors[:, 0]
        
        # Compute yaw (rotation around z-axis)
        yaw = np.arctan2(main_axis[1], main_axis[0])
        
        # Project points onto eigenvectors to get dimensions
        projected = np.dot(xyz - center, eigenvectors)
        min_point = np.min(projected, axis=0)
        max_point = np.max(projected, axis=0)
        
        # Dimensions of the oriented box
        dimensions = max_point - min_point
        
        # Return as [center_x, center_y, center_z, width, height, length, yaw]
        # Note: width = x-axis, height = z-axis, length = y-axis
        return [center[0], center[1], center[2], 
                dimensions[0], dimensions[2], dimensions[1], yaw]
    
    def detect(self, point_cloud_file, use_ml=False):
        """
        Main detection pipeline
        """
        # Load and preprocess point cloud
        points = self.load_point_cloud(point_cloud_file)
        # print(f"Loaded point cloud with {len(points)} points")
        filtered_points = self.preprocess_point_cloud(points)
        # print(f"Filtered point cloud with {len(filtered_points)} points")
        
        # Extract clusters
        clusters = self.extract_clusters(filtered_points)
        
        # Process each cluster
        results = []
        for cluster_points in clusters:
            # Compute features
            features = self.compute_cluster_features(cluster_points)
            
            # Classify cluster
            if use_ml and hasattr(self, 'model'):
                self.model.eval()
                cls_type, confidence = self.classify_cluster_ml(features)
            else:
                cls_type, confidence = self.classify_cluster_rule_based(cluster_points, features)
            
            # Skip if not a VRU
            if cls_type == "other" or confidence < 0.5:
                continue
            
            # Compute bounding box
            bbox = self.compute_oriented_bbox(cluster_points)
            
            # Add to results
            result = {
                'type': cls_type,
                'confidence': confidence,
                'bbox': bbox
            }
            results.append(result)
            
        return results
    
    def save_results(self, results, output_file):
        """
        Save detection results to a file
        Format: class_name, center_x, center_y, center_z, width, height, length, yaw
        """
        with open(output_file, 'w') as f:
            for result in results:
                bbox = result['bbox']
                line = f"{result['type']} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]} {bbox[4]} {bbox[5]} {bbox[6]}\n"
                f.write(line)


def process_dataset(input_dir, output_dir, use_ml=False, model_path=None):
    """
    Process an entire dataset of LiDAR scans
    """
    # Initialize detector
    detector = LiDARVRUDetector()
    
    # Load model if using ML approach
    if use_ml and model_path is not None:
        detector.load_model(model_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of scan files
    scan_files = sorted(glob.glob(os.path.join(input_dir, 'scans', '*.bin')))
    # Process each scan
    total_time = 0
    for i, scan_file in enumerate(scan_files):
        # Extract scan number
        scan_num = os.path.basename(scan_file).split('.')[0]
        
        # Process scan
        start_time = time.time()
        results = detector.detect(scan_file, use_ml=use_ml)
        end_time = time.time()
        
        # Calculate processing time
        process_time = end_time - start_time
        total_time += process_time
        
        # Save results
        output_file = os.path.join(output_dir, f'{scan_num}.txt')
        detector.save_results(results, output_file)
        
        # Print progress
        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{len(scan_files)} scans. Avg time: {total_time / (i+1):.4f}s")
    
    # Print statistics
    avg_time = total_time / len(scan_files)
    print(f"Processing complete! Processed {len(scan_files)} scans.")
    print(f"Average processing time: {avg_time:.4f}s ({1/avg_time:.2f} Hz)")


# def train_model(data_dir, model_save_path, epochs=10, batch_size=32):
#     return 
def main():
    """Main entry point for the program"""
    parser = argparse.ArgumentParser(description='LiDAR VRU Detection')
    parser.add_argument('--input_dir', type=str, default='data', help='Input directory containing scans and labels')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory for results')
    parser.add_argument('--mode', type=str, default='detect', choices=['detect', 'train'], 
                        help='Mode: detect or train')
    parser.add_argument('--model_path', default='models/model.pth', type=str, help='Path to model weights')
    parser.add_argument('--use_ml', action='store_true', help='Use ML-based approach')
    
    args = parser.parse_args()
    
    if args.mode == 'detect':
        process_dataset(args.input_dir, args.output_dir, args.use_ml, args.model_path)
    elif args.mode == 'train':
        from train import train_model
        train_model(args.input_dir, args.model_path)

if __name__ == "__main__":
    main()