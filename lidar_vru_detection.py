import numpy as np
import os
import glob
import time
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN
import argparse

# Configuration parameters
CONFIG = {
    'voxel_size': [0.1, 0.1, 0.1],         # Size of voxels for voxelization
    'point_cloud_range': [-50, -50, -5, 50, 50, 3],  # Range of point cloud to consider
    'max_points_per_voxel': 32,             # Maximum number of points in a voxel
    'max_voxels': 20000,                    # Maximum number of voxels
    'pedestrian_height_range': [0.5, 2.0],  # Typical height range for pedestrians
    'cyclist_height_range': [0.8, 2.3],     # Typical height range for cyclists
    'min_points_threshold': 10,             # Minimum points to consider a cluster
    'cluster_eps': 0.5,                     # DBSCAN epsilon parameter
    'cluster_min_samples': 5,               # DBSCAN min_samples parameter
    'ground_height_threshold': -1.5,        # Threshold for ground points
    'intensity_threshold': 0.1,             # Minimum intensity to consider
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
        Build a lightweight neural network model for VRU detection
        that can run on Jetson Nano.
        """
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 3)  # 3 classes: pedestrian, cyclist, motorcycle
        )
        return model.to(self.device)
    
    def load_model(self, model_path):
        """Load a pretrained model"""
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded from {model_path}")
        
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
        
        # Statistical features
        std_dev = np.std(xyz, axis=0)
        intensity_mean = np.mean(cluster_points[:, 3])
        intensity_std = np.std(cluster_points[:, 3])
        
        # Shape features
        eigenvalues = self._compute_eigenvalues(xyz)
        
        # Combine features
        features = np.concatenate([
            center,
            dimensions,
            std_dev,
            [intensity_mean, intensity_std],
            eigenvalues,
            [len(cluster_points)]  # Number of points in cluster
        ])
        
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
        Rule-based classification of clusters into VRU types:
        - pedestrian
        - cyclist
        - motorcycle
        - other (non-VRU)
        """
        xyz = cluster_points[:, :3]
        min_point = np.min(xyz, axis=0)
        max_point = np.max(xyz, axis=0)
        height = max_point[2] - min_point[2]
        width = max_point[0] - min_point[0]
        length = max_point[1] - min_point[1]
        
        # Check if it's a pedestrian
        if (height > self.config['pedestrian_height_range'][0] and
            height < self.config['pedestrian_height_range'][1] and
            width < 1.0 and length < 1.0):
            return "pedestrian", 0.8  # confidence score
            
        # Check if it's a cyclist
        if (height > self.config['cyclist_height_range'][0] and
            height < self.config['cyclist_height_range'][1] and
            width < 2.0 and length < 2.0):
            return "cyclist", 0.7
            
        # Check if it's a motorcycle
        if (height > 0.8 and height < 1.8 and
            width < 1.5 and length < 2.5):
            return "motorcycle", 0.6
            
        return "other", 0.5
        
    def classify_cluster_ml(self, features):
        """
        ML-based classification of clusters
        """
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
        filtered_points = self.preprocess_point_cloud(points)
        
        # Extract clusters
        clusters = self.extract_clusters(filtered_points)
        
        # Process each cluster
        results = []
        for cluster_points in clusters:
            # Compute features
            features = self.compute_cluster_features(cluster_points)
            
            # Classify cluster
            if use_ml and hasattr(self, 'model'):
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
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing scans and labels')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for results')
    parser.add_argument('--mode', type=str, default='detect', choices=['detect', 'train'], 
                        help='Mode: detect or train')
    parser.add_argument('--model_path', type=str, default=None, help='Path to model weights')
    parser.add_argument('--use_ml', action='store_true', help='Use ML-based approach')
    
    args = parser.parse_args()
    
    if args.mode == 'detect':
        process_dataset(args.input_dir, args.output_dir, args.use_ml, args.model_path)
    elif args.mode == 'train':
        from train import train_model
        train_model(args.input_dir, args.model_path)
        


if __name__ == "__main__":
    main()