import numpy as np
import os
import glob
import time
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN
import argparse

# Combined configuration with parameters from both scripts
CONFIG = {
    # Voxelization parameters
    'voxel_size': [0.05, 0.05, 0.05],
    'point_cloud_range': [-50, -50, -5, 50, 50, 3],
    'max_points_per_voxel': 64,
    'max_voxels': 40000,
    
    # Combined VRU dimension ranges
    'vru_height_range': [0.349, 2.744],  # Min from bicycle, max from pedestrian
    'vru_width_range': [0.233, 1.971],   # Min from bicycle, max from pedestrian
    'vru_length_range': [0.214, 4.409],  # Min from pedestrian, max from motorcycle
    
    # Clustering parameters
    'min_points_threshold': 10,
    'cluster_eps': 0.45,               # Compromise between the two scripts
    'cluster_min_samples': 7,
    'ground_height_threshold': -1.5,
    'intensity_threshold': 0.08,
    
    # Confidence scoring parameters
    'confidence_threshold': 0.6,       # Minimum confidence to accept a detection
    'density_threshold': 50,           # Minimum point density for VRUs
}

class VRUDetector:
    def __init__(self, config=None, use_ml=False):
        self.config = CONFIG if config is None else config
        self.use_ml = use_ml
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize ML model if using it
        if use_ml:
            self.model = self._build_model()
            print(f"Using device: {self.device}")
    
    def _build_model(self):
        """Build a lightweight neural network model for VRU detection."""
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            # Single output for binary VRU classification
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        return model.to(self.device)
    
    def load_model(self, model_path):
        """Load model weights from a file."""
        if not self.use_ml:
            return
            
        print(f"Loading model from {model_path}...")
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint, strict=False)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.use_ml = False
    
    def load_point_cloud(self, bin_file):
        """Load point cloud data from binary file."""
        points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 5)
        return points
    
    def preprocess_point_cloud(self, points):
        """Preprocess the point cloud data."""
        # Extract x, y, z, intensity
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        intensity = points[:, 3]
        
        # Filter by range
        pc_range = self.config['point_cloud_range']
        mask_x = np.logical_and(x >= pc_range[0], x <= pc_range[3])
        mask_y = np.logical_and(y >= pc_range[1], y <= pc_range[4])
        mask_z = np.logical_and(z >= pc_range[2], z <= pc_range[5])
        mask_range = np.logical_and(np.logical_and(mask_x, mask_y), mask_z)
        
        # Filter by intensity and ground points
        mask_intensity = intensity > self.config['intensity_threshold']
        mask_ground = z > self.config['ground_height_threshold']
        
        # Combine masks
        mask = np.logical_and(mask_range, np.logical_and(mask_intensity, mask_ground))
        filtered_points = points[mask]
        
        return filtered_points
    
    def extract_clusters(self, points):
        """Extract clusters from point cloud using DBSCAN."""
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
        """Compute features for a cluster of points."""
        # Geometric features
        xyz = cluster_points[:, :3]
        min_point = np.min(xyz, axis=0)
        max_point = np.max(xyz, axis=0)
        center = np.mean(xyz, axis=0)
        dimensions = max_point - min_point
        
        # Basic dimensions
        width = dimensions[0]
        length = dimensions[1]
        height = dimensions[2]
        
        # Point density
        volume = width * length * height
        density = len(cluster_points) / max(volume, 1e-6)
        
        # Aspect ratios
        width_height_ratio = width / (height + 1e-6)
        length_height_ratio = length / (height + 1e-6)
        width_length_ratio = width / (length + 1e-6)
        
        # Statistical features
        std_dev = np.std(xyz, axis=0)
        intensity_values = cluster_points[:, 3]
        intensity_mean = np.mean(intensity_values)
        intensity_std = np.std(intensity_values)
        
        # Distance features
        distance_to_center = np.linalg.norm(xyz - center, axis=1)
        mean_distance = np.mean(distance_to_center)
        std_distance = np.std(distance_to_center)
        
        # Height distribution
        height_percentiles = np.percentile(xyz[:, 2], [25, 50, 75])
        
        # Shape features
        eigenvalues = self._compute_eigenvalues(xyz)
        if len(eigenvalues) == 3 and np.sum(eigenvalues) > 0:
            normalized_evals = eigenvalues / np.sum(eigenvalues)
            linearity = (normalized_evals[0] - normalized_evals[1]) / (normalized_evals[0] + 1e-6)
            planarity = (normalized_evals[1] - normalized_evals[2]) / (normalized_evals[0] + 1e-6)
            sphericity = normalized_evals[2] / (normalized_evals[0] + 1e-6)
        else:
            linearity, planarity, sphericity = 0, 0, 0
        
        # Combine all features
        features = np.concatenate([
            [width, length, height],
            [density],
            [width_height_ratio, length_height_ratio, width_length_ratio],
            std_dev,
            [intensity_mean, intensity_std],
            [mean_distance, std_distance],
            height_percentiles,
            eigenvalues,
            [linearity, planarity, sphericity],
            [len(cluster_points)]
        ])
        
        # Pad to make 64 features for ML model
        if len(features) < 64:
            padding = np.zeros(64 - len(features))
            features = np.concatenate([features, padding])
        
        return features, {
            'center': center,
            'dimensions': dimensions,
            'width': width,
            'length': length,
            'height': height,
            'density': density,
            'num_points': len(cluster_points)
        }
    
    def _compute_eigenvalues(self, points):
        """Compute eigenvalues of the covariance matrix for shape analysis."""
        if len(points) < 3:
            return np.zeros(3)
            
        cov = np.cov(points, rowvar=False)
        eigenvalues, _ = np.linalg.eigh(cov)
        # Sort in descending order
        eigenvalues = np.sort(eigenvalues)[::-1]
        return eigenvalues
    
    def is_vru_rule_based(self, cluster_info):
        """Rule-based classification for VRU."""
        # Check dimensions against VRU ranges
        height = cluster_info['height']
        width = cluster_info['width']
        length = cluster_info['length']
        
        height_ok = (height >= self.config['vru_height_range'][0] and 
                      height <= self.config['vru_height_range'][1])
        
        width_ok = (width >= self.config['vru_width_range'][0] and 
                     width <= self.config['vru_width_range'][1])
        
        length_ok = (length >= self.config['vru_length_range'][0] and 
                      length <= self.config['vru_length_range'][1])
        
        # Additional criteria
        density_ok = cluster_info['density'] > self.config['density_threshold']
        point_count_ok = 10 <= cluster_info['num_points'] <= 1000
        height_position_ok = cluster_info['center'][2] > 0.2  # Above ground
        
        # Calculate confidence score based on matched criteria
        score = 0
        if height_ok: score += 0.2
        if width_ok: score += 0.2
        if length_ok: score += 0.2
        if density_ok: score += 0.2
        if point_count_ok: score += 0.1
        if height_position_ok: score += 0.1
        
        return score
    
    def predict_vru_ml(self, features):
        """ML-based prediction for VRU."""
        if not self.use_ml:
            return 0.0
            
        self.model.eval()
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            confidence = self.model(features_tensor).item()
        
        return confidence
    
    def compute_oriented_bbox(self, cluster_points):
        """Compute an oriented 3D bounding box for a cluster."""
        # Extract x, y, z coordinates
        xyz = cluster_points[:, :3]
        
        # Compute center
        center = np.mean(xyz, axis=0)
        
        # Compute PCA for orientation
        if len(xyz) < 3:
            # Not enough points for PCA, return axis-aligned box
            min_point = np.min(xyz, axis=0)
            max_point = np.max(xyz, axis=0)
            dimensions = max_point - min_point
            return center, dimensions, 0.0
        
        # Center the points
        centered_points = xyz - center
        
        # Compute covariance matrix
        cov = np.cov(centered_points, rowvar=False)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Sort eigenvectors by eigenvalues (in descending order)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Project points onto eigenvectors
        projected = np.dot(centered_points, eigenvectors)
        
        # Calculate min/max along principal axes
        min_point = np.min(projected, axis=0)
        max_point = np.max(projected, axis=0)
        
        # Dimensions along principal axes
        dimensions = max_point - min_point
        
        # Calculate yaw (rotation around z-axis)
        # First eigenvector gives principal direction in xy plane
        main_axis = eigenvectors[:, 0]
        yaw = np.arctan2(main_axis[1], main_axis[0])
        
        return center, dimensions, yaw
    
    def detect(self, point_cloud_file):
        """Main detection pipeline."""
        # Load and preprocess point cloud
        points = self.load_point_cloud(point_cloud_file)
        filtered_points = self.preprocess_point_cloud(points)
        
        # Extract clusters
        clusters = self.extract_clusters(filtered_points)
        
        # Process each cluster
        results = []
        for cluster_points in clusters:
            # Compute features and cluster info
            features, cluster_info = self.compute_cluster_features(cluster_points)
            
            # Get rule-based confidence
            rule_confidence = self.is_vru_rule_based(cluster_info)
            
            # Get ML-based confidence if available
            ml_confidence = self.predict_vru_ml(features) if self.use_ml else 0.0
            
            # Combine confidences (weighted average)
            if self.use_ml:
                # Give more weight to rule-based when ML is less accurate
                confidence = 0.7 * rule_confidence + 0.3 * ml_confidence
            else:
                confidence = rule_confidence
            
            # Skip if confidence is too low
            if confidence < self.config['confidence_threshold']:
                continue
            
            # Compute bounding box
            center, dimensions, yaw = self.compute_oriented_bbox(cluster_points)
            
            # Add to results
            results.append({
                'center': center,
                'width': dimensions[0],
                'height': dimensions[2],  # Height is the Z dimension
                'length': dimensions[1],
                'yaw': yaw,
                'confidence': confidence
            })
        
        return results
    
    def save_results(self, results, output_file):
        """Save detection results to a file."""
        with open(output_file, 'w') as f:
            for result in results:
                center = result['center']
                line = f"Vru {center[0]:.6f} {center[1]:.6f} {center[2]:.6f} {result['width']:.6f} {result['height']:.6f} {result['yaw']:.6f}\n"
                f.write(line)

def process_dataset(input_dir, output_dir, use_ml=False, model_path=None):
    """Process an entire dataset of LiDAR scans."""
    # Initialize detector
    detector = VRUDetector(use_ml=use_ml)
    
    # Load model if using ML approach
    if use_ml and model_path is not None:
        detector.load_model(model_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of scan files
    if os.path.isdir(os.path.join(input_dir, 'scans')):
        scan_files = sorted(glob.glob(os.path.join(input_dir, 'scans', '*.bin')))
    else:
        scan_files = sorted(glob.glob(os.path.join(input_dir, '*.bin')))
    
    # Process each scan
    total_time = 0
    for i, scan_file in enumerate(scan_files):
        # Extract scan number
        scan_num = os.path.basename(scan_file).split('.')[0]
        
        # Process scan
        start_time = time.time()
        results = detector.detect(scan_file)
        end_time = time.time()
        
        # Calculate processing time
        process_time = end_time - start_time
        total_time += process_time
        
        # Save results
        output_file = os.path.join(output_dir, f'{scan_num}.txt')
        detector.save_results(results, output_file)
        
        # Print progress
        if (i + 1) % 100 == 0 or i == 0:
            print(f"Processed {i+1}/{len(scan_files)} scans. Avg time: {total_time / (i+1):.4f}s")
    
    # Print statistics
    avg_time = total_time / len(scan_files)
    print(f"Processing complete! Processed {len(scan_files)} scans.")
    print(f"Average processing time: {avg_time:.4f}s ({1/avg_time:.2f} Hz)")

def main():
    """Main entry point for the program."""
    parser = argparse.ArgumentParser(description='LiDAR VRU Detection')
    parser.add_argument('--input_dir', type=str, default='data', help='Input directory containing scans')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory for results')
    parser.add_argument('--model_path', type=str, default='vru_model.pth', help='Path to model weights (optional)')
    parser.add_argument('--use_ml', action='store_true', help='Use ML-based approach in combination with rules')
    
    args = parser.parse_args()
    
    process_dataset(args.input_dir, args.output_dir, args.use_ml, args.model_path)

if __name__ == "__main__":
    main()