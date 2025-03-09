import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob
from sklearn.cluster import DBSCAN
from config import CLASS_MAP, SIMPLIFIED_CLASS_MAP

class LiDARVRUDataset(Dataset):
    def __init__(self, data_dir, split='train', config=None):
        self.data_dir = data_dir
        self.split = split
        self.scan_dir = os.path.join(data_dir, 'scans')
        self.label_dir = os.path.join(data_dir, 'labels')
        self.config = config
        
        # Get file paths
        self.scan_files = sorted(glob.glob(os.path.join(self.scan_dir, '*.bin')))
        
        # Dictionary to store labels
        self.labels = []
        self.processed_features = []
        self.processed_labels = []
        
        # Load and preprocess data
        self._preprocess_data()
        
    def __len__(self):
        return len(self.processed_features)
    
    def __getitem__(self, idx):
        features = self.processed_features[idx]
        label = self.processed_labels[idx]
        
        # Convert to tensors
        features = torch.FloatTensor(features)
        label = torch.LongTensor([label])[0]
        
        return {
            'features': features,
            'label': label,
            'cls_labels': torch.zeros(len(CLASS_MAP)),  # Placeholder for compatibility
            'box_labels': torch.zeros(7),  # Placeholder for compatibility
            'point_cloud': torch.zeros(100, 3)  # Placeholder for compatibility
        }
    
    def _preprocess_data(self):
        """Preprocess data and extract features"""
        print("Preprocessing dataset...")
        all_class_ids = set()  # Track all class IDs for debugging
        class_counts = {}  # Track counts of each class
        
        for scan_file in self.scan_files:
            # Extract scan ID
            scan_id = os.path.basename(scan_file).split('.')[0]
            
            # Load corresponding label file
            label_file = os.path.join(self.label_dir, f"{scan_id}.txt")
            
            if not os.path.exists(label_file):
                continue
                
            # Load point cloud
            points = self._load_point_cloud(scan_file)
            
            # Load labels
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                
                # Parse label data
                class_name = parts[0]
                center_x, center_y, center_z = float(parts[1]), float(parts[2]), float(parts[3])
                width, height, length = float(parts[4]), float(parts[5]), float(parts[6])
                yaw = float(parts[7]) if len(parts) > 7 else 0.0
                
                # Map class name to ID
                class_id = None
                for id, name in CLASS_MAP.items():
                    if name.split('.')[-1] in class_name:
                        class_id = id
                        break
                
                if class_id is None:
                    continue
                
                # Track class ID for debugging
                all_class_ids.add(class_id)
                
                # Count classes
                if class_id not in class_counts:
                    class_counts[class_id] = 0
                class_counts[class_id] += 1
                
                # Extract points in the bounding box
                bbox_points = self._extract_bbox_points(points, 
                                                      [center_x, center_y, center_z], 
                                                      [width, height, length], 
                                                      yaw)
                
                if len(bbox_points) < 5:
                    continue
                
                # Compute features for the points
                features = self.compute_features(bbox_points)
                
                # Add to processed data
                self.processed_features.append(features)
                self.processed_labels.append(class_id)
                
                # Add to global labels list
                self.labels.append({
                    'scan_id': scan_id,
                    'class_id': class_id,
                    'bbox': [center_x, center_y, center_z, width, height, length, yaw]
                })
        
        print(f"Dataset preprocessed: {len(self.processed_features)} samples")
        print("Class distribution:")
        
        # Debug: Print all found class IDs
        print(f"All class IDs found in dataset: {sorted(all_class_ids)}")
        
        # Print class distribution with safe access to class_map
        for cls, count in class_counts.items():
            try:
                class_name = CLASS_MAP[cls].split('.')[-1]  # Get shortened class name
            except KeyError:
                class_name = f"Unknown class {cls}"
            print(f"  {class_name}: {count} samples ({count/len(self.labels)*100:.2f}%)")
    
    def _load_point_cloud(self, bin_file):
        """Load point cloud from binary file"""
        points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 5)
        return points
    
    def _extract_bbox_points(self, points, center, dimensions, yaw):
        """Extract points inside a rotated 3D bounding box"""
        # Convert points to local coordinates
        local_points = points[:, :3] - np.array(center)
        
        # Rotation matrix for yaw
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        rotation_matrix = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])
        
        # Rotate points to align with box
        rotated_points = np.dot(local_points, rotation_matrix.T)
        
        # Check if points are inside the box
        half_width, half_height, half_length = dimensions[0]/2, dimensions[1]/2, dimensions[2]/2
        mask_x = np.abs(rotated_points[:, 0]) <= half_width
        mask_y = np.abs(rotated_points[:, 1]) <= half_length
        mask_z = np.abs(rotated_points[:, 2]) <= half_height
        mask = np.logical_and(np.logical_and(mask_x, mask_y), mask_z)
        
        # Return points inside the box
        return points[mask]
    
    def compute_features(self, points):
        """
        UNIFIED FEATURE EXTRACTION
        Compute consistent features for a cluster of points.
        Used by both training and inference code.
        """
        # Geometric features
        xyz = points[:, :3]
        min_point = np.min(xyz, axis=0)
        max_point = np.max(xyz, axis=0)
        center = np.mean(xyz, axis=0)
        dimensions = max_point - min_point
        
        # Calculate eigenvalues for shape analysis
        covariance = np.cov(xyz, rowvar=False)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            # Sort eigenvalues in descending order
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            
            # Normalize eigenvalues
            if np.sum(eigenvalues) > 0:
                norm_eigenvalues = eigenvalues / np.sum(eigenvalues)
                
                # Shape features based on eigenvalues
                linearity = (norm_eigenvalues[0] - norm_eigenvalues[1]) / (norm_eigenvalues[0] + 1e-6)
                planarity = (norm_eigenvalues[1] - norm_eigenvalues[2]) / (norm_eigenvalues[0] + 1e-6)
                sphericity = norm_eigenvalues[2] / (norm_eigenvalues[0] + 1e-6)
                anisotropy = (norm_eigenvalues[0] - norm_eigenvalues[2]) / (norm_eigenvalues[0] + 1e-6)
            else:
                norm_eigenvalues = np.zeros(3)
                linearity, planarity, sphericity, anisotropy = 0, 0, 0, 0
        except np.linalg.LinAlgError:
            eigenvalues = np.zeros(3)
            norm_eigenvalues = np.zeros(3)
            linearity, planarity, sphericity, anisotropy = 0, 0, 0, 0
        
        # Statistical features
        std_xyz = np.std(xyz, axis=0)
        intensity = points[:, 3]
        intensity_stats = [np.mean(intensity), np.std(intensity), np.min(intensity), np.max(intensity)]
        
        # Aspect ratios
        width_height_ratio = dimensions[0] / (dimensions[2] + 1e-6)
        length_height_ratio = dimensions[1] / (dimensions[2] + 1e-6)
        width_length_ratio = dimensions[0] / (dimensions[1] + 1e-6)
        
        # Point count and density
        point_count = len(points)
        volume = dimensions[0] * dimensions[1] * dimensions[2]
        density = point_count / (volume + 1e-10)
        
        # Height distribution (useful for pedestrians vs vehicles)
        height_percentiles = np.percentile(xyz[:, 2], [25, 50, 75]) if len(xyz) > 0 else np.zeros(3)
        
        # Distance features
        distance_to_center = np.linalg.norm(xyz - center, axis=1)
        mean_distance = np.mean(distance_to_center)
        std_distance = np.std(distance_to_center)
        max_distance = np.max(distance_to_center)
        
        # Combine all features
        features = np.concatenate([
            center,                       # 3 features
            dimensions,                   # 3 features
            [width_height_ratio, length_height_ratio, width_length_ratio],  # 3 features
            eigenvalues,                  # 3 features
            norm_eigenvalues,             # 3 features
            [linearity, planarity, sphericity, anisotropy],  # 4 features
            std_xyz,                      # 3 features
            intensity_stats,              # 4 features
            height_percentiles,           # 3 features
            [mean_distance, std_distance, max_distance],  # 3 features
            [point_count, density]        # 2 features
        ])
        
        # Pad to get to feature size of 64 (if needed for model)
        if len(features) < 64:
            padding = np.zeros(64 - len(features))
            features = np.concatenate([features, padding])
        elif len(features) > 64:
            features = features[:64]  # Truncate if too many features
        
        return features
    
    def get_split(self, split):
        """Return self for compatibility with Open3D-ML"""
        return self
    
    def extract_clusters(self, points, config):
        """Extract clusters from point cloud using DBSCAN"""
        # Use only x, y, z for clustering
        xyz = points[:, :3]
        
        # Apply DBSCAN clustering
        db = DBSCAN(
            eps=config['cluster_eps'],
            min_samples=config['cluster_min_samples'],
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
            if np.sum(cluster_mask) < config['min_points_threshold']:
                continue
                
            cluster_points = points[cluster_mask]
            clusters.append(cluster_points)
        
        return clusters


class VRUDetectionModel(nn.Module):
    """
    Unified VRU detection model used by both training and inference
    """
    def __init__(self, input_dim=64, num_classes=9):
        super(VRUDetectionModel, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        self.classifier = nn.Linear(128, num_classes)
        
    def forward(self, x):
        if isinstance(x, dict):
            # Handle input from DataLoader
            x = x['features']
            
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        
        # Format output to match Open3D-ML expectations
        return {
            'cls_preds': logits,
            'box_preds': torch.zeros((x.size(0), 7), device=x.device),  # Placeholder
        }


# Utility functions for both training and inference
def preprocess_point_cloud(points, config):
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
    pc_range = config['point_cloud_range']
    mask_x = np.logical_and(x >= pc_range[0], x <= pc_range[3])
    mask_y = np.logical_and(y >= pc_range[1], y <= pc_range[4])
    mask_z = np.logical_and(z >= pc_range[2], z <= pc_range[5])
    mask_range = np.logical_and(np.logical_and(mask_x, mask_y), mask_z)
    
    # Filter by intensity
    mask_intensity = intensity > config['intensity_threshold']
    
    # Filter ground points
    mask_not_ground = z > config['ground_height_threshold']
    
    # Combine masks
    mask = np.logical_and(np.logical_and(mask_range, mask_intensity), mask_not_ground)
    filtered_points = points[mask]
    
    return filtered_points

def compute_oriented_bbox(cluster_points):
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
