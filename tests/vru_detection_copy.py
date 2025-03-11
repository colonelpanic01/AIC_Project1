import numpy as np
import os
import glob
import time
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN
import argparse

# Updated configuration with more precise dimension ranges
CONFIG = {
    'voxel_size': [0.05, 0.05, 0.05],  # Finer voxel size for better resolution
    'point_cloud_range': [-50, -50, -5, 50, 50, 3],  # Range of point cloud to consider
    'max_points_per_voxel': 64,
    'max_voxels': 40000,
    
    # All pedestrian types with specific ranges
    'pedestrian': {
        'adult': {
            'width_range': [0.282, 1.505],
            'length_range': [0.214, 1.674],
            'height_range': [0.585, 2.744],
            'yaw_range': [-4.712371700070678, 1.570789171031985]
        },
        'construction_worker': {
            'width_range': [0.345, 1.971],
            'length_range': [0.293, 1.521],
            'height_range': [0.293, 2.573],
            'yaw_range': [-4.712387760134593, 1.5705286304974742]
        },
        'child': {
            'width_range': [0.295, 0.93],
            'length_range': [0.268, 0.995],
            'height_range': [0.724, 2.0],
            'yaw_range': [-4.710222062114154, 1.5624351131644398]
        },
        'wheelchair': {
            'width_range': [0.496, 0.876],
            'length_range': [0.682, 1.538],
            'height_range': [1.229, 1.532],
            'yaw_range': [-3.5224930470241946, 1.4828541376923488]
        },
        'personal_mobility': {
            'width_range': [0.298, 0.886],
            'length_range': [0.494, 2.239],
            'height_range': [0.846, 2.0],
            'yaw_range': [-4.6510599883954304, 1.505014419205605]
        },
        'police_officer': {
            'width_range': [0.527, 1.155],
            'length_range': [0.451, 1.024],
            'height_range': [1.394, 2.028],
            'yaw_range': [-4.709856889394767, 1.5693674589034963]
        },
        'stroller': {
            'width_range': [0.362, 0.87],
            'length_range': [0.418, 1.753],
            'height_range': [0.789, 1.888],
            'yaw_range': [-4.691451416352464, 1.5469162803724985]
        },
        # Aggregate ranges for any pedestrian
        'general': {
            'width_range': [0.282, 1.971],
            'length_range': [0.214, 2.239],
            'height_range': [0.293, 2.744],
            'yaw_range': [-4.712387760134593, 1.5705286304974742]
        }
    },
    
    'bicycle': {
        'width_range': [0.233, 1.661],
        'length_range': [0.454, 3.04],
        'height_range': [0.349, 2.223],
        'yaw_range': [-4.710947933322235, 1.5702342135720362]
    },
    
    'motorcycle': {
        'width_range': [0.351, 1.816],
        'length_range': [0.72, 4.409],
        'height_range': [0.791, 2.02],
        'yaw_range': [-4.7115172916358325, 1.5695307294768606]
    },
    
    # Clustering and filtering parameters
    'min_points_threshold': 20,  # Increased from 15 for more reliable clusters
    'cluster_eps': 0.35,  # Reduced for finer clustering
    'cluster_min_samples': 8,  # Increased from 7 for more robust clusters
    'ground_height_threshold': -1.5,
    'intensity_threshold': 0.075,  # Adjusted for better point filtering
    
    # Advanced parameters
    'use_height_clustering': True,  # Use height-based clustering for better separation
    'use_adaptive_clustering': True,  # Adapt clustering parameters based on distance
    'multi_scale_clustering': True,  # Use multiple scales for clustering
    'confidence_threshold': 0.6,  # Minimum confidence to report a detection
}

class LiDARVRUDetector:
    def __init__(self, config=None):
        self.config = CONFIG if config is None else config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = self._build_model()
        
    def _build_model(self):
        """
        Enhanced neural network model for VRU detection with more output classes
        """
        # More sophisticated network architecture with increased capacity
        model = nn.Sequential(
            # Input layer (expanded feature size to 96)
            nn.Linear(96, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # First hidden layer
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Second hidden layer
            nn.Linear(256, 512),  # Larger middle layer for more capacity
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Third hidden layer
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Fourth hidden layer
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Output layer - 9 classes for all VRU types
            # 7 pedestrian types + bicycle + motorcycle
            nn.Linear(128, 9)
        )
        return model.to(self.device)

    def load_model(self, model_path):
        """Load model weights from a file."""
        print(f"Loading model from {model_path}...")
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint, strict=False)  # Allow partial loading
            print("Model loaded successfully!")
        except Exception as e:
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
        Enhanced preprocessing with better filtering for VRU detection
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
        
        # Enhanced ground removal
        # Remove obvious ground points (assuming relatively flat surface)
        if 'ground_height_threshold' in self.config:
            mask_non_ground = z > self.config['ground_height_threshold']
            mask_range = np.logical_and(mask_range, mask_non_ground)
        
        # Filter by intensity (more reflective objects)
        mask_intensity = intensity > self.config['intensity_threshold']
        
        # Distance-adaptive intensity filtering 
        # Objects further away have lower intensity
        distance = np.sqrt(x**2 + y**2)
        distance_factor = np.clip(distance / 30.0, 0.5, 1.5)  # Normalize by 30m
        adaptive_intensity_threshold = self.config['intensity_threshold'] / distance_factor
        mask_adaptive_intensity = intensity > adaptive_intensity_threshold
        
        # Combine masks
        mask = np.logical_and(mask_range, np.logical_or(mask_intensity, mask_adaptive_intensity))
        filtered_points = points[mask]
        
        return filtered_points
    
    def extract_clusters(self, points):
        """
        Enhanced clustering with adaptive parameters and multi-scale approach
        """
        # Use only x, y, z for clustering
        xyz = points[:, :3]
        
        # Get distances from origin for adaptive clustering
        distances = np.sqrt(xyz[:, 0]**2 + xyz[:, 1]**2)
        
        # If adaptive clustering enabled, adjust eps based on distance
        if self.config.get('use_adaptive_clustering', False):
            # Points further away need larger eps
            distance_factor = np.mean(distances) / 20.0  # Normalize by 20m
            eps = self.config['cluster_eps'] * max(1.0, distance_factor)
        else:
            eps = self.config['cluster_eps']
        
        clusters = []
        
        # For multi-scale clustering
        if self.config.get('multi_scale_clustering', False):
            # Multiple scales to catch both small and large objects
            eps_scales = [eps * 0.7, eps, eps * 1.4]
            min_samples_scales = [
                int(self.config['cluster_min_samples'] * 1.2),
                self.config['cluster_min_samples'],
                max(4, int(self.config['cluster_min_samples'] * 0.8))
            ]
            
            # Combined clustering results
            all_labels = np.full(len(xyz), -1, dtype=int)
            next_label = 0
            
            for scale_eps, scale_min_samples in zip(eps_scales, min_samples_scales):
                # Apply DBSCAN at this scale
                db = DBSCAN(
                    eps=scale_eps,
                    min_samples=scale_min_samples,
                    n_jobs=-1
                ).fit(xyz)
                
                scale_labels = db.labels_
                
                # Add new clusters (avoiding duplicates)
                for label in np.unique(scale_labels):
                    if label == -1:
                        continue
                    
                    mask = scale_labels == label
                    # Skip if most points already assigned to a cluster
                    if np.mean(all_labels[mask] != -1) > 0.5:
                        continue
                    
                    # Add as new cluster
                    all_labels[mask] = next_label
                    next_label += 1
                
            # Process combined labels
            for label in range(next_label):
                mask = all_labels == label
                if np.sum(mask) < self.config['min_points_threshold']:
                    continue
                
                clusters.append(points[mask])
        else:
            # Standard DBSCAN clustering
            db = DBSCAN(
                eps=eps,
                min_samples=self.config['cluster_min_samples'],
                n_jobs=-1
            ).fit(xyz)
            
            labels = db.labels_
            unique_labels = np.unique(labels)
            
            # Collect clusters (excluding noise with label -1)
            for label in unique_labels:
                if label == -1:
                    continue
                    
                cluster_mask = labels == label
                if np.sum(cluster_mask) < self.config['min_points_threshold']:
                    continue
                    
                clusters.append(points[cluster_mask])
        
        # If enabled, use height-based splitting to separate stacked VRUs
        if self.config.get('use_height_clustering', False):
            refined_clusters = []
            for cluster_points in clusters:
                # Check if this could be vertically stacked VRUs
                z_values = cluster_points[:, 2]
                z_range = np.max(z_values) - np.min(z_values)
                
                # Only try to split tall clusters
                if z_range > 1.8:  # Taller than typical single VRU
                    # Try to split by height
                    z_db = DBSCAN(
                        eps=0.4,  # Smaller eps for height separation
                        min_samples=5,
                        n_jobs=-1
                    ).fit(z_values.reshape(-1, 1))
                    
                    z_labels = z_db.labels_
                    z_unique_labels = np.unique(z_labels)
                    
                    # If we found multiple height clusters
                    if len(z_unique_labels) > 1 and -1 not in z_unique_labels:
                        for z_label in z_unique_labels:
                            z_mask = z_labels == z_label
                            if np.sum(z_mask) < self.config['min_points_threshold'] / 2:
                                continue
                            
                            refined_clusters.append(cluster_points[z_mask])
                        continue  # Skip adding the original cluster
                
                # If not split, add original cluster
                refined_clusters.append(cluster_points)
                
            return refined_clusters
        
        return clusters
    
    def compute_cluster_features(self, cluster_points):
        """
        Compute enhanced features for a cluster of points
        """
        # Basic geometric features
        xyz = cluster_points[:, :3]
        min_point = np.min(xyz, axis=0)
        max_point = np.max(xyz, axis=0)
        center = np.mean(xyz, axis=0)
        dimensions = max_point - min_point
        
        # Distance from origin (important for context)
        distance_from_origin = np.sqrt(center[0]**2 + center[1]**2)
        
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
        
        # Height distribution (useful for separating VRU types)
        z_values = xyz[:, 2]
        height_percentiles = np.percentile(z_values, [5, 25, 50, 75, 95])
        height_histogram = np.histogram(z_values, bins=8, range=(min_point[2], max_point[2]))[0]
        height_histogram = height_histogram / (np.sum(height_histogram) + 1e-6)  # Normalize
        
        # Point distribution features
        distance_to_center = np.linalg.norm(xyz - center, axis=1)
        mean_distance = np.mean(distance_to_center)
        std_distance = np.std(distance_to_center)
        max_distance = np.max(distance_to_center)
        
        # Shape features from covariance
        eigenvalues, eigenvectors = self._compute_eigenvalues_vectors(xyz)
        if len(eigenvalues) == 3 and np.sum(eigenvalues) > 0:
            # Normalized eigenvalues for shape description
            normalized_evals = eigenvalues / np.sum(eigenvalues)
            linearity = (normalized_evals[0] - normalized_evals[1]) / (normalized_evals[0] + 1e-6)
            planarity = (normalized_evals[1] - normalized_evals[2]) / (normalized_evals[0] + 1e-6)
            sphericity = normalized_evals[2] / (normalized_evals[0] + 1e-6)
            anisotropy = (normalized_evals[0] - normalized_evals[2]) / (normalized_evals[0] + 1e-6)
            curvature = normalized_evals[2] / (np.sum(normalized_evals) + 1e-6)
            
            # Principal directions
            main_direction = eigenvectors[:, 0]
            secondary_direction = eigenvectors[:, 1]
            minor_direction = eigenvectors[:, 2]
            
            # Verticalness (dot product with up vector [0,0,1])
            verticalness = np.abs(np.dot(main_direction, [0, 0, 1]))
            
            # Orientation features
            yaw = np.arctan2(main_direction[1], main_direction[0])
            
            # Other shape descriptors
            surface_variation = eigenvalues[2] / (eigenvalues[0] + eigenvalues[1] + eigenvalues[2] + 1e-6)
            omnivariance = (eigenvalues[0] * eigenvalues[1] * eigenvalues[2] + 1e-6) ** (1/3)
        else:
            linearity, planarity, sphericity, anisotropy, curvature = 0, 0, 0, 0, 0
            verticalness, yaw = 0, 0
            surface_variation, omnivariance = 0, 0
            main_direction = [0, 0, 0]
            secondary_direction = [0, 0, 0]
            minor_direction = [0, 0, 0]
        
        # Density features
        volume = dimensions[0] * dimensions[1] * dimensions[2]
        point_density = len(cluster_points) / (volume + 1e-6)
        
        # Create 2D occupancy grid (top-down view)
        grid_size = 8
        x_bins = np.linspace(min_point[0], max_point[0], grid_size+1)
        y_bins = np.linspace(min_point[1], max_point[1], grid_size+1)
        
        # Compute histogram
        occupancy_grid, _, _ = np.histogram2d(
            xyz[:, 0], xyz[:, 1], 
            bins=[x_bins, y_bins]
        )
        # Normalize grid
        occupancy_grid = occupancy_grid / np.sum(occupancy_grid)
        # Flatten to 1D array
        occupancy_grid = occupancy_grid.flatten()
        
        # Intensity distribution across height
        # Correlation between height and intensity
        intensity_height_corr = np.corrcoef(z_values, intensity_values)[0, 1] if len(z_values) > 1 else 0
        
        # Combine all features into a single vector
        features = np.concatenate([
            center,  # 3 features
            dimensions,  # 3 features
            [width_height_ratio, length_height_ratio, width_length_ratio],  # 3 features
            std_dev,  # 3 features
            [intensity_mean, intensity_std, intensity_min, intensity_max],  # 4 features
            [mean_distance, std_distance, max_distance],  # 3 features
            height_percentiles,  # 5 features
            height_histogram,  # 8 features
            [point_density, distance_from_origin],  # 2 features
            eigenvalues,  # 3 features
            main_direction, secondary_direction, minor_direction,  # 9 features
            [verticalness, yaw],  # 2 features
            [linearity, planarity, sphericity, anisotropy, curvature],  # 5 features
            [surface_variation, omnivariance],  # 2 features
            occupancy_grid,  # grid_size^2 = 64 features
            [intensity_height_corr],  # 1 feature
            [len(cluster_points)]  # 1 feature - number of points
        ])  # Total: 121 features
        
        # Trim to 96 features (our model input size) by taking most important ones
        # We'll drop some of the less important occupancy grid cells
        if len(features) > 96:
            # Keep first 32 features and last feature (point count)
            keep_indices = list(range(32)) + list(range(len(features)-1, len(features)))
            # Add some occupancy grid features but not all
            keep_indices += list(range(57, 57+96-len(keep_indices)))
            features = features[keep_indices]
        
        # Pad if needed (shouldn't happen with our feature selection)
        if len(features) < 96:
            padding = np.zeros(96 - len(features))
            features = np.concatenate([features, padding])
        
        return features

    def _compute_eigenvalues_vectors(self, points):
        """Compute eigenvalues and eigenvectors of the covariance matrix for shape analysis"""
        if len(points) < 3:
            return np.zeros(3), np.zeros((3, 3))
            
        cov = np.cov(points, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Sort in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        return eigenvalues, eigenvectors
    
    def classify_cluster_rule_based(self, cluster_points, features):
        """
        Enhanced rule-based classification using precise dimension ranges
        """
        xyz = cluster_points[:, :3]
        min_point = np.min(xyz, axis=0)
        max_point = np.max(xyz, axis=0)
        height = max_point[2] - min_point[2]
        width = max_point[0] - min_point[0]
        length = max_point[1] - min_point[1]
        
        # Get orientation for better dimension alignment
        eigenvalues, eigenvectors = self._compute_eigenvalues_vectors(xyz)
        main_direction = eigenvectors[:, 0]
        yaw = np.arctan2(main_direction[1], main_direction[0])
        
        # Adjust dimensions based on orientation
        # For objects that are not axis-aligned, width and length might be swapped
        if abs(np.sin(yaw)) > 0.7:  # If object is rotated more than ~45 degrees
            # Swap width and length for better dimension matching
            width, length = length, width
        
        # Initialize scores for all VRU classes
        scores = {
            "pedestrian.adult": 0,
            "pedestrian.construction_worker": 0,
            "pedestrian.child": 0,
            "pedestrian.wheelchair": 0,
            "pedestrian.personal_mobility": 0,
            "pedestrian.police_officer": 0,
            "pedestrian.stroller": 0,
            "bicycle": 0,
            "motorcycle": 0
        }
        
        # Helper function to calculate dimension match score
        def dim_match_score(value, range_min, range_max):
            if value < range_min or value > range_max:
                return 0
            
            # Calculate how centered the value is within the range
            # 1.0 = perfect center, 0.0 = at the edge
            range_size = range_max - range_min
            center = range_min + range_size / 2
            distance_from_center = abs(value - center)
            normalized_distance = distance_from_center / (range_size / 2)
            return 1.0 - normalized_distance * 0.5  # Penalty is only 50% at the edge
        
        # Helper function to check if value is in range
        def in_range(value, range_min, range_max):
            return value >= range_min and value <= range_max
        
        # Check all pedestrian types
        for ped_type, ranges in self.config['pedestrian'].items():
            if ped_type == 'general':
                continue  # Skip the general range
                
            score = 0
            
            # Check dimensions
            width_score = dim_match_score(width, ranges['width_range'][0], ranges['width_range'][1])
            length_score = dim_match_score(length, ranges['length_range'][0], ranges['length_range'][1])
            height_score = dim_match_score(height, ranges['height_range'][0], ranges['height_range'][1])
            
            # Add scores with weights (height is most important)
            score += width_score * 0.25
            score += length_score * 0.25
            score += height_score * 0.5
            
            # Additional type-specific characteristics
            if ped_type == 'adult':
                # Adults typically have height > width
                if height > width * 1.5:
                    score += 0.2
                
                # Point count for adults is usually moderate
                if 30 <= len(cluster_points) <= 150:
                    score += 0.1
                    
            elif ped_type == 'construction_worker':
                # Construction workers often have high intensity (reflective gear)
                intensity_values = cluster_points[:, 3]
                if np.mean(intensity_values) > 0.3:
                    score += 0.3
                    
            elif ped_type == 'child':
                # Children are smaller
                if height < 1.5 and width < 0.6:
                    score += 0.3
                    
            elif ped_type == 'wheelchair':
                # Wheelchairs have distinctive width/height ratio
                if width > height * 0.5:
                    score += 0.3
                    
            elif ped_type == 'personal_mobility':
                # Personal mobility devices have distinctive length
                if length > width * 1.5:
                    score += 0.3
                    
            elif ped_type == 'police_officer':
                # Similar to adults but may have higher intensity
                intensity_values = cluster_points[:, 3]
                if np.mean(intensity_values) > 0.25:
                    score += 0.2
                    
            elif ped_type == 'stroller':
                # Strollers have distinctive height distribution
                z_values = xyz[:, 2]
                z_std = np.std(z_values)
                if z_std < 0.4:  # Less variation in height
                    score += 0.3
            
            # Cap score at 1.0
            score = min(1.0, score)
            scores[f"pedestrian.{ped_type}"] = score
            
        # Check bicycle
        bicycle_width_score = dim_match_score(width, self.config['bicycle']['width_range'][0], self.config['bicycle']['width_range'][1])
        bicycle_length_score = dim_match_score(length, self.config['bicycle']['length_range'][0], self.config['bicycle']['length_range'][1])
        bicycle_height_score = dim_match_score(height, self.config['bicycle']['height_range'][0], self.config['bicycle']['height_range'][1])
        
        bicycle_score = bicycle_width_score * 0.25 + bicycle_length_score * 0.4 + bicycle_height_score * 0.35
        
        # Additional bicycle characteristics
        # Bicycles typically have length > width
        if length > width * 1.5:
            bicycle_score += 0.2
            
        # Bicycles have distinctive point patterns
        if 20 <= len(cluster_points) <= 200:
            bicycle_score += 0.1
            
        scores["bicycle"] = min(1.0, bicycle_score)
        
        # Check motorcycle
        motorcycle_width_score = dim_match_score(width, self.config['motorcycle']['width_range'][0], self.config['motorcycle']['width_range'][1])
        motorcycle_length_score = dim_match_score(length, self.config['motorcycle']['length_range'][0], self.config['motorcycle']['length_range'][1])
        motorcycle_height_score = dim_match_score(height, self.config['motorcycle']['height_range'][0], self.config['motorcycle']['height_range'][1])
        
        motorcycle_score = motorcycle_width_score * 0.2 + motorcycle_length_score * 0.5 + motorcycle_height_score * 0.3
        
        # Additional motorcycle characteristics
        # Motorcycles typically have length > width
        if length > width * 2.0:
            motorcycle_score += 0.2
            
        # Motorcycles generally have more points than bicycles
        if len(cluster_points) > 100:
            motorcycle_score += 0.2
            
        # Motorcycles generally have higher intensity (metal parts)
        intensity_values = cluster_points[:, 3]
        if np.mean(intensity_values) > 0.3:
            motorcycle_score += 0.1
            
        scores["motorcycle"] = min(1.0, motorcycle_score)
        
        # Find best matching class
        best_class = max(scores.items(), key=lambda x: x[1])
        class_name, confidence = best_class
        
        # Only return high-confidence results
        if confidence > self.config.get('confidence_threshold', 0.5):
            # Map pedestrian subtypes to general pedestrian class if needed
            if class_name.startswith("pedestrian."):
                return "pedestrian", confidence, class_name.split(".", 1)[1]
            else:
                return class_name, confidence, None
        else:
            # See if general pedestrian matches better
            ped_ranges = self.config['pedestrian']['general']
            if (in_range(width, ped_ranges['width_range'][0], ped_ranges['width_range'][1]) and
                in_range(length, ped_ranges['length_range'][0], ped_ranges['length_range'][1]) and
                in_range(height, ped_ranges['height_range'][0], ped_ranges['height_range'][1])):
                return "pedestrian", 0.6, "general"
        return None, 0, None
    def classify_cluster_nn(self, features):
        """
        Neural network-based classification of cluster features
        """
        # Convert features to tensor
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Run inference
        with torch.no_grad():
            logits = self.model(features_tensor)
            probs = torch.nn.functional.softmax(logits, dim=1).squeeze(0)
            
        # Get predicted class
        class_idx = torch.argmax(probs).item()
        confidence = probs[class_idx].item()
        
        # Map class index to name
        class_names = [
            "pedestrian.adult",
            "pedestrian.construction_worker",
            "pedestrian.child",
            "pedestrian.wheelchair",
            "pedestrian.personal_mobility",
            "pedestrian.police_officer",
            "pedestrian.stroller",
            "bicycle",
            "motorcycle"
        ]
        
        class_name = class_names[class_idx]
        
        # Map pedestrian subtypes to general pedestrian class
        if class_name.startswith("pedestrian."):
            return "pedestrian", confidence, class_name.split(".", 1)[1]
        else:
            return class_name, confidence, None
    
    def predict(self, bin_file):
        """
        Main prediction function for VRU detection
        """
        # Start timing
        start_time = time.time()
        
        # Load and preprocess point cloud
        raw_points = self.load_point_cloud(bin_file)
        points = self.preprocess_point_cloud(raw_points)
        
        # Extract clusters
        clusters = self.extract_clusters(points)
        
        # Process all clusters
        detections = []
        
        for cluster_points in clusters:
            # Skip small clusters
            if len(cluster_points) < self.config['min_points_threshold']:
                continue
                
            # Extract features
            features = self.compute_cluster_features(cluster_points)
            
            # Classify using rule-based method first for validation
            rule_class, rule_conf, rule_subtype = self.classify_cluster_rule_based(
                cluster_points, features)
                
            # If model is loaded, use neural network for classification
            if hasattr(self, 'model'):
                nn_class, nn_conf, nn_subtype = self.classify_cluster_nn(features)
                
                # Ensemble approach - combine rule-based and NN predictions
                if rule_class is not None and nn_class is not None:
                    # If both methods agree on the base class
                    if rule_class == nn_class:
                        final_class = rule_class
                        final_conf = (rule_conf + nn_conf) / 2  # Average confidence
                        # Use NN subtype if available, otherwise use rule-based
                        final_subtype = nn_subtype if nn_subtype is not None else rule_subtype
                    # If they disagree, use the one with higher confidence
                    else:
                        if rule_conf > nn_conf:
                            final_class = rule_class
                            final_conf = rule_conf
                            final_subtype = rule_subtype
                        else:
                            final_class = nn_class
                            final_conf = nn_conf
                            final_subtype = nn_subtype
                # If only one method produced a result
                elif rule_class is not None:
                    final_class = rule_class
                    final_conf = rule_conf
                    final_subtype = rule_subtype
                elif nn_class is not None:
                    final_class = nn_class
                    final_conf = nn_conf
                    final_subtype = nn_subtype
                else:
                    continue  # Skip if both methods failed
            else:
                # Only use rule-based if no model is loaded
                if rule_class is None:
                    continue
                    
                final_class = rule_class
                final_conf = rule_conf
                final_subtype = rule_subtype
            
            # Skip low confidence detections
            if final_conf < self.config['confidence_threshold']:
                continue
                
            # Extract bounding box parameters
            xyz = cluster_points[:, :3]
            min_point = np.min(xyz, axis=0)
            max_point = np.max(xyz, axis=0)
            center = np.mean(xyz, axis=0)
            dimensions = max_point - min_point
            
            # Get orientation for better dimension alignment
            eigenvalues, eigenvectors = self._compute_eigenvalues_vectors(xyz)
            main_direction = eigenvectors[:, 0]
            yaw = np.arctan2(main_direction[1], main_direction[0])
            
            # Add detection to results
            detection = {
                'class': final_class,
                'subtype': final_subtype,
                'confidence': final_conf,
                'x': center[0],
                'y': center[1],
                'z': center[2],
                'width': dimensions[0],
                'length': dimensions[1],
                'height': dimensions[2],
                'yaw': yaw,
                'num_points': len(cluster_points)
            }
            detections.append(detection)
        
        # End timing
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        return detections, elapsed_time

    def train(self, data_dir, label_dir, output_model_path, epochs=50, batch_size=64, validation_split=0.2):
        """
        Train the model on labeled data
        """
        print(f"Starting training with {epochs} epochs...")
        
        # Find all input files
        bin_files = sorted(glob.glob(os.path.join(data_dir, '*.bin')))
        label_files = sorted(glob.glob(os.path.join(label_dir, '*.txt')))
        
        if len(bin_files) != len(label_files):
            print(f"Warning: Number of bin files ({len(bin_files)}) does not match number of label files ({len(label_files)})")
        
        if len(bin_files) == 0:
            print("Error: No training data found.")
            return
            
        print(f"Found {len(bin_files)} training samples.")
        
        # Process all files and prepare training data
        X = []  # Features
        y = []  # Labels
        
        # Class mapping for labels
        class_mapping = {
            "human.pedestrian.adult": 0,
            "human.pedestrian.construction_worker": 1,
            "human.pedestrian.child": 2,
            "human.pedestrian.wheelchair": 3,
            "human.pedestrian.personal_mobility": 4,
            "human.pedestrian.police_officer": 5,
            "human.pedestrian.stroller": 6,
            "vehicle.bicycle": 7,
            "vehicle.motorcycle": 8
        }
        
        # Process each sample
        for bin_file, label_file in zip(bin_files, label_files):
            print(f"Processing {os.path.basename(bin_file)}...")
            
            # Load and preprocess point cloud
            raw_points = self.load_point_cloud(bin_file)
            points = self.preprocess_point_cloud(raw_points)
            
            # Extract clusters
            clusters = self.extract_clusters(points)
            
            # Load ground truth labels
            with open(label_file, 'r') as f:
                gt_labels = [line.strip().split() for line in f.readlines()]
            
            # Process each ground truth label
            for label in gt_labels:
                # Parse label data
                class_name = label[0]
                
                # Skip if not in our classes of interest
                if class_name not in class_mapping:
                    continue
                    
                try:
                    # Parse bounding box parameters
                    x, y, z = float(label[1]), float(label[2]), float(label[3])
                    w, l, h = float(label[4]), float(label[5]), float(label[6])
                    yaw = float(label[7]) if len(label) > 7 else 0.0
                    
                    # Find the closest cluster to this ground truth
                    gt_center = np.array([x, y, z])
                    best_cluster = None
                    best_iou = 0
                    
                    for cluster_points in clusters:
                        # Skip small clusters
                        if len(cluster_points) < self.config['min_points_threshold']:
                            continue
                            
                        # Compute cluster center and dimensions
                        xyz = cluster_points[:, :3]
                        cluster_min = np.min(xyz, axis=0)
                        cluster_max = np.max(xyz, axis=0)
                        cluster_center = np.mean(xyz, axis=0)
                        cluster_dims = cluster_max - cluster_min
                        
                        # Compute IoU (simplified 3D IoU based on centers and dimensions)
                        dist = np.linalg.norm(gt_center - cluster_center)
                        
                        # If centers are close enough
                        if dist < (w + l + h) / 3:
                            # Calculate 3D IoU
                            iou = self._calculate_3d_iou(
                                [x, y, z, w, l, h, yaw],
                                [cluster_center[0], cluster_center[1], cluster_center[2], 
                                 cluster_dims[0], cluster_dims[1], cluster_dims[2], 0]
                            )
                            
                            if iou > best_iou:
                                best_iou = iou
                                best_cluster = cluster_points
                    
                    # If a good match was found
                    if best_cluster is not None and best_iou > 0.3:
                    # Extract features from this cluster
                        features = self.compute_cluster_features(best_cluster)
                        
                        # Make sure features is a list or numpy array, not a float
                        if isinstance(features, (int, float)):
                            print(f"Warning: Features for {class_name} is a scalar value: {features}. Skipping...")
                            continue
                        
                        # Add to training data
                        X.append(features)
                        y.append(class_mapping[class_name])

                except Exception as e:
                    print(f"Error processing label {label}: {e}")
        # Ensure we have training data
        if len(X) == 0:
            print("Error: No valid training examples found after processing.")
            return

            
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)

        print(f"Collected {len(X)} training examples")
        print(f"Feature shape: {X.shape}")

        
        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        # Create dataset and dataloader
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        
        # Split into train and validation
        train_size = int((1 - validation_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size)
        
        # Setup optimizer and loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            
            for batch_features, batch_labels in train_loader:
                batch_features, batch_labels = batch_features.to(self.device), batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == batch_labels).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = train_correct / len(train_dataset)
            
            # Validation phase
            self.model.eval()
            val_loss = 0
            val_correct = 0
            
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features, batch_labels = batch_features.to(self.device), batch_labels.to(self.device)
                    outputs = self.model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == batch_labels).sum().item()
            
            val_loss /= len(val_loader)
            val_acc = val_correct / len(val_dataset)
            
            # Update LR scheduler
            scheduler.step(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"Saving improved model to {output_model_path}")
                torch.save(self.model.state_dict(), output_model_path)
        
        print("Training complete!")

    def _calculate_3d_iou(self, box1, box2):
        """
        Calculate IoU between two 3D boxes
        Simplified implementation that does not account for rotation
        
        box format: [x, y, z, width, length, height, yaw]
        """
        # Extract parameters
        x1, y1, z1, w1, l1, h1, yaw1 = box1
        x2, y2, z2, w2, l2, h2, yaw2 = box2
        
        # For simplicity, we ignore rotation and use axis-aligned boxes
        # Calculate min and max for both boxes
        box1_min = np.array([x1 - w1/2, y1 - l1/2, z1 - h1/2])
        box1_max = np.array([x1 + w1/2, y1 + l1/2, z1 + h1/2])
        
        box2_min = np.array([x2 - w2/2, y2 - l2/2, z2 - h2/2])
        box2_max = np.array([x2 + w2/2, y2 + l2/2, z2 + h2/2])
        
        # Calculate intersection
        intersection_min = np.maximum(box1_min, box2_min)
        intersection_max = np.minimum(box1_max, box2_max)
        
        # If boxes don't overlap, return 0
        if np.any(intersection_min > intersection_max):
            return 0.0
            
        # Calculate volumes
        intersection_dims = intersection_max - intersection_min
        intersection_volume = intersection_dims[0] * intersection_dims[1] * intersection_dims[2]
        
        box1_volume = w1 * l1 * h1
        box2_volume = w2 * l2 * h2
        
        # Calculate union
        union_volume = box1_volume + box2_volume - intersection_volume
        
        # Calculate IoU
        iou = intersection_volume / union_volume
        
        return iou

    def evaluate(self, data_dir, label_dir):
        """
        Evaluate the detector on test data
        """
        print("Starting evaluation...")
        
        # Find all input files
        bin_files = sorted(glob.glob(os.path.join(data_dir, '*.bin')))
        label_files = sorted(glob.glob(os.path.join(label_dir, '*.txt')))
        
        if len(bin_files) != len(label_files):
            print(f"Warning: Number of bin files ({len(bin_files)}) does not match number of label files ({len(label_files)})")
        
        if len(bin_files) == 0:
            print("Error: No evaluation data found.")
            return
            
        print(f"Found {len(bin_files)} evaluation samples.")
        
        # Metrics
        total_gt = 0
        total_pred = 0
        total_tp = 0
        
        class_metrics = {}
        
        # Process each sample
        for bin_file, label_file in zip(bin_files, label_files):
            print(f"Evaluating {os.path.basename(bin_file)}...")
            
            # Get predictions
            detections, _ = self.predict(bin_file)
            
            # Load ground truth labels
            with open(label_file, 'r') as f:
                gt_labels = [line.strip().split() for line in f.readlines()]
            
            # Parse ground truth
            gt_boxes = []
            for label in gt_labels:
                class_name = label[0]
                
                # Map to general classes we handle
                if class_name.startswith("human.pedestrian"):
                    class_name = "pedestrian"
                elif class_name == "vehicle.bicycle":
                    class_name = "bicycle"
                elif class_name == "vehicle.motorcycle":
                    class_name = "motorcycle"
                else:
                    continue  # Skip other classes
                
                try:
                    # Parse bounding box parameters
                    x, y, z = float(label[1]), float(label[2]), float(label[3])
                    w, l, h = float(label[4]), float(label[5]), float(label[6])
                    yaw = float(label[7]) if len(label) > 7 else 0.0
                    
                    gt_boxes.append({
                        'class': class_name,
                        'box': [x, y, z, w, l, h, yaw]
                    })
                    
                    # Update counters
                    total_gt += 1
                    
                    # Initialize class counters if needed
                    if class_name not in class_metrics:
                        class_metrics[class_name] = {'gt': 0, 'pred': 0, 'tp': 0}
                    
                    class_metrics[class_name]['gt'] += 1
                    
                except Exception as e:
                    print(f"Error parsing label: {e}")
            
            # Parse predictions
            pred_boxes = []
            for det in detections:
                pred_boxes.append({
                    'class': det['class'],
                    'box': [det['x'], det['y'], det['z'], det['width'], det['length'], det['height'], det['yaw']]
                })
                
                # Update counters
                total_pred += 1
                
                # Initialize class counters if needed
                if det['class'] not in class_metrics:
                    class_metrics[det['class']] = {'gt': 0, 'pred': 0, 'tp': 0}
                
                class_metrics[det['class']]['pred'] += 1
            
            # Match predictions to ground truth
            matched_gt = [False] * len(gt_boxes)
            
            for pred in pred_boxes:
                best_iou = 0
                best_idx = -1
                
                for i, gt in enumerate(gt_boxes):
                    # Skip already matched ground truths
                    if matched_gt[i]:
                        continue
                        
                    # Skip different classes
                    if pred['class'] != gt['class']:
                        continue
                    
                    # Calculate IoU
                    iou = self._calculate_3d_iou(pred['box'], gt['box'])
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_idx = i
                
                # If a good match was found
                if best_idx >= 0 and best_iou > 0.5:
                    matched_gt[best_idx] = True
                    total_tp += 1
                    class_metrics[pred['class']]['tp'] += 1
        
        # Calculate overall metrics
        precision = total_tp / total_pred if total_pred > 0 else 0
        recall = total_tp / total_gt if total_gt > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print("\n--- Overall Evaluation Results ---")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Calculate per-class metrics
        print("\n--- Per-Class Results ---")
        for cls, metrics in class_metrics.items():
            cls_precision = metrics['tp'] / metrics['pred'] if metrics['pred'] > 0 else 0
            cls_recall = metrics['tp'] / metrics['gt'] if metrics['gt'] > 0 else 0
            cls_f1 = 2 * cls_precision * cls_recall / (cls_precision + cls_recall) if (cls_precision + cls_recall) > 0 else 0
            
            print(f"\n{cls}:")
            print(f"  Precision: {cls_precision:.4f}")
            print(f"  Recall: {cls_recall:.4f}")
            print(f"  F1 Score: {cls_f1:.4f}")
            print(f"  GT count: {metrics['gt']}")
            print(f"  Pred count: {metrics['pred']}")
            print(f"  True Positive count: {metrics['tp']}")
        
        return precision, recall, f1

def main():
    """Main function for CLI usage"""
    parser = argparse.ArgumentParser(description='LiDAR-based VRU Detection')
    parser.add_argument('--data_dir', type=str, help='Directory containing point cloud data', default='data/scans')
    parser.add_argument('--label_dir', type=str, help='Directory containing ground truth labels', default='data/labels')
    parser.add_argument('--model_path', type=str, help='Path to model file', default='models')
    parser.add_argument('--output_dir', type=str, help='Directory to save outputs', default='results')
    parser.add_argument('--mode', type=str, choices=['train', 'eval', 'predict'], default='train',
                        help='Mode: train, evaluate, or predict')
    parser.add_argument('--train_epochs', type=int, default=50, help='Number of training epochs')
    
    args = parser.parse_args()
    
    # Create detector
    detector = LiDARVRUDetector()
    
    if args.mode == 'train':
        # Create output directory if it doesn't exist
        # os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        
        # Train the model
        detector.train(args.data_dir, args.label_dir, args.model_path, epochs=args.train_epochs)
        
    elif args.mode == 'eval':
        # Load model
        if args.model_path:
            detector.load_model(args.model_path)
            
        # Evaluate
        detector.evaluate(args.data_dir, args.label_dir)
        
    elif args.mode == 'predict':
        # Create output directory
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            
        # Load model if provided
        if args.model_path:
            detector.load_model(args.model_path)
            
        # Process all bin files
        bin_files = sorted(glob.glob(os.path.join(args.data_dir, '*.bin')))
        
        for bin_file in bin_files:
            filename = os.path.basename(bin_file)
            print(f"Processing {filename}...")
            
            # Detect VRUs
            detections, elapsed_time = detector.predict(bin_file)
            
            print(f"Found {len(detections)} VRUs in {elapsed_time:.3f} seconds.")
            
            # Save results if output directory is provided
            if args.output_dir:
                output_file = os.path.join(args.output_dir, os.path.splitext(filename)[0] + '.txt')
                with open(output_file, 'w') as f:
                    for det in detections:
                        # Format: class x y z width length height yaw confidence
                        cls = det['class']
                        if det['subtype']:
                            cls += "." + det['subtype']
                            
                        f.write(f"{cls} {det['x']:.6f} {det['y']:.6f} {det['z']:.6f} "
                                f"{det['width']:.6f} {det['length']:.6f} {det['height']:.6f} "
                                f"{det['yaw']:.6f} {det['confidence']:.6f}\n")

if __name__ == "__main__":
    main()
