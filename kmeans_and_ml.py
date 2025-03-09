import numpy as np
import os
import glob
import time
import torch
import torch.nn as nn
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
import argparse

# Configuration parameters
CONFIG = {
    'voxel_size': [0.05, 0.05, 0.05],
    'point_cloud_range': [-50, -50, -5, 50, 50, 3],
    'max_points_per_voxel': 64,
    'max_voxels': 40000,

    'pedestrian_height_range': [0.585, 2.744],
    'pedestrian_width_range': [0.282, 1.505],
    'pedestrian_length_range': [0.214, 1.674],
    
    'bicycle_height_range': [0.349, 2.223],
    'bicycle_width_range': [0.233, 1.661],
    'bicycle_length_range': [0.454, 3.04],
    
    'motorcycle_height_range': [0.791, 2.02],
    'motorcycle_width_range': [0.351, 1.816],
    'motorcycle_length_range': [0.72, 4.409],
    
    'min_points_threshold': 15,
    'dbscan_eps': 0.4,
    'dbscan_min_samples': 7,
    'kmeans_min_clusters': 2,
    'kmeans_max_clusters': 30,
    'ground_height_threshold': -1.5,
    'intensity_threshold': 0.08,
    'confidence_threshold': 0.5,
    'iou_threshold': 0.2,
}

class LiDARModel(nn.Module):
    """Neural network model for VRU detection"""
    def __init__(self):
        super(LiDARModel, self).__init__()
        # Input features
        self.net = nn.Sequential(
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
            
            # Output layer - binary classification (VRU or not)
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

class CombinedVRUDetector:
    def __init__(self, config=None, use_ml=False, use_dbscan=True, use_kmeans=True):
        self.config = CONFIG if config is None else config
        self.use_ml = use_ml
        self.use_dbscan = use_dbscan
        self.use_kmeans = use_kmeans
        
        # Initialize device for PyTorch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize ML model if needed
        if self.use_ml:
            self.model = LiDARModel().to(self.device)
    
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
        """Preprocess point cloud by filtering out points."""
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
        
        # Remove ground points
        mask_ground = z > self.config['ground_height_threshold']
        
        # Combine masks
        mask = np.logical_and(np.logical_and(mask_range, mask_intensity), mask_ground)
        filtered_points = points[mask]
        
        return filtered_points
    
    def extract_dbscan_clusters(self, points):
        """Extract clusters using DBSCAN algorithm."""
        if len(points) < 10:
            return []
            
        # Use only x, y, z for clustering
        xyz = points[:, :3]
        
        # Apply DBSCAN clustering
        db = DBSCAN(
            eps=self.config['dbscan_eps'],
            min_samples=self.config['dbscan_min_samples'],
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
    
    def extract_kmeans_clusters(self, points):
        """Extract clusters using K-means algorithm."""
        if len(points) < 10:
            return []
        
        # Use only x, y, z for clustering
        xyz = points[:, :3]
        
        # Normalize the data
        scaler = StandardScaler()
        xyz_scaled = scaler.fit_transform(xyz)
        
        # Estimate the number of clusters based on point cloud density
        # More sophisticated methods could be used here
        estimated_clusters = max(
            self.config['kmeans_min_clusters'], 
            min(self.config['kmeans_max_clusters'], len(xyz) // 50)
        )
        
        # Apply K-means clustering
        kmeans = KMeans(
            n_clusters=estimated_clusters,
            random_state=0,
            n_init=10
        ).fit(xyz_scaled)
        
        labels = kmeans.labels_
        unique_labels = np.unique(labels)
        
        # Collect clusters
        clusters = []
        for label in unique_labels:
            cluster_mask = labels == label
            if np.sum(cluster_mask) < self.config['min_points_threshold']:
                continue
                
            cluster_points = points[cluster_mask]
            clusters.append(cluster_points)
        
        # Further refine clusters that might be too large
        refined_clusters = []
        for cluster_points in clusters:
            if len(cluster_points) > 300:  # Large cluster that might contain multiple objects
                # Apply a second round of K-means with smaller k for large clusters
                sub_xyz = cluster_points[:, :3]
                sub_xyz_scaled = scaler.transform(sub_xyz)
                
                # Use 2-5 subclusters for large clusters
                sub_k = max(2, min(5, len(cluster_points) // 60))
                
                sub_kmeans = KMeans(n_clusters=sub_k, random_state=0, n_init=10).fit(sub_xyz_scaled)
                sub_labels = sub_kmeans.labels_
                
                for sub_label in np.unique(sub_labels):
                    sub_mask = sub_labels == sub_label
                    if np.sum(sub_mask) < self.config['min_points_threshold']:
                        continue
                    
                    sub_points = cluster_points[sub_mask]
                    refined_clusters.append(sub_points)
            else:
                refined_clusters.append(cluster_points)
        
        return refined_clusters
    
    def compute_cluster_features(self, cluster_points):
        """Compute features for a cluster to be used by ML model."""
        # Geometric features
        xyz = cluster_points[:, :3]
        min_point = np.min(xyz, axis=0)
        max_point = np.max(xyz, axis=0)
        center = np.mean(xyz, axis=0)
        dimensions = max_point - min_point
        
        # Aspect ratios
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
        
        # Height distribution
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
        """Compute eigenvalues of the covariance matrix for shape analysis."""
        if len(points) < 3:
            return np.zeros(3)
            
        cov = np.cov(points, rowvar=False)
        eigenvalues, _ = np.linalg.eigh(cov)
        # Sort in descending order
        eigenvalues = np.sort(eigenvalues)[::-1]
        return eigenvalues
    
    def is_vru_rule_based(self, cluster_points):
        """Rule-based check if cluster is a VRU."""
        # Extract xyz coordinates
        xyz = cluster_points[:, :3]
        min_point = np.min(xyz, axis=0)
        max_point = np.max(xyz, axis=0)
        
        # Calculate dimensions
        width = max_point[0] - min_point[0]
        length = max_point[1] - min_point[1]
        height = max_point[2] - min_point[2]
        
        # Calculate center
        center = (min_point + max_point) / 2
        
        # Check if dimensions fall within any VRU category
        # Pedestrian check
        ped_width_ok = self.config['pedestrian_width_range'][0] <= width <= self.config['pedestrian_width_range'][1]
        ped_length_ok = self.config['pedestrian_length_range'][0] <= length <= self.config['pedestrian_length_range'][1]
        ped_height_ok = self.config['pedestrian_height_range'][0] <= height <= self.config['pedestrian_height_range'][1]
        is_pedestrian = ped_width_ok and ped_length_ok and ped_height_ok
        
        # Bicycle check
        bike_width_ok = self.config['bicycle_width_range'][0] <= width <= self.config['bicycle_width_range'][1]
        bike_length_ok = self.config['bicycle_length_range'][0] <= length <= self.config['bicycle_length_range'][1]
        bike_height_ok = self.config['bicycle_height_range'][0] <= height <= self.config['bicycle_height_range'][1]
        is_bicycle = bike_width_ok and bike_length_ok and bike_height_ok
        
        # Motorcycle check
        moto_width_ok = self.config['motorcycle_width_range'][0] <= width <= self.config['motorcycle_width_range'][1]
        moto_length_ok = self.config['motorcycle_length_range'][0] <= length <= self.config['motorcycle_length_range'][1]
        moto_height_ok = self.config['motorcycle_height_range'][0] <= height <= self.config['motorcycle_height_range'][1]
        is_motorcycle = moto_width_ok and moto_length_ok and moto_height_ok
        
        # Point count check
        point_count_ok = len(cluster_points) >= self.config['min_points_threshold']
        
        # Height from ground check
        not_ground = center[2] > 0.2
        
        # Combined check - any VRU type + basic validations
        return (is_pedestrian or is_bicycle or is_motorcycle) and point_count_ok and not_ground
    
    def predict_vru_ml(self, cluster_points):
        """ML-based prediction if cluster is a VRU."""
        if not self.use_ml:
            return False, 0.0
            
        # Compute features
        features = self.compute_cluster_features(cluster_points)
        
        # Convert to PyTorch tensor
        features_tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        # Get prediction
        self.model.eval()
        with torch.no_grad():
            confidence = self.model(features_tensor).item()
        
        # Return prediction and confidence
        is_vru = confidence >= self.config['confidence_threshold']
        return is_vru, confidence
    
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
            return [center[0], center[1], center[2], dimensions[0], dimensions[2], dimensions[1], 0.0]
            
        # Compute covariance matrix and eigenvectors for orientation
        cov = np.cov(xyz[:, :2], rowvar=False)  # Use only x,y for orientation
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Sort eigenvectors by eigenvalues in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Main axis is the first eigenvector
        main_axis = eigenvectors[:, 0]
        
        # Calculate yaw angle (orientation)
        yaw = np.arctan2(main_axis[1], main_axis[0])
        
        # Project points onto eigenvectors to get dimensions
        proj_points = np.dot(xyz[:, :2] - center[:2], eigenvectors)
        min_proj = np.min(proj_points, axis=0)
        max_proj = np.max(proj_points, axis=0)
        
        # Calculate width and length based on projections
        width = max_proj[0] - min_proj[0]
        length = max_proj[1] - min_proj[1]
        
        # Get height from max/min z
        height = np.max(xyz[:, 2]) - np.min(xyz[:, 2])
        
        return [center[0], center[1], center[2], width, length, height, yaw]
    
    def calculate_iou(self, box1, box2):
        """Calculate IoU between two axis-aligned bounding boxes for NMS."""
        # Extract box coordinates
        b1_x1, b1_y1 = box1[0] - box1[3]/2, box1[1] - box1[4]/2
        b1_x2, b1_y2 = box1[0] + box1[3]/2, box1[1] + box1[4]/2
        
        b2_x1, b2_y1 = box2[0] - box2[3]/2, box2[1] - box2[4]/2
        b2_x2, b2_y2 = box2[0] + box2[3]/2, box2[1] + box2[4]/2
        
        # Calculate intersection area
        intersect_x1 = max(b1_x1, b2_x1)
        intersect_y1 = max(b1_y1, b2_y1)
        intersect_x2 = min(b1_x2, b2_x2)
        intersect_y2 = min(b1_y2, b2_y2)
        
        if intersect_x2 < intersect_x1 or intersect_y2 < intersect_y1:
            return 0.0
            
        intersect_area = (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1)
        
        # Calculate union area
        b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        union_area = b1_area + b2_area - intersect_area
        
        # Calculate IoU
        iou = intersect_area / union_area if union_area > 0 else 0.0
        return iou
    
    def non_max_suppression(self, bboxes, confidences):
        """Apply non-maximum suppression to remove overlapping detections."""
        if len(bboxes) == 0:
            return []
            
        # Sort by confidence
        indices = np.argsort(confidences)[::-1]
        bboxes = [bboxes[i] for i in indices]
        confidences = [confidences[i] for i in indices]
        
        keep = []
        while len(bboxes) > 0:
            # Keep the highest confidence detection
            keep.append(bboxes[0])
            
            # Check for overlaps with remaining boxes
            remaining_boxes = []
            remaining_confidences = []
            
            for i in range(1, len(bboxes)):
                # Calculate IoU
                iou = self.calculate_iou(bboxes[0], bboxes[i])
                
                # Keep if IoU is below threshold
                if iou < self.config['iou_threshold']:
                    remaining_boxes.append(bboxes[i])
                    remaining_confidences.append(confidences[i])
            
            # Update boxes and confidences
            bboxes = remaining_boxes
            confidences = remaining_confidences
        
        return keep
    
    def detect_vrus(self, point_cloud_file):
        """Main detection method that processes a point cloud file."""
        # Load point cloud
        point_cloud = self.load_point_cloud(point_cloud_file)
        
        # Preprocess point cloud
        filtered_points = self.preprocess_point_cloud(point_cloud)
        
        if len(filtered_points) < 10:
            return []  # Not enough points
        
        # Extract clusters using different methods
        all_clusters = []
        
        if self.use_dbscan:
            dbscan_clusters = self.extract_dbscan_clusters(filtered_points)
            for cluster in dbscan_clusters:
                all_clusters.append(cluster)
        
        if self.use_kmeans:
            kmeans_clusters = self.extract_kmeans_clusters(filtered_points)
            for cluster in kmeans_clusters:
                all_clusters.append(cluster)
        
        # Process each cluster and detect VRUs
        vru_bboxes = []
        confidences = []
        
        for cluster_points in all_clusters:
            # Apply rule-based detection
            is_vru_rule = self.is_vru_rule_based(cluster_points)
            
            # Apply ML detection if available
            is_vru_ml, confidence_ml = self.predict_vru_ml(cluster_points) if self.use_ml else (False, 0.0)
            
            # Combine detections - if either method identifies a VRU
            is_vru = is_vru_rule or is_vru_ml
            
            # Calculate confidence - prioritize ML if available
            confidence = confidence_ml if self.use_ml else (0.7 if is_vru_rule else 0.0)
            
            if is_vru and confidence >= self.config['confidence_threshold']:
                # Generate bounding box for the detected VRU
                bbox = self.compute_oriented_bbox(cluster_points)
                vru_bboxes.append(bbox)
                confidences.append(confidence)
        
        # Apply non-maximum suppression to remove duplicates
        final_bboxes = self.non_max_suppression(vru_bboxes, confidences)
        
        return final_bboxes
    
    def process_directory(self, input_dir, output_dir=None):
        """Process all point cloud files in a directory."""
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        
        # Find all .bin files
        bin_files = glob.glob(os.path.join(input_dir, "*.bin"))
        
        results = []
        total_time = 0
        
        for bin_file in bin_files:
            print(f"Processing {bin_file}...")
            start_time = time.time()
            
            # Detect VRUs
            detections = self.detect_vrus(bin_file)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            total_time += processing_time
            
            print(f"Found {len(detections)} VRUs in {processing_time:.3f} seconds")
            
            # Optionally save results
            if output_dir is not None:
                base_name = os.path.basename(bin_file).replace('.bin', '.txt')
                output_file = os.path.join(output_dir, base_name)
                
                with open(output_file, 'w') as f:
                    for det in detections:
                        # Write in KITTI format: type x y z w l h yaw
                        f.write(f"Pedestrian {det[0]} {det[1]} {det[2]} {det[3]} {det[4]} {det[5]} {det[6]}\n")
            
            # Store results
            results.append({
                'file': bin_file,
                'detections': detections,
                'processing_time': processing_time
            })
        
        # Print summary
        avg_time = total_time / len(bin_files) if bin_files else 0
        print(f"Processed {len(bin_files)} files in {total_time:.3f} seconds (avg: {avg_time:.3f} sec/file)")
        
        return results


def main():
    """Main function to run the VRU detector."""
    parser = argparse.ArgumentParser(description='LiDAR-based VRU Detection')
    parser.add_argument('--input', type=str, default='data/scans', help='Input directory with point cloud files (.bin)')
    parser.add_argument('--output', type=str, default='output', help='Output directory for detection results')
    parser.add_argument('--model', type=str, default='vru_model.pth', help='Path to ML model weights (.pth)')
    parser.add_argument('--no-dbscan', action='store_true', help='Disable DBSCAN clustering')
    parser.add_argument('--no-kmeans', action='store_true', help='Disable KMeans clustering')
    
    args = parser.parse_args()
    
    # Create detector
    detector = CombinedVRUDetector(
        use_ml=args.model is not None,
        use_dbscan=not args.no_dbscan,
        use_kmeans=not args.no_kmeans
    )
    
    # Load model if specified
    if args.model is not None:
        detector.load_model(args.model)
    
    # Process files
    detector.process_directory(args.input, args.output)


if __name__ == "__main__":
    main()