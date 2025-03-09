import numpy as np
import os
import glob
import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def read_lidar_bin(bin_file):
    """
    Read LiDAR binary file.
    Format: X, Y, Z, Intensity, LiDAR Channel
    """
    scan = np.fromfile(bin_file, dtype=np.float32)
    scan = scan.reshape((-1, 5))  # Reshape to (N, 5)
    return scan

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
    
    # Remove ground points
    mask_ground = z > config['ground_height_threshold']
    
    # Combine masks
    mask = np.logical_and(np.logical_and(mask_range, mask_intensity), mask_ground)
    filtered_points = points[mask]
    
    return filtered_points

def extract_clusters_kmeans(points, config):
    """
    Extract clusters from point cloud using K-means
    Dynamically determines number of clusters based on point density
    """
    if len(points) < 10:
        return []
    
    # Use only x, y, z for clustering
    xyz = points[:, :3]
    
    # Normalize the data
    scaler = StandardScaler()
    xyz_scaled = scaler.fit_transform(xyz)
    
    # Estimate the number of clusters based on point cloud density
    # More sophisticated methods could be used here
    estimated_clusters = max(2, min(30, len(xyz) // 50))
    
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
        if np.sum(cluster_mask) < config['min_points_threshold']:
            continue
            
        cluster_points = points[cluster_mask]
        clusters.append(cluster_points)
    
    # Further refine clusters that might be too large
    refined_clusters = []
    for cluster_points in clusters:
        if len(cluster_points) > 300:  # Large cluster that might contain multiple objects
            # Apply a second round of K-means with smaller k for large clusters
            sub_clusters = extract_clusters_kmeans(cluster_points, {**config, 'min_points_threshold': 5})
            refined_clusters.extend(sub_clusters)
        else:
            refined_clusters.append(cluster_points)
    
    return refined_clusters

def is_vru(cluster_points, config):
    """
    Rule-based check if a cluster is a Vulnerable Road User (VRU).
    Using dimensions for pedestrians, bicycles, and motorcycles.
    """
    # Extract xyz coordinates
    xyz = cluster_points[:, :3]
    
    # Calculate min and max coordinates
    min_point = np.min(xyz, axis=0)
    max_point = np.max(xyz, axis=0)
    
    # Calculate dimensions
    dimensions = max_point - min_point
    width, length, height = dimensions[0], dimensions[1], dimensions[2]
    
    # Calculate center
    center = (min_point + max_point) / 2
    
    # Check if dimensions fall within VRU ranges
    # Pedestrian check
    ped_width_ok = config['pedestrian_width_range'][0] <= width <= config['pedestrian_width_range'][1]
    ped_length_ok = config['pedestrian_length_range'][0] <= length <= config['pedestrian_length_range'][1]
    ped_height_ok = config['pedestrian_height_range'][0] <= height <= config['pedestrian_height_range'][1]
    is_pedestrian = ped_width_ok and ped_length_ok and ped_height_ok
    
    # Bicycle check
    bike_width_ok = config['bicycle_width_range'][0] <= width <= config['bicycle_width_range'][1]
    bike_length_ok = config['bicycle_length_range'][0] <= length <= config['bicycle_length_range'][1]
    bike_height_ok = config['bicycle_height_range'][0] <= height <= config['bicycle_height_range'][1]
    is_bicycle = bike_width_ok and bike_length_ok and bike_height_ok
    
    # Motorcycle check
    moto_width_ok = config['motorcycle_width_range'][0] <= width <= config['motorcycle_width_range'][1]
    moto_length_ok = config['motorcycle_length_range'][0] <= length <= config['motorcycle_length_range'][1]
    moto_height_ok = config['motorcycle_height_range'][0] <= height <= config['motorcycle_height_range'][1]
    is_motorcycle = moto_width_ok and moto_length_ok and moto_height_ok
    
    # Additional checks
    point_count_ok = config['min_points_threshold'] <= len(cluster_points) <= 1000
    
    # Calculate point density
    volume = width * length * height
    density = len(cluster_points) / max(volume, 0.001)  # Avoid division by zero
    density_ok = density > 10  # VRUs tend to have higher point density
    
    # Check height from ground
    not_ground = center[2] > 0.2  # Z-coordinate should be above ground
    
    # Combined check for any VRU type
    is_vru = (is_pedestrian or is_bicycle or is_motorcycle) and point_count_ok and density_ok and not_ground
    
    return is_vru

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

def process_scan_kmeans(bin_file, output_file, config):
    """
    Process a single LiDAR scan using K-means clustering and output VRU detections.
    """
    # Read LiDAR data
    points = read_lidar_bin(bin_file)
    
    # Preprocess point cloud
    filtered_points = preprocess_point_cloud(points, config)
    
    # Extract clusters using K-means
    clusters = extract_clusters_kmeans(filtered_points, config)
    
    # Filter VRUs and compute bounding boxes
    results = []
    for cluster_points in clusters:
        if is_vru(cluster_points, config):
            bbox = compute_oriented_bbox(cluster_points)
            results.append({
                'type': 'Vru',
                'bbox': bbox
            })
    
    # Write detections to output file
    with open(output_file, 'w') as f:
        for result in results:
            bbox = result['bbox']
            line = f"{result['type']} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f} {bbox[6]:.6f}\n"
            f.write(line)
    
    return len(results)

def process_dataset_kmeans(input_dir, output_dir, config):
    """
    Process an entire dataset of LiDAR scans using K-means clustering
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of scan files
    scan_files = sorted(glob.glob(os.path.join(input_dir, '*.bin')))
    
    # Process each scan
    total_time = 0
    total_detections = 0
    
    for i, scan_file in enumerate(scan_files):
        # Extract scan number
        scan_num = os.path.basename(scan_file).split('.')[0]
        
        # Process scan
        start_time = time.time()
        num_detections = process_scan_kmeans(scan_file, os.path.join(output_dir, f'{scan_num}.txt'), config)
        end_time = time.time()
        
        # Calculate processing time
        process_time = end_time - start_time
        total_time += process_time
        total_detections += num_detections
        
        # Print progress
        if (i + 1) % 10 == 0:
            print(f"Processed {i+1}/{len(scan_files)} scans. Avg time: {total_time / (i+1):.4f}s")
    
    # Print statistics
    avg_time = total_time / len(scan_files)
    print(f"Processing complete! Processed {len(scan_files)} scans with {total_detections} VRU detections.")
    print(f"Average processing time: {avg_time:.4f}s ({1/avg_time:.2f} Hz)")

if __name__ == "__main__":
    import argparse
    
    # Configuration
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
        'intensity_threshold': 0.08,
        'ground_height_threshold': -1.5,
    }
    
    parser = argparse.ArgumentParser(description="K-means based VRU detection from LiDAR data")
    parser.add_argument("--input_dir", type=str, default='data/scans', help="Directory containing LiDAR bin files")
    parser.add_argument("--output_dir", type=str, default='output', help="Directory to save detection results")
    
    args = parser.parse_args()
    
    process_dataset_kmeans(args.input_dir, args.output_dir, CONFIG)