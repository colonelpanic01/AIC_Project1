import numpy as np
import os
import glob
from sklearn.cluster import DBSCAN

def read_lidar_bin(bin_file):
    """
    Read LiDAR binary file.
    Format: X, Y, Z, Intensity, LiDAR Channel
    """
    scan = np.fromfile(bin_file, dtype=np.float32)
    scan = scan.reshape((-1, 5))  # Reshape to (N, 5)
    return scan

def cluster_points(points, eps=0.5, min_samples=5):
    """
    Cluster points using DBSCAN algorithm.
    Returns clusters with labels.
    """
    # Use only XYZ for clustering
    xyz = points[:, :3]
    
    # Apply DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(xyz)
    labels = db.labels_
    
    # Number of clusters (excluding noise with label -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    return labels, n_clusters

def get_cluster_objects(points, labels):
    """
    Extract individual clusters and their properties.
    """
    unique_labels = set(labels)
    clusters = []
    
    for label in unique_labels:
        # Skip noise points
        if label == -1:
            continue
            
        # Get points belonging to this cluster
        cluster_points = points[labels == label]
        
        # Calculate cluster dimensions
        min_coords = np.min(cluster_points[:, :3], axis=0)
        max_coords = np.max(cluster_points[:, :3], axis=0)
        
        # Calculate width, length, height
        dimensions = max_coords - min_coords
        width, length, height = dimensions[0], dimensions[1], dimensions[2]
        
        # Calculate center
        center = (min_coords + max_coords) / 2
        
        # Calculate point density (points per cubic meter)
        volume = width * length * height
        density = len(cluster_points) / max(volume, 0.001)  # Avoid division by zero
        
        # Store cluster info
        cluster_info = {
            'points': cluster_points,
            'center': center,
            'width': width,
            'length': length,
            'height': height,
            'num_points': len(cluster_points),
            'density': density
        }
        
        clusters.append(cluster_info)
    
    return clusters

def is_vru(cluster):
    """
    Rule-based check if a cluster is a Vulnerable Road User (VRU).
    Using the min/max dimensions provided for pedestrians, bicycles, and motorcycles.
    """
    # Combined min-max ranges for all VRU types from the provided data
    min_width, max_width = 0.233, 1.971
    min_length, max_length = 0.214, 4.409
    min_height, max_height = 0.349, 2.744
    
    # Get cluster dimensions
    width = cluster['width']
    length = cluster['length']
    height = cluster['height']
    
    # Check if dimensions are within VRU ranges (with some tolerance)
    width_ok = min_width * 0.9 <= width <= max_width * 1.1
    length_ok = min_length * 0.9 <= length <= max_length * 1.1
    height_ok = min_height * 0.9 <= height <= max_height * 1.1
    
    # Additional rules
    density_ok = cluster['density'] > 50  # VRUs tend to have higher point density
    point_count_ok = 10 <= cluster['num_points'] <= 1000  # VRUs typically have reasonable point counts
    
    # Check for ground objects
    not_ground = cluster['center'][2] > 0.2  # Z-coordinate (height from ground) should be above 0.2m
    
    return width_ok and length_ok and height_ok and density_ok and point_count_ok and not_ground

def estimate_yaw(points):
    """
    Estimate yaw angle using principal component analysis.
    """
    # Use only X and Y for PCA
    xy_points = points[:, :2]
    
    # Center data
    xy_centered = xy_points - np.mean(xy_points, axis=0)
    
    # Compute covariance matrix
    cov = np.cov(xy_centered.T)
    
    # Compute eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    
    # Get the principal direction (largest eigenvalue)
    idx = np.argmax(eigenvalues)
    principal_vec = eigenvectors[:, idx]
    
    # Calculate yaw (angle with x-axis)
    yaw = np.arctan2(principal_vec[1], principal_vec[0])
    
    return yaw

def process_scan(bin_file, output_file):
    """
    Process a single LiDAR scan and output VRU detections.
    """
    # Read LiDAR data
    points = read_lidar_bin(bin_file)
    
    # Cluster points
    labels, n_clusters = cluster_points(points, eps=0.5, min_samples=10)
    
    # Get cluster objects
    clusters = get_cluster_objects(points, labels)
    
    # Filter VRUs
    vru_clusters = [cluster for cluster in clusters if is_vru(cluster)]
    
    # Write detections to output file
    with open(output_file, 'w') as f:
        for cluster in vru_clusters:
            center_x, center_y, center_z = cluster['center']
            width = cluster['width']
            height = cluster['height']
            yaw = estimate_yaw(cluster['points'])
            
            # Format: Vru CenterX CenterY CenterZ Width Height Yaw
            f.write(f"Vru {center_x:.6f} {center_y:.6f} {center_z:.6f} {width:.6f} {height:.6f} {yaw:.6f}\n")

def process_all_scans(input_dir, output_dir):
    """
    Process all LiDAR scans in input directory and output results to output directory.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all bin files
    bin_files = glob.glob(os.path.join(input_dir, "*.bin"))
    
    for bin_file in bin_files:
        # Get file basename (without extension)
        basename = os.path.basename(bin_file).replace('.bin', '')
        
        # Create output file path
        output_file = os.path.join(output_dir, f"{basename}.txt")
        
        # Process scan
        process_scan(bin_file, output_file)
        
        print(f"Processed {bin_file} -> {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Rule-based VRU detection from LiDAR data")
    parser.add_argument("--input_dir", type=str, default='data/scans', help="Directory containing LiDAR bin files")
    parser.add_argument("--output_dir", type=str, default='results', help="Directory to save detection results")
    
    args = parser.parse_args()
    
    process_all_scans(args.input_dir, args.output_dir)