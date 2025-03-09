import numpy as np
import open3d as o3d

# Load LiDAR scan (binary file)
def load_lidar_scan(bin_file):
    scan = np.fromfile(bin_file, dtype=np.float32)
    scan = scan.reshape(-1, 5)  # Reshape to Nx5 (X, Y, Z, Intensity, LiDAR Channel)
    return scan

# Load 3D bounding box (text file)
def load_bounding_box(txt_file):
    with open(txt_file, 'r') as f:
        lines = f.readlines()
    boxes = []
    for line in lines:
        parts = line.strip().split()
        label = parts[0]
        x, y, z, w, l, h, yaw = map(float, parts[1:])
        boxes.append((label, x, y, z, w, l, h, yaw))
    return boxes

# Example usage
bin_file = 'data/scans/000000.bin'
txt_file = 'data/labels/000000.txt'

lidar_scan = load_lidar_scan(bin_file)
bounding_boxes = load_bounding_box(txt_file)

# Convert to Open3D point cloud
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(lidar_scan[:, :3])  # Use only X, Y, Z

# Visualize
o3d.visualization.draw_geometries([pcd])