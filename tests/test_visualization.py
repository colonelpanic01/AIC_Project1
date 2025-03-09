import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import argparse
from pathlib import Path
import time

from lidar_vru_detection import LiDARVRUDetector

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    print("Open3D not found. Installing with pip...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "open3d"])
    try:
        import open3d as o3d
        OPEN3D_AVAILABLE = True
    except ImportError:
        print("Warning: Open3D installation failed. Falling back to matplotlib for visualization.")
        OPEN3D_AVAILABLE = False

def load_labels(label_file):
    """Load ground truth labels from file"""
    labels = []
    if not os.path.exists(label_file):
        return labels
        
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
                
            cls_type = parts[0] 
            bbox = [float(p) for p in parts[1:8]]
            labels.append({'type': cls_type, 'bbox': bbox})
    return labels

def create_bounding_box_lines(bbox):
    """Create lines for 3D bounding box visualization"""
    center_x, center_y, center_z, width, height, length, yaw = bbox
    
    # Half dimensions
    hw, hh, hl = width/2, height/2, length/2
    
    # Rotation matrix around z-axis
    R = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Define the 8 corners of the box in local coordinates
    corners_local = np.array([
        [ hw,  hl,  hh],  # 0: top right front
        [ hw,  hl, -hh],  # 1: bottom right front
        [ hw, -hl, -hh],  # 2: bottom right rear
        [ hw, -hl,  hh],  # 3: top right rear
        [-hw,  hl,  hh],  # 4: top left front
        [-hw,  hl, -hh],  # 5: bottom left front
        [-hw, -hl, -hh],  # 6: bottom left rear
        [-hw, -hl,  hh],  # 7: top left rear
    ])
    
    # Transform corners to global coordinates
    corners_global = np.zeros_like(corners_local)
    for i, corner in enumerate(corners_local):
        # Rotate the corner
        rotated = R @ corner
        # Translate to center position
        corners_global[i] = rotated + [center_x, center_y, center_z]
    
    # Define the 12 lines connecting corners (each line is defined by start and end point indices)
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Bottom face
        [4, 5], [5, 6], [6, 7], [7, 4],  # Top face
        [0, 4], [1, 5], [2, 6], [3, 7]   # Connecting lines
    ]
    
    line_sets = []
    for start_idx, end_idx in lines:
        line_sets.append([corners_global[start_idx], corners_global[end_idx]])
    
    return line_sets

def get_color_by_class(class_name):
    """Get color based on class name"""
    if 'pedestrian' in class_name.lower():
        return [1, 0, 0]  # Red for pedestrians
    elif 'cyclist' in class_name.lower() or 'bicycle' in class_name.lower():
        return [0, 1, 0]  # Green for cyclists
    elif 'motorcycle' in class_name.lower():
        return [0, 0, 1]  # Blue for motorcycles
    else:
        return [1, 1, 0]  # Yellow for others

def visualize_detections_open3d(points, detections, gt_labels=None):
    """Visualize LiDAR points and detections using Open3D"""
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    # Color points by intensity
    intensity = points[:, 3]
    normalized_intensity = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity) + 1e-10)
    colors = np.zeros((len(normalized_intensity), 3))
    colors[:, 0] = normalized_intensity
    colors[:, 1] = normalized_intensity
    colors[:, 2] = normalized_intensity
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    
    # Add detection bounding boxes
    for detection in detections:
        bbox = detection['bbox']
        class_name = detection['type']
        color = get_color_by_class(class_name)
        
        # Create lines for bounding box
        line_sets = create_bounding_box_lines(bbox)
        
        # Create line set geometry
        for i, line in enumerate(line_sets):
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(line)
            line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
            line_set.colors = o3d.utility.Vector3dVector([color])
            vis.add_geometry(line_set)
    
    # Add ground truth bounding boxes if provided
    if gt_labels:
        for label in gt_labels:
            bbox = label['bbox']
            class_name = label['type']
            
            # Use different color scheme for ground truth (add transparency)
            color = get_color_by_class(class_name)
            color = [c * 0.7 + 0.3 for c in color]  # Make it slightly brighter
            
            # Create lines for bounding box
            line_sets = create_bounding_box_lines(bbox)
            
            # Create line set geometry
            for i, line in enumerate(line_sets):
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(line)
                line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
                line_set.colors = o3d.utility.Vector3dVector([color])
                vis.add_geometry(line_set)
    
    # Configure view
    view_control = vis.get_view_control()
    view_control.set_front([0, 0, -1])  # Look at the scene from the front
    view_control.set_lookat([0, 0, 0])  # Look at the origin
    view_control.set_up([0, -1, 0])     # Set up direction
    view_control.set_zoom(0.8)
    
    # Run visualizer
    vis.run()
    vis.destroy_window()

def visualize_detections_matplotlib(points, detections, gt_labels=None, max_points=10000):
    """Visualize LiDAR points and detections using Matplotlib"""
    fig = plt.figure(figsize=(16, 12))
    
    # Create 3D plot
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    
    # Subsample points if too many
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        vis_points = points[indices]
    else:
        vis_points = points
    
    # Plot 3D view
    ax1.scatter(vis_points[:, 0], vis_points[:, 1], vis_points[:, 2], c=vis_points[:, 3], 
               cmap='viridis', s=1, alpha=0.5)
    
    # Add bounding boxes to 3D view
    for detection in detections:
        bbox = detection['bbox']
        class_name = detection['type']
        color = get_color_by_class(class_name)
        
        # Get lines for the bounding box
        line_sets = create_bounding_box_lines(bbox)
        
        # Plot each line
        for line in line_sets:
            xs = [line[0][0], line[1][0]]
            ys = [line[0][1], line[1][1]]
            zs = [line[0][2], line[1][2]]
            ax1.plot(xs, ys, zs, color=color)
    
    ax1.set_title('3D View')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    
    # Plot top-down view (XY)
    ax2.scatter(vis_points[:, 0], vis_points[:, 1], c=vis_points[:, 3], 
               cmap='viridis', s=1, alpha=0.5)
    
    # Add bounding boxes to top view
    for detection in detections:
        bbox = detection['bbox']
        class_name = detection['type']
        color = get_color_by_class(class_name)
        
        # Get corners for top-down view
        center_x, center_y, _, width, _, length, yaw = bbox
        corners = []
        
        # Corners in local coordinates (top-down view)
        hw, hl = width/2, length/2
        local_corners = [
            [hw, hl], [hw, -hl], [-hw, -hl], [-hw, hl], [hw, hl]
        ]
        
        # Rotation matrix
        R = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw), np.cos(yaw)]
        ])
        
        # Transform corners to global coordinates
        global_corners = []
        for corner in local_corners:
            rotated = R @ np.array(corner)
            global_corners.append([rotated[0] + center_x, rotated[1] + center_y])
        
        # Plot bounding box
        global_corners = np.array(global_corners)
        ax2.plot(global_corners[:, 0], global_corners[:, 1], color=color)
        
        # Add label
        ax2.text(center_x, center_y, class_name.split('.')[-1], 
                 color='white', fontsize=8, weight='bold',
                 bbox=dict(facecolor=color, alpha=0.7))
    
    ax2.set_title('Top View (XY)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.grid(True)
    ax2.axis('equal')
    
    # Plot front view (XZ)
    ax3.scatter(vis_points[:, 0], vis_points[:, 2], c=vis_points[:, 3], 
               cmap='viridis', s=1, alpha=0.5)
    
    # Add bounding boxes to front view
    for detection in detections:
        bbox = detection['bbox']
        class_name = detection['type']
        color = get_color_by_class(class_name)
        
        # Front view shows X-Z plane
        center_x, _, center_z, width, height, _, yaw = bbox
        hw, hh = width/2, height/2
        
        # Simplified front view (no rotation around Y axis considered)
        corners_x = [center_x - hw, center_x + hw, center_x + hw, center_x - hw, center_x - hw]
        corners_z = [center_z - hh, center_z - hh, center_z + hh, center_z + hh, center_z - hh]
        
        ax3.plot(corners_x, corners_z, color=color)
    
    ax3.set_title('Front View (XZ)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z')
    ax3.grid(True)
    
    # Plot side view (YZ)
    ax4.scatter(vis_points[:, 1], vis_points[:, 2], c=vis_points[:, 3], 
               cmap='viridis', s=1, alpha=0.5)
    
    # Add bounding boxes to side view
    for detection in detections:
        bbox = detection['bbox']
        class_name = detection['type']
        color = get_color_by_class(class_name)
        
        # Side view shows Y-Z plane
        _, center_y, center_z, _, height, length, yaw = bbox
        hl, hh = length/2, height/2
        
        # Simplified side view (no rotation around X axis considered)
        corners_y = [center_y - hl, center_y + hl, center_y + hl, center_y - hl, center_y - hl]
        corners_z = [center_z - hh, center_z - hh, center_z + hh, center_z + hh, center_z - hh]
        
        ax4.plot(corners_y, corners_z, color=color)
    
    ax4.set_title('Side View (YZ)')
    ax4.set_xlabel('Y')
    ax4.set_ylabel('Z')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()

def test_detector(scan_file, label_file=None, use_ml=False, model_path=None):
    """Test detector on a single scan file"""
    # Initialize detector
    detector = LiDARVRUDetector()
    
    # Load model if using ML approach
    if use_ml and model_path:
        detector.load_model(model_path)
    
    # Start timing
    start_time = time.time()
    
    # Run detection
    results = detector.detect(scan_file, use_ml=use_ml)
    
    # End timing
    end_time = time.time()
    process_time = end_time - start_time
    print(f"Detection completed in {process_time:.4f} seconds")
    
    # Load point cloud for visualization
    points = detector.load_point_cloud(scan_file)
    
    # Load ground truth labels if provided
    gt_labels = None
    if label_file and os.path.exists(label_file):
        gt_labels = load_labels(label_file)
        
        # Filter for VRUs only in ground truth
        gt_labels = [l for l in gt_labels if any(vru in l['type'].lower() 
                                               for vru in ['pedestrian', 'cyclist', 'bicycle', 'motorcycle'])]
        
        print(f"Loaded {len(gt_labels)} ground truth VRU labels")
    
    # Print detection results
    print(f"Found {len(results)} VRUs:")
    for i, result in enumerate(results):
        bbox = result['bbox']
        print(f"  {i+1}. {result['type']} (confidence: {result['confidence']:.2f})")
        print(f"     Position: ({bbox[0]:.2f}, {bbox[1]:.2f}, {bbox[2]:.2f})")
        print(f"     Dimensions: {bbox[3]:.2f} x {bbox[4]:.2f} x {bbox[5]:.2f}")
        print(f"     Yaw: {bbox[6]:.4f}")
    
    # Visualize results
    if OPEN3D_AVAILABLE:
        print("Visualizing with Open3D (3D interactive view)...")
        visualize_detections_open3d(points, results, gt_labels)
    else:
        print("Visualizing with Matplotlib...")
        visualize_detections_matplotlib(points, results, gt_labels)
    
    return results

def test_multiple_scans(data_dir, num_scans=5, use_ml=False, model_path=None):
    """Test detector on multiple scan files"""
    # Get scan files
    scan_dir = os.path.join(data_dir, 'scans')
    label_dir = os.path.join(data_dir, 'labels')
    
    scan_files = sorted(os.listdir(scan_dir))[:num_scans]
    
    for scan_file in scan_files:
        scan_path = os.path.join(scan_dir, scan_file)
        label_path = os.path.join(label_dir, scan_file.replace('.bin', '.txt'))
        
        print(f"\nProcessing {scan_file}...")
        test_detector(scan_path, label_path, use_ml, model_path)
        
        # Ask if user wants to continue
        if scan_file != scan_files[-1]:
            user_input = input("\nPress Enter to continue to next scan, or 'q' to quit: ")
            if user_input.lower() == 'q':
                break

def main():
    parser = argparse.ArgumentParser(description='Test LiDAR VRU Detection')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing scans and labels')
    parser.add_argument('--scan_file', type=str, default=None,
                        help='Specific scan file to process')
    parser.add_argument('--label_file', type=str, default=None,
                        help='Ground truth label file for the scan')
    parser.add_argument('--use_ml', action='store_true',
                        help='Use ML-based detection instead of rule-based')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained model weights')
    parser.add_argument('--num_scans', type=int, default=5,
                        help='Number of scans to process when testing multiple')
                        
    args = parser.parse_args()
    
    if args.scan_file:
        # Test on a single file
        test_detector(args.scan_file, args.label_file, args.use_ml, args.model_path)
    else:
        # Test on multiple files
        test_multiple_scans(args.data_dir, args.num_scans, args.use_ml, args.model_path)
        
if __name__ == "__main__":
    main()