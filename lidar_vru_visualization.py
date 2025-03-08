import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import os
import glob
from tqdm import tqdm

def load_point_cloud(bin_file):
    """Load LiDAR point cloud from binary file"""
    points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 5)
    return points

def load_labels(label_file):
    """Load ground truth labels"""
    labels = []
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            cls_type = parts[0]
            bbox = [float(p) for p in parts[1:8]]
            labels.append({'type': cls_type, 'bbox': bbox})
    return labels

def visualize_point_cloud(points, labels=None, max_points=10000):
    """
    Visualize point cloud using Open3D
    """
    # Subsample points if too many
    if len(points) > max_points:
        indices = np.random.choice(len(points), max_points, replace=False)
        points = points[indices]
    
    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:, :3])
    
    # Set colors based on intensity
    intensity = points[:, 3]
    normalized_intensity = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))
    colors = np.zeros((len(normalized_intensity), 3))
    colors[:, 0] = normalized_intensity  # Red channel
    colors[:, 1] = normalized_intensity  # Green channel
    colors[:, 2] = normalized_intensity  # Blue channel
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    
    # Add bounding boxes if labels provided
    if labels is not None:
        for label in labels:
            bbox = label['bbox']
            center = bbox[:3]
            dimensions = bbox[3:6]
            yaw = bbox[6]
            
            # Create bounding box
            box = o3d.geometry.OrientedBoundingBox()
            box.center = center
            box.extent = dimensions
            
            # Apply rotation around Z-axis
            R = np.array([
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1]
            ])
            box.R = R
            
            # Set color based on label type
            if 'pedestrian' in label['type'].lower():
                box.color = (1, 0, 0)  # Red for pedestrians
            elif 'cyclist' in label['type'].lower() or 'bicycle' in label['type'].lower():
                box.color = (0, 1, 0)  # Green for cyclists
            elif 'motorcycle' in label['type'].lower():
                box.color = (0, 0, 1)  # Blue for motorcycles
            else:
                box.color = (1, 1, 0)  # Yellow for others
                
            vis.add_geometry(box)
    
    # Set viewpoint
    ctr = vis.get_view_control()
    ctr.set_zoom(0.8)
    ctr.set_front([0, 0, -1])  # Looking at XY plane
    ctr.set_lookat([0, 0, 0])
    ctr.set_up([0, -1, 0])
    
    # Run visualization
    vis.run()
    vis.destroy_window()

def plot_lidar_2d(points, labels=None, xlim=(-20, 20), ylim=(-20, 20)):
    """Plot LiDAR points in 2D (top-down view)"""
    plt.figure(figsize=(10, 10))
    
    # Plot points
    plt.scatter(points[:, 0], points[:, 1], c=points[:, 3], 
                cmap='viridis', s=1, alpha=0.5)
    
    # Plot bounding boxes if provided
    if labels is not None:
        for label in labels:
            bbox = label['bbox']
            center_x, center_y = bbox[0], bbox[1]
            width, length = bbox[3], bbox[5]
            yaw = bbox[6]
            
            # Compute corners of the bounding box
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            
            # Half-width and half-length
            hw, hl = width/2, length/2
            
            # Corners in local coordinate system
            corners_local = np.array([
                [ hw,  hl],
                [ hw, -hl],
                [-hw, -hl],
                [-hw,  hl],
                [ hw,  hl]  # Close the loop
            ])
            
            # Rotate and translate corners
            corners_global = np.zeros_like(corners_local)
            for i, (x, y) in enumerate(corners_local):
                corners_global[i, 0] = center_x + x * cos_yaw - y * sin_yaw
                corners_global[i, 1] = center_y + x * sin_yaw + y * cos_yaw
            
            # Plot the bounding box
            if 'pedestrian' in label['type'].lower():
                color = 'r'  # Red for pedestrians
            elif 'cyclist' in label['type'].lower() or 'bicycle' in label['type'].lower():
                color = 'g'  # Green for cyclists
            elif 'motorcycle' in label['type'].lower():
                color = 'b'  # Blue for motorcycles
            else:
                color = 'y'  # Yellow for others
                
            plt.plot(corners_global[:, 0], corners_global[:, 1], color, linewidth=2)
            plt.text(center_x, center_y, label['type'].split('.')[-1], 
                     color='white', fontsize=8, weight='bold',
                     bbox=dict(facecolor=color, alpha=0.5))
    
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.grid(True)
    plt.title('LiDAR Point Cloud - Top Down View')
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.axis('equal')
    plt.colorbar(label='Intensity')
    plt.show()

def evaluate_detections(gt_dir, pred_dir, iou_threshold=0.5):
    """
    Evaluate detection performance by comparing predictions to ground truth
    """
    # Get list of files
    gt_files = sorted(glob.glob(os.path.join(gt_dir, '*.txt')))
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    class_metrics = {
        'pedestrian': {'tp': 0, 'fp': 0, 'fn': 0},
        'cyclist': {'tp': 0, 'fp': 0, 'fn': 0},
        'motorcycle': {'tp': 0, 'fp': 0, 'fn': 0}
    }
    
    for gt_file in tqdm(gt_files, desc="Evaluating"):
        # Get base filename
        base_name = os.path.basename(gt_file)
        pred_file = os.path.join(pred_dir, base_name)
        
        # Skip if prediction file doesn't exist
        if not os.path.exists(pred_file):
            print(f"Warning: No prediction file for {base_name}")
            continue
        
        # Load ground truth and predictions
        gt_labels = load_labels(gt_file)
        pred_labels = load_labels(pred_file)
        
        # Filter for VRUs only
        gt_vru = [l for l in gt_labels if any(vru in l['type'].lower() 
                                             for vru in ['pedestrian', 'cyclist', 'bicycle', 'motorcycle'])]
        
        # Match predictions to ground truth
        matched_gt = [False] * len(gt_vru)
        
        for pred in pred_labels:
            # Get prediction class
            if 'pedestrian' in pred['type'].lower():
                pred_class = 'pedestrian'
            elif 'cyclist' in pred['type'].lower() or 'bicycle' in pred['type'].lower():
                pred_class = 'cyclist'
            elif 'motorcycle' in pred['type'].lower():
                pred_class = 'motorcycle'
            else:
                pred_class = 'other'
                
            if pred_class == 'other':
                continue
                
            # Find best matching ground truth
            best_iou = 0
            best_gt_idx = -1
            
            for i, gt in enumerate(gt_vru):
                if matched_gt[i]:
                    continue
                    
                # Calculate IoU
                iou = calculate_3d_iou(pred['bbox'], gt['bbox'])
                
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = i
            
            # Check if match found
            if best_gt_idx >= 0 and best_iou >= iou_threshold:
                # Get ground truth class
                gt = gt_vru[best_gt_idx]
                if 'pedestrian' in gt['type'].lower():
                    gt_class = 'pedestrian'
                elif 'cyclist' in gt['type'].lower() or 'bicycle' in gt['type'].lower():
                    gt_class = 'cyclist'
                elif 'motorcycle' in gt['type'].lower():
                    gt_class = 'motorcycle'
                else:
                    gt_class = 'other'
                
                # Check class match
                if pred_class == gt_class:
                    total_tp += 1
                    class_metrics[pred_class]['tp'] += 1
                    matched_gt[best_gt_idx] = True
                else:
                    total_fp += 1
                    class_metrics[pred_class]['fp'] += 1
            else:
                total_fp += 1
                class_metrics[pred_class]['fp'] += 1
        
        # Count false negatives
        for i, matched in enumerate(matched_gt):
            if not matched:
                total_fn += 1
                
                # Get ground truth class
                gt = gt_vru[i]
                if 'pedestrian' in gt['type'].lower():
                    gt_class = 'pedestrian'
                elif 'cyclist' in gt['type'].lower() or 'bicycle' in gt['type'].lower():
                    gt_class = 'cyclist'
                elif 'motorcycle' in gt['type'].lower():
                    gt_class = 'motorcycle'
                else:
                    gt_class = 'other'
                    
                if gt_class != 'other':
                    class_metrics[gt_class]['fn'] += 1
    
    # Calculate overall metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate per-class metrics
    class_results = {}
    for cls, metrics in class_metrics.items():
        cls_precision = metrics['tp'] / (metrics['tp'] + metrics['fp']) if (metrics['tp'] + metrics['fp']) > 0 else 0
        cls_recall = metrics['tp'] / (metrics['tp'] + metrics['fn']) if (metrics['tp'] + metrics['fn']) > 0 else 0
        cls_f1 = 2 * cls_precision * cls_recall / (cls_precision + cls_recall) if (cls_precision + cls_recall) > 0 else 0
        
        class_results[cls] = {
            'precision': cls_precision,
            'recall': cls_recall,
            'f1': cls_f1
        }
    
    # Print results
    print("\n===== Detection Evaluation Results =====")
    print(f"Overall Precision: {precision:.4f}")
    print(f"Overall Recall: {recall:.4f}")
    print(f"Overall F1-Score: {f1:.4f}")
    print("\nPer-Class Results:")
    for cls, results in class_results.items():
        print(f"  {cls.capitalize()}:")
        print(f"    Precision: {results['precision']:.4f}")
        print(f"    Recall: {results['recall']:.4f}")
        print(f"    F1-Score: {results['f1']:.4f}")
    
    return {
        'overall': {
            'precision': precision,
            'recall': recall,
            'f1': f1
        },
        'class': class_results
    }

def calculate_3d_iou(bbox1, bbox2):
    """
    Calculate IoU between two 3D bounding boxes
    This is a simplified implementation that doesn't handle rotation perfectly
    """
    # Extract box parameters
    center1 = np.array(bbox1[:3])
    dim1 = np.array(bbox1[3:6])
    yaw1 = bbox1[6]
    
    center2 = np.array(bbox2[:3])
    dim2 = np.array(bbox2[3:6])
    yaw2 = bbox2[6]
    
    # If yaw angles are very different, IOU will be low
    if abs(yaw1 - yaw2) > np.pi/4 and abs(yaw1 - yaw2) < 7*np.pi/4:
        return 0.0
    
    # Calculate min/max points for each box
    half_dim1 = dim1 / 2
    half_dim2 = dim2 / 2
    
    min1 = center1 - half_dim1
    max1 = center1 + half_dim1
    
    min2 = center2 - half_dim2
    max2 = center2 + half_dim2
    
    # Calculate intersection volume
    intersection_min = np.maximum(min1, min2)
    intersection_max = np.minimum(max1, max2)