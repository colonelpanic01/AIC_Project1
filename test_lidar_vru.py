import os
import numpy as np
import matplotlib.pyplot as plt
from lidar_vru_detection import LiDARVRUDetector
from lidar_vru_visualization import visualize_point_cloud, plot_lidar_2d

def create_synthetic_data(output_dir, samples=5):
    # """Create synthetic data for testing"""
    # os.makedirs(os.path.join(output_dir, 'scans'), exist_ok=True)
    # os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
    
    # for i in range(num_samples):
    #     # Create point cloud with random points
    #     num_points = np.random.randint(1000, 5000)
        
    #     # Background points (random distribution)
    #     background_points = np.random.uniform(-50, 50, (num_points, 3))
    #     background_intensity = np.random.uniform(0, 0.5, (num_points, 1))
    #     background_channel = np.zeros((num_points, 1))
        
    #     # Add pedestrians
    #     num_pedestrians = np.random.randint(1, 4)
    #     pedestrians = []
    #     pedestrian_points = []
        
    #     for p in range(num_pedestrians):
    #         # Random position
    #         center_x = np.random.uniform(-20, 20)
    #         center_y = np.random.uniform(-20, 20)
    #         center_z = np.random.uniform(-0.2, 0.2)
            
    #         # Random dimensions (pedestrian-like)
    #         width = np.random.uniform(0.3, 0.5)
    #         length = np.random.uniform(0.3, 0.5)
    #         height = np.random.uniform(1.5, 1.9)
            
    #         # Random orientation
    #         yaw = np.random.uniform(0, 2*np.pi)
            
    #         # Generate points for this pedestrian
    #         num_ped_points = np.random.randint(50, 200)
            
    #         # Points distributed in the pedestrian volume
    #         ped_x = np.random.uniform(-width/2, width/2, num_ped_points) + center_x
    #         ped_y = np.random.uniform(-length/2, length/2, num_ped_points) + center_y
    #         ped_z = np.random.uniform(0, height, num_ped_points) + center_z
            
    #         ped_pts = np.column_stack((ped_x, ped_y, ped_z))
    #         ped_intensity = np.random.uniform(0.5, 1.0, (num_ped_points, 1))
    #         ped_channel = np.ones((num_ped_points, 1))
            
    #         pedestrian_points.append(np.column_stack((ped_pts, ped_intensity, ped_channel)))
    #         pedestrians.append(f"human.pedestrian.adult {center_x} {center_y} {center_z} {width} {height} {length} {yaw}")
        
    #     # Add cyclists
    #     num_cyclists = np.random.randint(0, 2)
    #     cyclists = []
    #     cyclist_points = []
        
    #     for c in range(num_cyclists):
    #         # Random position
    #         center_x = np.random.uniform(-20, 20)
    #         center_y = np.random.uniform(-20, 20)
    #         center_z = np.random.uniform(-0.2, 0.2)
            
    #         # Random dimensions (cyclist-like)
    #         width = np.random.uniform(0.6, 0.8)
    #         length = np.random.uniform(1.5, 1.8)
    #         height = np.random.uniform(1.5, 1.8)
            
    #         # Random orientation
    #         yaw = np.random.uniform(0, 2*np.pi)
            
    #         # Generate points for this cyclist
    #         num_cyc_points = np.random.randint(100, 300)
            
    #         # Points distributed in the cyclist volume
    #         cyc_x = np.random.uniform(-width/2, width/2, num_cyc_points) + center_x
    #         cyc_y = np.random.uniform(-length/2, length/2, num_cyc_points) + center_y
    #         cyc_z = np.random.uniform(0, height, num_cyc_points) + center_z
            
    #         cyc_pts = np.column_stack((cyc_x, cyc_y, cyc_z))
    #         cyc_intensity = np.random.uniform(0.5, 1.0, (num_cyc_points, 1))
    #         cyc_channel = np.ones((num_cyc_points, 1)) * 2
            
    #         cyclist_points.append(np.column_stack((cyc_pts, cyc_intensity, cyc_channel)))
    #         cyclists.append(f"vehicle.bicycle {center_x} {center_y} {center_z} {width} {height} {length} {yaw}")
        
    #     # Combine all points
    #     all_points = [np.column_stack((background_points, background_intensity, background_channel))]
    #     all_points.extend(pedestrian_points)
    #     all_points.extend(cyclist_points)
        
    #     combined_points = np.vstack(all_points)
        
    #    # Save the point cloud
    #    #scan_file = os.path.join(output_dir, 'scans', f'{i:06d}.bin')
    #    combined_points.astype(np.float32).tofile(scan_file)
        
    # Save the labels
    for i in range(samples):
        label_file = os.path.join(output_dir, 'labels', f'{i:06d}.txt')
    
    with open(label_file, 'w') as f:
        for ped in pedestrians:
            f.write(f"{ped}\n")
        for cyc in cyclists:
            f.write(f"{cyc}\n")
    
    return output_dir

def test_detector(data_dir, output_dir):
    """Test the detector on sample data"""
    # Initialize detector
    detector = LiDARVRUDetector()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process a sample scan
    sample_file = os.path.join(data_dir, 'scans', '000000.bin')
    
    if not os.path.exists(sample_file):
        print(f"Sample file {sample_file} not found!")
        return
    
    # Load point cloud
    points = detector.load_point_cloud(sample_file)
    
    # Preprocess
    filtered_points = detector.preprocess_point_cloud(points)
    
    # Extract clusters
    clusters = detector.extract_clusters(filtered_points)
    print(f"Found {len(clusters)} clusters")
    
    # Process each cluster
    results = []
    for i, cluster_points in enumerate(clusters):
        # Compute features
        features = detector.compute_cluster_features(cluster_points)
        
        # Classify cluster
        cls_type, confidence = detector.classify_cluster_rule_based(cluster_points, features)
        
        # Compute bounding box
        bbox = detector.compute_oriented_bbox(cluster_points)
        
        # Add to results
        result = {
            'type': cls_type,
            'confidence': confidence,
            'bbox': bbox
        }
        results.append(result)
        
        print(f"Cluster {i}: {cls_type} (conf: {confidence:.2f})")
    
    # Save results
    output_file = os.path.join(output_dir, '000000.txt')
    detector.save_results(results, output_file)
    
    # Load ground truth for comparison
    label_file = os.path.join(data_dir, 'labels', '000000.txt')
    gt_labels = []
    
    if os.path.exists(label_file):
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                cls_type = parts[0]
                bbox = [float(p) for p in parts[1:8]]
                gt_labels.append({'type': cls_type, 'bbox': bbox})
    
    # Visualize results
    print("Displaying point cloud with detections...")
    plot_lidar_2d(points, results)
    
    # Compare with ground truth if available
    if gt_labels:
        print("Displaying point cloud with ground truth...")
        plot_lidar_2d(points, gt_labels)
    
    # Full 3D visualization
    print("Starting 3D visualization (close window to continue)...")
    visualize_point_cloud(points, results)
    
    return results

def test_full_pipeline(data_dir, output_dir):
    """Test the complete detection pipeline"""
    print("Testing complete detection pipeline...")
    
    # Process sample data
    from lidar_vru_detection import process_dataset
    
    # Process just a few samples
    sample_files = [os.path.join(data_dir, 'scans', f) 
                    for f in os.listdir(os.path.join(data_dir, 'scans'))
                    if f.endswith('.bin')][:5]
    
    # Create a temporary input directory with just the samples
    temp_dir = os.path.join(data_dir, 'temp')
    os.makedirs(os.path.join(temp_dir, 'scans'), exist_ok=True)
    os.makedirs(os.path.join(temp_dir, 'labels'), exist_ok=True)
    
    for sample_file in sample_files:
        file_name = os.path.basename(sample_file)
        os.system(f"cp {sample_file} {os.path.join(temp_dir, 'scans', file_name)}")
        
        # Copy corresponding label file if it exists
        label_file = os.path.join(data_dir, 'labels', file_name.replace('.bin', '.txt'))
        if os.path.exists(label_file):
            os.system(f"cp {label_file} {os.path.join(temp_dir, 'labels', file_name.replace('.bin', '.txt'))}")
    
    # Process the dataset
    process_dataset(temp_dir, output_dir)
    
    # Clean up
    os.system(f"rm -rf {temp_dir}")
    
    print(f"Results saved to {output_dir}")

def main():
    """Main test function"""
    # Create or use test data directory
    test_data_dir = 'data'
    output_dir = 'test_output'
    
    # Check if data directory exists
    if not os.path.exists(test_data_dir):
        print("Creating synthetic test data...")
        create_synthetic_data(test_data_dir, num_samples=5)
    
    # Test detector components
    print("\n=== Testing detector components ===")
    test_detector(test_data_dir, output_dir)
    
    # Test full pipeline
    print("\n=== Testing full pipeline ===")
    test_full_pipeline(test_data_dir, output_dir)
    
    print("\nTesting complete!")

if __name__ == "__main__":
    main()