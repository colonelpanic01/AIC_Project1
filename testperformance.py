import time
import numpy as np
from lidar_vru_detection import LiDARVRUDetector

def benchmark_detector_components(point_cloud_file, num_runs=5):
    """Benchmark different components of the detector"""
    detector = LiDARVRUDetector()
    
    # Load point cloud
    points = detector.load_point_cloud(point_cloud_file)
    
    # Benchmark preprocessing
    start = time.time()
    for _ in range(num_runs):
        filtered_points = detector.preprocess_point_cloud(points)
    preprocess_time = (time.time() - start) / num_runs
    
    # Benchmark clustering
    start = time.time()
    for _ in range(num_runs):
        clusters = detector.extract_clusters(filtered_points)
    clustering_time = (time.time() - start) / num_runs
    
    # Benchmark feature computation and classification
    start = time.time()
    results = []
    for _ in range(num_runs):
        results = []
        for cluster_points in clusters:
            features = detector.compute_cluster_features(cluster_points)
            cls_type, confidence = detector.classify_cluster_rule_based(cluster_points, features)
            bbox = detector.compute_oriented_bbox(cluster_points)
            results.append({'type': cls_type, 'confidence': confidence, 'bbox': bbox})
    postprocess_time = (time.time() - start) / num_runs
    
    # Total time
    total_time = preprocess_time + clustering_time + postprocess_time
    
    # Print results
    print("\n===== Detector Performance Benchmark =====")
    print(f"Input point cloud: {len(points)} points")
    print(f"Filtered points: {len(filtered_points)} points")
    print(f"Clusters found: {len(clusters)}")
    print(f"Objects detected: {len(results)}")
    print("\nPerformance breakdown:")
    print(f"  Preprocessing:  {preprocess_time*1000:.2f} ms ({preprocess_time/total_time*100:.1f}%)")
    print(f"  Clustering:     {clustering_time*1000:.2f} ms ({clustering_time/total_time*100:.1f}%)")
    print(f"  Postprocessing: {postprocess_time*1000:.2f} ms ({postprocess_time/total_time*100:.1f}%)")
    print(f"  Total time:     {total_time*1000:.2f} ms ({1/total_time:.1f} Hz)")
    
    return {
        'preprocess_time': preprocess_time,
        'clustering_time': clustering_time,
        'postprocess_time': postprocess_time,
        'total_time': total_time,
        'frequency': 1/total_time
    }

def optimize_parameters(point_cloud_file):
    """Test different parameter configurations to find optimal settings"""
    # Define parameter ranges to test
    params_to_test = {
        'cluster_eps': [0.3, 0.5, 0.7, 1.0],
        'cluster_min_samples': [3, 5, 10, 15],
        'min_points_threshold': [5, 10, 20, 30]
    }
    
    # Load point cloud
    detector = LiDARVRUDetector()
    points = detector.load_point_cloud(point_cloud_file)
    filtered_points = detector.preprocess_point_cloud(points)
    
    # Store results
    results = []
    
    # Test each parameter combination
    print("\n===== Parameter Optimization =====")
    for eps in params_to_test['cluster_eps']:
        for min_samples in params_to_test['cluster_min_samples']:
            for min_points in params_to_test['min_points_threshold']:
                # Update configuration
                config = detector.config.copy()
                config['cluster_eps'] = eps
                config['cluster_min_samples'] = min_samples
                config['min_points_threshold'] = min_points
                
                # Create detector with this configuration
                test_detector = LiDARVRUDetector(config)
                
                # Time clustering
                start = time.time()
                clusters = test_detector.extract_clusters(filtered_points)
                cluster_time = time.time() - start
                
                # Store results
                results.append({
                    'eps': eps,
                    'min_samples': min_samples,
                    'min_points': min_points,
                    'num_clusters': len(clusters),
                    'time': cluster_time
                })
                
                print(f"eps={eps}, min_samples={min_samples}, min_points={min_points}: "
                      f"{len(clusters)} clusters in {cluster_time*1000:.1f} ms")
    
    # Find best configuration
    # Sort by number of clusters (more is better) and then by time (less is better)
    results.sort(key=lambda x: (-x['num_clusters'], x['time']))
    
    print("\nTop 3 configurations:")
    for i in range(min(3, len(results))):
        r = results[i]
        print(f"{i+1}. eps={r['eps']}, min_samples={r['min_samples']}, "
              f"min_points={r['min_points']}: {r['num_clusters']} clusters in {r['time']*1000:.1f} ms")
    
    return results