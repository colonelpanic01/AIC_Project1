import os
import sys
import time
import glob
import argparse
import numpy as np
import torch
import torch.nn as nn

# Add parent directory to path to import detector module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import detector
from lidar_vru_detection import LiDARVRUDetector

def optimize_for_jetson():
    """Optimize settings for Jetson Nano performance"""
    # Set CPU affinity to use all cores
    try:
        import os
        os.system('taskset -p 0xff %d' % os.getpid())
    except:
        print("Warning: Could not set CPU affinity")
    
    # Set PyTorch to use TensorRT if available
    try:
        import torch_tensorrt
        print("TensorRT available for acceleration")
    except ImportError:
        print("TensorRT not available, using standard PyTorch")
    
    # Configure CUDA settings
    if torch.cuda.is_available():
        # Enable cuDNN autotuner
        torch.backends.cudnn.benchmark = True
        # Use fastest implementation
        torch.backends.cudnn.fastest = True
        # Print CUDA info
        device_name = torch.cuda.get_device_name(0)
        print(f"Using CUDA device: {device_name}")
        print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("CUDA not available, using CPU")

def process_dataset(detector, input_dir, output_dir, use_ml=False):
    """Process a dataset on Jetson Nano with performance monitoring"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of scan files
    scan_files = sorted(glob.glob(os.path.join(input_dir, 'scans', '*.bin')))
    
    # Performance metrics
    total_time = 0
    frame_times = []
    
    # Process each scan
    for i, scan_file in enumerate(scan_files):
        # Extract scan number
        scan_num = os.path.basename(scan_file).split('.')[0]
        
        # Process scan with timing
        start_time = time.time()
        results = detector.detect(scan_file, use_ml=use_ml)
        process_time = time.time() - start_time
        
        # Track metrics
        total_time += process_time
        frame_times.append(process_time)
        
        # Save results
        output_file = os.path.join(output_dir, f'{scan_num}.txt')
        detector.save_results(results, output_file)
        
        # Print progress
        if (i + 1) % 10 == 0:
            elapsed = total_time
            fps = (i + 1) / elapsed
            print(f"Processed {i+1}/{len(scan_files)} scans. "
                  f"Avg time: {elapsed / (i+1):.4f}s ({fps:.2f} FPS)")
    
    # Print final statistics
    if len(scan_files) > 0:
        avg_time = total_time / len(scan_files)
        fps = 1.0 / avg_time
        
        # Calculate percentiles for frame times
        frame_times = np.array(frame_times)
        p50 = np.percentile(frame_times, 50) * 1000
        p95 = np.percentile(frame_times, 95) * 1000
        p99 = np.percentile(frame_times, 99) * 1000
        
        print("\n===== Performance Statistics =====")
        print(f"Processed {len(scan_files)} scans in {total_time:.2f} seconds")
        print(f"Average processing time: {avg_time:.4f}s ({fps:.2f} FPS)")
        print(f"Frame time (ms) - P50: {p50:.2f}, P95: {p95:.2f}, P99: {p99:.2f}")

def main():
    """Main function for Jetson deployment"""
    parser = argparse.ArgumentParser(description='LiDAR VRU Detection on Jetson Nano')
    parser.add_argument('--input_dir', type=str, required=True, 
                        help='Input directory containing scans folder')
    parser.add_argument('--output_dir', type=str, required=True, 
                        help='Output directory for detection results')
    parser.add_argument('--use_ml', action='store_true', 
                        help='Use ML-based detection (requires model_path)')
    parser.add_argument('--model_path', type=str, default=None, 
                        help='Path to trained model weights')
    parser.add_argument('--optimize', action='store_true', 
                        help='Apply optimizations for Jetson Nano')
    
    args = parser.parse_args()
    
    # Apply optimizations if requested
    if args.optimize:
        print("Applying Jetson Nano optimizations...")
        optimize_for_jetson()
    
    # Initialize detector
    print("Initializing LiDAR VRU detector...")
    detector = LiDARVRUDetector()
    
    # Load model if using ML approach
    if args.use_ml:
        if args.model_path is None or not os.path.exists(args.model_path):
            print("Error: Model path required for ML-based detection")
            return
        
        print(f"Loading model from {args.model_path}...")
        detector.load_model(args.model_path)
    
    # Process dataset
    print(f"Processing dataset from {args.input_dir}...")
    process_dataset(detector, args.input_dir, args.output_dir, args.use_ml)
    
    print(f"Processing complete! Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
