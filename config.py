import yaml
import os

# Default configuration
DEFAULT_CONFIG = {
    # Point cloud preprocessing
    'voxel_size': [0.05, 0.05, 0.05],
    'point_cloud_range': [-50, -50, -5, 50, 50, 3],
    'max_points_per_voxel': 64,
    'max_voxels': 40000,
    'intensity_threshold': 0.08,
    'ground_height_threshold': -1.5,
    
    # Clustering parameters
    'cluster_eps': 0.4,
    'cluster_min_samples': 7,
    'min_points_threshold': 15,
    
    # Dimension ranges for all classes in CLASS_MAP
    'dimension_ranges': {
        # Pedestrian variants
        'human.pedestrian.adult': {
            'width_range': [0.282, 1.505],
            'length_range': [0.214, 1.674],
            'height_range': [0.585, 2.744],
            'yaw_range': [-4.712371700070678, 1.570789171031985]
        },
        'human.pedestrian.construction_worker': {
            'width_range': [0.345, 1.971],
            'length_range': [0.293, 1.521],
            'height_range': [0.293, 2.573],
            'yaw_range': [-4.712387760134593, 1.5705286304974742]
        },
        'human.pedestrian.child': {
            'width_range': [0.295, 0.93],
            'length_range': [0.268, 0.995],
            'height_range': [0.724, 2.0],
            'yaw_range': [-4.710222062114154, 1.5624351131644398]
        },
        'human.pedestrian.wheelchair': {
            'width_range': [0.496, 0.876],
            'length_range': [0.682, 1.538],
            'height_range': [1.229, 1.532],
            'yaw_range': [-3.5224930470241946, 1.4828541376923488]
        },
        'human.pedestrian.personal_mobility': {
            'width_range': [0.298, 0.886],
            'length_range': [0.494, 2.239],
            'height_range': [0.846, 2.0],
            'yaw_range': [-4.6510599883954304, 1.505014419205605]
        },
        'human.pedestrian.police_officer': {
            'width_range': [0.527, 1.155],
            'length_range': [0.451, 1.024],
            'height_range': [1.394, 2.028],
            'yaw_range': [-4.709856889394767, 1.5693674589034963]
        },
        'human.pedestrian.stroller': {
            'width_range': [0.362, 0.87],
            'length_range': [0.418, 1.753],
            'height_range': [0.789, 1.888],
            'yaw_range': [-4.691451416352464, 1.5469162803724985]
        },
        'vehicle.motorcycle': {
            'width_range': [0.351, 1.816],
            'length_range': [0.72, 4.409],
            'height_range': [0.791, 2.02],
            'yaw_range': [-4.7115172916358325, 1.5695307294768606]
        },
        'vehicle.bicycle': {
            'width_range': [0.233, 1.661],
            'length_range': [0.454, 3.04],
            'height_range': [0.349, 2.223],
            'yaw_range': [-4.710947933322235, 1.5702342135720362]
        }
    },
    
    # Model parameters
    'input_feature_dim': 64,
    'num_classes': 9,
    'batch_size': 32,
    'learning_rate': 0.001,
    'weight_decay': 0.0001,
    'num_epochs': 50
}

# Class mapping - consistent across training and inference
CLASS_MAP = {
    0: "human.pedestrian.adult",
    1: "human.pedestrian.construction_worker",
    2: "human.pedestrian.child",
    3: "human.pedestrian.wheelchair",
    4: "human.pedestrian.personal_mobility",
    5: "human.pedestrian.police_officer",
    6: "human.pedestrian.stroller",
    7: "vehicle.motorcycle",
    8: "vehicle.bicycle"
}

# Simplified class mapping for rule-based detection
SIMPLIFIED_CLASS_MAP = {
    "pedestrian": [0, 1, 2, 3, 4, 5, 6],  # All pedestrian classes
    "motorcycle": [7],
    "bicycle": [8]
}

def load_config(config_path=None):
    """Load configuration from file or use defaults"""
    config = DEFAULT_CONFIG.copy()
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    # Update config with values from file
                    config.update(file_config)
            print(f"Configuration loaded from {config_path}")
        except Exception as e:
            print(f"Error loading config from {config_path}: {e}")
            print("Using default configuration")
    
    return config

def filter_detection_by_class(detection, config):
    """
    Filter detections based on class-specific dimension thresholds
    Returns True if detection passes the filter, False otherwise
    """
    class_name = CLASS_MAP.get(detection.class_id)
    
    # Reject if class not in our defined CLASS_MAP
    if not class_name:
        return False
        
    # Get dimension ranges for this class
    class_ranges = config['dimension_ranges'].get(class_name)
    if not class_ranges:
        return False
    
    # Check if dimensions are within specified ranges
    width, length, height = detection.dimensions
    
    width_min, width_max = class_ranges['width_range']
    length_min, length_max = class_ranges['length_range']
    height_min, height_max = class_ranges['height_range']
    
    # Return True only if all dimensions are within range
    return (width_min <= width <= width_max and 
            length_min <= length <= length_max and 
            height_min <= height <= height_max)

def save_config(config, config_path):
    """Save configuration to a YAML file"""
    try:
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        print(f"Configuration saved to {config_path}")
        return True
    except Exception as e:
        print(f"Error saving config to {config_path}: {e}")
        return False