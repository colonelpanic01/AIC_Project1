import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import glob

# Define class mapping
class_map = {
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

# Custom LiDAR VRU Dataset
class LiDARVRUDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.data_dir = data_dir
        self.split = split
        self.scan_dir = os.path.join(data_dir, 'scans')
        self.label_dir = os.path.join(data_dir, 'labels')
        
        # Get file paths
        self.scan_files = sorted(glob.glob(os.path.join(self.scan_dir, '*.bin')))
        
        # Dictionary to store labels
        self.labels = []
        
        # Load and preprocess data
        self._preprocess_data()
        
    def __len__(self):
        return len(self.processed_features)
    
    def __getitem__(self, idx):
        features = self.processed_features[idx]
        label = self.processed_labels[idx]
        
        # Convert to tensors
        features = torch.FloatTensor(features)
        label = torch.LongTensor([label])[0]  # Convert label to tensor and index into it
        
        return {
            'features': features,
            'label': label,
            'cls_labels': torch.zeros(len(class_map)),  # Placeholder for compatibility
            'box_labels': torch.zeros(7),  # Placeholder for compatibility
            'point_cloud': torch.zeros(100, 3)  # Placeholder for compatibility
        }
    
    def _preprocess_data(self):
        """Preprocess data and extract features"""
        self.processed_features = []
        self.processed_labels = []
        
        print("Preprocessing dataset...")
        all_class_ids = set()  # Track all class IDs for debugging
        class_counts = {}  # Track counts of each class
        
        for scan_file in self.scan_files:
            # Extract scan ID
            scan_id = os.path.basename(scan_file).split('.')[0]
            
            # Load corresponding label file
            label_file = os.path.join(self.label_dir, f"{scan_id}.txt")
            
            if not os.path.exists(label_file):
                continue
                
            # Load point cloud
            points = self._load_point_cloud(scan_file)
            
            # Load labels
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                parts = line.strip().split()
                
                # Parse label data
                class_name = parts[0]
                center_x, center_y, center_z = float(parts[1]), float(parts[2]), float(parts[3])
                width, height, length = float(parts[4]), float(parts[5]), float(parts[6])
                yaw = float(parts[7]) if len(parts) > 7 else 0.0
                
                # Map class name to ID
                class_id = None
                for id, name in class_map.items():
                    if name.split('.')[-1] in class_name:
                        class_id = id
                        break
                
                if class_id is None:
                    continue
                
                # Track class ID for debugging
                all_class_ids.add(class_id)
                
                # Count classes
                if class_id not in class_counts:
                    class_counts[class_id] = 0
                class_counts[class_id] += 1
                
                # Extract points in the bounding box
                bbox_points = self._extract_bbox_points(points, 
                                                      [center_x, center_y, center_z], 
                                                      [width, height, length], 
                                                      yaw)
                
                if len(bbox_points) < 5:
                    continue
                
                # Compute features for the points
                features = self._compute_features(bbox_points)
                
                # Add to processed data
                self.processed_features.append(features)
                self.processed_labels.append(class_id)
                
                # Add to global labels list
                self.labels.append({
                    'scan_id': scan_id,
                    'class_id': class_id,
                    'bbox': [center_x, center_y, center_z, width, height, length, yaw]
                })
        
        print(f"Dataset preprocessed: {len(self.processed_features)} samples")
        print("Class distribution:")
        
        # Debug: Print all found class IDs
        print(f"All class IDs found in dataset: {sorted(all_class_ids)}")
        
        # Print class distribution with safe access to class_map
        for cls, count in class_counts.items():
            try:
                class_name = class_map[cls].split('.')[-1]  # Get shortened class name
            except KeyError:
                class_name = f"Unknown class {cls}"
            print(f"  {class_name}: {count} samples ({count/len(self.labels)*100:.2f}%)")
    
    def _load_point_cloud(self, bin_file):
        """Load point cloud from binary file"""
        points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 5)
        return points
    
    def _extract_bbox_points(self, points, center, dimensions, yaw):
        """Extract points inside a rotated 3D bounding box"""
        # Convert points to local coordinates
        local_points = points[:, :3] - np.array(center)
        
        # Rotation matrix for yaw
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        rotation_matrix = np.array([
            [cos_yaw, -sin_yaw, 0],
            [sin_yaw, cos_yaw, 0],
            [0, 0, 1]
        ])
        
        # Rotate points to align with box
        rotated_points = np.dot(local_points, rotation_matrix.T)
        
        # Check if points are inside the box
        half_width, half_height, half_length = dimensions[0]/2, dimensions[1]/2, dimensions[2]/2
        mask_x = np.abs(rotated_points[:, 0]) <= half_width
        mask_y = np.abs(rotated_points[:, 1]) <= half_length
        mask_z = np.abs(rotated_points[:, 2]) <= half_height
        mask = np.logical_and(np.logical_and(mask_x, mask_y), mask_z)
        
        # Return points inside the box
        return points[mask]
    
    def _compute_features(self, points):
        """Compute features for points in a bounding box"""
        # Basic geometric features
        xyz = points[:, :3]
        min_point = np.min(xyz, axis=0)
        max_point = np.max(xyz, axis=0)
        center = np.mean(xyz, axis=0)
        dimensions = max_point - min_point
        
        # Calculate eigenvalues for shape analysis
        covariance = np.cov(xyz, rowvar=False)
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(covariance)
            # Sort eigenvalues in descending order
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
        except np.linalg.LinAlgError:
            eigenvalues = np.zeros(3)
        
        # Normalize eigenvalues
        if np.sum(eigenvalues) > 0:
            norm_eigenvalues = eigenvalues / np.sum(eigenvalues)
        else:
            norm_eigenvalues = np.zeros(3)
        
        # Statistical features
        std_xyz = np.std(xyz, axis=0)
        intensity = points[:, 3]
        intensity_stats = [np.mean(intensity), np.std(intensity), np.min(intensity), np.max(intensity)]
        
        # Point count and density
        point_count = len(points)
        volume = dimensions[0] * dimensions[1] * dimensions[2]
        density = point_count / (volume + 1e-10)
        
        # Combine all features
        features = np.concatenate([
            center,                 # 3 features
            dimensions,             # 3 features
            eigenvalues,            # 3 features
            norm_eigenvalues,       # 3 features
            std_xyz,                # 3 features
            intensity_stats,        # 4 features
            [point_count, density]  # 2 features
        ])
        
        # Pad to get to feature size of 64 (if needed for model)
        if len(features) < 64:
            padding = np.zeros(64 - len(features))
            features = np.concatenate([features, padding])
        
        return features
    
    def get_split(self, split):
        """Return self for compatibility with Open3D-ML"""
        return self

# Focal Loss implementation
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return focal_loss.mean()

# Custom VRU detection model
class VRUDetectionModel(nn.Module):
    def __init__(self, input_dim=64, num_classes=9):
        super(VRUDetectionModel, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        self.classifier = nn.Linear(128, num_classes)
        
    def forward(self, x):
        if isinstance(x, dict):
            # Handle input from DataLoader
            x = x['features']
            
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        
        # Format output to match Open3D-ML expectations
        return {
            'cls_preds': logits,
            'box_preds': torch.zeros((x.size(0), 7), device=x.device),  # Placeholder
        }

# Define validation function
def validate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in val_loader:
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(features)
            logits = outputs['cls_preds']
            
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    
    print(f"Validation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

def train_model(data_dir, model_save_path, config_path=None):
    print(f"Training model using data from {data_dir}")
    
    # Load configuration
    config = None
    if config_path:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize dataset
    full_dataset = LiDARVRUDataset(data_dir)
    
    # Split into train and validation sets
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Create data loaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_dim = 64  # Match the feature dimension
    model = VRUDetectionModel(input_dim=input_dim, num_classes=len(class_map))
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    
    # Training parameters
    num_epochs = 50
    best_val_accuracy = 0.0
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for i, batch in enumerate(train_loader):
            # Get data
            features = batch['features'].to(device)
            labels = batch['label'].to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(features)
            logits = outputs['cls_preds']
            
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            # Print statistics
            running_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {running_loss/10:.4f}")
                running_loss = 0.0
        
        # Validate after each epoch
        val_loss, val_accuracy = validate(model, val_loader, device)
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved with accuracy: {val_accuracy:.2f}%")
    
    print("Training complete!")

def main():
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='data', help='Path to the dataset')
    parser.add_argument('--config', type=str, default='configs/pointpillars.yml', help='Path to the config file')
    parser.add_argument('--checkpoint', type=str, default="models/model.pth", help='Path to save the model')
    args = parser.parse_args()

    # Train the model
    train_model(args.dataset_path, args.checkpoint, args.config)

if __name__ == '__main__':
    main()