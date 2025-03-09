import numpy as np
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from lidar_vru_detection import LiDARVRUDetector

class LiDARVRUDataset(Dataset):
    """
    Dataset for LiDAR VRU detection
    """
    def __init__(self, data_dir, transform=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        self.data_dir = data_dir
        self.transform = transform
        self.scan_files = sorted(glob.glob(os.path.join(data_dir, 'scans', '*.bin')))
        self.label_files = sorted(glob.glob(os.path.join(data_dir, 'labels', '*.txt')))
        
        # Ensure matching files
        assert len(self.scan_files) == len(self.label_files), "Mismatch between scan and label files"
        
        # Store features and labels
        self.features = []
        self.labels = []
        
        # Load and preprocess data
        self._preprocess_data()
        
    def _preprocess_data(self):
        """
        Preprocess all data files to extract features and labels
        """
        print("Preprocessing dataset...")
        detector = LiDARVRUDetector()
        
        for i, (scan_file, label_file) in enumerate(tqdm(zip(self.scan_files, self.label_files), total=len(self.scan_files))):
            # Only process a subset for faster development
            if i > 100:  # Process first 1000 files
                break
                
            # Load point cloud
            points = detector.load_point_cloud(scan_file)
            filtered_points = detector.preprocess_point_cloud(points)
            
            # Extract clusters
            clusters = detector.extract_clusters(filtered_points)
            
            # Load ground truth labels
            gt_labels = self._load_labels(label_file)
            
            # Match clusters to ground truth based on IoU
            for cluster_points in clusters:
                # Compute cluster features
                features = detector.compute_cluster_features(cluster_points)
                
                # Compute bounding box for the cluster
                cluster_bbox = detector.compute_oriented_bbox(cluster_points)
                
                # Find best matching ground truth
                best_iou = 0.3  # Minimum IoU threshold
                best_label = "other"
                
                for gt in gt_labels:
                    # Calculate IoU between cluster and ground truth
                    iou = self._calculate_iou(cluster_bbox, gt['bbox'])
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_label = gt['type']
                
                # Map label to class index
                class_idx = self._label_to_idx(best_label)
                
                # Add to dataset
                self.features.append(features)
                self.labels.append(class_idx)
        
        # Convert to numpy arrays
        self.features = np.array(self.features, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        
        print(f"Dataset preprocessed: {len(self.features)} samples")
        
        # Print class distribution
        classes, counts = np.unique(self.labels, return_counts=True)
        print("Class distribution:")
        class_map = {0: "pedestrian", 1: "cyclist", 2: "motorcycle", 3: "other"}
        for cls, count in zip(classes, counts):
            print(f"  {class_map[cls]}: {count} samples ({count/len(self.labels)*100:.2f}%)")
    
    def _load_labels(self, label_file):
        """Load ground truth labels from file"""
        labels = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                cls_type = parts[0]
                bbox = [float(p) for p in parts[1:8]]
                
                # Only keep VRU classes
                if any(vru in cls_type.lower() for vru in ['pedestrian', 'cyclist', 'bicycle', 'motorcycle']):
                    labels.append({'type': cls_type, 'bbox': bbox})
        return labels
    
    def _calculate_iou(self, bbox1, bbox2):
        """
        Calculate IoU between two 3D bounding boxes (simplified)
        """
        # Extract box parameters
        center1 = np.array(bbox1[:3])
        dim1 = np.array(bbox1[3:6])
        
        center2 = np.array(bbox2[:3])
        dim2 = np.array(bbox2[3:6])
        
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
        
        if np.any(intersection_max < intersection_min):
            return 0.0
        
        intersection_volume = np.prod(intersection_max - intersection_min)
        
        # Calculate box volumes
        volume1 = np.prod(dim1)
        volume2 = np.prod(dim2)
        
        # Calculate IoU
        union_volume = volume1 + volume2 - intersection_volume
        iou = intersection_volume / union_volume if union_volume > 0 else 0.0
        
        return iou
    
    def _label_to_idx(self, label):
        """Convert label string to class index"""
        if 'pedestrian' in label.lower():
            return 0
        elif 'cyclist' in label.lower() or 'bicycle' in label.lower():
            return 1
        elif 'motorcycle' in label.lower():
            return 2
        else:
            return 3  # Other
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        features = torch.FloatTensor(self.features[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        if self.transform:
            features = self.transform(features)
            
        return features, label


def train_model(data_dir, model_save_path, epochs=10, batch_size=32, lr=0.001):
    """
    Train the ML model using labeled data from the data directory
    """
    print(f"Training model using data from {data_dir}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare dataset
    dataset = LiDARVRUDataset(data_dir)
    
    # Split dataset
    train_indices, val_indices = train_test_split(
        np.arange(len(dataset)),
        test_size=0.2,
        random_state=42,
        stratify=dataset.labels
    )
    
    # Select features for model input (exclude some redundant features)
    input_dim = min(64, dataset.features.shape[1])  # Limit to 64 features
    
    # Create model
    model = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Dropout(0.3),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.BatchNorm1d(64),
        nn.Linear(64, 4)  # 4 classes: pedestrian, cyclist, motorcycle, other
    ).to(device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # Create data loaders
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(train_indices),
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=torch.utils.data.SubsetRandomSampler(val_indices),
        num_workers=4,
        pin_memory=True
    )
    
    # Training loop
    best_val_acc = 0.0
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            # Ensure features have the right shape
            features = features[:, :input_dim].to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for features, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                # Ensure features have the right shape
                features = features[:, :input_dim].to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                # Track metrics
                val_loss += loss.item() * features.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print metrics
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"Saving best model with validation accuracy: {val_acc:.4f}")
            torch.save(model.state_dict(), model_save_path)
    
    print(f"Training complete! Best validation accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {model_save_path}")
    
    return model
if __name__ == '__main__':
    train_model("data", "/models/model.pth")