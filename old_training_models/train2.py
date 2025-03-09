import numpy as np
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from tqdm import tqdm
from lidar_vru_detection import LiDARVRUDetector
from torch.utils.data.sampler import WeightedRandomSampler
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import Counter

class LiDARVRUDataset(Dataset):
    """
    Improved Dataset for LiDAR VRU detection
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
        self.original_clusters = []  # Store original cluster points for augmentation
        
        # Load and preprocess data
        self._preprocess_data()
        
    def _preprocess_data(self):
        """
        Preprocess all data files to extract features and labels
        """
        print("Preprocessing dataset...")
        detector = LiDARVRUDetector()
        
        # Process all files (adjust the limit if needed for faster development)
        max_files_to_process = len(self.scan_files)  # Process all files
        
        for i, (scan_file, label_file) in enumerate(tqdm(zip(self.scan_files, self.label_files), total=min(max_files_to_process, len(self.scan_files)))):
            # Optional limit for development
            if i >= max_files_to_process:
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
                
                # Map label to class index (only keeping pedestrian, cyclist, motorcycle)
                class_idx = self._label_to_idx(best_label)
                
                # Only add if it's one of our three target classes (0, 1, 2)
                if class_idx < 3:
                    self.features.append(features)
                    self.labels.append(class_idx)
                    self.original_clusters.append(cluster_points)  # Store for augmentation
        
        # Convert to numpy arrays
        self.features = np.array(self.features, dtype=np.float32)
        self.labels = np.array(self.labels, dtype=np.int64)
        
        print(f"Dataset preprocessed: {len(self.features)} samples")
        
        # Print class distribution
        classes, counts = np.unique(self.labels, return_counts=True)
        print("Class distribution:")
        class_map = {0: "pedestrian", 1: "cyclist", 2: "motorcycle"}
        for cls, count in zip(classes, counts):
            if cls in class_map:
                print(f"  {class_map[cls]}: {count} samples ({count/len(self.labels)*100:.2f}%)")
        
        # Perform data augmentation for underrepresented classes
        self._augment_data()
        
        # Feature normalization - important for neural network training
        self._normalize_features()
    
    def _normalize_features(self):
        """
        Normalize features using z-score normalization (mean=0, std=1)
        """
        # Calculate mean and std for each feature dimension
        means = np.mean(self.features, axis=0)
        stds = np.std(self.features, axis=0)
        
        # Replace zeros in stds to avoid division by zero
        stds[stds == 0] = 1.0
        
        # Normalize
        self.features = (self.features - means) / stds
        
        # Store normalization parameters for inference
        self.norm_params = {
            'means': means,
            'stds': stds
        }
    
    def _augment_data(self):
        """
        Augment data for underrepresented classes
        """
        classes, counts = np.unique(self.labels, return_counts=True)
        
        # Find the class with maximum samples
        max_samples = np.max(counts)
        
        # Create augmented data to balance classes
        augmented_features = []
        augmented_labels = []
        
        for cls_idx in range(3):  # 0: pedestrian, 1: cyclist, 2: motorcycle
            # Get indices for this class
            cls_indices = np.where(self.labels == cls_idx)[0]
            
            if len(cls_indices) == 0:
                continue
                
            # Number of augmented samples needed
            n_augment = max_samples - len(cls_indices)
            
            # Only augment if more samples are needed
            if n_augment <= 0:
                continue
                
            print(f"Augmenting {n_augment} samples for class {cls_idx}")
            
            # Sample with replacement from this class
            augment_indices = np.random.choice(cls_indices, size=n_augment, replace=True)
            
            for idx in augment_indices:
                # Get original feature and cluster
                orig_feature = self.features[idx]
                orig_cluster = self.original_clusters[idx]
                
                # Apply augmentation strategies
                aug_feature, aug_cluster = self._augment_sample(orig_feature, orig_cluster)
                
                # Add augmented sample
                augmented_features.append(aug_feature)
                augmented_labels.append(cls_idx)
        
        # Add augmented data to original data
        if augmented_features:
            self.features = np.vstack([self.features, np.array(augmented_features)])
            self.labels = np.append(self.labels, augmented_labels)
            
            # Print augmented class distribution
            print(f"After augmentation: {len(self.features)} samples")
            classes, counts = np.unique(self.labels, return_counts=True)
            class_map = {0: "pedestrian", 1: "cyclist", 2: "motorcycle"}
            for cls, count in zip(classes, counts):
                if cls in class_map:
                    print(f"  {class_map[cls]}: {count} samples ({count/len(self.labels)*100:.2f}%)")
    
    def _augment_sample(self, feature, cluster):
        """Apply augmentation to a single sample"""
        # Create a copy to avoid modifying the original
        aug_feature = feature.copy()
        aug_cluster = cluster.copy()
        
        # Apply random rotation around z-axis
        theta = np.random.uniform(-np.pi/6, np.pi/6)  # ±30 degrees
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        
        # Rotate cluster points
        centroid = np.mean(aug_cluster[:, :3], axis=0)
        aug_cluster[:, :3] = (aug_cluster[:, :3] - centroid) @ R.T + centroid
        
        # Apply small random scaling
        scale = np.random.uniform(0.9, 1.1, 3)  # Scale x, y, z by ±10%
        aug_cluster[:, :3] = (aug_cluster[:, :3] - centroid) * scale + centroid
        
        # Add small random noise to intensity
        noise_scale = 0.05
        aug_cluster[:, 3] += np.random.normal(0, noise_scale, aug_cluster[:, 3].shape)
        aug_cluster[:, 3] = np.clip(aug_cluster[:, 3], 0, 1)  # Keep intensity in valid range
        
        # Recalculate features from augmented cluster
        detector = LiDARVRUDetector()
        aug_feature = detector.compute_cluster_features(aug_cluster)
        
        return aug_feature, aug_cluster
    
    def _load_labels(self, label_file):
        """Load ground truth labels from file"""
        labels = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 8:  # Skip invalid lines
                    continue
                    
                cls_type = parts[0]
                bbox = [float(p) for p in parts[1:8]]
                
                # Only keep our three target classes
                if 'pedestrian' in cls_type.lower():
                    labels.append({'type': 'pedestrian', 'bbox': bbox})
                elif 'bicycle' in cls_type.lower() or 'cyclist' in cls_type.lower():
                    labels.append({'type': 'cyclist', 'bbox': bbox})
                elif 'motorcycle' in cls_type.lower():
                    labels.append({'type': 'motorcycle', 'bbox': bbox})
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
        
        # Consider rotation for more accurate IoU
        # For simplicity, we'll approximate with axis-aligned boxes
        
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
        
        intersection_volume = np.prod(np.maximum(intersection_max - intersection_min, 0))
        
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


def calculate_f1_score(y_true, y_pred):
    """
    Calculate macro F1 score
    """
    return f1_score(y_true, y_pred, average='macro')


def print_class_metrics(y_true, y_pred):
    """
    Print per-class precision, recall, and F1 score
    """
    class_map = {0: "pedestrian", 1: "cyclist", 2: "motorcycle"}
    
    # Calculate metrics
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    f1 = f1_score(y_true, y_pred, average=None)
    
    # Print per-class metrics
    print("\nPer-class metrics:")
    for cls_idx in range(len(precision)):
        if cls_idx in class_map:
            print(f"  {class_map[cls_idx]}: Precision={precision[cls_idx]:.4f}, Recall={recall[cls_idx]:.4f}, F1={f1[cls_idx]:.4f}")
    
    # Print confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    

def visualize_features(features, labels, title="Feature Visualization", save_path=None):
    """
    Visualize features using t-SNE
    """
    # Apply t-SNE for dimensionality reduction
    print("Applying t-SNE for visualization...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    reduced_features = tsne.fit_transform(features)
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        reduced_features[:, 0], 
        reduced_features[:, 1], 
        c=labels, 
        cmap='viridis', 
        alpha=0.6,
        s=20
    )
    
    # Add legend
    class_map = {0: "pedestrian", 1: "cyclist", 2: "motorcycle"}
    legend_labels = [class_map[i] for i in range(3) if i in np.unique(labels)]
    legend = plt.legend(
        handles=scatter.legend_elements()[0],
        labels=legend_labels,
        title="Classes",
        loc="upper right"
    )
    
    plt.title(title)
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def plot_training_history(history, save_path=None):
    """
    Plot training history
    """
    plt.figure(figsize=(15, 5))
    
    # Plot training & validation accuracy
    plt.subplot(1, 3, 1)
    plt.plot(history['train_acc'], label='Training')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot training & validation loss
    plt.subplot(1, 3, 2)
    plt.plot(history['train_loss'], label='Training')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot training & validation F1 score
    plt.subplot(1, 3, 3)
    plt.plot(history['train_f1'], label='Training')
    plt.plot(history['val_f1'], label='Validation')
    plt.title('Model F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


class FocalLoss(nn.Module):
    """
    Focal Loss to handle class imbalance better than standard CrossEntropyLoss
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(weight=alpha, reduction='none')
        
    def forward(self, inputs, targets):
        ce_loss = self.ce_loss(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def train_model(data_dir, model_save_path, epochs=20, batch_size=64, lr=0.001):
    """
    Train the ML model using labeled data with improved training process
    """
    # Create output directory for model and visualizations
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    print(f"Training model using data from {data_dir}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Fix random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Prepare dataset
    dataset = LiDARVRUDataset(data_dir)
    
    # Split dataset
    train_indices, val_indices = train_test_split(
        np.arange(len(dataset)),
        test_size=0.2,
        random_state=42,
        stratify=dataset.labels
    )
    
    # Calculate class weights for weighted loss
    class_counts = np.bincount(dataset.labels)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / np.sum(class_weights) * len(class_weights)
    class_weights_tensor = torch.FloatTensor(class_weights).to(device)
    
    # Select features for model input
    input_dim = min(64, dataset.features.shape[1])
    print(f"Using {input_dim} features for model input")
    
    # Visualize features before training
    print("Visualizing feature distribution...")
    # Sample a subset for visualization if dataset is large
    max_vis_samples = 5000
    if len(dataset.features) > max_vis_samples:
        vis_indices = np.random.choice(len(dataset.features), max_vis_samples, replace=False)
        vis_features = dataset.features[vis_indices, :input_dim]
        vis_labels = dataset.labels[vis_indices]
    else:
        vis_features = dataset.features[:, :input_dim]
        vis_labels = dataset.labels
    
    visualize_features(
        vis_features, 
        vis_labels, 
        "Feature Distribution (Before Training)",
        os.path.join(os.path.dirname(model_save_path), "features_visualization.png")
    )
    
    # Create improved model
    model = nn.Sequential(
        # Input layer
        nn.Linear(input_dim, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.2),
        
        # First hidden layer
        nn.Linear(128, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.3),
        
        # Second hidden layer
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.2),
        
        # Third hidden layer
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        
        # Output layer - just 3 classes instead of 4
        nn.Linear(64, 3)  # pedestrian, cyclist, motorcycle (removed "other")
    ).to(device)
    
    # Define weighted loss and optimizer
    criterion = FocalLoss(alpha=class_weights_tensor, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=3, factor=0.5, verbose=True
    )
    
    # Create weighted sampler for training data to handle class imbalance
    train_labels = [dataset.labels[i] for i in train_indices]
    class_sample_count = np.bincount(train_labels)
    weight = 1. / class_sample_count
    samples_weight = weight[train_labels]
    samples_weight = torch.from_numpy(samples_weight).float()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
    
    # Create data loaders
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,  # Use weighted sampler
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
    best_val_f1 = 0.0
    patience = 7
    patience_counter = 0
    
    # Save training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'train_f1': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        all_preds = []
        all_labels = []
        
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
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item() * features.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # Collect for F1 score calculation
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # Calculate F1 scores
        train_f1 = calculate_f1_score(all_labels, all_preds)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_labels = []
        
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
                
                # Collect for F1 score calculation
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
        
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        val_f1 = calculate_f1_score(all_val_labels, all_val_preds)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Print metrics
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
        
        # Print per-class metrics
        print_class_metrics(all_val_labels, all_val_preds)
        
        # Save best model (using F1 score instead of just accuracy)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            patience_counter = 0
            print(f"Saving best model with validation F1: {val_f1:.4f}, accuracy: {val_acc:.4f}")
            
            # Save model state
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
                'input_dim': input_dim,
                'norm_params': dataset.norm_params,  # Save normalization parameters
                'class_weights': class_weights
            }, model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping after {patience} epochs without improvement")
                break
    
    # Plot training history
    plot_training_history(
        history,
        os.path.join(os.path.dirname(model_save_path), "training_history.png")
    )
    
    # Load best model for final evaluation
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation on validation set
    model.eval()
    val_correct = 0
    val_total = 0
    all_val_preds = []
    all_val_labels = []
    all_val_probs = []
    
    with torch.no_grad():
        for features, labels in tqdm(val_loader, desc="Final Evaluation"):
            features = features[:, :input_dim].to(device)
            labels = labels.to(device)
            
            outputs = model(features)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
            all_val_preds.extend(predicted.cpu().numpy())
            all_val_labels.extend(labels.cpu().numpy())
            all_val_probs.extend(probs.cpu().numpy())
    
    final_acc = val_correct / val_total
    final_f1 = calculate_f1_score(all_val_labels, all_val_preds)
    
    print(f"\nFinal Evaluation:")
    print(f"Accuracy: {final_acc:.4f}")
    print(f"F1 Score: {final_f1:.4f}")
    print_class_metrics(all_val_labels, all_val_preds)
    
    # Save model summary and evaluation results
    model_info = {
        'accuracy': final_acc,
        'f1_score': final_f1,
        'precision': precision_score(all_val_labels, all_val_preds, average=None).tolist(),
        'recall': recall_score(all_val_labels, all_val_preds, average=None).tolist(),
        'confusion_matrix': confusion_matrix(all_val_labels, all_val_preds).tolist(),
        'class_distribution': Counter(all_val_labels)
    }
    
    # Save model info to a text file
    info_path = os.path.join(os.path.dirname(model_save_path), "model_info.txt")
    with open(info_path, 'w') as f:
        f.write("Model Summary:\n")
        f.write(f"Accuracy: {final_acc:.4f}\n")
        f.write(f"F1 Score: {final_f1:.4f}\n\n")
        # Save model info to a text file
    info_path = os.path.join(os.path.dirname(model_save_path), "model_info.txt")
    with open(info_path, 'w') as f:
        f.write("Model Summary:\n")
        f.write(f"Accuracy: {final_acc:.4f}\n")
        f.write(f"F1 Score: {final_f1:.4f}\n\n")
        
        # Add per-class metrics
        class_map = {0: "pedestrian", 1: "cyclist", 2: "motorcycle"}
        f.write("Per-Class Metrics:\n")
        for i, (prec, rec, f1_val) in enumerate(zip(
            model_info['precision'], 
            model_info['recall'], 
            f1_score(all_val_labels, all_val_preds, average=None)
        )):
            if i in class_map:
                f.write(f"{class_map[i]}: Precision={prec:.4f}, Recall={rec:.4f}, F1={f1_val:.4f}\n")
        
        # Add confusion matrix
        f.write("\nConfusion Matrix:\n")
        cm = model_info['confusion_matrix']
        for row in cm:
            f.write(f"{row}\n")
        
        # Add class distribution
        f.write("\nClass Distribution:\n")
        for cls, count in model_info['class_distribution'].items():
            if cls in class_map:
                f.write(f"{class_map[cls]}: {count} samples\n")
    
    print(f"Model information saved to {info_path}")
    
    return model, model_info


def inference(model_path, features):
    """
    Run inference using the trained model
    
    Args:
        model_path: Path to the saved model
        features: Numpy array of features (shape: [n_samples, n_features])
        
    Returns:
        Predicted class labels and probabilities
    """
    # Load the model
    checkpoint = torch.load(model_path)
    input_dim = checkpoint['input_dim']
    norm_params = checkpoint['norm_params']
    
    # Initialize model architecture (same as in training)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = nn.Sequential(
        nn.Linear(input_dim, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.2),
        
        nn.Linear(128, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.3),
        
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.2),
        
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        
        nn.Linear(64, 3)
    ).to(device)
    
    # Load the trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Apply normalization using saved parameters
    means = norm_params['means']
    stds = norm_params['stds']
    
    # Ensure we only use the features needed by the model
    if features.shape[1] > input_dim:
        features = features[:, :input_dim]
    
    # Normalize features
    features_norm = (features - means[:input_dim]) / stds[:input_dim]
    
    # Convert to tensor
    features_tensor = torch.FloatTensor(features_norm).to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(features_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
    
    # Convert to numpy arrays
    predicted = predicted.cpu().numpy()
    probabilities = probabilities.cpu().numpy()
    
    return predicted, probabilities


def main():
    """
    Main function to run training or inference
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='LiDAR VRU Detection Model')
    parser.add_argument('--mode', type=str, default='train', help='Mode: train or inference')
    parser.add_argument('--data_dir', type=str, default='data', help='Data directory')
    parser.add_argument('--model_path', type=str, default='models/vru_detector.pth', help='Path to save/load model')
    parser.add_argument('--epochs', type=int, default=30, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    
    if args.mode == 'train':
        # Train the model
        print(f"Training model with data from {args.data_dir}")
        train_model(
            args.data_dir,
            args.model_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.learning_rate
        )
        
    elif args.mode == 'inference':
        print("Inference mode not implemented in this script.")
        print("Please use the LiDARVRUDetector class for real-time inference.")
    
    else:
        print(f"Unknown mode: {args.mode}")
        print("Available modes: train, inference")


if __name__ == "__main__":
    main()