import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import numpy as np
from vru_core import LiDARVRUDataset, VRUDetectionModel
from config import load_config, CLASS_MAP

def validate(model, val_loader, device):
    """Validate model on validation dataset"""
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
    """Train the VRU detection model"""
    print(f"Training model using data from {data_dir}")
    
    # Load configuration
    config = load_config(config_path)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize dataset
    full_dataset = LiDARVRUDataset(data_dir, config=config)
    
    # Split into train and validation sets
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create data loaders
    batch_size = config['batch_size']
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_dim = config['input_feature_dim']
    num_classes = config['num_classes']
    model = VRUDetectionModel(input_dim=input_dim, num_classes=num_classes)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay']
    )
    
    # Training parameters
    num_epochs = config['num_epochs']
    best_val_accuracy = 0.0
    
    # Ensure model_save_path directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
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
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'class_map': CLASS_MAP,
                'epoch': epoch,
                'accuracy': val_accuracy,
                'input_dim': input_dim,
                'num_classes': num_classes
            }, model_save_path)
            print(f"New best model saved with accuracy: {val_accuracy:.2f}%")
    
    print("Training complete!")

def main():
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='data', help='Path to the dataset')
    parser.add_argument('--config', type=str, default='configs/vru_config.yml', help='Path to the config file')
    parser.add_argument('--checkpoint', type=str, default="models/vru_model.pth", help='Path to save the model')
    args = parser.parse_args()

    # Train the model
    train_model(args.dataset_path, args.checkpoint, args.config)

if __name__ == '__main__':
    main()
