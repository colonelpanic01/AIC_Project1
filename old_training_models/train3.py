import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from open3d.ml.datasets import CustomLidarDataset
from open3d.ml.torch.models import PointPillars

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

# Custom loss function combining Smooth L1, Focal, and Cross-Entropy losses
def compute_loss(outputs, targets, config):
    # Extract predictions and targets
    cls_preds = outputs['cls_preds']  # Class predictions
    box_preds = outputs['box_preds']  # Bounding box predictions
    cls_targets = targets['cls_targets']  # Class labels
    box_targets = targets['box_targets']  # Bounding box targets

    # Focal Loss for classification
    focal_loss = FocalLoss(
        gamma=config['model']['loss']['focal']['gamma'],
        alpha=config['model']['loss']['focal']['alpha']
    )
    cls_loss = focal_loss(cls_preds, cls_targets)

    # Smooth L1 Loss for bounding box regression
    smooth_l1_loss = nn.SmoothL1Loss(beta=config['model']['loss']['smooth_l1']['beta'])
    box_loss = smooth_l1_loss(box_preds, box_targets)

    # Cross-Entropy Loss for direction classification (if applicable)
    if 'dir_preds' in outputs:
        dir_preds = outputs['dir_preds']
        dir_targets = targets['dir_targets']
        ce_loss = nn.CrossEntropyLoss()
        dir_loss = ce_loss(dir_preds, dir_targets)
    else:
        dir_loss = 0.0

    # Combine losses with weights from config
    total_loss = (
        config['model']['loss']['focal']['loss_weight'] * cls_loss +
        config['model']['loss']['smooth_l1']['loss_weight'] * box_loss +
        config['model']['loss']['cross_entropy']['loss_weight'] * dir_loss
    )
    return total_loss

# Validation function
def validate(model, val_loader, config):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(batch['point_cloud'])
            targets = {
                'cls_targets': batch['cls_labels'],
                'box_targets': batch['box_labels'],
                'dir_targets': batch['dir_labels'] if 'dir_labels' in batch else None
            }
            loss = compute_loss(outputs, targets, config)
            total_loss += loss.item()
    avg_loss = total_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss

def main():
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--checkpoint', type=str, default="modelsV2", help='Path to the checkpoint file')
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize dataset
    dataset = CustomLidarDataset(dataset_path=args.dataset_path)
    train_split = dataset.get_split('train')
    val_split = dataset.get_split('val')

    # Create DataLoader for training and validation
    train_loader = DataLoader(train_split, batch_size=config['pipeline']['batch_size'], shuffle=True)
    val_loader = DataLoader(val_split, batch_size=config['pipeline']['val_batch_size'], shuffle=False)

    # Initialize model
    model = PointPillars(**config['model'])
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))

    # Set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['pipeline']['optimizer']['lr'])

    # Training loop
    for epoch in range(config['pipeline']['max_epoch']):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            # Forward pass
            outputs = model(batch['point_cloud'])
            targets = {
                'cls_targets': batch['cls_labels'],
                'box_targets': batch['box_labels'],
                'dir_targets': batch['dir_labels'] if 'dir_labels' in batch else None
            }
            loss = compute_loss(outputs, targets, config)
            epoch_loss += loss.item()

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Print training loss
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{config['pipeline']['max_epoch']}], Loss: {avg_epoch_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % config['pipeline']['save_ckpt_freq'] == 0:
            checkpoint_path = f"checkpoint_epoch_{epoch + 1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

        # Validation
        if (epoch + 1) % config['pipeline']['val_freq'] == 0:
            validate(model, val_loader, config)

if __name__ == '__main__':
    main()