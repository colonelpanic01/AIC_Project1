import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

# Simple Neural Network Model (Modify as needed)
class ObjectDetectionModel(nn.Module):
    def __init__(self, input_dim=3, output_dim=7):  # X, Y, Z → Object Attributes
        super(ObjectDetectionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, output_dim)  # Predict width, height, length, etc.

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Function to load point clouds and labels
def load_data(data_dir):
    scans_path = os.path.join(data_dir, "scans")
    labels_path = os.path.join(data_dir, "labels")

    X, Y = [], []
    for filename in os.listdir(scans_path):
        if filename.endswith(".bin"):
            scan_file = os.path.join(scans_path, filename)
            label_file = os.path.join(labels_path, filename.replace(".bin", ".txt"))

            # Load point cloud data
            points = np.fromfile(scan_file, dtype=np.float32).reshape(-1, 4)[:, :3]  # Ignore intensity
            labels = []
            if os.path.exists(label_file):
                with open(label_file, "r") as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        labels.append([float(x) for x in parts[1:]])  # Ignore object type

            # Use centroid of points for training (simplified)
            centroid = np.mean(points, axis=0) if len(points) > 0 else np.zeros(3)
            if labels:
                for label in labels:
                    X.append(centroid)
                    Y.append(label)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

# Train Function
def train_model(data_dir, model_path, epochs=10, lr=0.001):
    model = ObjectDetectionModel()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X, Y = load_data(data_dir)
    dataset = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = loss_fn(predictions, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

train_model(data_dir="data", model_path="model")