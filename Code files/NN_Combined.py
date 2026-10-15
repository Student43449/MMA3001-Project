import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from pathlib import Path

# Define the Multi-Output Neural Network Architecture (predicts 2 target variables)
class GolfNeuralNet(nn.Module):
    def __init__(self, input_dim):
        super(GolfNeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 500)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(500, 500)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(500, 2)  # Output layer predicting 2 values (Side Total, Total)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

# Load Training and Validation datasets directly
train_df = pd.read_csv('/content/MMA3001-Project/data files/Training data.csv')
val_df = pd.read_csv('/content/MMA3001-Project/data files/Validation data.csv')

# Split into features (X) and targets (y) using available, verified columns
features = [
    'Club Speed', 'Attack Angle', 'Club Path', 'Low Point', 'Swing Plane',
    'Swing Direction', 'Dyn. Loft', 'Face Angle', 'Face To Path', 'Ball Speed',
    'Smash Factor', 'Launch Angle', 'Launch Direction', 'Spin Rate', 'Spin Axis'
]
targets = ['Side Total', 'Total']

X_train = train_df[features]
y_train = train_df[targets]
X_val = val_df[features]
y_val = val_df[targets]

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Convert to PyTorch Tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32)

# Instantiate Model, Loss Function, and Optimizer
model = GolfNeuralNet(len(features))
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the Model
print("\nTraining Multi-Output Neural Network Model...")
epochs = 150
train_losses = []
val_losses = []

for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    # Evaluate on Validation set
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_loss = criterion(val_outputs, y_val_tensor)

    train_losses.append(loss.item())
    val_losses.append(val_loss.item())

    if epoch == 1 or epoch % 25 == 0:
        print(f"Epoch {epoch:03d}/{epochs} | Train Loss (MSE): {loss.item():.4f} | Val Loss (MSE): {val_loss.item():.4f}")

# Model evaluation
model.eval()
with torch.no_grad():
    train_preds_np = model(X_train_tensor).numpy()
    val_preds_np = model(X_val_tensor).numpy()

# Plot actual vs predicted values side-by-side
for idx, target_name in enumerate(targets):
    plt.figure(figsize=(12, 5))

    # Train plot
    plt.subplot(1, 2, 1)
    tr_true = y_train.values[:, idx]
    tr_pred = train_preds_np[:, idx]
    mae_tr = mean_absolute_error(tr_true, tr_pred)
    r2_tr = r2_score(tr_true, tr_pred)
    plt.scatter(tr_true, tr_pred, alpha=0.3, color='blue')
    plt.plot([tr_true.min(), tr_true.max()], [tr_true.min(), tr_true.max()], 'r--', lw=2)
    plt.title(f'Train: Actual vs Pred ({target_name})\nMAE: {mae_tr:.2f}, R²: {r2_tr:.4f}')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')

    # Val plot
    plt.subplot(1, 2, 2)
    v_true = y_val.values[:, idx]
    v_pred = val_preds_np[:, idx]
    mae_val = mean_absolute_error(v_true, v_pred)
    r2_val = r2_score(v_true, v_pred)
    plt.scatter(v_true, v_pred, alpha=0.3, color='green')
    plt.plot([v_true.min(), v_true.max()], [v_true.min(), v_true.max()], 'r--', lw=2)
    plt.title(f'Val: Actual vs Pred ({target_name})\nMAE: {mae_val:.2f}, R²: {r2_val:.4f}')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')

    plt.tight_layout()
    plt.show()