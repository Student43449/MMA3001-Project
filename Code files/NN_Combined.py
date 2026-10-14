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

class GolfNeuralNet(nn.Module):
    def __init__(self, input_dim):
        super(GolfNeuralNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2)
        )
    def forward(self, x):
        return self.network(x)

def train_and_evaluate(train_path, val_path):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    # Dynamically select all features (including Club Speed, Ball Speed, etc.)
    features = [c for c in train_df.columns if c not in ['Club', 'Side Total', 'Total']]
    targets = ['Side Total', 'Total']
    
    X_train, y_train = train_df[features], train_df[targets]
    X_val, y_val = val_df[features], val_df[targets]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_train_t, y_train_t = torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train.values)
    X_val_t, y_val_t = torch.FloatTensor(X_val_scaled), torch.FloatTensor(y_val.values)
    model = GolfNeuralNet(X_train_scaled.shape[1])
    criterion, optimizer = nn.MSELoss(), optim.Adam(model.parameters(), lr=0.001)
    print("Training Multi-Output Neural Network Model...")
    epochs = 150
    train_losses, val_losses = [], []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t)
            val_loss = criterion(val_outputs, y_val_t)
            val_losses.append(val_loss.item())
        if epoch % 25 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs} | Train Loss (MSE): {loss.item():.4f} | Val Loss (MSE): {val_loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        train_preds, val_preds = model(X_train_t).numpy(), model(X_val_t).numpy()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vis_dir = os.path.join(base_dir, 'Raw data visualization')
    os.makedirs(vis_dir, exist_ok=True)

    # Generate Plots
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', color='teal')
    plt.plot(range(1, epochs + 1), val_losses, label='Val Loss', color='coral')
    plt.title('Loss Convergence (Error vs Epoch Count)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'loss_convergence.png'))
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(y_train['Side Total'], train_preds[:, 0], color='green', alpha=0.5, label='Train')
    plt.plot([y_train['Side Total'].min(), y_train['Side Total'].max()], [y_train['Side Total'].min(), y_train['Side Total'].max()], 'r--')
    plt.title('Side Total: Actual vs Predicted (Train)')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.scatter(y_val['Side Total'], val_preds[:, 0], color='blue', alpha=0.5, label='Val')
    plt.plot([y_val['Side Total'].min(), y_val['Side Total'].max()], [y_val['Side Total'].min(), y_val['Side Total'].max()], 'r--')
    plt.title('Side Total: Actual vs Predicted (Validation)')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'actual_vs_predicted_side_total.png'))
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(y_train['Total'], train_preds[:, 1], color='green', alpha=0.5, label='Train')
    plt.plot([y_train['Total'].min(), y_train['Total'].max()], [y_train['Total'].min(), y_train['Total'].max()], 'r--')
    plt.title('Total Distance: Actual vs Predicted (Train)')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.scatter(y_val['Total'], val_preds[:, 1], color='blue', alpha=0.5, label='Val')
    plt.plot([y_val['Total'].min(), y_val['Total'].max()], [y_val['Total'].min(), y_val['Total'].max()], 'r--')
    plt.title('Total Distance: Actual vs Predicted (Validation)')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'actual_vs_predicted_total_distance.png'))
    plt.close()

    for i, target in enumerate(targets):
        mae_tr = mean_absolute_error(y_train[target], train_preds[:, i])
        r2_tr = r2_score(y_train[target], train_preds[:, i])
        mae_val = mean_absolute_error(y_val[target], val_preds[:, i])
        r2_val = r2_score(y_val[target], val_preds[:, i])
        print(f"\n=== {target} Training Evaluation ===")
        print(pd.DataFrame({'Metric': ['MAE', 'R2 Score'], 'Value': [mae_tr, r2_tr]}).to_string(index=False))
        print(f"\n=== {target} Validation Evaluation ===")
        print(pd.DataFrame({'Metric': ['MAE', 'R2 Score'], 'Value': [mae_val, r2_val]}).to_string(index=False))

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, 'data files', 'Training_data.csv')
    val_path = os.path.join(base_dir, 'data files', 'Validation_data.csv')
    train_and_evaluate(train_path, val_path)
