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

# Single-Output Neural Network (1 output node for Total Distance)
class GolfNeuralNet(nn.Module):
    def __init__(self, input_dim):
        super(GolfNeuralNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # One output node: Total Distance
        )

    def forward(self, x):
        return self.network(x)

def prompt_multiple_files(dataset_type):
    file_paths = []
    print(f"\n--- Configure {dataset_type} Datasets ---")
    while True:
        while True:
            path = input(f"Enter path to {dataset_type} dataset: ").strip()
            if os.path.exists(path):
                file_paths.append(path)
                break
            print(f"Error: File '{path}' does not exist. Please try again.\n")

        another = input("Is there another file? (YES, NO): ").strip().upper()
        if another != 'YES':
            break
    return file_paths

def load_and_combine_data(file_paths, feature_start=3, feature_end=18, target_col='Total'):
    dfs = []
    for path in file_paths:
        try:
            df = pd.read_csv(path)
            if target_col not in df.columns:
                print(f"Check dataset: {target_col} not found")
                sys.exit(1)
            dfs.append(df)
        except Exception:
            print("Check dataset")
            sys.exit(1)

    combined_df = pd.concat(dfs, ignore_index=True)
    feature_cols = combined_df.columns[feature_start:feature_end]

    X = combined_df[feature_cols].apply(pd.to_numeric, errors='coerce')
    y = combined_df[[target_col]].apply(pd.to_numeric, errors='coerce')

    aligned = pd.concat([X, y], axis=1).dropna()
    X = aligned[feature_cols]
    y = aligned[target_col]

    if len(X) == 0:
        print("Check dataset: Empty after alignment")
        sys.exit(1)

    return X, y

if __name__ == '__main__':
    train_files = prompt_multiple_files("Training")
    val_files = prompt_multiple_files("Validation")

    target = 'Total'
    X_train, y_train = load_and_combine_data(train_files, target_col=target)
    X_val, y_val = load_and_combine_data(val_files, target_col=target)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    X_train_t = torch.FloatTensor(X_train_scaled)
    y_train_t = torch.FloatTensor(y_train.values).unsqueeze(1)
    X_val_t = torch.FloatTensor(X_val_scaled)
    y_val_t = torch.FloatTensor(y_val.values).unsqueeze(1)

    model = GolfNeuralNet(X_train_scaled.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 150
    train_losses = []
    val_losses = []

    print(f"\nTraining Neural Network Model for {target}...")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        predictions = model(X_train_t)
        loss = criterion(predictions, y_train_t)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = criterion(val_preds, y_val_t)

        train_losses.append(loss.item())
        val_losses.append(val_loss.item())

        if epoch % 25 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs} | Train Loss (MSE): {loss.item():.4f} | Val Loss (MSE): {val_loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        train_preds_np = model(X_train_t).numpy().flatten()
        val_preds_np = model(X_val_t).numpy().flatten()

    # Plot Loss Convergence
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', color='teal')
    plt.plot(range(1, epochs + 1), val_losses, label='Val Loss', color='coral')
    plt.title(f'{target} Loss Convergence (Error vs Epoch Count)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.grid(True)
    plt.legend()
    plt.show()

    mae_tr = mean_absolute_error(y_train.values, train_preds_np)
    r2_tr = r2_score(y_train.values, train_preds_np)

    mae_val = mean_absolute_error(y_val.values, val_preds_np)
    r2_val = r2_score(y_val.values, val_preds_np)

    print(f"\n=== Training Set: {target} Evaluation Error Table ===")
    print(pd.DataFrame({'Metric': ['MAE', 'R² Score'], 'Value': [mae_tr, r2_tr]}).to_string(index=False))

    print(f"\n=== Validation Set: {target} Evaluation Error Table ===")
    print(pd.DataFrame({'Metric': ['MAE', 'R² Score'], 'Value': [mae_val, r2_val]}).to_string(index=False))

    # Plot Train Scatter
    plt.figure(figsize=(8, 6))
    plt.scatter(y_train.values, train_preds_np, color='green', alpha=0.5, edgecolors='k', label='Train (Total Distance Only)')
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2, label='Perfect Fit')
    plt.title(f'{target}: Actual Vs Predicted Training')
    plt.xlabel(f'Actual {target}')
    plt.ylabel(f'Predicted {target}')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Plot Val Scatter
    plt.figure(figsize=(8, 6))
    plt.scatter(y_val.values, val_preds_np, color='blue', alpha=0.5, edgecolors='k', label='Val (Total Distance Only)')
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2, label='Perfect Fit')
    plt.title(f'{target}: Actual Vs Predicted Validation')
    plt.xlabel(f'Actual {target}')
    plt.ylabel(f'Predicted {target}')
    plt.grid(True)
    plt.legend()
    plt.show()
