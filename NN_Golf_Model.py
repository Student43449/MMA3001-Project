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

# Define the Neural Network Architecture
class GolfNeuralNet(nn.Module):
    def __init__(self, input_dim):
        super(GolfNeuralNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.network(x)

def prompt_multiple_files(dataset_type):
    """
    Prompts the user to enter paths for one or more files and combines them.
    """
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

def load_and_combine_data(file_paths, feature_start=1, feature_end=10, target_col='Side Total'):
    """
    Loads multiple CSV files, drops NaNs, converts to numeric, and combines them.
    """
    dfs = []
    for path in file_paths:
        try:
            df = pd.read_csv(path)
            if target_col not in df.columns:
                print("check data set")
                sys.exit(1)
            dfs.append(df)
        except Exception:
            print("check data set")
            sys.exit(1)
            
    combined_df = pd.concat(dfs, ignore_index=True)
    
    feature_cols = combined_df.columns[feature_start:feature_end]
    
    X = combined_df[feature_cols].apply(pd.to_numeric, errors='coerce')
    y = combined_df[target_col].apply(pd.to_numeric, errors='coerce')
    
    aligned = pd.concat([X, y], axis=1).dropna()
    X = aligned[feature_cols]
    y = aligned[target_col]
    
    if len(X) == 0:
        print("check data set")
        sys.exit(1)
        
    return X, y

if __name__ == '__main__':
    # Get file pathways
    train_files = prompt_multiple_files("Training")
    val_files = prompt_multiple_files("Validation")
    
    # Load and clean datasets
    X_train, y_train = load_and_combine_data(train_files)
    X_val, y_val = load_and_combine_data(val_files)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Convert to PyTorch tensors
    X_train_t = torch.FloatTensor(X_train_scaled)
    y_train_t = torch.FloatTensor(y_train.values).view(-1, 1)
    X_val_t = torch.FloatTensor(X_val_scaled)
    y_val_t = torch.FloatTensor(y_val.values).view(-1, 1)
    
    # Model initialization
    model = GolfNeuralNet(X_train_scaled.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    # Training Loop
    epochs = 150  # Reasonable limit of epochs
    train_losses = []
    val_losses = []
    
    print("\nTraining Neural Network Model...")
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        predictions = model(X_train_t)
        loss = criterion(predictions, y_train_t)
        loss.backward()
        optimizer.step()
        
        # Validation Loss calculation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = criterion(val_preds, y_val_t)
            
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        
        if epoch % 25 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs} | Train Loss (MSE): {loss.item():.4f} | Val Loss (MSE): {val_loss.item():.4f}")
            
    # Model Evaluation Metrics
    model.eval()
    with torch.no_grad():
        train_preds_np = model(X_train_t).numpy().flatten()
        val_preds_np = model(X_val_t).numpy().flatten()
        
    mae_train = mean_absolute_error(y_train, train_preds_np)
    r2_train = r2_score(y_train, train_preds_np)
    
    mae_val = mean_absolute_error(y_val, val_preds_np)
    r2_val = r2_score(y_val, val_preds_np)
    
    # Display tables
    error_tr_df = pd.DataFrame({
        'Metric': ['Mean Absolute Error (MAE)', 'R² Score'],
        'Value': [mae_train, r2_train]
    })
    print("\n=== Training Set: Neural Network Evaluation Error Table ===")
    print(error_tr_df.to_string(index=False))
    
    error_val_df = pd.DataFrame({
        'Metric': ['Mean Absolute Error (MAE)', 'R² Score'],
        'Value': [mae_val, r2_val]
    })
    print("\n=== Validation Set: Neural Network Evaluation Error Table ===")
    print(error_val_df.to_string(index=False))
    
    # Plot 1: Training Loss vs Validation Loss over Epochs
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', color='teal')
    plt.plot(range(1, epochs + 1), val_losses, label='Val Loss', color='coral')
    plt.title('Loss Convergence (Error vs Epoch Count)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Plot 2: Training Scatter Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_train, train_preds_np, color='green', alpha=0.5, edgecolors='k', label='Train (Combined)')
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', lw=2, label='Perfect Fit')
    plt.title('Side Total: Actual Vs Predicted Training')
    plt.xlabel('Actual Side Total')
    plt.ylabel('Predicted Side Total')
    plt.grid(True)
    plt.legend()
    plt.show()
    
    # Plot 3: Validation Scatter Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_val, val_preds_np, color='blue', alpha=0.5, edgecolors='k', label='Val (Combined)')
    plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2, label='Perfect Fit')
    plt.title('Side Total: Actual Vs Predicted Validation')
    plt.xlabel('Actual Side Total')
    plt.ylabel('Predicted Side Total')
    plt.grid(True)
    plt.legend()
    plt.show()
