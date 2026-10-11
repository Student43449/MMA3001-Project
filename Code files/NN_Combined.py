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

# Define the Multi-Output Neural Network Architecture (predicts 2 target variables)
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
            nn.Linear(500, 2)  # Two outputs: [Side Total, Total]
       )

    def forward(self, x):
        return self.network(x)

def prompt_multiple_files(dataset_type):
    file_paths = []
    print(f"\n--- Configure {dataset_type} Datasets ---")
    while True:
        while True:
            if os.path.exists(path):
                file_paths.append(path)
                break
            print(f"Error: File '{path}' does not exist. Please try again.\n")

        if another != 'YES':
            break
    return file_paths

def load_and_combine_data(file_paths, feature_start=3, feature_end=18, targets=['Side Total', 'Total']):
    dfs = []
    for path in file_paths:
        try:
            df = pd.read_csv(path)
            for target in targets:
                if target not in df.columns:
                    print(f"Check dataset: {target} not found")
                    sys.exit(1)
            dfs.append(df)
        except Exception:
            print("Check dataset")
            sys.exit(1)

    combined_df = pd.concat(dfs, ignore_index=True)
    feature_cols = combined_df.columns[feature_start:feature_end]

    X = combined_df[feature_cols].apply(pd.to_numeric, errors='coerce')
    y = combined_df[targets].apply(pd.to_numeric, errors='coerce')

    aligned = pd.concat([X, y], axis=1).dropna()
    X = aligned[feature_cols]
    y = aligned[targets]

    if len(X) == 0:
        print("Check dataset: Empty after alignment")
        sys.exit(1)

    return X, y

if __name__ == '__main__':
    train_files = [
        '/content/MMA3001-Project/data files/divided_data_1.csv',
        '/content/MMA3001-Project/data files/divided_data_2.csv'
    ]
    val_files = [
        '/content/MMA3001-Project/data files/divided_data_3.csv',
        '/content/MMA3001-Project/data files/divided_data_4.csv'
    ]

    targets = ['Side Total', 'Total']
    X_train, y_train = load_and_combine_data(train_files, targets=targets)
    X_val, y_val = load_and_combine_data(val_files, targets=targets)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    X_train_t = torch.FloatTensor(X_train_scaled)
    y_train_t = torch.FloatTensor(y_train.values)
    X_val_t = torch.FloatTensor(X_val_scaled)
    y_val_t = torch.FloatTensor(y_val.values)

    model = GolfNeuralNet(X_train_scaled.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 150
    train_losses = []
    val_losses = []

    print("\nTraining Multi-Output Neural Network Model...")
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
        train_preds_np = model(X_train_t).numpy()
        val_preds_np = model(X_val_t).numpy()

    # Plot 1: Combined Loss Convergence over Epochs
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Train Loss', color='teal')
    plt.plot(range(1, epochs + 1), val_losses, label='Val Loss', color='coral')
    plt.title('Multi-Output Loss Convergence (Error vs Epoch Count)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.grid(True)
    plt.legend()
    plt.show()

    # Generate tables and scatter plots for each output target
    for idx, target_name in enumerate(targets):
        tr_true = y_train.iloc[:, idx].values
        tr_pred = train_preds_np[:, idx]
        v_true = y_val.iloc[:, idx].values
        v_pred = val_preds_np[:, idx]

        mae_tr = mean_absolute_error(tr_true, tr_pred)
        r2_tr = r2_score(tr_true, tr_pred)

        mae_val = mean_absolute_error(v_true, v_pred)
        r2_val = r2_score(v_true, v_pred)

        print(f"\n=== Training Set: {target_name} Evaluation Error Table ===")
        print(pd.DataFrame({'Metric': ['MAE', 'R² Score'], 'Value': [mae_tr, r2_tr]}).to_string(index=False))

        print(f"\n=== Validation Set: {target_name} Evaluation Error Table ===")
        print(pd.DataFrame({'Metric': ['MAE', 'R² Score'], 'Value': [mae_val, r2_val]}).to_string(index=False))

        # Plot 2: Scatter plot (Train)
        plt.figure(figsize=(8, 6))
        plt.scatter(tr_true, tr_pred, color='green', alpha=0.5, edgecolors='k', label='Train (Combined)')
        plt.plot([tr_true.min(), tr_true.max()], [tr_true.min(), tr_true.max()], 'r--', lw=2, label='Perfect Fit')
        plt.title(f'{target_name}: Actual Vs Predicted Training')
        plt.xlabel(f'Actual {target_name}')
        plt.ylabel(f'Predicted {target_name}')
        plt.grid(True)
        plt.legend()
        plt.show()

        # Plot 3: Scatter plot (Val)
        plt.figure(figsize=(8, 6))
        plt.scatter(v_true, v_pred, color='blue', alpha=0.5, edgecolors='k', label='Val (Combined)')
        plt.plot([v_true.min(), v_true.max()], [v_true.min(), v_true.max()], 'r--', lw=2, label='Perfect Fit')
        plt.title(f'{target_name}: Actual Vs Predicted Validation')
        plt.xlabel(f'Actual {target_name}')
        plt.ylabel(f'Predicted {target_name}')
        plt.grid(True)
        plt.legend()
        plt.show()
