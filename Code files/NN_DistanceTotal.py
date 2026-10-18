"""Train a neural network to predict golf-shot total distance.

The script loads the prepared training and validation datasets, standardises
the named input features, trains a single-output PyTorch model, and reports
training and validation MAE and R² scores. It also plots loss convergence and
actual-versus-predicted results.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

# Explicit predictors shared with the combined neural network. Target columns are
# deliberately excluded to prevent the model from seeing the answers it predicts.
FEATURE_COLUMNS = [
    "Club Speed", "Attack Angle", "Club Path", "Low Point", "Swing Plane",
    "Swing Direction", "Dyn. Loft", "Face Angle", "Face To Path", "Ball Speed",
    "Smash Factor", "Launch Angle", "Launch Direction", "Spin Rate", "Spin Axis",
]

# Single-output network: its final layer returns one prediction per golf shot.
# Single-Output Neural Network (1 output node for Total Distance)
class GolfNeuralNet(nn.Module):
    def __init__(self, input_dim):
        """Build hidden layers and a single total-distance output."""
        super(GolfNeuralNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 1)  # One output node: Total Distance
        )

    def forward(self, x):
        """Return the predicted total distance for each input row."""
        return self.network(x)

def load_and_combine_data(file_paths, target_col='Total'):
    """Load CSV files and return the named numeric predictors and target.

    Explicit feature names prevent CSV column-order changes from silently
    changing the model inputs or accidentally including a target as a feature.
    """
    if not file_paths:
        raise ValueError("At least one dataset file must be provided.")

    frames = []
    for path in file_paths:
        try:
            frame = pd.read_csv(path)
        except Exception as exc:
            raise RuntimeError(f"Could not read dataset '{path}': {exc}") from exc

        required_columns = FEATURE_COLUMNS + [target_col]
        missing_columns = [column for column in required_columns if column not in frame.columns]
        if missing_columns:
            raise ValueError(
                f"Dataset '{path}' is missing required columns: {missing_columns}"
            )
        frames.append(frame)

    combined_df = pd.concat(frames, ignore_index=True)
    X = combined_df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(combined_df[target_col], errors="coerce")

    # Drop rows with missing or non-numeric values in any predictor or the target.
    aligned = pd.concat([X, y.rename(target_col)], axis=1).dropna()
    if aligned.empty:
        raise ValueError(
            f"No complete numeric rows remain when preparing target '{target_col}'."
        )

    return aligned[FEATURE_COLUMNS], aligned[target_col]

if __name__ == '__main__':
    # Find the data folder relative to this script, not a machine-specific location.
    repo_root = Path(__file__).resolve().parent.parent
    data_folder = repo_root / "data files"
    train_file = data_folder / "Training data.csv"
    val_file = data_folder / "Validation data.csv"

    for required_file in (train_file, val_file):
        if not required_file.is_file():
            raise FileNotFoundError(
                f"Required dataset not found: {required_file}. "
                "Run Data_processing.py first."
            )

    target = 'Total'
    X_train, y_train = load_and_combine_data([train_file], target_col=target)
    X_val, y_val = load_and_combine_data([val_file], target_col=target)

    # Standardise each feature using training-set means and standard deviations only.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    X_train_t = torch.FloatTensor(X_train_scaled)
    y_train_t = torch.FloatTensor(y_train.values).unsqueeze(1)
    X_val_t = torch.FloatTensor(X_val_scaled)
    y_val_t = torch.FloatTensor(y_val.values).unsqueeze(1)

    model = GolfNeuralNet(X_train_scaled.shape[1])
    # MSE is the optimisation objective; Adam updates weights to reduce this loss.
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    epochs = 150
    train_losses = []
    val_losses = []

    print(f"\nTraining Neural Network Model for {target}...")
    # Repeat the training and validation evaluation for the configured number of epochs.
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        predictions = model(X_train_t)
        loss = criterion(predictions, y_train_t)
        # Calculate gradients from the training loss, then apply one optimiser step.
        loss.backward()
        optimizer.step()

        model.eval()
        # Evaluate validation loss without tracking gradients or changing model weights.
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

    # Compare training and validation loss over time to help spot poor generalisation.
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

    # Compare predictions with measured values; the dashed diagonal is the perfect-fit line.
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
