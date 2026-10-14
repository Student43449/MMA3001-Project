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
    # Load datasets directly
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    # Drop 'Club' if it is non-numeric string, handle features/targets
    features = [c for c in train_df.columns if c not in ['Club', 'Side Total', 'Total']]
    targets = ['Side Total', 'Total']
    
    X_train = train_df[features]
    y_train = train_df[targets]
    X_val = val_df[features]
    y_val = val_df[targets]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    X_train_t = torch.FloatTensor(X_train_scaled)
    y_train_t = torch.FloatTensor(y_train.values)
    X_val_t = torch.FloatTensor(X_val_scaled)
    y_val_t = torch.FloatTensor(y_val.values)
    
    model = GolfNeuralNet(X_train_scaled.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Training Multi-Output Neural Network Model...")
    epochs = 150
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        
        if epoch % 25 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t)
            print(f"Epoch {epoch:03d}/{epochs} | Train Loss (MSE): {loss.item():.4f} | Val Loss (MSE): {val_loss.item():.4f}")
            
    model.eval()
    with torch.no_grad():
        train_preds = model(X_train_t).numpy()
        val_preds = model(X_val_t).numpy()
        
    for i, target in enumerate(targets):
        mae_tr = mean_absolute_error(y_train[target], train_preds[:, i])
        r2_tr = r2_score(y_train[target], train_preds[:, i])
        mae_val = mean_absolute_error(y_val[target], val_preds[:, i])
        r2_val = r2_score(y_val[target], val_preds[:, i])
        
        print(f"
=== {target} Training Evaluation ===")
        print(pd.DataFrame({'Metric': ['MAE', 'R² Score'], 'Value': [mae_tr, r2_tr]}).to_string(index=False))
        print(f"
=== {target} Validation Evaluation ===")
        print(pd.DataFrame({'Metric': ['MAE', 'R² Score'], 'Value': [mae_val, r2_val]}).to_string(index=False))

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, 'data files', 'Training_data.csv')
    val_path = os.path.join(base_dir, 'data files', 'Validation_data.csv')
    train_and_evaluate(train_path, val_path)
