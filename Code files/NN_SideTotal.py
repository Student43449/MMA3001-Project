import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

class SideNet(nn.Module):
    def __init__(self, input_dim):
        super(SideNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 1)
        )
    def forward(self, x):
        return self.network(x)

def train_and_evaluate(train_path, val_path):
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    features = [
        'Attack Angle', 'Club Path', 'Low Point', 'Swing Plane', 'Swing Direction',
        'Dyn. Loft', 'Face Angle', 'Face To Path', 'Ball Speed', 'Smash Factor',
        'Launch Angle', 'Launch Direction', 'Spin Rate', 'Spin Axis'
    ]
    X_train = train_df[features]
    y_train = train_df[['Side Total']]
    X_val = val_df[features]
    y_val = val_df[['Side Total']]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    model = SideNet(X_train_scaled.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    X_tr_t = torch.FloatTensor(X_train_scaled)
    y_tr_t = torch.FloatTensor(y_train.values)
    X_va_t = torch.FloatTensor(X_val_scaled)
    print("Training Side Total Neural Network Model...")
    for epoch in range(1, 101):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_tr_t), y_tr_t)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        preds = model(X_va_t).numpy()
    print(f"Side Total Val MAE: {mean_absolute_error(y_val, preds):.4f}")

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, 'data files', 'Training_data.csv')
    val_path = os.path.join(base_dir, 'data files', 'Validation_data.csv')
    train_and_evaluate(train_path, val_path)
