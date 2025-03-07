import numpy as np
import pandas as pd
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import torch
from torch import nn 
from visualization import display_biggest_variations

DATA_PATH = "../data/processed/exoplanet_data_clean.csv"
MODEL_PATH = "../models"

def load_and_split_data(test_size=0.2, random_state=42):
    """
    Loads and splits dataset into training and testing sets.
    """
    df = pd.read_csv(DATA_PATH)

    # Features and target
    X = df.drop(columns=['habitability_score', 'pl_name'])  # Drop non-numeric and target
    y = df['habitability_score']

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    # Keep track of pl_name for predictions
    pl_names = df.loc[X_test.index, 'pl_name']

    return X_train, X_test, y_train, y_test, pl_names

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), 
            nn.Linear(64, 32), nn.ReLU(), 
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.model(x)

def train_mlp(X_train, X_test, y_train, y_test):
    """
    Trains an MLP model.
    """
    print("Training MLP...")
    start_time = time.time()

    # Convert data to PyTorch tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_torch = torch.tensor(X_train.values, dtype=torch.float32).to(device)
    y_train_torch = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)
    X_test_torch = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    y_test_torch = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(device)

    model = MLP(X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(100):
        optimizer.zero_grad()
        y_pred = model(X_train_torch)
        loss = criterion(y_pred, y_train_torch)
        loss.backward()
        optimizer.step()
    
    # Evaluate model
    model.eval()
    y_pred_test = model(X_test_torch).cpu().detach().numpy().flatten()
    mse = mean_squared_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)

    print(f"MLP - MSE: {mse:.4f}, R²: {r2:.4f}")

    # Save model
    torch.save(model.state_dict(), f"{MODEL_PATH}/mlp_model.pth")
    return y_pred_test



def train_xgboost(X_train, X_test, y_train, y_test, pl_names):
    """
    Trains an XGBoost model.
    """
    print("Training XGBoost...")
    start_time = time.time()

    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, max_depth=6, learning_rate=0.05)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"XGBoost - MSE: {mse:.4f}, R²: {r2:.4f}, Time: {time.time() - start_time:.2f}s")
    # display_biggest_variations(y_test, y_pred, pl_names)

    # Save model
    model.save_model(f"{MODEL_PATH}/xgboost_model.json")
    return model

def train_random_forest(X_train, X_test, y_train, y_test, pl_names):
    """
    Trains a Random Forest model.
    """
    print("Training Random Forest...")
    start_time = time.time()

    model = RandomForestRegressor(n_estimators=200, max_depth=6)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Random Forest - MSE: {mse:.4f}, R²: {r2:.4f}, Time: {time.time() - start_time:.2f}s")
    # display_biggest_variations(y_test, y_pred, pl_names)

    # Save model
    joblib.dump(model, f"{MODEL_PATH}/random_forest_model.pkl")
    return model

def train_meta_learner(X_train, X_test, y_train, y_test):
    """
    Stacking with meta-learner.
    """
    pass

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, pl_names  = load_and_split_data()

    y_pred_mlp = train_mlp(X_train, X_test, y_train, y_test)
    y_pred_xgb = train_xgboost(X_train, X_test, y_train, y_test, pl_names)
    y_pred_rf = train_random_forest(X_train, X_test, y_train, y_test, pl_names)

    # Train meta-learner
