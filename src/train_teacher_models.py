import numpy as np
import pandas as pd
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import torch
from torch import nn 
from visualization import display_biggest_variations, display_habitability_rankings

DATA_PATH = "../data/processed/exoplanet_data_clean.csv"
MODEL_PATH = "../models"

def load_and_split_data(test_size=0.2, random_state=42):
    """
    Loads and splits dataset into training and testing sets.
    """
    df = pd.read_csv(DATA_PATH)

    # Features and target
    X = df.drop(columns=['habitable', 'habitability_score', 'pl_name'])
    y = df['habitable']

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    pl_names = df.loc[X_test.index, 'pl_name']

    return X_train, X_test, y_train, y_test, pl_names

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes=[256, 512, 256, 128], dropout_rate=0.3):
        super(MLP, self).__init__()
        
    # Input layer
        layers = [nn.Linear(input_size, hidden_sizes[0]),
                  nn.BatchNorm1d(hidden_sizes[0]),
                  nn.ReLU(),
                  nn.Dropout(dropout_rate)]
        
        # Hidden layers with residual connections where possible
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.BatchNorm1d(hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
 
def train_mlp(X_train, X_test, y_train, y_test, pl_names, epochs=3000, patience=150):
    """
    Trains an MLP model. 
    """
    print("Training MLP...")
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Convert to tensors and move to GPU if available
    X_train_tensor = torch.FloatTensor(X_train.values).to(device)
    X_test_tensor = torch.FloatTensor(X_test.values).to(device)
    y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1).to(device)
    y_test_tensor = torch.FloatTensor(y_test.values).unsqueeze(1).to(device)

    # Create model and move it to GPU
    input_size = X_train.shape[1]
    model = MLP(input_size, hidden_sizes=[256, 512, 512, 256, 128]).to(device)

    # Loss, optimizer, and scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Early stopping variables
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_state = None

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        # Backward pass and optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Prevent gradient explosion
        optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_tensor)
            val_loss = criterion(val_outputs, y_test_tensor)

            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}')

            # Check early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                early_stop_counter += 1

            if early_stop_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        y_pred = torch.sigmoid(model(X_test_tensor)).detach().cpu().numpy().flatten()
        y_pred = (y_pred >= 0.5).astype(int)  # Convert to binary
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

    print(f"MLP - MSE: {mse:.10f}, R²: {r2:.10f}, Time: {time.time() - start_time:.2f}s")
    display_biggest_variations(y_test, y_pred, pl_names)
    display_habitability_rankings(y_test, y_pred, pl_names)

    torch.save(model.state_dict(), f"{MODEL_PATH}/mlp_model.pt")
    return model, y_pred

def train_xgboost(X_train, X_test, y_train, y_test, pl_names):
    """
    Trains an XGBoost model.
    """
    print("Training XGBoost...")
    start_time = time.time()

    model = xgb.XGBClassifier(objective='reg:squarederror', n_estimators=200, max_depth=0, learning_rate=0.05)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"XGBoost - MSE: {mse:.10f}, R²: {r2:.10f}, Time: {time.time() - start_time:.10f}s")
    display_biggest_variations(y_test, y_pred, pl_names)
    display_habitability_rankings(y_test, y_pred, pl_names)

    model.save_model(f"{MODEL_PATH}/xgboost_model.json")
    return model

def train_random_forest(X_train, X_test, y_train, y_test, pl_names):
    """
    Trains a Random Forest model.
    """
    print("Training Random Forest...")
    start_time = time.time()

    model = RandomForestClassifier(n_estimators=200, max_depth=5000)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Random Forest - MSE: {mse:.10f}, R²: {r2:.10f}, Time: {time.time() - start_time:.10f}s")
    display_biggest_variations(y_test, y_pred, pl_names)
    display_habitability_rankings(y_test, y_pred, pl_names)

    joblib.dump(model, f"{MODEL_PATH}/random_forest_model.pkl")
    return model

def train_meta_learner(X_train, X_test, y_train, y_test):
    """
    Stacking with meta-learner.
    """
    pass

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, pl_names  = load_and_split_data()

    y_pred_mlp = train_mlp(X_train, X_test, y_train, y_test, pl_names)
    # y_pred_xgb = train_xgboost(X_train, X_test, y_train, y_test, pl_names)
    # y_pred_rf = train_random_forest(X_train, X_test, y_train, y_test, pl_names)

    # Train meta-learner
