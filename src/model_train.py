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
    def __init__(self, input_size, hidden_sizes=[256, 512, 256, 128], dropout_rate=0.3):
        super(MLP, self).__init__()
        
        # Input layer
        layers = [nn.Linear(input_size, hidden_sizes[0]),
                  nn.BatchNorm1d(hidden_sizes[0]),
                  nn.ReLU(),
                  nn.Dropout(dropout_rate)]
        
        # Hidden layers with residual connections where possible
        for i in range(len(hidden_sizes) - 1):
            # Main path
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]))
            layers.append(nn.BatchNorm1d(hidden_sizes[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            
        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x).squeeze()
 
def train_mlp(X_train, X_test, y_train, y_test, pl_names, epochs=3000, patience=150):
    """
    Trains an MLP model with advanced features.
    """
    print("Training MLP...")
    start_time = time.time()
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train.values)
    X_test_tensor = torch.FloatTensor(X_test.values)
    y_train_tensor = torch.FloatTensor(y_train.values)
    y_test_tensor = torch.FloatTensor(y_test.values)
    
    # Create model
    input_size = X_train.shape[1]
    model = MLP(input_size, hidden_sizes=[256, 512, 512, 256, 128])
    
    # Use mean squared error loss and AdamW optimizer with cosine annealing
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # Early stopping
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_state = None
    
    # Training loop
    for epoch in range(epochs):
        # Training mode
        model.train()
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        # Backward pass and optimize
        loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test_tensor).numpy()
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
    
    print(f"Improved MLP - MSE: {mse:.10f}, R²: {r2:.10f}, Time: {time.time() - start_time:.2f}s")
    display_biggest_variations(y_test, y_pred, pl_names)

    # Save model
    torch.save(model.state_dict(), f"{MODEL_PATH}/mlp_model.pt")
    
    return model, y_pred

def train_xgboost(X_train, X_test, y_train, y_test, pl_names):
    """
    Trains an XGBoost model.
    """
    print("Training XGBoost...")
    start_time = time.time()

    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, max_depth=0, learning_rate=0.05)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"XGBoost - MSE: {mse:.10f}, R²: {r2:.10f}, Time: {time.time() - start_time:.10f}s")
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

    model = RandomForestRegressor(n_estimators=200, max_depth=5000)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Random Forest - MSE: {mse:.10f}, R²: {r2:.10f}, Time: {time.time() - start_time:.10f}s")
    display_biggest_variations(y_test, y_pred, pl_names)

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

    y_pred_mlp = train_mlp(X_train, X_test, y_train, y_test, pl_names)
    y_pred_xgb = train_xgboost(X_train, X_test, y_train, y_test, pl_names)
    y_pred_rf = train_random_forest(X_train, X_test, y_train, y_test, pl_names)

    # Train meta-learner
