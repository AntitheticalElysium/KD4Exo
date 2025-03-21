import numpy as np
import pandas as pd
import time
import joblib
import yaml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
# import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import torch
from torch import nn 
from visualization import display_biggest_variations, display_habitability_rankings
from core_preprocessing import load_config

def load_and_split_data(config):
    """
    Loads and splits dataset into training and testing sets.
    """
    df = pd.read_csv(config['data']['processed_path'])

    # Features and target
    X = df.drop(columns=['habitable', 'habitability_score', 'pl_name'])
    y = df['habitable']

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config['training']['test_size'], 
        random_state=config['training']['random_state']
    )
    pl_names = df.loc[X_test.index, 'pl_name']

    return X_train, X_test, y_train, y_test, pl_names

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rate):
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

def train_mlp( X_train, X_test, y_train, y_test, pl_names, config, hidden_sizes=None, dropout_rate=None, learning_rate=None, weight_decay=None, 
              epochs=None, patience=None, verbose=True):
    """
    Trains an MLP model.
    """
    if verbose:
        print("Training MLP...")
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Extract MLP config and allow overrides
    mlp_config = config['training']['teacher']['mlp']
    hidden_sizes = hidden_sizes or mlp_config['hidden_sizes']
    dropout_rate = dropout_rate if dropout_rate is not None else mlp_config['dropout_rate']
    learning_rate = learning_rate if learning_rate is not None else mlp_config['learning_rate']
    weight_decay = weight_decay if weight_decay is not None else mlp_config['weight_decay']
    epochs = epochs if epochs is not None else mlp_config['epochs']
    patience = patience if patience is not None else mlp_config['patience']

    # Convert to tensors and move to GPU if available
    X_train_tensor = torch.FloatTensor(X_train.values).to(device)
    X_test_tensor = torch.FloatTensor(X_test.values).to(device)
    y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1).to(device)
    y_test_tensor = torch.FloatTensor(y_test.values).unsqueeze(1).to(device)

    # Create model and move it to GPU
    input_size = X_train.shape[1]
    model = MLP(input_size, hidden_sizes, dropout_rate).to(device)

    # Loss, optimizer, and scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
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

            if verbose and (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}')

            # Check early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                early_stop_counter += 1
            
            if verbose and early_stop_counter >= patience:
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

    if verbose:
        print(f"MLP - MSE: {mse:.10f}, R²: {r2:.10f}, Time: {time.time() - start_time:.2f}s")
        display_biggest_variations(y_test, y_pred, pl_names)
        display_habitability_rankings(y_test, y_pred, pl_names)

    torch.save(model.state_dict(), f"{config['data']['models_path']}/mlp_model.pt")
    return model, y_pred

def train_xgboost(X_train, X_test, y_train, y_test, pl_names, config, n_estimators=None, learning_rate=None, max_depth=None, verbose=True):
    """
    Trains an XGBoost model.
    """
    if verbose:
        print("Training XGBoost...")
    start_time = time.time()

    # Extract XGBoost config and allow overrides
    xgb_config = config['training']['teacher']['xgboost']
    n_estimators = n_estimators or xgb_config['n_estimators']
    learning_rate = learning_rate if learning_rate is not None else xgb_config['learning_rate']
    max_depth = max_depth or xgb_config['max_depth']

    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    if verbose:
        print(f"XGBoost - MSE: {mse:.10f}, R²: {r2:.10f}, Time: {time.time() - start_time:.2f}s")
        display_biggest_variations(y_test, y_pred, pl_names)
        display_habitability_rankings(y_test, y_pred, pl_names)

    joblib.dump(model, f"{config['data']['models_path']}/xgboost_model.pkl")
    return model, y_pred

def train_random_forest(X_train, X_test, y_train, y_test, pl_names, config, n_estimators=None, max_depth=None, verbose=True):
    """
    Trains a Random Forest model.
    """
    if verbose:
        print("Training Random Forest...")
    start_time = time.time()

    # Extract Random Forest config and allow overrides
    rf_config = config['training']['teacher']['random_forest']
    n_estimators = n_estimators or rf_config['n_estimators']
    max_depth = max_depth or rf_config['max_depth']

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    if verbose:
        print(f"Random Forest - MSE: {mse:.10f}, R²: {r2:.10f}, Time: {time.time() - start_time:.2f}s")
        display_biggest_variations(y_test, y_pred, pl_names)
        display_habitability_rankings(y_test, y_pred, pl_names)

    joblib.dump(model, f"{config['data']['models_path']}/random_forest_model.pkl")
    return model, y_pred

if __name__ == "__main__":
    config = load_config()
    X_train, X_test, y_train, y_test, pl_names = load_and_split_data(config)

    model_mlp, y_pred_mlp = train_mlp(X_train, X_test, y_train, y_test, pl_names, config)
    # model_xgb, y_pred_xgb = train_xgboost(X_train, X_test, y_train, y_test, pl_names, config)
    model_rf, y_pred_rf = train_random_forest(X_train, X_test, y_train, y_test, pl_names, config)
