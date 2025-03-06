import numpy as np
import pandas as pd
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor

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

def train_neural_network(X_train, X_test, y_train, y_test, pl_names):
    """
    Trains a Neural Network model.
    """
    pass 

def display_biggest_variations(y_test, y_pred, pl_names):
    """
    Displays the planets with the biggest variations between actual and predicted scores.
    """
    # Calculate the absolute differences between actual and predicted values
    abs_diff = np.abs(y_test - y_pred)

    # Print model predictions alongside pl_name, habitability_score, and absolute differences
    results = pd.DataFrame({
        'pl_name': pl_names,
        'actual_habitability_score': y_test,
        'predicted_habitability_score': y_pred,
        'absolute_difference': abs_diff
    })

    pd.set_option('display.float_format', '{:.6f}'.format)
    # Sort results by absolute difference in descending order to see the biggest variations
    results_sorted = results.sort_values(by='absolute_difference', ascending=False)
    print("Planets with the biggest variations between actual and predicted habitability scores:")
    print(results_sorted.head(50))

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, pl_names  = load_and_split_data()

    xgb_model = train_xgboost(X_train, X_test, y_train, y_test, pl_names)
    rf_model = train_random_forest(X_train, X_test, y_train, y_test, pl_names)
