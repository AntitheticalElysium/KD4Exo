import numpy as np
import pandas as pd
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb

# Paths
DATA_PATH = "../data/processed/exoplanet_data_clean.csv"
MODEL_PATH = "../models"

def load_data():
    """Load preprocessed exoplanet data."""
    df = pd.read_csv(DATA_PATH)
    
    # Features and target
    X = df.drop(columns=['habitability_score', 'pl_name'])
    y = df['habitability_score']
    
    return X, y

def prepare_data(X, y, test_size=0.2, random_state=42):
    """Split data into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

def load_teacher_model():
    """Load the pre-trained XGBoost teacher model."""
    teacher_model = xgb.XGBRegressor()
    teacher_model.load_model(f"{MODEL_PATH}/xgboost_model.json")
    return teacher_model
    
if __name__ == "__main__":
    # Load and prepare data
    X, y = load_data()
    X_train, X_test, y_train, y_test = prepare_data(X, y)
    
    # Load teacher model
    teacher_model = load_teacher_model()
