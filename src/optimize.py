from train_teacher_models import train_mlp, train_xgboost, train_random_forest, load_and_split_data
from core_preprocessing import load_config
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import itertools
import random
import yaml
import torch


def optimize_mlp(X_train, X_test, y_train, y_test, pl_names, config, strategy='grid', n_trials=20):
    """
    Optimize hyperparameters for the MLP model.
    """
    print("Optimizing MLP hyperparameters...")
    
    # Get current hyperparameters
    mlp_config = config['training']['teacher']['mlp']
    
    # Define hyperparameter search space
    param_grid = {
        'hidden_sizes': [
            [128, 256, 128],
            [256, 512, 256],
            [256, 512, 512, 256],
            [256, 512, 512, 256, 128],
            [512, 1024, 512]
        ],
        'dropout_rate': [0.2, 0.3, 0.4, 0.5],
        'learning_rate': [0.001, 0.005, 0.01, 0.02],
        'weight_decay': [0.0001, 0.0005, 0.001],
        'batch_size': [32, 64, 128],
        'patience': [100, 150, 200]
    }
    fixed_params = {
        'epochs': mlp_config['epochs']
    }
    
    # Results tracking
    best_score = -float('inf')
    best_params = None
    results = []
    
    # Determine search strategy
    if strategy == 'grid':
        # Expensive
        keys = list(param_grid.keys())
        param_values = [param_grid[key] for key in keys]
        param_combinations = list(itertools.product(*param_values))
        print(f"Running grid search with {len(param_combinations)} combinations")
        
        for i, values in enumerate(param_combinations):
            current_params = {keys[j]: values[j] for j in range(len(keys))}
            current_params.update(fixed_params)
            
            print(f"Combination {i+1}/{len(param_combinations)}: {current_params}")
            
            # Train model with current parameters
            score, model = evaluate_mlp_params(X_train, X_test, y_train, y_test, pl_names, config, current_params)
            
            # Store result
            results.append((score, current_params))
            
            if score > best_score:
                best_score = score
                best_params = current_params
                print(f"New best score: {best_score} with params: {best_params}")
    
    elif strategy == 'random':
        # More efficient for high-dimensional spaces
        print(f"Running random search with {n_trials} trials")
        
        for i in range(n_trials):
            # Randomly sample parameters
            current_params = {
                'hidden_sizes': random.choice(param_grid['hidden_sizes']),
                'dropout_rate': random.choice(param_grid['dropout_rate']),
                'learning_rate': random.choice(param_grid['learning_rate']),
                'weight_decay': random.choice(param_grid['weight_decay']),
                'batch_size': random.choice(param_grid['batch_size']),
                'patience': random.choice(param_grid['patience'])
            }
            current_params.update(fixed_params)
            
            print(f"Trial {i+1}/{n_trials}: {current_params}")
            
            # Train model with current parameters
            score, model = evaluate_mlp_params(X_train, X_test, y_train, y_test, pl_names, config, current_params)
            
            # Store result
            results.append((score, current_params))
            
            if score > best_score:
                best_score = score
                best_params = current_params
                print(f"New best score: {best_score} with params: {best_params}")
    
    # Sort results by score
    results.sort(reverse=True, key=lambda x: x[0])
    
    print("\nTop 5 parameter combinations:")
    for i, (score, params) in enumerate(results[:5]):
        print(f"{i+1}. Score: {score:.4f}, Params: {params}")
    
    # Update configuration with best params
    update_config_with_best_params(config, best_params)
    
    return best_params, best_score


def evaluate_mlp_params(X_train, X_test, y_train, y_test, pl_names, config, params):
    """
    Evaluate MLP with given hyperparameters.
        model: Trained model
    """
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    # Train model with current parameters
    model, y_pred = train_mlp(
        X_train, X_test, y_train, y_test, pl_names, config,
        hidden_sizes=params['hidden_sizes'],
        dropout_rate=params['dropout_rate'],
        learning_rate=params['learning_rate'],
        weight_decay=params['weight_decay'],
        epochs=params['epochs'],
        patience=params['patience'],
        verbose=False  # Disable verbose output during optimization
    )
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    # Here we use F1 score which is good for imbalanced classes
    score = f1
    
    print(f"Parameters: {params}")
    print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
    
    return score, model


def update_config_with_best_params(config, best_params):
    """
    Update the configuration with the best parameters and save to file.
    """
    for key, value in best_params.items():
        config['training']['teacher']['mlp'][key] = value
    
    with open('../config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("Updated configuration saved to config.yaml")


def optimize_random_forest(X_train, X_test, y_train, y_test, pl_names, config, n_trials=20):
    """
    Optimize hyperparameters for the Random Forest model.
    """
    print("Optimizing Random Forest hyperparameters...")
    
    # Define hyperparameter search space
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [None, 10, 20, 30, 50, 100]
    }
    
    # Results tracking
    best_score = -float('inf')
    best_params = None
    results = []
    
    # Random search
    for i in range(n_trials):
        # Randomly sample parameters
        current_params = {
            'n_estimators': random.choice(param_grid['n_estimators']),
            'max_depth': random.choice(param_grid['max_depth'])
        }
        
        # Train model with current parameters
        model, y_pred = train_random_forest(
            X_train, X_test, y_train, y_test, pl_names, config,
            n_estimators=current_params['n_estimators'],
            max_depth=current_params['max_depth'],
            verbose=False
        )
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        score = f1
        results.append((score, current_params))
        
        if score > best_score:
            best_score = score
            best_params = current_params
            print(f"New best score: {best_score} with params: {best_params}")
    
    for key, value in best_params.items():
        config['training']['teacher']['random_forest'][key] = value
    
    with open('config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return best_params, best_score


def optimize_xgboost(X_train, X_test, y_train, y_test, pl_names, config, n_trials=20):
    """
    Optimize hyperparameters for the XGBoost model.
    """
    print("Optimizing XGBoost hyperparameters...")
    
    # Define hyperparameter search space
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1, 0.2]
    }
    
    # Results tracking
    best_score = -float('inf')
    best_params = None
    results = []
    
    # Random search
    for i in range(n_trials):
        # Randomly sample parameters
        current_params = {
            'n_estimators': random.choice(param_grid['n_estimators']),
            'max_depth': random.choice(param_grid['max_depth']),
            'learning_rate': random.choice(param_grid['learning_rate'])
        }
        
        # Train model with current parameters
        model, y_pred = train_xgboost(
            X_train, X_test, y_train, y_test, pl_names, config,
            n_estimators=current_params['n_estimators'],
            max_depth=current_params['max_depth'],
            learning_rate=current_params['learning_rate'],
            verbose=False
        )
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        score = f1
        results.append((score, current_params))
        
        if score > best_score:
            best_score = score
            best_params = current_params
            print(f"New best score: {best_score} with params: {best_params}")
    
    for key, value in best_params.items():
        config['training']['teacher']['xgboost'][key] = value
    
    with open('config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return best_params, best_score


if __name__ == "__main__":
    config = load_config()
    X_train, X_test, y_train, y_test, pl_names = load_and_split_data(config)
    
    # Choose which models to optimize
    optimize_mlp_model = True
    optimize_rf_model = False
    optimize_xgb_model = False  # to fix
    
    if optimize_mlp_model:
        # Use random search for faster optimization
        best_mlp_params, best_mlp_score = optimize_mlp(
            X_train, X_test, y_train, y_test, pl_names, config, 
            strategy='random', n_trials=20
        )
        print(f"Best MLP parameters: {best_mlp_params}")
        print(f"Best MLP score: {best_mlp_score}")
    
    if optimize_rf_model:
        best_rf_params, best_rf_score = optimize_random_forest(
            X_train, X_test, y_train, y_test, pl_names, config, 
            n_trials=15
        )
        print(f"Best Random Forest parameters: {best_rf_params}")
        print(f"Best Random Forest score: {best_rf_score}")
    
    if optimize_xgb_model:
        best_xgb_params, best_xgb_score = optimize_xgboost(
            X_train, X_test, y_train, y_test, pl_names, config, 
            n_trials=15
        )
        print(f"Best XGBoost parameters: {best_xgb_params}")
        print(f"Best XGBoost score: {best_xgb_score}")
