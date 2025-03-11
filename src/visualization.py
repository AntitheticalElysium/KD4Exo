import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

def plot_feature_distributions(df, exclude_cols=None):
    """
    Plots distribution of numerical features.
    """
    if exclude_cols is None:
        exclude_cols = ['pl_name']
    
    # Select numeric columns to plot
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    plot_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Calculate grid dimensions
    n_cols = len(plot_cols)
    n_rows = math.ceil(n_cols / 3)
    n_cols_per_row = min(3, n_cols)
    
    # Create plots
    plt.figure(figsize=(5 * n_cols_per_row, 4 * n_rows))
    
    for i, col in enumerate(plot_cols, 1):
        plt.subplot(n_rows, n_cols_per_row, i)
        sns.histplot(df[col], bins=30, kde=True)
        plt.title(col)
        plt.tight_layout()
    
    plt.show()

def display_habitability_rankings(y_test, y_pred, pl_names):
    """
    Display the habitability rankings of non-synthetic planets.
    """
    # Exclude synthetic planets (Terra-* and Thanatos-*)
    non_synthetic = ~pl_names.str.startswith("Terra") & ~pl_names.str.startswith("Thanatos")
    y_test = y_test[non_synthetic]
    y_pred = y_pred[non_synthetic]
    pl_names = pl_names[non_synthetic]

    # Display habitability rankings 
    results = pd.DataFrame({
        'pl_name': pl_names,
        'actual_habitability_score': y_test,
        'predicted_habitability_score': y_pred
    })

    results_sorted = results.sort_values(by='actual_habitability_score', ascending=False)
    print("Habitability rankings of non-synthetic planets:")
    print(results_sorted.head(20))

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
