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

def plot_habitability_rankings(df, top_n=20):
    """
    Plots the top N most habitable planets.
    """
    # Sort by habitability index
    df_sorted = df.sort_values(by='habitability_score', ascending=False)
    top_planets = df_sorted.head(top_n)
    
    # Create bar plot
    plt.figure(figsize=(12, 8))
    bars = plt.barh(top_planets['pl_name'], top_planets['habitability_score'])
    plt.xlabel('Habitability Score')
    plt.ylabel('Planet Name')
    plt.title(f'Top {top_n} Most Earth-like Exoplanets')
    plt.xlim(0, 1)
    plt.gca().invert_yaxis()  # Display highest value at the top
    plt.tight_layout()
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                 f'{width:.3f}', va='center')
    
    plt.show()

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
