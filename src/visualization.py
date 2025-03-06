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

