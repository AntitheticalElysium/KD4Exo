import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from scipy.stats import boxcox

RAW_DATA_PATH = "../data/raw/exoplanet_data.csv"
CLEAN_DATA_PATH = "../data/processed/exoplanet_data_clean.csv"

def preprocess_exoplanet_data():
    """
    Main function to load and process exoplanet data, calculating habitability scores.
    """
    # Load the dataset
    df = pd.read_csv(RAW_DATA_PATH)
    
    # Columns to retain based on habitability relevance
    retained_columns = [
        'pl_name', 'pl_rade', 'pl_bmasse', 'pl_orbsmax', 'pl_orbeccen',
        'st_lum', 'st_mass', 'sy_snum', 'sy_pnum', 'pl_dens', 'st_teff', 'st_rad'
    ]
    df = df[retained_columns]
    
    # Process missing data using physical relationships
    from physical_calculations import process_missing_data
    df = process_missing_data(df)
    
    # Compute habitability index
    from habitability_scoring import calculate_habitability_score
    df = calculate_habitability_score(df)
    
    return df

def handle_outliers(df, exclude_cols=None):
    """
    Removes outliers using IQR method.
    Excludes specified columns and non-numeric data.
    """
    if exclude_cols is None:
        exclude_cols = ['pl_name', 'habitability_score']
    
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    numeric_cols = [col for col in numeric_df.columns if col not in exclude_cols]
    
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    
    # Report outliers found
    outlier_counts = ((df[numeric_cols] < (Q1 - 1.5 * IQR)) | 
                      (df[numeric_cols] > (Q3 + 1.5 * IQR))).sum()
    print("Outliers per column:")
    print(outlier_counts[outlier_counts > 0])
    
    # Create bounds for clipping
    lower_bounds = Q1 - 1.5 * IQR
    upper_bounds = Q3 + 1.5 * IQR
    
    # Clip outliers without affecting excluded columns
    for col in numeric_cols:
        df[col] = df[col].clip(lower=lower_bounds[col], upper=upper_bounds[col])
    
    return df

def handle_skewness(df, threshold=0.75, exclude_cols=None):
    """
    Applies Box-Cox transformation to reduce skewness in numerical features.
    """
    if exclude_cols is None:
        exclude_cols = ['pl_name', 'habitability_score', 'sy_snum', 'sy_pnum']
    
    # Calculate skewness for numeric columns
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    skewness = numeric_df.skew()
    
    # Find columns with skewness above threshold
    skewed_cols = [col for col in skewness[abs(skewness) > threshold].index 
                  if col not in exclude_cols]
    
    if skewed_cols:
        print("Skewness before transformation:")
        print(skewness[skewed_cols])
        
        # Apply Box-Cox transformation
        for col in skewed_cols:
            # Ensure all values are positive for Box-Cox
            if df[col].min() <= 0:
                df[col] = df[col] - df[col].min() + 0.01
            df[col], _ = boxcox(df[col])
        
        print("Skewness after transformation:")
        print(df[skewed_cols].skew())
    else:
        print("No features with significant skewness found.")
    
    return df

def scale_features(df, exclude_cols=None):
    """
    Standardizes features using StandardScaler.
    """
    if exclude_cols is None:
        exclude_cols = ['pl_name', 'habitability_score']
    
    # Select numeric columns to scale
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    cols_to_scale = [col for col in numeric_cols if col not in exclude_cols]
    
    # Apply scaling
    scaler = StandardScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    # Save the scaler for later inference
    joblib.dump(scaler, "../models/scaler.pkl")
    print("Scaler saved to ../models/scaler.pkl")
    
    return df

if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    # Load and preprocess data
    df = preprocess_exoplanet_data()
    
    # Display habitability statistics
    print(f"Average habitability score: {df['habitability_score'].mean():.4f}")
    print(f"Maximum habitability score: {df['habitability_score'].max():.4f}")
    
    # Optional processing steps
    df = handle_outliers(df)
    df = handle_skewness(df)
    df = scale_features(df)
    
    # Display top habitable planets
    df_sorted = df.sort_values(by='habitability_score', ascending=False)
    print("\nTop 50 most Earth-like planets:")
    print(df_sorted[['pl_name', 'habitability_score', 'pl_rade', 'atm_retention_prob']].head(50))
    
    # Save processed data
    df.to_csv(CLEAN_DATA_PATH, index=False)
    
    # Visualizations
    # from visualization import plot_feature_distributions, plot_habitability_rankings
    # plot_feature_distributions(df)
    # plot_habitability_rankings(df)
