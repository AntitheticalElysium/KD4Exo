import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from scipy.stats import boxcox
from physical_calculations import process_missing_data
from habitability_scoring import calculate_habitability_score
from visualization import plot_feature_distributions, plot_habitability_rankings

RAW_DATA_PATH = "../data/raw/exoplanet_data.csv"
CLEAN_DATA_PATH = "../data/processed/exoplanet_data_clean.csv"

def preprocess_exoplanet_data():
    """
    Load and process exoplanet data, calculating habitability scores.
    """
    df = pd.read_csv(RAW_DATA_PATH)
    
    retained_columns = [
        'pl_name', 'pl_rade', 'pl_bmasse', 'pl_orbsmax', 'pl_orbeccen',
        'st_lum', 'st_mass', 'sy_snum', 'sy_pnum', 'pl_dens', 'st_teff', 'st_rad'
    ]
    df = df[retained_columns]
    
    df = process_missing_data(df)
    df = calculate_habitability_score(df)
    
    return df

def process_data(df, skip_outliers=False, skip_skewness=False, skip_scaling=False):
    """
    Process data, optionally skipping outlier handling, skewness correction, or scaling
    """
    if not skip_outliers:
        df = handle_outliers(df)
    
    if not skip_skewness:
        df = handle_skewness(df)
    
    if not skip_scaling:
        df = scale_features(df)
    
    return df

def handle_outliers(df, exclude_cols=None):
    """
    Handle outliers using IQR method.
    """
    if exclude_cols is None:
        exclude_cols = ['pl_name', 'habitability_score']
    
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    numeric_cols = [col for col in numeric_df.columns if col not in exclude_cols]
    
    Q1 = df[numeric_cols].quantile(0.25)
    Q3 = df[numeric_cols].quantile(0.75)
    IQR = Q3 - Q1
    
    outlier_counts = ((df[numeric_cols] < (Q1 - 1.5 * IQR)) | 
                     (df[numeric_cols] > (Q3 + 1.5 * IQR))).sum()
    print("Outliers per column:")
    print(outlier_counts[outlier_counts > 0])
    
    for col in numeric_cols:
        df[col] = df[col].clip(lower=Q1[col] - 1.5 * IQR[col], 
                              upper=Q3[col] + 1.5 * IQR[col])
    
    return df

def handle_skewness(df, threshold=0.75, exclude_cols=None):
    """
    Correct skewness in features using Box-Cox transformation.
    """
    if exclude_cols is None:
        exclude_cols = ['pl_name', 'habitability_score', 'sy_snum', 'sy_pnum']
    
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    skewness = numeric_df.skew()
    
    skewed_cols = [col for col in skewness[abs(skewness) > threshold].index 
                  if col not in exclude_cols]
    
    if skewed_cols:
        print("Skewness before transformation:")
        print(skewness[skewed_cols])
        
        for col in skewed_cols:
            # Ensure no zero or negative values
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
    Scale features using StandardScaler.
    """
    if exclude_cols is None:
        exclude_cols = ['pl_name', 'habitability_score']
    
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    cols_to_scale = [col for col in numeric_cols if col not in exclude_cols]
    
    scaler = StandardScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    
    joblib.dump(scaler, "../models/scaler.pkl")
    print("Scaler saved to ../models/scaler.pkl")
    
    return df

if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    df = preprocess_exoplanet_data()
    
    print(f"Average habitability score: {df['habitability_score'].mean():.4f}")
    print(f"Maximum habitability score: {df['habitability_score'].max():.4f}")
    
    df = process_data(df)
    
    df_sorted = df.sort_values(by='habitability_score', ascending=False)
    print("\nTop 50 most Earth-like planets:")
    print(df_sorted[['pl_name', 'habitability_score', 'pl_rade', 'atm_retention_prob']].head(50))
    
    df.to_csv(CLEAN_DATA_PATH, index=False)

    # plot_feature_distributions(df)
    # plot_habitability_rankings(df)
