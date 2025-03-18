import numpy as np
import pandas as pd
import joblib
import yaml
from sklearn.preprocessing import StandardScaler
from scipy.stats import boxcox
from physical_calculations import process_missing_data
from habitability_scoring import calculate_habitability
from visualization import plot_feature_distributions
from generate_planets import generate_habitable_planets, generate_non_habitable_planets

def load_config():
    """
    Load configuration from config.yaml
    """
    with open("../config.yaml", "r") as f:
        config = yaml.safe_load(f)
    return config

def preprocess_exoplanet_data(config):
    """
    Load and process exoplanet data, calculating habitability scores.
    """
    RAW_DATA_PATH = config['data']['raw_path']
    CLEAN_DATA_PATH = config['data']['processed_path']
    
    df = pd.read_csv(RAW_DATA_PATH)
    
    retained_columns = [
        'pl_name', 'pl_rade', 'pl_bmasse', 'pl_orbsmax', 'pl_orbeccen',
        'st_lum', 'st_mass', 'sy_snum', 'sy_pnum', 'pl_dens', 'st_teff', 'st_rad'
    ]
    df = df[retained_columns]
    
    df = process_missing_data(df)
    df = calculate_habitability(df)

    # Check class distribution
    habitable_count = df['habitable'].sum()
    total_count = len(df)
    print(f"Original dataset: {habitable_count}/{total_count} habitable planets ({habitable_count/total_count:.1%})")
    
    # Generate synthetic habitable planets if needed
    habitable_threshold = config['augmentation']['habitable_threshold']
    if habitable_count < habitable_threshold:
        synthetic_planets = generate_habitable_planets(df, num_to_generate=habitable_threshold)
        synthetic_planets['habitable'] = 1  # All synthetics are habitable by design
        df = pd.concat([df, synthetic_planets], ignore_index=True)
        print(f"After augmentation: {df['habitable'].sum()}/{len(df)} habitable planets ({df['habitable'].sum()/len(df):.1%})")

    # Generate synthetic non-habitable planets if needed
    non_habitable_threshold = config['augmentation']['non_habitable_threshold']
    non_habitable_count = total_count - habitable_count
    if non_habitable_count < non_habitable_threshold:
        synthetic_planets = generate_non_habitable_planets(df, num_to_generate=non_habitable_threshold)
        synthetic_planets['habitable'] = 0  # All synthetics are non-habitable by design
        df = pd.concat([df, synthetic_planets], ignore_index=True)
        print(f"After augmentation: {df['habitable'].sum()}/{len(df)} habitable planets ({df['habitable'].sum()/len(df):.1%})")
    
    return df

def process_data(df, config):
    """
    Process data, optionally skipping outlier handling, skewness correction, or scaling
    based on configuration
    """
    if config['preprocessing']['handle_outliers']:
        df = handle_outliers(df)
    
    if config['preprocessing']['handle_skewness']:
        df = handle_skewness(df, threshold=config['preprocessing']['skewness_threshold'])
    
    if config['preprocessing']['scaling']:
        df = scale_features(df)
    
    return df

def handle_outliers(df, exclude_cols=None):
    """
    Handle outliers using IQR method.
    """
    if exclude_cols is None:
        exclude_cols = ['pl_name', 'habitable', 'habitability_score']
    
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
        exclude_cols = ['pl_name', 'habitable', 'habitability_score', 'sy_snum', 'sy_pnum', 'hz_position']
    
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
        exclude_cols = ['pl_name', 'habitable', 'habitability_score']
    
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    cols_to_scale = [col for col in numeric_cols if col not in exclude_cols]
    
    scaler = StandardScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    
    joblib.dump(scaler, "../models/scaler.pkl")
    print("Scaler saved to ../models/scaler.pkl")
    
    return df

if __name__ == "__main__":
    pd.set_option("display.max_columns", None)
    config = load_config()
    
    df = preprocess_exoplanet_data(config)
    
    # display the planets with a habitable of 1, sorted by habitability scores and excluding the synthetic planets
    habitable_planets = df[(df['habitable'] == 1) & ~df['pl_name'].str.startswith("Terra")]
    habitable_planets = habitable_planets.sort_values(by='habitability_score', ascending=False)
    print("Habitability rankings of non-synthetic planets:")
    print(habitable_planets.head(20))

    print(f"Average habitability score: {df['habitability_score'].mean():.4f}")
    print(f"Maximum habitability score: {df['habitability_score'].max():.4f}")
    
    df = process_data(df, config)
    
    df.to_csv(config['data']['processed_path'], index=False)

    # plot_feature_distributions(df)
    # plot_habitability_rankings(df)
