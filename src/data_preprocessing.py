import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox
import math
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preprocess_exoplanet_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    # Columns to retain based on habitability relevance
    retained_columns = [
        'pl_name', 'pl_rade', 'pl_bmasse', 'pl_orbsmax', 'pl_orbeccen',
        'st_lum', 'st_mass', 'sy_snum', 'sy_pnum', 'pl_dens', 'st_teff', 'st_rad'
    ]
    df = df[retained_columns]
    
    # Process missing data using physical relationships instead of simple imputation
    df = process_missing_data(df)
    # Compute habitability index
    df = calculate_habitability_score(df)

    return df

def process_missing_data(df, drop_threshold=0):
    """
    Process missing values in exoplanet data using physical relationships between parameters.
    """
    result_df = df.copy()
    
    key_features = ['pl_rade', 'pl_bmasse', 'pl_orbsmax', 'pl_orbeccen', 
                    'st_lum', 'st_mass', 'pl_dens', 'st_teff', 'st_rad']

    missing_counts = result_df[key_features].isnull().sum(axis=1)
    too_many_missing = missing_counts > (len(key_features) * drop_threshold)
    result_df = result_df[~too_many_missing].copy()
    
    # Ensure luminosity is positive
    result_df.loc[result_df['st_lum'] < 0, 'st_lum'] = np.nan

    # Mass-Radius relationship for rocky planets
    mask_radius = result_df['pl_rade'].notna() & result_df['pl_bmasse'].isna()
    result_df.loc[mask_radius, 'pl_bmasse'] = result_df.loc[mask_radius, 'pl_rade']**3
    
    mask_mass = result_df['pl_bmasse'].notna() & result_df['pl_rade'].isna()
    result_df.loc[mask_mass, 'pl_rade'] = result_df.loc[mask_mass, 'pl_bmasse']**(1/3)

    # Density calculations
    mask_dens = result_df['pl_dens'].isna() & result_df['pl_rade'].notna() & result_df['pl_bmasse'].notna()
    earth_density = 5.51  # g/cm³
    planet_volume = (result_df.loc[mask_dens, 'pl_rade']**3)
    result_df.loc[mask_dens, 'pl_dens'] = (result_df.loc[mask_dens, 'pl_bmasse'] / planet_volume) * earth_density

    # Mass from density and radius
    mask_mass_from_dens = result_df['pl_bmasse'].isna() & result_df['pl_dens'].notna() & result_df['pl_rade'].notna()
    volume_ratio = result_df.loc[mask_mass_from_dens, 'pl_rade']**3
    result_df.loc[mask_mass_from_dens, 'pl_bmasse'] = result_df.loc[mask_mass_from_dens, 'pl_dens'] * volume_ratio / earth_density

    # Stellar luminosity from mass
    valid_mass = result_df['st_mass'] > 0
    mask_lum = result_df['st_lum'].isna() & valid_mass
    result_df.loc[mask_lum, 'st_lum'] = result_df.loc[mask_lum, 'st_mass']**3.5

    # Stellar mass from luminosity
    valid_lum = result_df['st_lum'] > 0
    mask_mass_star = result_df['st_mass'].isna() & valid_lum
    result_df.loc[mask_mass_star, 'st_mass'] = result_df.loc[mask_mass_star, 'st_lum']**(1/3.5)

    # Eccentricity: Assume slightly elliptical orbits if missing
    result_df['pl_orbeccen'] = result_df['pl_orbeccen'].fillna(0.05)

    # Surface Temperature Estimation (Equilibrium Temperature)
    albedo = 0.3  # Earth-like assumption
    mask_temp = result_df['st_teff'].notna() & result_df['st_rad'].notna() & result_df['pl_orbsmax'].notna()
    result_df.loc[mask_temp, 'pl_temp'] = result_df.loc[mask_temp, 'st_teff'] * \
        np.sqrt(result_df.loc[mask_temp, 'st_rad'] / (2 * result_df.loc[mask_temp, 'pl_orbsmax'])) * (1 - albedo) ** 0.25
    
    # Atmospheric Retention Probability
    #G = 6.67430e-11  # m^3/kg/s^2
    #earth_mass_kg = 5.97e24
    #earth_radius_m = 6371e3
    #
    #mask_escape = result_df['pl_rade'].notna() & result_df['pl_bmasse'].notna()
    #result_df.loc[mask_escape, 'pl_escape_vel'] = np.sqrt(2 * G * result_df.loc[mask_escape, 'pl_bmasse'] * earth_mass_kg / \
    #                                                       (result_df.loc[mask_escape, 'pl_rade'] * earth_radius_m))

    # Fill remaining missing values with group median
    if result_df[key_features].isnull().any().any():
        for feature in key_features:
            if result_df[feature].isnull().any():
                grouped_median = result_df.groupby(['sy_snum', 'sy_pnum'])[feature].transform('median')
                result_df[feature] = result_df[feature].fillna(grouped_median)
                result_df[feature] = result_df[feature].fillna(result_df[feature].median())

    return result_df

def calculate_habitability_score(df):
    """
    Calculates a habitability score from 0-1 for exoplanets. WIP
    """
    # Store planet names
    planet_names = df['pl_name'].copy() if 'pl_name' in df.columns else None
    
    result_df = df.copy()
    
    # 1. HABITABLE ZONE ASSESSMENT
    # Calculate habitable zone boundaries based on stellar luminosity
    # Using Kopparapu et al. (2014) conservative habitable zone estimates
   
    # Inner edge (runaway greenhouse)
    result_df['hz_inner'] = 0.97 * np.sqrt(result_df['st_lum'])
    
    # Outer edge (maximum greenhouse)
    result_df['hz_outer'] = 1.67 * np.sqrt(result_df['st_lum'])
    
    # Normalized position in habitable zone (0-1 for planets in HZ)
    result_df['hz_position'] = (result_df['pl_orbsmax'] - result_df['hz_inner']) / (result_df['hz_outer'] - result_df['hz_inner'])

    # Calculate habitable zone score - highest at center of zone, declining toward edges
    # Using a modified Gaussian function centered on the middle of the habitable zone
    result_df['hz_score'] = np.exp(-4 * ((result_df['hz_position'] - 0.5) ** 2))
    
    # Handle planets outside habitable zone
    outside_hz = (result_df['hz_position'] < 0) | (result_df['hz_position'] > 1)
    result_df.loc[outside_hz, 'hz_score'] = result_df.loc[outside_hz, 'hz_score'] * 0.1
    
    # 2. PLANETARY MASS AND RADIUS
    # Earth-like size planets are more likely to be habitable
    
    # Radius score - peaks at Earth radius (1.0), declines as radius diverges
    result_df['radius_score'] = np.exp(-((result_df['pl_rade'] - 1.0) ** 2) / 0.5)
    
    # Mass score - peaks near Earth mass (1.0), with wider acceptance range for super-Earths
    result_df['mass_score'] = np.exp(-(np.log10(result_df['pl_bmasse']) ** 2) / 0.8)
    
    # 3. ORBITAL STABILITY
    # Low eccentricity orbits provide more stable climates
    result_df['stability_score'] = np.exp(-5 * result_df['pl_orbeccen'])
    
    # 4. DENSITY - COMPOSITION INDICATOR
    # Earth-like density suggests rocky composition with potential for atmosphere
    earth_density = 5.51  # g/cm³
    result_df['density_score'] = np.exp(-((result_df['pl_dens'] - earth_density) ** 2) / 10)
    
    # 5. STELLAR TYPE SUITABILITY
    # K and G type stars provide longer habitable timescales than M dwarfs or massive stars
    result_df['stellar_lifetime_score'] = np.exp(-((result_df['st_mass'] - 0.8) ** 2) / 0.3)
    
    # 6. TIDAL LOCKING LIKELIHOOD
    # Planets too close to their stars may be tidally locked (synchronous rotation)
    # Calculate a simplified tidal locking parameter (lower is better)
    result_df['tidal_parameter'] = result_df['st_mass'] / (result_df['pl_orbsmax'] ** 3)
    max_tidal = result_df['tidal_parameter'].max()
    result_df['tidal_score'] = np.exp(-result_df['tidal_parameter'] / (max_tidal / 5))
    
    # 7. SYSTEM ARCHITECTURE ASSESSMENT
    # Number of stars in system (binary/multiple star systems can destabilize planetary orbits)
    # Single star systems (sy_snum=1) are preferred for habitability
    result_df['star_system_score'] = np.where(
        result_df['sy_snum'] == 1, 
        1.0,  # Single star gets full score
        np.exp(-(result_df['sy_snum'] - 1))  # Exponential penalty for multiple stars
    )
    
    # Number of planets (moderate number is best - too few may indicate instability,
    # too many may lead to more frequent perturbations)
    # Systems with 2-5 planets appear most stable in simulations
    result_df['planet_system_score'] = np.where(
        (result_df['sy_pnum'] >= 2) & (result_df['sy_pnum'] <= 5),
        1.0,  # Optimal range gets full score
        np.where(
            result_df['sy_pnum'] == 1,
            0.8,  # Single planet systems get slightly lower score
            0.7   # Systems with many planets (>5) get lower score
        )
    )

    # 8. ATMOSPHERIC RETENTION PROBABILITY

    # 9. ESTIMATED EQUILIBRIUM TEMPERATURE
    earth_temp = 288  # K (Earth's average surface temperature)
    result_df['temp_score'] = np.exp(-((result_df['pl_temp'] - earth_temp) ** 2) / 5000)

    # 10. FINAL HABITABILITY SCORE
    # Weight the factors according to their importance for habitability
    result_df['habitability_score'] = (
        0.25 * result_df['hz_score'] +           # Most important: being in habitable zone
        0.15 * result_df['temp_score'] +         # Having Earth-like surface temp
        0.10 * result_df['radius_score'] +       # Having Earth-like radius
        0.10 * result_df['mass_score'] +         # Having Earth-like mass  
        0.10 * result_df['stability_score'] +    # Stable orbit
        0.10 * result_df['density_score'] +      # Earth-like composition
        0.08 * result_df['stellar_lifetime_score'] + # Long-lived, stable star
        0.05 * result_df['tidal_score'] +        # Avoid tidal locking
        0.04 * result_df['star_system_score'] +  # Single-star systems preferred
        0.03 * result_df['planet_system_score']  # Moderate number of planets preferred
    )
    
    # Normalize scores to 0-1 range (though it should already be close to this)
    result_df['habitability_score'] = np.clip(result_df['habitability_score'], 0, 1)
    
    # Ensure Earth would get a high score if it were in the dataset
    earth_like = (
        (result_df['pl_rade'].between(0.9, 1.1)) & 
        (result_df['pl_bmasse'].between(0.9, 1.1)) & 
        (result_df['pl_orbsmax'].between(0.95, 1.05)) & 
        (result_df['pl_orbeccen'] < 0.02) &
        (result_df['st_mass'].between(0.95, 1.05)) &
        (result_df['sy_snum'] == 1)
    )
    result_df.loc[earth_like, 'habitability_score'] = np.maximum(
        result_df.loc[earth_like, 'habitability_score'], 
        0.95
    )
    
    # Drop intermediate calculation columns
    columns_to_drop = [
        'hz_inner', 'hz_outer', 'hz_position', 'hz_score',
        'radius_score', 'mass_score', 'stability_score', 
        'density_score', 'stellar_lifetime_score', 
        'tidal_parameter', 'tidal_score',
        'star_system_score', 'planet_system_score', 'temp_score'
    ]
    result_df = result_df.drop(columns=[col for col in columns_to_drop if col in result_df.columns])
    
    # Restore planet names if they were present
    if planet_names is not None:
        result_df['pl_name'] = planet_names
    
    return result_df

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
    
    return df

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

if __name__ == "__main__":
    # Load and preprocess data
    df = preprocess_exoplanet_data("../data/raw/exoplanet_data.csv")
    
    # Display habitability statistics
    print(f"Average habitability score: {df['habitability_score'].mean():.4f}")
    print(f"Maximum habitability score: {df['habitability_score'].max():.4f}")
    
    # Optional processing steps
    df = handle_outliers(df)
    df = handle_skewness(df)
    df = scale_features(df)
    
    # Display top habitable planets
    df_sorted = df.sort_values(by='habitability_score', ascending=False)
    print("\nTop 20 most Earth-like planets:")
    print(df_sorted[['pl_name', 'habitability_score', 'pl_rade', 'pl_orbsmax']].head(20))
    
    # Save processed data
    df.to_csv('../data/processed/exoplanet_data_clean.csv', index=False)
    
    # Visualizations
    plot_feature_distributions(df)
    # plot_habitability_rankings(df)
