import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox
import math
from sklearn.preprocessing import StandardScaler

# Physical constants
CONSTANTS = {
    'G': 6.67430e-11,              # Gravitational constant (m^3/kg/s^2)
    'EARTH_MASS_KG': 5.97e24,      # Earth mass in kg
    'EARTH_RADIUS_M': 6371e3,      # Earth radius in m
    'EARTH_DENSITY': 5.51,         # Earth density in g/cm³
    'EARTH_TEMP': 288,             # Earth average surface temperature in K
    'SOLAR_RADIUS_M' : 6.955e8,    # SRAD to m
    'AU_TO_M' : 1.496e11,          # AU to m
    'K_BOLTZMANN': 1.380649e-23,   # Boltzmann constant (J/K)
    'H_MASS': 1.6735575e-27,       # Mass of hydrogen atom (kg)
    'HE_MASS': 6.6464764e-27,      # Mass of helium atom (kg)
    'O_MASS': 2.6566962e-26,       # Mass of oxygen atom (kg)
    'N_MASS': 2.3258671e-26,       # Mass of nitrogen atom (kg)
    'ALBEDO': 0.3                  # Default Earth-like albedo
}

# Gas retention parameters
JEANS_THRESHOLDS = {
    'H': 3.0,    # Hydrogen is easily lost
    'He': 10.0,  # Helium requires higher escape parameter
    'N': 15.0,   # Nitrogen requires even higher
    'O': 20.0    # Oxygen requires highest
}

def preprocess_exoplanet_data(file_path):
    """
    Main function to load and process exoplanet data, calculating habitability scores.
    """
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Columns to retain based on habitability relevance
    retained_columns = [
        'pl_name', 'pl_rade', 'pl_bmasse', 'pl_orbsmax', 'pl_orbeccen',
        'st_lum', 'st_mass', 'sy_snum', 'sy_pnum', 'pl_dens', 'st_teff', 'st_rad'
    ]
    df = df[retained_columns]
    
    # Process missing data using physical relationships
    df = process_missing_data(df)
    
    # Compute habitability index
    df = calculate_habitability_score(df)
    
    return df

def process_missing_data(df, drop_threshold=0):
    """
    Process missing values in exoplanet data using physical relationships.
    """
    result_df = df.copy()
    
    # Define key features for evaluation
    key_features = ['pl_rade', 'pl_bmasse', 'pl_orbsmax', 'pl_orbeccen', 
                    'st_lum', 'st_mass', 'pl_dens', 'st_teff', 'st_rad']
    
    # Remove rows with too many missing values
    if drop_threshold > 0:
        missing_counts = result_df[key_features].isnull().sum(axis=1)
        too_many_missing = missing_counts > (len(key_features) * drop_threshold)
        result_df = result_df[~too_many_missing].copy()
    
    # Data cleaning
    result_df.loc[result_df['st_lum'] < 0, 'st_lum'] = np.nan

    # Apply physical relationships to fill missing values
    result_df = _apply_mass_radius_relationship(result_df)
    result_df = _calculate_density_from_mass_radius(result_df)
    result_df = _calculate_mass_from_density_radius(result_df)
    result_df = _apply_stellar_mass_luminosity_relationship(result_df)
    
    # Eccentricity: Assume slightly elliptical orbits if missing
    result_df['pl_orbeccen'] = result_df['pl_orbeccen'].fillna(0.05)
    
    # Calculate equilibrium temperature where data is available
    result_df = _estimate_equilibrium_temperature(result_df)
    
    # Fill remaining missing values with group medians
    result_df = _fill_remaining_with_medians(result_df, key_features)
    
    return result_df

def _apply_mass_radius_relationship(df):
    """Apply mass-radius relationship for rocky planets."""
    # For planets with radius but no mass data
    mask_radius = df['pl_rade'].notna() & df['pl_bmasse'].isna()
    df.loc[mask_radius, 'pl_bmasse'] = df.loc[mask_radius, 'pl_rade']**3
    
    # For planets with mass but no radius data
    mask_mass = df['pl_bmasse'].notna() & df['pl_rade'].isna()
    df.loc[mask_mass, 'pl_rade'] = df.loc[mask_mass, 'pl_bmasse']**(1/3)
    
    return df

def _calculate_density_from_mass_radius(df):
    """Calculate density from mass and radius."""
    mask_dens = df['pl_dens'].isna() & df['pl_rade'].notna() & df['pl_bmasse'].notna()
    planet_volume = (df.loc[mask_dens, 'pl_rade']**3)
    df.loc[mask_dens, 'pl_dens'] = (df.loc[mask_dens, 'pl_bmasse'] / planet_volume) * CONSTANTS['EARTH_DENSITY']
    return df

def _calculate_mass_from_density_radius(df):
    """Calculate mass from density and radius."""
    mask_mass_from_dens = df['pl_bmasse'].isna() & df['pl_dens'].notna() & df['pl_rade'].notna()
    volume_ratio = df.loc[mask_mass_from_dens, 'pl_rade']**3
    df.loc[mask_mass_from_dens, 'pl_bmasse'] = df.loc[mask_mass_from_dens, 'pl_dens'] * volume_ratio / CONSTANTS['EARTH_DENSITY']
    return df

def _apply_stellar_mass_luminosity_relationship(df):
    """Apply stellar mass-luminosity relationship."""
    # Stellar luminosity from mass
    valid_mass = df['st_mass'] > 0
    mask_lum = df['st_lum'].isna() & valid_mass
    df.loc[mask_lum, 'st_lum'] = df.loc[mask_lum, 'st_mass']**3.5
    
    # Stellar mass from luminosity
    valid_lum = df['st_lum'] > 0
    mask_mass_star = df['st_mass'].isna() & valid_lum
    df.loc[mask_mass_star, 'st_mass'] = df.loc[mask_mass_star, 'st_lum']**(1/3.5)
    
    return df

def _estimate_equilibrium_temperature(df):
    """Estimate planet equilibrium temperature."""
    mask_temp = df['st_teff'].notna() & df['st_rad'].notna() & df['pl_orbsmax'].notna()
    
    df.loc[mask_temp, 'pl_temp'] = df.loc[mask_temp, 'st_teff'] * \
        np.sqrt(df.loc[mask_temp, 'st_rad'] / (2 * df.loc[mask_temp, 'pl_orbsmax'])) * \
        (1 - CONSTANTS['ALBEDO']) ** 0.25
    
    return df

def _fill_remaining_with_medians(df, key_features):
    """Fill remaining missing values with group medians first, then overall medians."""
    if df[key_features].isnull().any().any():
        for feature in key_features:
            if df[feature].isnull().any():
                # Try to fill based on system properties first
                grouped_median = df.groupby(['sy_snum', 'sy_pnum'])[feature].transform('median')
                df[feature] = df[feature].fillna(grouped_median)
                
                # Fill remaining with overall median
                df[feature] = df[feature].fillna(df[feature].median())
    
    return df

def calculate_habitability_score(df):
    """
    Calculate a habitability score from 0-1 for exoplanets.
    """
    # First estimate atmospheric retention probability
    df = estimate_atmosphere_probability(df)
    
    # Store planet names for later
    planet_names = df['pl_name'].copy() if 'pl_name' in df.columns else None
    
    # Calculate various habitability factors
    result_df = df.copy()
    result_df = _calculate_habitable_zone_score(result_df)
    result_df = _calculate_planetary_properties_scores(result_df)
    result_df = _calculate_system_architecture_scores(result_df)
    result_df = _calculate_temperature_score(result_df)
    result_df = _calculate_viability_factors(result_df)
    
    # Calculate final habitability score
    result_df = _combine_habitability_factors(result_df)
    
    # Apply special case adjustments
    result_df = _adjust_special_cases(result_df)
    
    # Clean up intermediate columns
    result_df = _cleanup_intermediate_columns(result_df)
    
    # Restore planet names if they were present
    if planet_names is not None:
        result_df['pl_name'] = planet_names
    
    return result_df

def estimate_atmosphere_probability(df):
    """
    Estimate the probability that an exoplanet has retained a substantial atmosphere.
    """
    result_df = df.copy()
    
    # Calculate escape velocity
    result_df = _calculate_escape_velocity(result_df)
    
    # Ensure we have temperature data
    result_df = _ensure_temperature_data(result_df)
    
    # Calculate Jeans escape parameters for different gases
    result_df = _calculate_jeans_parameters(result_df)
    
    # Assess stellar activity impact on atmospheric stripping
    result_df = _assess_stellar_activity(result_df)
    
    # Calculate atmospheric retention probability
    result_df = _calculate_atmosphere_retention(result_df)
    
    # Clean up intermediate columns
    columns_to_drop = [
        'escape_vel', 'jeans_H', 'jeans_He', 'jeans_N', 'jeans_O',
        'H_retention', 'He_retention', 'N_retention', 'O_retention',
        'stellar_activity_factor'
    ]
    result_df = result_df.drop(columns=[col for col in columns_to_drop if col in result_df.columns])
    
    return result_df

def _calculate_escape_velocity(df):
    """Calculate planetary escape velocity in m/s."""
    mask_escape = df['pl_rade'].notna() & df['pl_bmasse'].notna()
    
    df.loc[mask_escape, 'escape_vel'] = np.sqrt(
        2 * CONSTANTS['G'] * df.loc[mask_escape, 'pl_bmasse'] * CONSTANTS['EARTH_MASS_KG'] / 
        (df.loc[mask_escape, 'pl_rade'] * CONSTANTS['EARTH_RADIUS_M'])
    )
    
    return df

def _ensure_temperature_data(df):
    """Ensure temperature data is available, estimate if needed."""
    # Estimate equilibrium temperature
    mask_temp = df['st_teff'].notna() & df['st_rad'].notna() & df['pl_orbsmax'].notna()
        
    df.loc[mask_temp, 'pl_temp'] = df.loc[mask_temp, 'st_teff'] * \
        np.sqrt((df.loc[mask_temp, 'st_rad'] * CONSTANTS['SOLAR_RADIUS_M']) / \
                (2 * df.loc[mask_temp, 'pl_orbsmax'] * CONSTANTS['AU_TO_M'])) * \
        (1 - CONSTANTS['ALBEDO'])**0.25
        
    # Fill remaining with median
    df['pl_temp'] = df['pl_temp'].fillna(df['pl_temp'].median())
    return df

def _calculate_jeans_parameters(df):
    """Calculate Jeans escape parameters for different atmospheric gases."""
    mask_params = df['escape_vel'].notna() & df['pl_temp'].notna()

    # Calculate for hydrogen (most easily lost)
    df.loc[mask_params, 'jeans_H'] = (
        df.loc[mask_params, 'escape_vel']**2 * CONSTANTS['H_MASS'] / 
        (2 * CONSTANTS['K_BOLTZMANN'] * df.loc[mask_params, 'pl_temp'])
    )
    
    # Calculate for helium
    df.loc[mask_params, 'jeans_He'] = (
        df.loc[mask_params, 'escape_vel']**2 * CONSTANTS['HE_MASS'] / 
        (2 * CONSTANTS['K_BOLTZMANN'] * df.loc[mask_params, 'pl_temp'])
    )
    
    # Calculate for nitrogen
    df.loc[mask_params, 'jeans_N'] = (
        df.loc[mask_params, 'escape_vel']**2 * CONSTANTS['N_MASS'] / 
        (2 * CONSTANTS['K_BOLTZMANN'] * df.loc[mask_params, 'pl_temp'])
    )
    
    # Calculate for oxygen
    df.loc[mask_params, 'jeans_O'] = (
        df.loc[mask_params, 'escape_vel']**2 * CONSTANTS['O_MASS'] / 
        (2 * CONSTANTS['K_BOLTZMANN'] * df.loc[mask_params, 'pl_temp'])
    )
    
    return df

def _assess_stellar_activity(df):
    """Assess the impact of stellar activity on atmospheric retention."""
    # Initialize activity factor
    df['stellar_activity_factor'] = 1.0
    
    # Identify M-dwarfs and close-in planets
    m_dwarfs = df['st_teff'] < 3800
    close_in = df['pl_orbsmax'] < 0.1
    
    # Penalize close-in planets around M-dwarfs
    df.loc[m_dwarfs & close_in, 'stellar_activity_factor'] = 0.3
    df.loc[m_dwarfs & ~close_in, 'stellar_activity_factor'] = 0.7
    
    return df

def _calculate_atmosphere_retention(df):
    """Calculate the final atmospheric retention probability."""
    # Calculate retention probability components for each gas
    for gas, threshold in JEANS_THRESHOLDS.items():
        df[f'{gas}_retention'] = 1 / (1 + np.exp(-(df[f'jeans_{gas}'] - threshold)))

    # Combine retention probabilities with weights
    # Emphasize N and O for habitability (Earth-like atmosphere)
    df['atm_retention_prob'] = (
        0.1 * df['H_retention'] +
        0.1 * df['He_retention'] +
        0.4 * df['N_retention'] +
        0.4 * df['O_retention']
    ) * df['stellar_activity_factor']
    
    # Clip to 0-1 range
    df['atm_retention_prob'] = np.clip(df['atm_retention_prob'], 0, 1)
    return df

def _calculate_habitable_zone_score(df):
    """Calculate habitable zone position and score."""
    # Calculate habitable zone boundaries based on stellar luminosity
    # Using Kopparapu et al. (2014) conservative habitable zone estimates
    df['hz_inner'] = 0.97 * np.sqrt(df['st_lum'])
    df['hz_outer'] = 1.67 * np.sqrt(df['st_lum'])
    
    # Normalized position in habitable zone (0-1 for planets in HZ)
    df['hz_position'] = (df['pl_orbsmax'] - df['hz_inner']) / (df['hz_outer'] - df['hz_inner'])
    
    # Calculate habitable zone score - highest at center of zone, declining toward edges
    df['hz_score'] = np.exp(-4 * ((df['hz_position'] - 0.5) ** 2))
    
    # Handle planets outside habitable zone
    outside_hz = (df['hz_position'] < 0) | (df['hz_position'] > 1)
    df.loc[outside_hz, 'hz_score'] = df.loc[outside_hz, 'hz_score'] * 0.1
    
    return df

def _calculate_planetary_properties_scores(df):
    """Calculate scores related to planetary physical properties."""
    # Radius score - peaks at Earth radius (1.0), declines as radius diverges
    df['radius_score'] = np.exp(-((df['pl_rade'] - 1.0) ** 2) / 0.5)
    
    # Mass score - peaks near Earth mass (1.0), with wider acceptance range for super-Earths
    df['mass_score'] = np.exp(-(np.log10(df['pl_bmasse']) ** 2) / 0.8)
    
    # Orbital stability - low eccentricity orbits provide more stable climates
    df['stability_score'] = np.exp(-5 * df['pl_orbeccen'])
    
    # Density - Earth-like density suggests rocky composition with potential for atmosphere
    df['density_score'] = np.exp(-((df['pl_dens'] - CONSTANTS['EARTH_DENSITY']) ** 2) / 10)
    
    # Stellar type suitability - K and G type stars provide longer habitable timescales
    df['stellar_lifetime_score'] = np.exp(-((df['st_mass'] - 0.8) ** 2) / 0.3)
    
    # Tidal locking likelihood
    df['tidal_parameter'] = df['st_mass'] / (df['pl_orbsmax'] ** 3)
    max_tidal = df['tidal_parameter'].max()
    df['tidal_score'] = np.exp(-df['tidal_parameter'] / (max_tidal / 5))
    
    return df

def _calculate_system_architecture_scores(df):
    """Calculate scores related to the planetary system architecture."""
    # Star system score (single star systems preferred)
    df['star_system_score'] = np.where(
        df['sy_snum'] == 1, 
        1.0,  # Single star gets full score
        np.exp(-(df['sy_snum'] - 1))  # Exponential penalty for multiple stars
    )
    
    # Planet system score (systems with 2-5 planets appear most stable)
    df['planet_system_score'] = np.where(
        (df['sy_pnum'] >= 2) & (df['sy_pnum'] <= 5),
        1.0,  # Optimal range gets full score
        np.where(
            df['sy_pnum'] == 1,
            0.8,  # Single planet systems get slightly lower score
            0.7   # Systems with many planets (>5) get lower score
        )
    )
    
    return df

def _calculate_temperature_score(df):
    """Calculate temperature suitability score."""
    df['temp_score'] = np.exp(-((df['pl_temp'] - CONSTANTS['EARTH_TEMP']) ** 2) / 5000)
    return df

def _calculate_viability_factors(df):
    """Calculate hard requirement factors that can rule out habitability."""
    # Temperature viability (too hot or too cold is eliminatory)
    df['temp_viability'] = np.where(
        (df['pl_temp'] > 373) | (df['pl_temp'] < 180),  # Water phase transitions
        0.01,  # Almost eliminatory if outside temperature range for liquid water
        np.exp(-((df['pl_temp'] - CONSTANTS['EARTH_TEMP']) ** 2) / 5000)
    )
    
    # Mass viability (gas giants are eliminatory)
    df['mass_viability'] = np.where(
        df['pl_bmasse'] > 10,  # Above ~10 Earth masses likely becomes gas-dominated
        0.01,  # Almost eliminatory for gas giants
        np.exp(-(np.log10(df['pl_bmasse']) ** 2) / 0.8)
    )
    
    # Radiation environment (extreme radiation is eliminatory)
    m_dwarfs_close = (df['st_teff'] < 3800) & (df['pl_orbsmax'] < 0.05)
    df['radiation_viability'] = np.where(
        m_dwarfs_close,
        0.1,  # Severe penalty for very close planets around M-dwarfs
        1.0   # No penalty otherwise
    )
    
    return df

def _combine_habitability_factors(df):
    """Combine all habitability factors into a single score using weighted geometric mean."""
    pd.set_option("display.max_columns", None)
    print(df.loc[[3194]])
    # Geometric mean of viability factors with weights
    viability_score = (
        df['hz_score'] ** 0.3 *              # Habitable zone position
        df['temp_viability'] ** 0.3 *        # Temperature suitability
        df['mass_viability'] ** 0.2 *        # Mass appropriateness
        df['radiation_viability'] ** 0.2     # Radiation environment
    )
    
    # Arithmetic mean of other desirable factors
    other_factors = (
        0.25 * df['atm_retention_prob'] +    # Atmosphere is very important
        0.15 * df['density_score'] +         # Earth-like composition
        0.15 * df['stability_score'] +       # Stable orbit
        0.15 * df['radius_score'] +          # Earth-like size
        0.10 * df['stellar_lifetime_score'] + # Long-lived star
        0.10 * df['tidal_score'] +           # Avoid tidal locking
        0.05 * df['star_system_score'] +     # Stable star system
        0.05 * df['planet_system_score']     # Stable planetary system
    )
    
    # Combine the two components
    df['habitability_score'] = (viability_score * other_factors) ** 0.2 # Scale out the values
    
    return df

def _adjust_special_cases(df):
    """Adjust scores for special cases like Earth-analogs and gas giants."""
    # Ensure Earth-analogs get very high scores
    earth_like = (
        (df['pl_rade'].between(0.9, 1.1)) & 
        (df['pl_bmasse'].between(0.9, 1.1)) & 
        (df['pl_orbsmax'].between(0.95, 1.05)) & 
        (df['pl_orbeccen'] < 0.02) &
        (df['st_mass'].between(0.95, 1.05)) &
        (df['sy_snum'] == 1) &
        (df['atm_retention_prob'] > 0.8)  # Good atmosphere retention
    )
    df.loc[earth_like, 'habitability_score'] = np.maximum(
        df.loc[earth_like, 'habitability_score'], 
        0.95
    )
    
    ## Make sure gas giants and extremely hot/cold planets get very low scores
    #extreme_cases = (
    #    (df['pl_bmasse'] > 50) |                  # Definite gas giants
    #    (df['pl_temp'] > 500) |                   # Too hot
    #    (df['pl_temp'] < 100) |                   # Too cold
    #    (df['pl_orbsmax'] < 0.01) |               # Extremely close orbits
    #    (df['atm_retention_prob'] < 0.1)          # Cannot retain atmosphere
    #)
    #df.loc[extreme_cases, 'habitability_score'] = np.minimum(
    #    df.loc[extreme_cases, 'habitability_score'],
    #    0.05
    #)
    
    # Normalize scores to 0-1 range
    df['habitability_score'] = np.clip(df['habitability_score'], 0, 1)
    
    return df

def _cleanup_intermediate_columns(df):
    """Remove intermediate calculation columns to clean up the results."""
    columns_to_drop = [
        'hz_inner', 'hz_outer', 'hz_position', 'hz_score',
        'radius_score', 'mass_score', 'stability_score', 
        'density_score', 'stellar_lifetime_score', 
        'tidal_parameter', 'tidal_score',
        'star_system_score', 'planet_system_score', 'temp_score',
        'temp_viability', 'mass_viability', 'radiation_viability'
    ]
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
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
    #df = handle_outliers(df)
    #df = handle_skewness(df)
    #df = scale_features(df)
    
    # Display top habitable planets
    df_sorted = df.sort_values(by='habitability_score', ascending=False)
    print("\nTop 20 most Earth-like planets:")
    print(df_sorted[['pl_name', 'habitability_score', 'pl_rade', 'atm_retention_prob']].head(20))
    
    # Save processed data
    df.to_csv('../data/processed/exoplanet_data_clean.csv', index=False)
    
    # Visualizations
    # plot_feature_distributions(df)
    # plot_habitability_rankings(df)
