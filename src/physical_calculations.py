import numpy as np
import pandas as pd

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

def calculate_escape_velocity(df):
    """Calculate planetary escape velocity in m/s."""
    mask_escape = df['pl_rade'].notna() & df['pl_bmasse'].notna()
    
    df.loc[mask_escape, 'escape_vel'] = np.sqrt(
        2 * CONSTANTS['G'] * df.loc[mask_escape, 'pl_bmasse'] * CONSTANTS['EARTH_MASS_KG'] / 
        (df.loc[mask_escape, 'pl_rade'] * CONSTANTS['EARTH_RADIUS_M'])
    )
    
    return df

def ensure_temperature_data(df):
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

def calculate_jeans_parameters(df):
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

def assess_stellar_activity(df):
    """Assess the impact of stellar activity on atmospheric retention."""
    # Initialize activity factor
    df['stellar_activity_factor'] = 0.0
    
    # Identify M-dwarfs and close-in planets
    m_dwarfs = df['st_teff'] < 3800
    close_in = df['pl_orbsmax'] < 0.1
    
    # Penalize close-in planets around M-dwarfs
    df.loc[m_dwarfs & close_in, 'stellar_activity_factor'] = 0.25
    df.loc[m_dwarfs & ~close_in, 'stellar_activity_factor'] = 0.5
    
    return df

def calculate_magnetic_field_protection(df):
    """
    Calculate a magnetic field protection factor for atmospheric retention.
    """
    # Initialize magnetic protection factor
    df['magnetic_protection'] = 1.0
    
    # Mass factor - exponential growth of magnetic field potential with mass
    # Using a more aggressive scaling that increases rapidly with mass
    # Peaks around 1-3 Earth masses, then levels off
    df['mass_magnetic_factor'] = np.where(
        df['pl_bmasse'] <= 3,
        1 - np.exp(-df['pl_bmasse']),  # Rapid increase for smaller masses
        1 - np.exp(-3) * np.exp(-(df['pl_bmasse'] - 3) / 10)  # Slower increase after 3 Earth masses
    )
    
    # Density factor - higher density suggests more metallic core
    # Earth's density is ~5.51 g/cm³, use this as a reference
    df['density_magnetic_factor'] = np.exp(
        -((df['pl_dens'] - CONSTANTS['EARTH_DENSITY']) ** 2) / 5  # Broader, gentler curve
    )
    
    # Rotation period factor
    # Optimal rotation period around 20-30 hours
    # Sharp peaks around this range, drops off quickly outside
    df['estimated_rotation_period'] = 24 * np.sqrt(df['pl_bmasse']) / np.sqrt(1.0)
    df['rotation_magnetic_factor'] = np.exp(
        -((df['estimated_rotation_period'] - 25) ** 2) / 200
    )
    
    # Radius factor - larger planets have more potential for complex internal dynamics
    # Use log scale to handle wide range of planetary radii
    df['radius_magnetic_factor'] = np.exp(
        -((np.log10(df['pl_rade']) - np.log10(1.0)) ** 2) / 0.5
    )
    
    # Combine factors
    df['magnetic_protection'] = (
        0.4 * df['mass_magnetic_factor'] +
        0.2 * df['density_magnetic_factor'] +
        0.2 * df['rotation_magnetic_factor'] +
        0.2 * df['radius_magnetic_factor']
    )
    
    # Adjust magnetic protection based on stellar activity
    if 'stellar_activity_factor' in df.columns:
        df['magnetic_protection'] = (
            df['magnetic_protection'] * (1 - df['stellar_activity_factor'])
        )

    # Clip to 0-1 range
    df['magnetic_protection'] = np.clip(df['magnetic_protection'], 0, 1)   
 
    return df
