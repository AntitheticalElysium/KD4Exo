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
    'ALBEDO': 0.3,                 # Default Earth-like albedo
    'SOLAR_TEMP': 5778             # Solar temperature in K
}

# Habital zone constants
HZ_CONSTANTS = {
    'SOLAR_TEMP': 5778,                                         # Solar temperature in K
    'INNER_BASE': 1.107,                                        # Inner edge (runaway greenhouse) base value
    'INNER_COEF': [1.332e-4, 1.58e-8, -8.308e-12, -1.931e-15],  # Inner edge coefficients
    'OUTER_BASE': 0.356,                                        # Outer edge (maximum greenhouse) base value
    'OUTER_COEF': [6.171e-5, 1.698e-9, -3.198e-12, -5.575e-16], # Outer edge coefficients
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
    Process missing data in the dataset.
    """
    result_df = df.copy()
    
    key_features = ['pl_rade', 'pl_bmasse', 'pl_orbsmax', 'pl_orbeccen', 
                    'st_lum', 'st_mass', 'pl_dens', 'st_teff', 'st_rad']
    
    # Drop rows with too many missing values
    if drop_threshold > 0:
        missing_counts = result_df[key_features].isnull().sum(axis=1)
        too_many_missing = missing_counts > (len(key_features) * drop_threshold)
        result_df = result_df[~too_many_missing].copy()
    
    result_df.loc[result_df['st_lum'] < 0, 'st_lum'] = np.nan
    
    # Mass-radius relationship for rocky planets
    mask_radius = result_df['pl_rade'].notna() & result_df['pl_bmasse'].isna()
    result_df.loc[mask_radius, 'pl_bmasse'] = result_df.loc[mask_radius, 'pl_rade']**3
    
    mask_mass = result_df['pl_bmasse'].notna() & result_df['pl_rade'].isna()
    result_df.loc[mask_mass, 'pl_rade'] = result_df.loc[mask_mass, 'pl_bmasse']**(1/3)
    
    # Density from mass and radius
    mask_dens = result_df['pl_dens'].isna() & result_df['pl_rade'].notna() & result_df['pl_bmasse'].notna()
    planet_volume = (result_df.loc[mask_dens, 'pl_rade']**3)
    result_df.loc[mask_dens, 'pl_dens'] = (result_df.loc[mask_dens, 'pl_bmasse'] / planet_volume) * CONSTANTS['EARTH_DENSITY']
    
    mask_mass_from_dens = result_df['pl_bmasse'].isna() & result_df['pl_dens'].notna() & result_df['pl_rade'].notna()
    volume_ratio = result_df.loc[mask_mass_from_dens, 'pl_rade']**3
    result_df.loc[mask_mass_from_dens, 'pl_bmasse'] = result_df.loc[mask_mass_from_dens, 'pl_dens'] * volume_ratio / CONSTANTS['EARTH_DENSITY']
    
    # Stellar mass-luminosity
    valid_mass = result_df['st_mass'] > 0
    mask_lum = result_df['st_lum'].isna() & valid_mass
    result_df.loc[mask_lum, 'st_lum'] = result_df.loc[mask_lum, 'st_mass']**3.5
    
    valid_lum = result_df['st_lum'] > 0
    mask_mass_star = result_df['st_mass'].isna() & valid_lum
    result_df.loc[mask_mass_star, 'st_mass'] = result_df.loc[mask_mass_star, 'st_lum']**(1/3.5)
    
    # Assume slightly eliptical orbits if missing 
    result_df['pl_orbeccen'] = result_df['pl_orbeccen'].fillna(0.05)
    
    # Temperature equilibrium estimate
    mask_temp = result_df['st_teff'].notna() & result_df['st_rad'].notna() & result_df['pl_orbsmax'].notna()
    
    result_df.loc[mask_temp, 'pl_temp'] = result_df.loc[mask_temp, 'st_teff'] * \
        np.sqrt(result_df.loc[mask_temp, 'st_rad'] / (2 * result_df.loc[mask_temp, 'pl_orbsmax'])) * \
        (1 - CONSTANTS['ALBEDO']) ** 0.25
    
    # Fill remaining with medians
    for feature in key_features:
        if result_df[feature].isnull().any():
            grouped_median = result_df.groupby(['sy_snum', 'sy_pnum'])[feature].transform('median')
            result_df[feature] = result_df[feature].fillna(grouped_median)
            result_df[feature] = result_df[feature].fillna(result_df[feature].median())
    
    return result_df

def calculate_escape_velocity(df):
    """
    Calculate planetary escape velocity.
    """
    result_df = df.copy()
    mask_escape = result_df['pl_rade'].notna() & result_df['pl_bmasse'].notna()
    
    result_df.loc[mask_escape, 'escape_vel'] = np.sqrt(
        2 * CONSTANTS['G'] * result_df.loc[mask_escape, 'pl_bmasse'] * CONSTANTS['EARTH_MASS_KG'] / 
        (result_df.loc[mask_escape, 'pl_rade'] * CONSTANTS['EARTH_RADIUS_M'])
    )
    
    return result_df

def ensure_temperature_data(df):
    """
    Ensure that temperature data is available, estimate if necessary.
    """
    result_df = df.copy()
    mask_temp = result_df['st_teff'].notna() & result_df['st_rad'].notna() & result_df['pl_orbsmax'].notna()
        
    result_df.loc[mask_temp, 'pl_temp'] = result_df.loc[mask_temp, 'st_teff'] * \
        np.sqrt((result_df.loc[mask_temp, 'st_rad'] * CONSTANTS['SOLAR_RADIUS_M']) / \
                (2 * result_df.loc[mask_temp, 'pl_orbsmax'] * CONSTANTS['AU_TO_M'])) * \
        (1 - CONSTANTS['ALBEDO'])**0.25
    
    # Fill remaining with medians
    result_df['pl_temp'] = result_df['pl_temp'].fillna(result_df['pl_temp'].median())
    return result_df

def calculate_jeans_parameters(df):
    """
    Calculate Jeans escape parameters for different atmospheric gases.
    """
    result_df = df.copy()
    mask_params = result_df['escape_vel'].notna() & result_df['pl_temp'].notna()

    gases = {
        'H': CONSTANTS['H_MASS'],
        'He': CONSTANTS['HE_MASS'],
        'N': CONSTANTS['N_MASS'],
        'O': CONSTANTS['O_MASS']
    }
    
    for gas, mass in gases.items():
        result_df.loc[mask_params, f'jeans_{gas}'] = (
            result_df.loc[mask_params, 'escape_vel']**2 * mass / 
            (2 * CONSTANTS['K_BOLTZMANN'] * result_df.loc[mask_params, 'pl_temp'])
        )
    
    return result_df

def assess_stellar_activity(df):
    """
    Assess the impact of stellar activity on atmospheric retention.
    """
    result_df = df.copy()
    result_df['stellar_activity_factor'] = 0.0
    
    m_dwarfs = result_df['st_teff'] < 3800
    close_in = result_df['pl_orbsmax'] < 0.1
    
    # Penalize close-in planets around M dwarfs - likely to tear off atmosphere
    result_df.loc[m_dwarfs & close_in, 'stellar_activity_factor'] = 0.25
    result_df.loc[m_dwarfs & ~close_in, 'stellar_activity_factor'] = 0.5
    
    return result_df

def calculate_magnetic_field_protection(df):
    """
    Calculate the magnetic field protection factor.
    """
    result_df = df.copy()
    result_df['magnetic_protection'] = 1.0
    
    # Mass factor - exponential growth of magnetic field potential with mass (peaks around 1-3 Earth masses then levels off)
    result_df['mass_magnetic_factor'] = np.where(
        result_df['pl_bmasse'] <= 3,
        1 - np.exp(-result_df['pl_bmasse']),
        1 - np.exp(-3) * np.exp(-(result_df['pl_bmasse'] - 3) / 10)
    )
    
    # Density factor - higher density suggests more metallic core
    result_df['density_magnetic_factor'] = np.exp(
        -((result_df['pl_dens'] - CONSTANTS['EARTH_DENSITY']) ** 2) / 5
    )
    
    # Rotation factor - faster rotation can generate stronger magnetic fields (optimal is 20-30 hours)
    result_df['estimated_rotation_period'] = 24 * np.sqrt(result_df['pl_bmasse']) / np.sqrt(1.0)
    result_df['rotation_magnetic_factor'] = np.exp(
        -((result_df['estimated_rotation_period'] - 25) ** 2) / 200
    )
    
    # Radius factor - larger planets have more potential for complex internal dynamics
    result_df['radius_magnetic_factor'] = np.exp(
        -((np.log10(result_df['pl_rade']) - np.log10(1.0)) ** 2) / 0.5
    )
    
    # Combined protection factor
    result_df['magnetic_protection'] = (
        0.4 * result_df['mass_magnetic_factor'] +
        0.2 * result_df['density_magnetic_factor'] +
        0.2 * result_df['rotation_magnetic_factor'] +
        0.2 * result_df['radius_magnetic_factor']
    )
    
    # Stellar activity will impact magnetic protection 
    if 'stellar_activity_factor' in result_df.columns:
        result_df['magnetic_protection'] = (
            result_df['magnetic_protection'] * (1 - result_df['stellar_activity_factor'])
        )
    
    result_df['magnetic_protection'] = np.clip(result_df['magnetic_protection'], 0, 1)
    
    return result_df
