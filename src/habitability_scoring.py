import numpy as np
import pandas as pd
from physical_calculations import (
    CONSTANTS, JEANS_THRESHOLDS,
    calculate_escape_velocity, ensure_temperature_data, 
    calculate_jeans_parameters, assess_stellar_activity,
    calculate_magnetic_field_protection
)

def calculate_habitability_score(df):
    """
    Calculate habitability score for exoplanets.
    """
    result_df = df.copy()
    
    result_df = estimate_atmosphere_probability(result_df)
    planet_names = result_df['pl_name'].copy() if 'pl_name' in result_df.columns else None
    
    result_df = calculate_scores(result_df)
    result_df = adjust_special_cases(result_df)
    
    columns_to_keep = ['pl_name', 'pl_rade', 'pl_bmasse', 'pl_orbsmax', 'pl_orbeccen',
                      'st_lum', 'st_mass', 'sy_snum', 'sy_pnum', 'pl_dens', 'st_teff', 
                      'st_rad', 'pl_temp', 'habitability_score', 'atm_retention_prob',
                      'radiation_viability']
    
    result_cols = [col for col in columns_to_keep if col in result_df.columns]
    result_df = result_df[result_cols]
    
    if planet_names is not None:
        result_df['pl_name'] = planet_names
    
    return result_df

def estimate_atmosphere_probability(df):
    """
    Estimate the probability of atmosphere retention.
    """
    result_df = df.copy()
    
    result_df = calculate_escape_velocity(result_df)
    result_df = ensure_temperature_data(result_df)
    result_df = calculate_jeans_parameters(result_df)
    result_df = assess_stellar_activity(result_df)
    
    # Calculate retention probabilities for each gas
    for gas, threshold in JEANS_THRESHOLDS.items():
        result_df[f'{gas}_retention'] = 1 / (1 + np.exp(-(result_df[f'jeans_{gas}'] - threshold)))

    # Emphasize N and O for an Earth-like atmosphere
    result_df['atm_retention_prob'] = (
        0.1 * result_df['H_retention'] +
        0.1 * result_df['He_retention'] +
        0.4 * result_df['N_retention'] +
        0.4 * result_df['O_retention']
    )
    
    result_df = calculate_magnetic_field_protection(result_df)
    
    # A strong magnetic field can protect the atmosphere
    result_df['atm_retention_prob'] = np.clip(
        result_df['atm_retention_prob'] * result_df['magnetic_protection'],
        0, 1
    )

    cols_to_drop = ['escape_vel', 'jeans_H', 'jeans_He', 'jeans_N', 'jeans_O',
                   'H_retention', 'He_retention', 'N_retention', 'O_retention',
                   'estimated_rotation_period', 'mass_magnetic_factor', 
                   'density_magnetic_factor', 'rotation_magnetic_factor', 
                   'radius_magnetic_factor', 'magnetic_protection', 'stellar_activity_factor']
    
    result_df = result_df.drop(columns=[col for col in cols_to_drop if col in result_df.columns])
    
    return result_df

def calculate_scores(df):
    """
    Calculate habitability scores based on physical properties. 
    """
    result_df = df.copy()
    
    # Habitable zone calculation
    result_df['hz_inner'] = 0.97 * np.sqrt(result_df['st_lum'])
    result_df['hz_outer'] = 1.67 * np.sqrt(result_df['st_lum'])
    result_df['hz_position'] = (result_df['pl_orbsmax'] - result_df['hz_inner']) / (result_df['hz_outer'] - result_df['hz_inner'])
    result_df['hz_score'] = np.exp(-4 * ((result_df['hz_position'] - 0.5) ** 2))
    
    # Penalize planets outside the habitable zone
    outside_hz = (result_df['hz_position'] < 0) | (result_df['hz_position'] > 1)
    result_df.loc[outside_hz, 'hz_score'] = result_df.loc[outside_hz, 'hz_score'] * 0.1
    
    # Planetary properties scores
    # Radius score - peaks at Earth radius (1.0), declines as radius diverges
    result_df['radius_score'] = np.exp(-((result_df['pl_rade'] - 1.0) ** 2) / 0.3)
    # Mass score - peaks near Earth mass (1.0), with wider acceptance range for super-Earths
    result_df['mass_score'] = np.exp(-(np.log10(result_df['pl_bmasse']) ** 2) / 0.8)
    # Orbital stability - low eccentricity orbits provide more stable climates
    result_df['stability_score'] = np.exp(-5 * result_df['pl_orbeccen'])
    # Density - Earth-like density suggests rocky composition with potential for atmosphere
    result_df['density_score'] = np.exp(-((result_df['pl_dens'] - CONSTANTS['EARTH_DENSITY']) ** 2) / 10)
    # Stellar type suitability - K and G type stars provide longer habitable timescales
    result_df['stellar_lifetime_score'] = np.exp(-((result_df['st_mass'] - 0.8) ** 2) / 0.3)
    
    # Tidally locked planets
    result_df['tidal_parameter'] = result_df['st_mass'] / (result_df['pl_orbsmax'] ** 3)
    max_tidal = result_df['tidal_parameter'].max()
    result_df['tidal_score'] = np.exp(-np.log1p(result_df['tidal_parameter']) / (np.log1p(max_tidal) / 5))
    
    # Star system score - single star systems preferred
    result_df['star_system_score'] = np.where(
        result_df['sy_snum'] == 1, 
        1.0,
        np.exp(-(result_df['sy_snum'] - 1))
    )
    
    # Planetary system score - 2-5 planets preferred
    result_df['planet_system_score'] = np.where(
        (result_df['sy_pnum'] >= 2) & (result_df['sy_pnum'] <= 5),
        1.0,
        np.where(
            result_df['sy_pnum'] == 1,
            0.8,
            0.7
        )
    )
    
    # Temperature scores
    result_df['temp_score'] = np.exp(-((result_df['pl_temp'] - CONSTANTS['EARTH_TEMP']) ** 2) / 5000)
    
    # Viability factors
    result_df['temp_viability'] = np.where(
        (result_df['pl_temp'] > 373) | (result_df['pl_temp'] < 180),
        0.01,
        np.exp(-((result_df['pl_temp'] - CONSTANTS['EARTH_TEMP']) ** 2) / 5000)
    )
    
    result_df['mass_viability'] = np.where(
        result_df['pl_bmasse'] > 10,
        0.01,
        np.exp(-(np.log10(result_df['pl_bmasse']) ** 2) / 0.8)
    )
    
    m_dwarfs_close = (result_df['st_teff'] < 3800) & (result_df['pl_orbsmax'] < 0.1)
    result_df['radiation_viability'] = np.where(
        m_dwarfs_close,
        0.01,
        1.0
    )
    
    # Combine into final habitability score
    viability_score = (
        result_df['hz_score'] ** 0.3 *           # HZ position
        result_df['temp_viability'] ** 0.3 *     # Temperature suitability
        result_df['atm_retention_prob'] ** 0.3 * # Amtosphere retention
        result_df['mass_viability'] ** 0.2 *     # Mass suitability
        result_df['radiation_viability'] ** 0.2  # Radiation environment
    )
    
    other_factors = (
        0.25 * result_df['atm_retention_prob'] +     # Has an atmosphere
        0.15 * result_df['density_score'] +          # Earth-like density
        0.15 * result_df['stability_score'] +        # Stable orbit
        0.15 * result_df['radius_score'] +           # Earth-like radius
        0.10 * result_df['stellar_lifetime_score'] + # Long-lived star
        0.10 * result_df['tidal_score'] +            # Not tidally locked
        0.05 * result_df['star_system_score'] +      # Single star system
        0.05 * result_df['planet_system_score']      # 2-5 planets in system
    )
    
    result_df['habitability_score'] = (viability_score * other_factors) ** 0.1 # Scale out the values
    
    return result_df

def adjust_special_cases(df):
    """
    Adjust habitability scores for special cases.
    """
    result_df = df.copy()
    
    earth_like = (
        (result_df['pl_rade'].between(0.9, 1.1)) & 
        (result_df['pl_bmasse'].between(0.9, 1.1)) & 
        (result_df['pl_orbsmax'].between(0.95, 1.05)) & 
        (result_df['pl_orbeccen'] < 0.02) &
        (result_df['st_mass'].between(0.95, 1.05)) &
        (result_df['sy_snum'] == 1) &
        (result_df['atm_retention_prob'] > 0.8)
    )
    # Earth-like planets should have a high habitability score
    result_df.loc[earth_like, 'habitability_score'] = np.maximum(
        result_df.loc[earth_like, 'habitability_score'], 
        0.95
    )

    extreme_cases = (
        (result_df['pl_bmasse'] > 50) |         # Definite gas giants
        (result_df['pl_rade'] > 2) |            # Likely no solid surface
        (result_df['pl_temp'] > 500) |          # Too hot for life
        (result_df['pl_temp'] < 100) |          # Too cold for life
        (result_df['atm_retention_prob'] < 0.5) # Unlikely to have an atmosphere
    )
    # Extreme cases should have a low habitability score
    result_df.loc[extreme_cases, 'habitability_score'] = np.minimum(
        result_df.loc[extreme_cases, 'habitability_score'],
        0.05
    )
    
    result_df['habitability_score'] = np.clip(result_df['habitability_score'], 0, 1)
    
    return result_df
