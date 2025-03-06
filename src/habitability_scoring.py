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
    result_df = calculate_escape_velocity(result_df)
    
    # Ensure we have temperature data
    result_df = ensure_temperature_data(result_df)
    
    # Calculate Jeans escape parameters for different gases
    result_df = calculate_jeans_parameters(result_df)
    
    # Assess stellar activity impact on atmospheric stripping
    result_df = assess_stellar_activity(result_df)
    
    # Calculate atmospheric retention probability
    result_df = _calculate_atmosphere_retention(result_df)

    # Add magnetic field protection assessment
    result_df = calculate_magnetic_field_protection(result_df)
    
    # Modify atmospheric retention probability
    # Combine existing atm_retention_prob with magnetic protection
    result_df['atm_retention_prob'] = np.clip(
        result_df['atm_retention_prob'] * result_df['magnetic_protection'],
        0, 1
    )

    # Clean up intermediate columns
    columns_to_drop = [
        'escape_vel', 'jeans_H', 'jeans_He', 'jeans_N', 'jeans_O',
        'H_retention', 'He_retention', 'N_retention', 'O_retention',
        'estimated_rotation_period', 'mass_magnetic_factor', 
        'density_magnetic_factor', 'rotation_magnetic_factor', 
        'radius_magnetic_factor', 'magnetic_protection', 'stellar_activity_factor'
    ]
    result_df = result_df.drop(columns=[col for col in columns_to_drop if col in result_df.columns])
    
    return result_df

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
    )
    
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
    df['radius_score'] = np.exp(-((df['pl_rade'] - 1.0) ** 2) / 0.3)
    
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
    df['tidal_score'] = np.exp(-np.log1p(df['tidal_parameter']) / (np.log1p(max_tidal) / 5))

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
    m_dwarfs_close = (df['st_teff'] < 3800) & (df['pl_orbsmax'] < 0.1)
    df['radiation_viability'] = np.where(
        m_dwarfs_close,
        0.01,  # Severe penalty for very close planets around M-dwarfs
        1.0   # No penalty otherwise
    )
    
    return df

def _combine_habitability_factors(df):
    """Combine all habitability factors into a single score using weighted geometric mean."""
    #Geometric mean of viability factors with weights
    viability_score = (
        df['hz_score'] ** 0.3 *              # Habitable zone position
        df['temp_viability'] ** 0.3 *        # Temperature suitability
        df['atm_retention_prob'] ** 0.3 *    # Atmosphere is very important
        df['mass_viability'] ** 0.2 *        # Mass appropriateness
        df['radiation_viability'] ** 0.2     # Radiation environment
    )
    
    # Arithmetic mean of other desirable factors
    other_factors = (
        0.25 * df['atm_retention_prob'] +    # Has an atmosphere 
        0.15 * df['density_score'] +         # Earth-like composition
        0.15 * df['stability_score'] +       # Stable orbit
        0.15 * df['radius_score'] +          # Earth-like size
        0.10 * df['stellar_lifetime_score'] + # Long-lived star
        0.10 * df['tidal_score'] +           # Avoid tidal locking
        0.05 * df['star_system_score'] +     # Stable star system
        0.05 * df['planet_system_score']     # Stable planetary system
    )
    
    # Combine the two components
    df['habitability_score'] = (viability_score * other_factors) ** 0.1 # Scale out the values
    
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
    
    # Make sure gas giants and extremely hot/cold planets get very low scores
    extreme_cases = (
        (df['pl_bmasse'] > 50) |                  # Definite gas giants
        (df['pl_rade'] > 2) |                     # Likely no solid surface
        (df['pl_temp'] > 500) |                   # Too hot
        (df['pl_temp'] < 100) |                   # Too cold
        (df['atm_retention_prob'] < 0.5)          # Cannot retain atmosphere
        # (df['radiation_viability'] < 0.2)
    )
    df.loc[extreme_cases, 'habitability_score'] = np.minimum(
        df.loc[extreme_cases, 'habitability_score'],
        0.05
    )
    
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
