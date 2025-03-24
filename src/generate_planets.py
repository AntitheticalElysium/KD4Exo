import pandas as pd 
import numpy as np
from habitability_scoring import calculate_habitability

def generate_habitable_planets(df, num_to_generate=100):
    """
    Generate diverse synthetic habitable planets efficiently using vectorized operations.
    """
    habitable_planets = df[df['habitable'] == 1].reset_index(drop=True)
    
    # Randomly select template planets
    template_indices = np.random.randint(0, len(habitable_planets), num_to_generate)
    templates = habitable_planets.iloc[template_indices].copy()
    
    # Generate random factors
    templates['pl_rade'] *= np.random.uniform(0.8, 1.2, num_to_generate)
    templates['pl_bmasse'] *= np.random.uniform(0.7, 1.3, num_to_generate)
    templates['pl_dens'] *= np.random.uniform(0.9, 1.1, num_to_generate)
    templates['pl_temp'] *= np.random.uniform(0.9, 1.1, num_to_generate)
    
    templates['pl_orbsmax'] *= np.random.uniform(0.9, 1.1, num_to_generate)
    templates['pl_orbeccen'] = np.minimum(0.4, templates['pl_orbeccen'] * np.random.uniform(0.7, 1.3, num_to_generate))
    
    templates['st_lum'] *= np.random.uniform(0.9, 1.1, num_to_generate)
    templates['st_mass'] *= np.random.uniform(0.95, 1.05, num_to_generate)
    templates['st_rad'] *= np.random.uniform(0.95, 1.05, num_to_generate)
    templates['st_teff'] *= np.random.uniform(0.95, 1.05, num_to_generate)
    
    # Adjust system parameters
    templates['sy_snum'] = np.maximum(1, templates['sy_snum'] + np.random.choice([-1, 0, 1], num_to_generate))
    templates['sy_pnum'] = np.maximum(1, templates['sy_pnum'] + np.random.choice([-1, 0, 1, 2], num_to_generate))
    
    # Assign new planet names
    templates['pl_name'] = [f"Terra-{i+1}" for i in range(num_to_generate)]
    
    return templates

def generate_non_habitable_planets(df, num_to_generate=4000):
    """
    Efficiently generate diverse synthetic non-habitable planets using vectorized operations.
    """
    # Filter non-habitable planets
    non_habitable_planets = df[df['habitable'] == 0].reset_index(drop=True)
    
    # Randomly select template planets
    template_indices = np.random.randint(0, len(non_habitable_planets), num_to_generate)
    templates = non_habitable_planets.iloc[template_indices].copy()
    
    # Define extreme planet types and their effects
    extreme_types = np.random.choice(
        ["scorched", "frozen", "gas_giant", "toxic", "dense", "volatile", "unstable_orbit", "death_star", "micro_planet"], 
        num_to_generate
    )
    
    # Generate random modification factors
    pl_temp_factors = np.ones(num_to_generate)
    pl_orbsmax_factors = np.ones(num_to_generate)
    pl_rade_factors = np.ones(num_to_generate)
    pl_bmasse_factors = np.ones(num_to_generate)
    pl_dens_factors = np.ones(num_to_generate)
    pl_orbeccen_values = templates['pl_orbeccen'].values
    st_teff_factors = np.ones(num_to_generate)
    st_lum_factors = np.ones(num_to_generate)
    sy_snum_changes = np.zeros(num_to_generate, dtype=int)
    
    # Apply transformations based on planet type
    scorched_mask = extreme_types == "scorched"
    pl_temp_factors[scorched_mask] = np.random.uniform(2.0, 5.0, np.sum(scorched_mask))
    pl_orbsmax_factors[scorched_mask] = np.random.uniform(0.1, 0.4, np.sum(scorched_mask))
    st_teff_factors[scorched_mask] = np.random.uniform(1.2, 1.5, np.sum(scorched_mask))
    st_lum_factors[scorched_mask] = np.random.uniform(1.5, 3.0, np.sum(scorched_mask))
    
    frozen_mask = extreme_types == "frozen"
    pl_temp_factors[frozen_mask] = np.random.uniform(0.05, 0.3, np.sum(frozen_mask))
    pl_orbsmax_factors[frozen_mask] = np.random.uniform(3.0, 10.0, np.sum(frozen_mask))
    st_lum_factors[frozen_mask] = np.random.uniform(0.3, 0.8, np.sum(frozen_mask))
    
    gas_giant_mask = extreme_types == "gas_giant"
    pl_rade_factors[gas_giant_mask] = np.random.uniform(8.0, 15.0, np.sum(gas_giant_mask))
    pl_bmasse_factors[gas_giant_mask] = np.random.uniform(50.0, 300.0, np.sum(gas_giant_mask))
    pl_dens_factors[gas_giant_mask] = np.random.uniform(0.1, 0.3, np.sum(gas_giant_mask))
    
    toxic_mask = extreme_types == "toxic"
    pl_temp_factors[toxic_mask] = np.random.uniform(1.2, 1.8, np.sum(toxic_mask))
    pl_dens_factors[toxic_mask] = np.random.uniform(1.3, 2.0, np.sum(toxic_mask))
    st_teff_factors[toxic_mask] = np.random.uniform(1.1, 1.3, np.sum(toxic_mask))
    
    dense_mask = extreme_types == "dense"
    pl_dens_factors[dense_mask] = np.random.uniform(2.5, 5.0, np.sum(dense_mask))
    pl_rade_factors[dense_mask] = np.random.uniform(0.8, 1.5, np.sum(dense_mask))
    pl_bmasse_factors[dense_mask] = np.random.uniform(3.0, 8.0, np.sum(dense_mask))
    
    volatile_mask = extreme_types == "volatile"
    pl_temp_factors[volatile_mask] = np.random.uniform(1.3, 2.0, np.sum(volatile_mask))
    pl_dens_factors[volatile_mask] = np.random.uniform(1.1, 1.5, np.sum(volatile_mask))
    
    unstable_mask = extreme_types == "unstable_orbit"
    pl_orbeccen_values[unstable_mask] = np.random.uniform(0.7, 0.99, np.sum(unstable_mask))
    sy_snum_changes[unstable_mask] = np.random.choice([1, 2], np.sum(unstable_mask))
    
    death_star_mask = extreme_types == "death_star"
    st_teff_factors[death_star_mask] = np.random.uniform(1.5, 3.0, np.sum(death_star_mask))
    st_lum_factors[death_star_mask] = np.random.uniform(3.0, 10.0, np.sum(death_star_mask))
    
    micro_planet_mask = extreme_types == "micro_planet"
    pl_rade_factors[micro_planet_mask] = np.random.uniform(0.05, 0.2, np.sum(micro_planet_mask))
    pl_bmasse_factors[micro_planet_mask] = np.random.uniform(0.01, 0.1, np.sum(micro_planet_mask))
    
    # Apply transformations in a vectorized manner
    templates['pl_temp'] *= pl_temp_factors
    templates['pl_orbsmax'] *= pl_orbsmax_factors
    templates['pl_rade'] *= pl_rade_factors
    templates['pl_bmasse'] *= pl_bmasse_factors
    templates['pl_dens'] *= pl_dens_factors
    templates['pl_orbeccen'] = pl_orbeccen_values
    templates['st_teff'] *= st_teff_factors
    templates['st_lum'] *= st_lum_factors
    templates['sy_snum'] += sy_snum_changes
    templates['sy_snum'] = np.maximum(1, templates['sy_snum'])
    templates['sy_pnum'] = np.maximum(1, templates['sy_pnum'] + np.random.choice([-1, 0, 1, 2], num_to_generate))
    
    # Assign unique names
    templates['pl_name'] = [f"Thanatos-{i+1}" for i in range(num_to_generate)]
    
    # Recalculate habitability scores
    templates = calculate_habitability(templates)
    
    return templates
