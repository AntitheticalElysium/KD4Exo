import pandas as pd 
import numpy as np
from habitability_scoring import calculate_habitability

def generate_habitable_planets(df, num_to_generate=100):
    """
    Generate diverse synthetic habitable planets by modifying all available parameters.
    """
    # Find any planets that are already classified as habitable
    habitable_planets = df[df['habitable'] == 1]
    
    # Create empty list for synthetic planets
    synthetic_planets = []
    
    for i in range(num_to_generate):
        # Randomly select a template planet
        template_idx = np.random.randint(0, len(habitable_planets))
        template = habitable_planets.iloc[template_idx].copy()
        
        # Create a new synthetic planet
        new_planet = template.copy()
        new_planet['pl_name'] = f"Terra-{i+1}"
        
        # Modify all planetary parameters within habitable ranges
        # Planet characteristics
        new_planet['pl_rade'] *= np.random.uniform(0.8, 1.2)  # Earth radii
        new_planet['pl_bmasse'] *= np.random.uniform(0.7, 1.3)  # Earth masses
        new_planet['pl_dens'] *= np.random.uniform(0.9, 1.1)  # Density
        new_planet['pl_temp'] *= np.random.uniform(0.9, 1.1)  # Temperature
        
        # Orbital characteristics
        new_planet['pl_orbsmax'] *= np.random.uniform(0.9, 1.1)  # Semi-major axis
        new_planet['pl_orbeccen'] = min(0.4, new_planet['pl_orbeccen'] * np.random.uniform(0.7, 1.3))  # Eccentricity
        
        # Star characteristics
        new_planet['st_lum'] *= np.random.uniform(0.9, 1.1)  # Luminosity
        new_planet['st_mass'] *= np.random.uniform(0.95, 1.05)  # Stellar mass
        new_planet['st_rad'] *= np.random.uniform(0.95, 1.05)  # Stellar radius
        new_planet['st_teff'] *= np.random.uniform(0.95, 1.05)  # Effective temperature
        
        # System characteristics - discrete values, handle carefully
        # Slightly adjust number of stars and planets in system
        new_planet['sy_snum'] = max(1, int(new_planet['sy_snum'] + np.random.choice([-1, 0, 1])))
        new_planet['sy_pnum'] = max(1, int(new_planet['sy_pnum'] + np.random.choice([-1, 0, 1, 2])))
        
        synthetic_planets.append(new_planet)
    
    synthetic_planets_df = pd.DataFrame(synthetic_planets)
    
    # Recalculate habitability scores
    synthetic_planets_df = calculate_habitability(synthetic_planets_df)
    
    return synthetic_planets_df

def generate_non_habitable_planets(df, num_to_generate=4000):
    """
    Generate diverse synthetic non-habitable planets by modifying all available parameters.
    """
    # Find planets that are classified as non-habitable
    non_habitable_planets = df[df['habitable'] == 0]
    
    # Create empty list for synthetic planets
    synthetic_planets = []
    
    # Define extreme planet types
    extreme_types = [
        "scorched", "frozen", "gas_giant", "toxic", "dense", "volatile", 
        "unstable_orbit", "death_star", "micro_planet"
    ]
    
    for i in range(num_to_generate):
        # Randomly select a template planet
        template_idx = np.random.randint(0, len(non_habitable_planets))
        template = non_habitable_planets.iloc[template_idx].copy()
        
        # Create a new synthetic planet
        new_planet = template.copy()
        new_planet['pl_name'] = f"Thanatos-{i+1}"
        
        # Choose an extreme planet type
        extreme_type = np.random.choice(extreme_types)
        
        # Modify parameters based on extreme type
        if extreme_type == "scorched":
            # Very hot planets, close to star
            new_planet['pl_temp'] *= np.random.uniform(2.0, 5.0)
            new_planet['pl_orbsmax'] *= np.random.uniform(0.1, 0.4)
            new_planet['st_teff'] *= np.random.uniform(1.2, 1.5)
            new_planet['st_lum'] *= np.random.uniform(1.5, 3.0)
            
        elif extreme_type == "frozen":
            # Very cold planets, far from star
            new_planet['pl_temp'] *= np.random.uniform(0.05, 0.3)
            new_planet['pl_orbsmax'] *= np.random.uniform(3.0, 10.0)
            new_planet['st_lum'] *= np.random.uniform(0.3, 0.8)
            
        elif extreme_type == "gas_giant":
            # Massive gas giants
            new_planet['pl_rade'] *= np.random.uniform(8.0, 15.0)
            new_planet['pl_bmasse'] *= np.random.uniform(50.0, 300.0)
            new_planet['pl_dens'] *= np.random.uniform(0.1, 0.3)  # Low density
            
        elif extreme_type == "toxic":
            # Planets with extreme atmospheric conditions
            new_planet['pl_temp'] *= np.random.uniform(1.2, 1.8)
            new_planet['pl_dens'] *= np.random.uniform(1.3, 2.0)
            new_planet['st_teff'] *= np.random.uniform(1.1, 1.3)
            
        elif extreme_type == "dense":
            # Super-dense rocky planets
            new_planet['pl_dens'] *= np.random.uniform(2.5, 5.0)
            new_planet['pl_rade'] *= np.random.uniform(0.8, 1.5)
            new_planet['pl_bmasse'] *= np.random.uniform(3.0, 8.0)
            
        elif extreme_type == "volatile":
            # Planets with extreme volcanic/tectonic activity
            new_planet['pl_temp'] *= np.random.uniform(1.3, 2.0)
            new_planet['pl_dens'] *= np.random.uniform(1.1, 1.5)
            
        elif extreme_type == "unstable_orbit":
            # Planets with highly eccentric or unstable orbits
            new_planet['pl_orbeccen'] = np.random.uniform(0.7, 0.99)
            new_planet['sy_snum'] = max(1, int(new_planet['sy_snum'] + np.random.choice([1, 2])))  # More stars = unstable
            
        elif extreme_type == "death_star":
            # Planets orbiting dangerous stars
            new_planet['st_teff'] *= np.random.uniform(1.5, 3.0)
            new_planet['st_lum'] *= np.random.uniform(3.0, 10.0)
            new_planet['st_mass'] *= np.random.uniform(1.5, 3.0)
            
        elif extreme_type == "micro_planet":
            # Tiny planets that can't hold atmosphere
            new_planet['pl_rade'] *= np.random.uniform(0.05, 0.2)
            new_planet['pl_bmasse'] *= np.random.uniform(0.01, 0.1)
            
        # Modify remaining parameters randomly
        # System characteristics - discrete values
        new_planet['sy_pnum'] = max(1, int(new_planet['sy_pnum'] * np.random.uniform(0.5, 3.0)))
        
        # Add the synthetic planet to the list
        synthetic_planets.append(new_planet)
    
    synthetic_planets_df = pd.DataFrame(synthetic_planets)
    
    # Recalculate habitability scores
    synthetic_planets_df = calculate_habitability(synthetic_planets_df)
    
    return synthetic_planets_df
