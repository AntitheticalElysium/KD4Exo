import pandas as pd 
import numpy as np
from habitability_scoring import calculate_habitability

def generate_habitable_planets(df, num_to_generate=100):
    """
    Generate synthetic habitable planets by modifying parameters of existing habitable ones.
    """
    # Find any planets that are already classified as habitable
    habitable_planets = df[df['habitable'] == 1]
    
    if len(habitable_planets) == 0:
        raise ValueError("No habitable planets found to use as templates.")
    
    # Create empty dataframe for synthetic planets
    synthetic_planets = []
    
    for i in range(num_to_generate):
        # Randomly select a template planet
        template_idx = np.random.randint(0, len(habitable_planets))
        template = habitable_planets.iloc[template_idx].copy()
        
        # Create a new synthetic planet
        new_planet = template.copy()
        new_planet['pl_name'] = f"Synthetic-{i+1}"
        
        # Introduce small variations in key properties within a habitable range
        new_planet['pl_rade'] *= np.random.uniform(0.9, 1.1)
        new_planet['pl_bmasse'] *= np.random.uniform(0.8, 1.2)
        new_planet['pl_orbsmax'] *= np.random.uniform(0.95, 1.05)
        new_planet['pl_orbeccen'] *= np.random.uniform(0.8, 1.2)
        new_planet['pl_temp'] *= np.random.uniform(0.95, 1.05)
        
        synthetic_planets.append(new_planet)
    
    synthetic_planets_df = pd.DataFrame(synthetic_planets)
    
    # Recalculate habitability scores
    synthetic_planets_df = calculate_habitability(synthetic_planets_df)
    
    return synthetic_planets_df

