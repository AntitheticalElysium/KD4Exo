import pandas as pd

# Load the raw dataset
df = pd.read_csv("../data/raw/exoplanet_data.csv")

# Columns retained for estimating exoplanet habitability:
# 
# 1. Planetary Properties 
#    - 'pl_rade' (Planet Radius [Earth radii]): Indicates if the planet is terrestrial or gaseous.
#    - 'pl_bmasse' (Planet Mass [Earth masses]): Affects gravity, atmosphere retention, and surface conditions.
#    - 'pl_orbsmax' (Orbital Semi-Major Axis [AU]): Distance from the host star, influencing temperature.
#    - 'pl_orbper' (Orbital Period [days]): Helps estimate climate cycles and seasonal variations.
#    - 'pl_orbeccen' (Orbital Eccentricity): Higher values cause extreme climate swings.
#    - 'pl_insol' (Insolation Flux [Earth units]): Radiation received from the star, affecting temperature.
#    - 'pl_eqt' (Equilibrium Temperature [K]): Estimated surface temperature assuming no atmosphere.
#
# 2. Stellar Properties 
#    - 'st_spectype' (Spectral Type): Determines the type of radiation emitted by the star.
#    - 'st_teff' (Stellar Effective Temperature [K]): Defines the habitable zone around the star.
#    - 'st_rad' (Stellar Radius [Solar radii]): Affects the extent of the habitable zone.
#    - 'st_lum' (Stellar Luminosity [Solar units]): Directly impacts planetary climate.
#    - 'st_mass' (Stellar Mass [Solar masses]): Influences stellar lifespan and stability.
#    - 'st_met' (Stellar Metallicity [dex]): Affects planet formation and atmospheric composition.
#
# 3. System Properties
#    - 'sy_snum' (Number of Stars in System): Binary or multiple-star systems may have unstable orbits.
#    - 'sy_pnum' (Number of Planets in System): Can indicate gravitational influences on habitability.
#    - 'sy_dist' (Distance from Earth [parsecs]): Useful for determining observation feasibility.
#
# 4. Additional Properties?
#    - 'pl_dens' (Planet Density [g/cm^3]): Helps distinguish between rocky and gaseous planets.
#    - 'pl_trandep' (Transit Depth [%]): Can hint at atmospheric properties.
#    - 'pl_orbincl' (Orbital Inclination [degrees]): Useful for detection but not directly linked to habitability.

print(df[["pl_rade", "pl_bmasse", "pl_orbsmax", "pl_orbper", "pl_orbeccen", "pl_insol", "pl_eqt",
          "st_spectype", "st_teff", "st_rad", "st_lum", "st_mass", "st_met",
          "sy_snum", "sy_pnum", "sy_dist",
          "pl_dens", "pl_trandep", "pl_orbincl"]].isnull().sum())

