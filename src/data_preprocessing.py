import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import boxcox
import math
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

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

def compute_habitability_index(df):
    """
    Computes a habitability index based on how close each planet's properties are to ideal habitable conditions.
    The index is continuous, meaning values closer to 1 are more habitable.
    """
    # Define optimal values based on Earth's conditions and astrophysical research
    optimal_values = {
        'pl_rade': 1.0,      # Earth-like radius
        'pl_bmasse': 1.0,    # Earth mass
        'pl_orbsmax': 1.0,   # 1 AU (Habitable zone reference point)
        'pl_orbper': 365.25, # Earth year
        'pl_orbeccen': 0.0167, # Earth's eccentricity
        'st_teff': 5778,     # Sun's effective temperature in Kelvin
        'st_rad': 1.0,       # Sun's radius
        'st_lum': 1.0,       # Solar luminosity
        'st_mass': 1.0,      # Solar mass
        'st_met': 0.0,       # Solar metallicity (dex)
        'sy_snum': 1,        # Single star system
        'sy_pnum': 1,        # No extreme planetary interactions
        'sy_dist': 10,       # Close enough for good observation (arbitrary, not influencing habitability itself)
        'pl_dens': 5.51      # Earth's density in g/cm^3
    }
    
    # Normalize the impact of each feature by scaling distances from optimal values
    scaler = MinMaxScaler()
    scaled_df = df.copy()
    
    for col, opt_value in optimal_values.items():
        if col in df.columns:
            scaled_df[col] = 1 - np.abs(df[col] - opt_value) / (df[col].max() - df[col].min())
    
    # Compute habitability index as the mean of all feature scores
    df['habitability_index'] = scaled_df[optimal_values.keys()].mean(axis=1)
    
    return df

def preprocess_exoplanet_data(file_path):
    """
    Loads and preprocesses the exoplanet dataset.
    Drops columns with more than 25% missing values.
    Fills remaining missing values with the median.
    """
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Columns to retain
    retained_columns = [
        'pl_rade', 'pl_bmasse', 'pl_orbsmax', 'pl_orbper', 'pl_orbeccen',
        'pl_insol', 'pl_eqt', 'st_spectype', 'st_teff', 'st_rad', 'st_lum',
        'st_mass', 'st_met', 'sy_snum', 'sy_pnum', 'sy_dist', 'pl_dens', 'pl_trandep', 'pl_orbincl'
    ]
    df = df[retained_columns]
    
    #print("Missing values before preprocessing:")
    #print(df.isnull().sum())
    
    # Drop columns with more than 25% missing values
    columns_to_drop = ['st_spectype', 'pl_insol', 'pl_eqt', 'pl_trandep', 'pl_orbincl']
    df.drop(columns=columns_to_drop, inplace=True)
    
    # Fill remaining missing values with median
    num_features = ['pl_rade', 'pl_bmasse', 'pl_orbsmax', 'pl_orbper', 'pl_orbeccen',
                    'st_teff', 'st_rad', 'st_lum', 'st_mass', 'st_met', 'sy_dist', 'pl_dens']
    df[num_features] = df[num_features].fillna(df[num_features].median())
    
    #print(df.isnull().sum())

    df = compute_habitability_index(df)
    
    return df


def plot_feature_distributions(df):
    num_features = df.select_dtypes(include=['float64', 'int64']).columns
    num_features_count = len(num_features)

    # Calculate total plot dimensions
    rows = math.ceil(num_features_count / 3)
    cols = min(3, num_features_count)  
    plt.figure(figsize=(5 * cols, 4 * rows))

    for i, col in enumerate(num_features, 1):
        plt.subplot(rows, cols, i)
        sns.histplot(df[col], bins=50, kde=True)
        plt.title(col)

    plt.tight_layout()
    plt.show()

#def handle_correlated_features(df, threshold=0.9):
    #plt.figure(figsize=(12, 8))
    #sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    #plt.title("Feature Correlation Heatmap")
    #plt.show()

def handle_skewness(df):
    exclude_cols = {'sy_snum', 'sy_pnum', 'pl_orbccen'} # Skewness is natural here
    skewness = df.skew()

    # Print features with high skewness (above abs(0.75))
    skewed_cols = [col for col in skewness[abs(skewness) > 0.75].index if col not in exclude_cols]
    print(skewness[skewed_cols])

    for col in skewed_cols:
        df[col], _ = boxcox(df[col] + 1)

    print(df.skew()[skewed_cols])

def handle_outliers(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
    print(outliers[outliers > 0])  # Features with outliers

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df = df.clip(lower=lower_bound, upper=upper_bound, axis=1)

def handle_scaling(df):
    scaler = StandardScaler()
    # Scale every col
    df.loc[:, :] = scaler.fit_transform(df)

def split_data(df):
    X = df.drop(columns=["target_column"])  # Replace with actual target column
    y = df["target_column"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

df = preprocess_exoplanet_data("../data/raw/exoplanet_data.csv")
print(df['habitability_index'].mean())
#handle_correlated_features(df)
#handle_outliers(df)
#handle_skewness(df)
#handle_scaling(df)

#df.to_csv('../data/processed/exoplanet_data_clean.csv')

#plot_feature_distributions(df)
