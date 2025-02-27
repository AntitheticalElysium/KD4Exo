import pandas as pd

# Load the CSV file
df = pd.read_csv("../data/raw/exoplanet_data.csv")

print(df.isnull().sum())
