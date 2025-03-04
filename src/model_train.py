import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset():
    df = pd.read_csv('../data/processed/exoplanet_data_clean.csv')
    print(df.shape)
