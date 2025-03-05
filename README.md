# KD4Exo
Knowledge Distillation for Exoplanet Regression (KD4Exo)

## Project Overview

This project aims to develop a machine learning pipeline for predicting the habitability of exoplanets using data preprocessing, feature engineering, and machine learning techniques. Utilizing the NASA Exoplanet dataset, the primary goal is to create a robust habitability prediction model and explore knowledge distillation across multiple teacher models.

## Key Objectives

- Preprocess and engineer features from NASA Exoplanet data
- Develop machine learning models for habitability prediction
- Implement knowledge distillation using multiple teacher models

## Technology Stack

- Python
- XGBoost
- Scikit-learn
- Pandas
- Numpy

## Knowledge Distillation Approach

- Teacher Models: 
  - XGBoost Regressor
  - Additional machine learning models to be explored
- Goal: Transfer knowledge to a more compact student model while maintaining predictive performance

## Getting Started

```bash
# Clone repository
git clone https://github.com/yourusername/exoplanet-habitability.git

# Install dependencies
pip install -r requirements.txt

# Run preprocessing
python src/data_preprocessing.py

# Train models
python src/model_train.py
```
