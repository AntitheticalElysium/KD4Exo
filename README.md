# KD4Exo: Knowledge Distillation for Exoplanet Classification

A machine learning pipeline for predicting exoplanet habitability using knowledge distillation techniques.

## Project Overview

KD4Exo builds a sophisticated pipeline for analyzing exoplanet data to:

1. Predict the habitability of exoplanets based on physical properties
2. Apply knowledge distillation to create efficient, lightweight models
3. Compare various distillation approaches for optimal performance

The project combines physics-based feature engineering with state-of-the-art ML techniques to build accurate and efficient models.

## Features

- **Data Preprocessing**: Handles missing values, outliers, and feature engineering
- **Physics-Based Modeling**: Calculates escape velocity, atmospheric retention, and habitable zone positions
- **Habitability Scoring**: Multi-factor scoring system for comprehensive habitability assessment
- **Synthetic Data Generation**: Creates balanced datasets with realistic planet properties
- **Teacher-Student Architecture**: Implements knowledge distillation techniques:
  - Standard logit distillation
  - Relational distillation (inter-sample relationships)
  - Feature-based distillation (intermediate representations)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/KD4Exo.git
cd KD4Exo

# Install dependencies
pip install -r requirements.txt
```

## Configuration

The project uses a `config.yaml` file to centralize all configuration parameters:

- Physical constants used in calculations
- Habitable zone thresholds
- Gas retention parameters
- Data preprocessing options
- Model hyperparameters
- Training settings
- Habitability classification criteria

To customize the pipeline, edit the parameters in `config.yaml` rather than modifying the source code. This simplifies experimentation and parameter tuning.

## Usage

### Data Preparation

```bash
# Process the raw exoplanet data, calculate habitability scores
python src/core_preprocessing.py
```

### Training Teacher Models

```bash
# Train complex teacher models (MLP, XGBoost, Random Forest)
python src/train_teacher_models.py
```

### Knowledge Distillation

```bash
# Train student models with knowledge distillation
python src/train_student_models.py
```

## Methodology

### 1. Feature Engineering

The pipeline calculates several physics-based features:
- Atmospheric retention probability based on escape velocity
- Habitable zone positioning relative to the host star
- Temperature modeling for liquid water assessment
- Magnetic field protection from stellar radiation

### 2. Teacher Models

Complex models are trained to achieve high accuracy:
- Multi-layer perceptron
- Random Forest
- XGBoost

### 3. Student Models

Lightweight models are trained using knowledge distillation:
- Shallow neural networks 
- Various distillation techniques to transfer knowledge

### 4. Evaluation

Models are evaluated based on:
- Mean squared error (MSE)
- R² score
- Teacher-student agreement percentage
- Habitability rankings comparison

## Results

Student models achieve comparable performance to teacher models while being significantly smaller and faster:
- Logit distillation: ~95% agreement with teacher
- Feature distillation: Best balance of size and accuracy
- Relational distillation: Improved generalization on edge cases

## Acknowledgments

- NASA Exoplanet Archive for the dataset
- Knowledge distillation techniques inspired by Hinton et al. (2015)
