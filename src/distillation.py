#1. **Sequential distillation**: Start with a large, complex model (teacher) trained on your original or sparse-autoencoded features, then distill to progressively smaller models.
# This can reveal how much knowledge is retained at each compression level.
#
#2. **Feature-focused distillation**: Rather than just matching the final output predictions, force the student model to match intermediate representations of key features. 
# This could help preserve physical meaning in your habitable exoplanet predictions.
#
#3. **Ensemble teacher approach**: Train multiple teacher models with different architectures (e.g., deep neural networks, gradient boosting, random forests) that each 
# excel at different aspects of the prediction task, then distill their collective knowledge into a single student model.
#
#4. **Temperature-scaled distillation**: Experiment with different temperature parameters in the softmax to control how much of the teacher's uncertainty is transferred to 
# the student. This is especially relevant for borderline habitable/non-habitable cases.
#
#5. **Attention distillation**: If using attention-based models for your teacher, have the student also learn the attention patterns. This could help it focus on the 
# same critical planetary features that drive habitability.
#
#6. **Physics-informed distillation**: Add regularization terms that ensure your distilled model respects physical constraints related to planetary habitability, rather 
# than just matching the teacher's predictions blindly.
#
#7. **Data augmentation during distillation**: Generate synthetic exoplanet data based on your physical models to expand the training set, helping the student model 
# generalize better.
#
#8. **Dual-objective training**: Have your student model simultaneously learn from the teacher and directly from the raw data, potentially allowing it to correct some 
# teacher model errors.
#
#For your specific exoplanet context, I'd be particularly interested in comparing how well different distillation approaches preserve the physical relationships 
# between features that you've carefully modeled in your preprocessing steps.
import numpy as np
import pandas as pd
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
import torch
from torch import nn 
from visualization import display_biggest_variations

class ShallowNN(nn.Module):
    def __init__(self, input_dim):
        super(ShallowNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.model(x)
