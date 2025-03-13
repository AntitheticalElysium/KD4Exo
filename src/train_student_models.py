import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import time
from train_teacher_models import load_and_split_data, MLP
from visualization import display_biggest_variations, display_habitability_rankings

DATA_PATH = "../data/processed/exoplanet_data_clean.csv"
MODEL_PATH = "../models"
TEMPERATURE = 2.0  # Softmax temperature for distillation

# Student model
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

def distill_knowledge(X_train, X_test, y_train, y_test, pl_names, alpha=0.5, epochs=1000, patience=50):
    """
    Distill knowledge from trained teacher model to student model.
    """
    print("Starting knowledge distillation...")
    start_time = time.time()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Convert data to tensors
    X_train_tensor = torch.FloatTensor(X_train.values).to(device)
    X_test_tensor = torch.FloatTensor(X_test.values).to(device)
    y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1).to(device)
    y_test_tensor = torch.FloatTensor(y_test.values).unsqueeze(1).to(device)
    
    # Load teacher model
    input_size = X_train.shape[1]
    teacher_model = MLP(input_size, hidden_sizes=[256, 512, 512, 256, 128]).to(device)
    teacher_model.load_state_dict(torch.load(f"{MODEL_PATH}/mlp_model.pt", map_location=device))
    teacher_model.eval()  # Set to evaluation mode
    
    student_model = ShallowNN(input_size).to(device)
    
    # Loss functions and optimizer
    criterion_hard = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.005, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    
    # Early stopping variables
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_state = None
    
    # Get teacher's logits for training data
    with torch.no_grad():
        teacher_logits = teacher_model(X_train_tensor)
    
    # Training loop
    for epoch in range(epochs):
        student_model.train()
        optimizer.zero_grad()
        
        # Forward pass through student model
        student_logits = student_model(X_train_tensor)
        
        # Hard loss - comparing with actual labels
        hard_loss = criterion_hard(student_logits, y_train_tensor)
        # Soft loss - comparing with teacher's logits (MSE on logits)
        soft_loss = F.mse_loss(student_logits, teacher_logits)
        # Weighted loss - combines hard and soft losses
        loss = (1 - alpha) * hard_loss + alpha * soft_loss
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        # Validation
        student_model.eval()
        with torch.no_grad():
            val_outputs = student_model(X_test_tensor)
            val_loss = criterion_hard(val_outputs, y_test_tensor)
            
            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Hard Loss: {hard_loss.item():.6f}, '
                      f'Soft Loss: {soft_loss.item():.6f}, Val Loss: {val_loss.item():.6f}')
            
            # Check early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                best_model_state = student_model.state_dict().copy()
            else:
                early_stop_counter += 1
            
            if early_stop_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
        
        # Update learning rate
        scheduler.step(val_loss)
    
    if best_model_state is not None:
        student_model.load_state_dict(best_model_state)
    
    # Final evaluation
    student_model.eval()
    with torch.no_grad():
        student_preds = torch.sigmoid(student_model(X_test_tensor)).cpu().numpy().flatten()
        student_preds = (student_preds >= 0.5).astype(int)  # Convert to binary
        student_binary_preds = (student_preds >= 0.5).astype(int)
        
        teacher_preds = torch.sigmoid(teacher_model(X_test_tensor)).cpu().numpy().flatten()
        teacher_preds = (teacher_preds >= 0.5).astype(int)  # Convert to binary
        teacher_binary_preds = (teacher_preds >= 0.5).astype(int)
        
        # Calculate metrics
        student_mse = mean_squared_error(y_test, student_binary_preds)
        student_r2 = r2_score(y_test, student_binary_preds)
        teacher_mse = mean_squared_error(y_test, teacher_binary_preds)
        teacher_r2 = r2_score(y_test, teacher_binary_preds)
    
    print(f"Training time: {time.time() - start_time:.2f}s")
    print(f"Teacher - MSE: {teacher_mse:.10f}, R²: {teacher_r2:.10f}")
    print(f"Student - MSE: {student_mse:.10f}, R²: {student_r2:.10f}")
    
    agreement = np.mean(teacher_binary_preds == student_binary_preds) * 100
    print(f"Teacher-Student agreement: {agreement:.2f}%")

    display_biggest_variations(y_test, student_preds, pl_names)
    display_habitability_rankings(y_test, student_preds, pl_names)
    
    torch.save(student_model.state_dict(), f"{MODEL_PATH}/shallow_nn_distilled.pt")
    return student_model

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, pl_names = load_and_split_data()
    
    # Perform knowledge distillation
    student_model = distill_knowledge(X_train, X_test, y_train, y_test, pl_names, alpha=0.7)
