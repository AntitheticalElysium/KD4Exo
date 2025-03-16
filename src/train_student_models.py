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

def train_baseline_student_model(X_train, X_test, y_train, y_test, pl_names, epochs=1000, patience=50):
    """
    Trains a baseline student NN model.
    """
    print("Training ShallowNN...")
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Convert to tensors and move to GPU if available
    X_train_tensor = torch.FloatTensor(X_train.values).to(device)
    X_test_tensor = torch.FloatTensor(X_test.values).to(device)
    y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1).to(device)
    y_test_tensor = torch.FloatTensor(y_test.values).unsqueeze(1).to(device)

    # Create model and move it to GPU
    input_size = X_train.shape[1]
    model = ShallowNN(input_size).to(device)

    # Loss, optimizer, and scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Early stopping variables
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_state = None

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        # Backward pass and optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Prevent gradient explosion
        optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test_tensor)
            val_loss = criterion(val_outputs, y_test_tensor)

            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}')

            # Check early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stop_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                early_stop_counter += 1

            if early_stop_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        y_pred = torch.sigmoid(model(X_test_tensor)).detach().cpu().numpy().flatten()
        y_pred = (y_pred >= 0.5).astype(int)  # Convert to binary
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

    print(f"ShallowNN - MSE: {mse:.10f}, R²: {r2:.10f}, Time: {time.time() - start_time:.2f}s")
    display_biggest_variations(y_test, y_pred, pl_names)
    display_habitability_rankings(y_test, y_pred, pl_names)

    torch.save(model.state_dict(), f"{MODEL_PATH}/baseline_shallow_nn_model.pt")
    return model, y_pred
   

def logit_distillation(X_train, X_test, y_train, y_test, pl_names, alpha=0.25, epochs=1000, patience=50):
    """
    Implement logit knowledge distillation from teacher to student model.
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
        # Soft loss - comparing with teacher's logits
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
    
    torch.save(student_model.state_dict(), f"{MODEL_PATH}/distilled_shallow_nn_model.pt")
    return student_model

def relational_distillation(X_train, X_test, y_train, y_test, pl_names, alpha=0.5, beta=0.3, epochs=1000, patience=50):
    """
    Implement relational knowledge distillation from teacher to student model.
    """
    print("Starting relational knowledge distillation...")
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
    
    # Create student model
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
    
    # Compute pairwise distances for relational knowledge
    def compute_relation_matrix(logits):
        batch_size = logits.size(0)
        # Compute pairwise distances between all samples
        logits_expanded = logits.expand(batch_size, batch_size, -1)
        logits_expanded_t = logits_expanded.transpose(0, 1)
        # L2 distance between pairs of logits
        relations = ((logits_expanded - logits_expanded_t)**2).sum(dim=2)
        return relations
    
    teacher_relations = compute_relation_matrix(teacher_logits)
    
    # Training loop
    for epoch in range(epochs):
        student_model.train()
        optimizer.zero_grad()
        
        # Forward pass through student model
        student_logits = student_model(X_train_tensor)
        
        # Compute student relations
        student_relations = compute_relation_matrix(student_logits)
        
        # Compute losses
        hard_loss = criterion_hard(student_logits, y_train_tensor)
        soft_loss = F.mse_loss(student_logits, teacher_logits)
        relation_loss = F.mse_loss(student_relations, teacher_relations)
        
        # Weighted combined loss
        loss = (1 - alpha - beta) * hard_loss + alpha * soft_loss + beta * relation_loss
        
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
                      f'Soft Loss: {soft_loss.item():.6f}, Relation Loss: {relation_loss.item():.6f}, '
                      f'Val Loss: {val_loss.item():.6f}')
            
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
        student_preds = (student_preds >= 0.5).astype(int)
        
        teacher_preds = torch.sigmoid(teacher_model(X_test_tensor)).cpu().numpy().flatten()
        teacher_preds = (teacher_preds >= 0.5).astype(int)
        
        # Calculate metrics
        student_mse = mean_squared_error(y_test, student_preds)
        student_r2 = r2_score(y_test, student_preds)
        teacher_mse = mean_squared_error(y_test, teacher_preds)
        teacher_r2 = r2_score(y_test, teacher_preds)
    
    print(f"Training time: {time.time() - start_time:.2f}s")
    print(f"Teacher - MSE: {teacher_mse:.10f}, R²: {teacher_r2:.10f}")
    print(f"Student - MSE: {student_mse:.10f}, R²: {student_r2:.10f}")
    
    agreement = np.mean(teacher_preds == student_preds) * 100
    print(f"Teacher-Student agreement: {agreement:.2f}%")

    display_biggest_variations(y_test, student_preds, pl_names)
    display_habitability_rankings(y_test, student_preds, pl_names)
    
    torch.save(student_model.state_dict(), f"{MODEL_PATH}/relational_distilled_nn_model.pt")
    return student_model

def feature_distillation(X_train, X_test, y_train, y_test, pl_names, alpha=0.5, gamma=0.3, epochs=1000, patience=50):
    """
    Implement feature-based knowledge distillation from teacher to student model.
    """
    print("Starting feature knowledge distillation...")
    start_time = time.time()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Convert data to tensors
    X_train_tensor = torch.FloatTensor(X_train.values).to(device)
    X_test_tensor = torch.FloatTensor(X_test.values).to(device)
    y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1).to(device)
    y_test_tensor = torch.FloatTensor(y_test.values).unsqueeze(1).to(device)
    
    # Load teacher model with hooks to extract intermediate features
    input_size = X_train.shape[1]
    teacher_model = MLP(input_size, hidden_sizes=[256, 512, 512, 256, 128]).to(device)
    teacher_model.load_state_dict(torch.load(f"{MODEL_PATH}/mlp_model.pt", map_location=device))
    teacher_model.eval()
    
    # Student model with feature extraction hooks
    class StudentModelWithFeatures(nn.Module):
        def __init__(self, input_dim):
            super(StudentModelWithFeatures, self).__init__()
            self.layer1 = nn.Linear(input_dim, 64)
            self.relu1 = nn.ReLU()
            self.bn1 = nn.BatchNorm1d(64)
            
            self.layer2 = nn.Linear(64, 32)
            self.relu2 = nn.ReLU()
            self.bn2 = nn.BatchNorm1d(32)
            
            self.layer3 = nn.Linear(32, 1)
            
            self.features = {}
        
        def forward(self, x):
            # First layer
            x = self.layer1(x)
            x = self.relu1(x)
            x = self.bn1(x)
            self.features['layer1'] = x.clone()
            
            # Second layer
            x = self.layer2(x)
            x = self.relu2(x)
            x = self.bn2(x)
            self.features['layer2'] = x.clone()
            
            # Output layer
            x = self.layer3(x)
            
            return x
    
    student_model = StudentModelWithFeatures(input_size).to(device)
    
    # Register hooks for teacher model to extract features
    teacher_features = {}
    
    def get_teacher_features(name):
        def hook(model, input, output):
            teacher_features[name] = output.detach()
        return hook
    
    # Apply hooks to teacher model layers
    teacher_model.model[1].register_forward_hook(get_teacher_features('layer1'))
    teacher_model.model[4].register_forward_hook(get_teacher_features('layer2')) 
    
    # Loss functions and optimizer
    criterion_hard = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.005, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    
    # Early stopping variables
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_state = None
    
    # Adaptation layer for feature matching
    adaptation_layer1 = nn.Linear(256, 64).to(device)
    adaptation_layer2 = nn.Linear(512, 32).to(device)
    
    # Get teacher's logits and features for training data
    with torch.no_grad():
        teacher_logits = teacher_model(X_train_tensor)
    
    # Training loop
    for epoch in range(epochs):
        student_model.train()
        optimizer.zero_grad()
        
        # Forward pass through student model
        student_logits = student_model(X_train_tensor)
        
        # Compute losses
        hard_loss = criterion_hard(student_logits, y_train_tensor)
        soft_loss = F.mse_loss(student_logits, teacher_logits)
        
        # Feature distillation losses
        feature_loss1 = F.mse_loss(student_model.features['layer1'], 
                                  adaptation_layer1(teacher_features['layer1']))
        feature_loss2 = F.mse_loss(student_model.features['layer2'], 
                                  adaptation_layer2(teacher_features['layer2']))
        feature_loss = feature_loss1 + feature_loss2
        
        # Weighted combined loss
        loss = (1 - alpha - gamma) * hard_loss + alpha * soft_loss + gamma * feature_loss
        
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
                      f'Soft Loss: {soft_loss.item():.6f}, Feature Loss: {feature_loss.item():.6f}, '
                      f'Val Loss: {val_loss.item():.6f}')
            
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
        student_preds = (student_preds >= 0.5).astype(int)
        
        teacher_preds = torch.sigmoid(teacher_model(X_test_tensor)).cpu().numpy().flatten()
        teacher_preds = (teacher_preds >= 0.5).astype(int)
        
        # Calculate metrics
        student_mse = mean_squared_error(y_test, student_preds)
        student_r2 = r2_score(y_test, student_preds)
        teacher_mse = mean_squared_error(y_test, teacher_preds)
        teacher_r2 = r2_score(y_test, teacher_preds)
    
    print(f"Training time: {time.time() - start_time:.2f}s")
    print(f"Teacher - MSE: {teacher_mse:.10f}, R²: {teacher_r2:.10f}")
    print(f"Student - MSE: {student_mse:.10f}, R²: {student_r2:.10f}")

    agreement = np.mean(teacher_preds == student_preds) * 100
    print(f"Teacher-Student agreement: {agreement:.2f}%")

    display_biggest_variations(y_test, student_preds, pl_names)
    display_habitability_rankings(y_test, student_preds, pl_names)

    torch.save(student_model.state_dict(), f"{MODEL_PATH}/feature_distilled_nn_model.pt")
    return student_model

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, pl_names = load_and_split_data()

    # Train baseline student model 
    student_model = train_baseline_student_model(X_train, X_test, y_train, y_test, pl_names)
    
    # Perform knowledge distillation
    student_model = logit_distillation(X_train, X_test, y_train, y_test, pl_names)
    # GPU LOL
    # student_model = relational_distillation(X_train, X_test, y_train, y_test, pl_names)
    student_model = feature_distillation(X_train, X_test, y_train, y_test, pl_names)
