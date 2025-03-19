import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from visualization import display_biggest_variations, display_habitability_rankings
import time
import yaml
from train_teacher_models import load_and_split_data, MLP
from core_preprocessing import load_config

# Student model
class ShallowNN(nn.Module):
    def __init__(self, input_dim, hidden_sizes=None):
        super(ShallowNN, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(current_dim, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_size))
            current_dim = hidden_size
        
        layers.append(nn.Linear(current_dim, 1))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


# Student model with feature extraction hooks
class StudentModelWithFeatures(nn.Module):
    def __init__(self, input_dim, hidden_sizes=None):
        super(StudentModelWithFeatures, self).__init__()

        self.layer1 = nn.Linear(input_dim, hidden_sizes[0])
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(hidden_sizes[0])
            
        self.layer2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(hidden_sizes[1])
            
        self.layer3 = nn.Linear(hidden_sizes[1], 1)
            
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
    

def train_baseline_student_model(X_train, X_test, y_train, y_test, pl_names):
    """
    Trains a baseline student NN model.
    """
    print("Training ShallowNN...")
    start_time = time.time()
    
    # Get config parameters
    student_config = config["training"]["student"]["shallow_nn"]
    hidden_sizes = student_config["hidden_sizes"]
    learning_rate = student_config["learning_rate"]
    weight_decay = student_config["weight_decay"]
    epochs = student_config["epochs"]
    patience = student_config["patience"]
    batch_size = student_config["batch_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Convert to tensors and move to GPU if available
    X_train_tensor = torch.FloatTensor(X_train.values).to(device)
    X_test_tensor = torch.FloatTensor(X_test.values).to(device)
    y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1).to(device)
    y_test_tensor = torch.FloatTensor(y_test.values).unsqueeze(1).to(device)

    # Create model and move it to GPU
    input_size = X_train.shape[1]
    model = ShallowNN(input_size, hidden_sizes).to(device)

    # Loss, optimizer, and scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
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

    torch.save(model.state_dict(), f"{config['data']['models_path']}/baseline_nn_model.pt")
    return model, y_pred
   

def logit_distillation(X_train, X_test, y_train, y_test, pl_names):
    """
    Implement logit knowledge distillation from teacher to student model.
    """
    print("Starting knowledge distillation...")
    start_time = time.time()
    torch.cuda.empty_cache()
    
    # Get config parameters
    logit_config = config["training"]["student"]["distillation"]["logit"]
    student_config = config["training"]["student"]["shallow_nn"]
    alpha = logit_config["alpha"]
    temperature = logit_config["temperature"]
    hidden_sizes = student_config["hidden_sizes"]
    learning_rate = student_config["learning_rate"]
    weight_decay = student_config["weight_decay"]
    epochs = student_config["epochs"]
    patience = student_config["patience"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Convert data to tensors
    X_train_tensor = torch.FloatTensor(X_train.values).to(device)
    X_test_tensor = torch.FloatTensor(X_test.values).to(device)
    y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1).to(device)
    y_test_tensor = torch.FloatTensor(y_test.values).unsqueeze(1).to(device)
    
    # Load teacher model
    input_size = X_train.shape[1]
    teacher_config = config["training"]["teacher"]["mlp"]
    teacher_hidden_sizes = teacher_config["hidden_sizes"]
    teacher_dropout_rate = teacher_config["dropout_rate"]
    teacher_model = MLP(input_size, hidden_sizes=teacher_hidden_sizes, dropout_rate=teacher_dropout_rate).to(device)
    teacher_model.load_state_dict(torch.load(f"{config['data']['models_path']}/mlp_model.pt", map_location=device))
    teacher_model.eval()  # Set to evaluation mode
    
    student_model = ShallowNN(input_size, hidden_sizes).to(device)
    
    # Loss functions and optimizer
    criterion_hard = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
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
        # Apply temperature scaling for softer probabilities
        soft_student_logits = student_logits / temperature
        soft_teacher_logits = teacher_logits / temperature
        soft_loss = F.mse_loss(soft_student_logits, soft_teacher_logits)
        
        # Weighted loss - combines hard and soft losses
        loss = (1 - alpha) * hard_loss + alpha * soft_loss * (temperature ** 2)
        
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
        student_binary_preds = (student_preds >= 0.5).astype(int)  # Convert to binary
        
        teacher_preds = torch.sigmoid(teacher_model(X_test_tensor)).cpu().numpy().flatten()
        teacher_binary_preds = (teacher_preds >= 0.5).astype(int)  # Convert to binary
        
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

    display_biggest_variations(y_test, student_binary_preds, pl_names)
    display_habitability_rankings(y_test, student_binary_preds, pl_names)
    
    torch.save(student_model.state_dict(), f"{config['data']['models_path']}/logit_distilled_nn_model.pt")
    return student_model

def relational_distillation(X_train, X_test, y_train, y_test, pl_names):
    """
    Implement relational knowledge distillation from teacher to student model.
    """
    print("Starting relational knowledge distillation...")
    start_time = time.time()
    torch.cuda.empty_cache()
    
    # Get config parameters
    rel_config = config["training"]["student"]["distillation"]["relational"]
    student_config = config["training"]["student"]["shallow_nn"]
    alpha = rel_config["alpha"]
    beta = rel_config["beta"]
    temperature = rel_config["temperature"]
    hidden_sizes = student_config["hidden_sizes"]
    learning_rate = student_config["learning_rate"]
    weight_decay = student_config["weight_decay"]
    epochs = student_config["epochs"]
    patience = student_config["patience"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Convert data to tensors
    X_train_tensor = torch.FloatTensor(X_train.values).to(device)
    X_test_tensor = torch.FloatTensor(X_test.values).to(device)
    y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1).to(device)
    y_test_tensor = torch.FloatTensor(y_test.values).unsqueeze(1).to(device)
    
    # Load teacher model
    input_size = X_train.shape[1]
    teacher_config = config["training"]["teacher"]["mlp"]
    teacher_hidden_sizes = teacher_config["hidden_sizes"]
    teacher_dropout_rate = teacher_config["dropout_rate"]
    teacher_model = MLP(input_size, hidden_sizes=teacher_hidden_sizes, dropout_rate=teacher_dropout_rate).to(device)
    teacher_model.load_state_dict(torch.load(f"{config['data']['models_path']}/mlp_model.pt", map_location=device))
    teacher_model.eval()  # Set to evaluation mode
    
    # Create student model
    student_model = ShallowNN(input_size, hidden_sizes).to(device)
    
    # Loss functions and optimizer
    criterion_hard = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
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
        
        # Apply temperature scaling for softer probabilities
        soft_student_logits = student_logits / temperature
        soft_teacher_logits = teacher_logits / temperature
        
        # Compute losses
        hard_loss = criterion_hard(student_logits, y_train_tensor)
        soft_loss = F.mse_loss(soft_student_logits, soft_teacher_logits)
        relation_loss = F.mse_loss(student_relations, teacher_relations)
        
        # Weighted combined loss
        loss = (1 - alpha - beta) * hard_loss + alpha * soft_loss * (temperature ** 2) + beta * relation_loss
        
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
        student_binary_preds = (student_preds >= 0.5).astype(int)
        
        teacher_preds = torch.sigmoid(teacher_model(X_test_tensor)).cpu().numpy().flatten()
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

    display_biggest_variations(y_test, student_binary_preds, pl_names)
    display_habitability_rankings(y_test, student_binary_preds, pl_names)
    
    torch.save(student_model.state_dict(), f"{config['data']['models_path']}/relational_distilled_nn_model.pt")
    return student_model

def feature_distillation(X_train, X_test, y_train, y_test, pl_names):
    """
    Implement feature-based knowledge distillation from teacher to student model.
    """
    print("Starting feature knowledge distillation...")
    start_time = time.time()
    torch.cuda.empty_cache()
    
    # Get config parameters
    feat_config = config["training"]["student"]["distillation"]["feature"]
    student_config = config["training"]["student"]["shallow_nn"]
    alpha = feat_config["alpha"]
    beta = feat_config["beta"]
    feature_weights = feat_config["feature_weights"]
    hidden_sizes = student_config["hidden_sizes"]
    learning_rate = student_config["learning_rate"]
    weight_decay = student_config["weight_decay"]
    epochs = student_config["epochs"]
    patience = student_config["patience"]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Convert data to tensors
    X_train_tensor = torch.FloatTensor(X_train.values).to(device)
    X_test_tensor = torch.FloatTensor(X_test.values).to(device)
    y_train_tensor = torch.FloatTensor(y_train.values).unsqueeze(1).to(device)
    y_test_tensor = torch.FloatTensor(y_test.values).unsqueeze(1).to(device)
    
    # Load teacher model with hooks to extract intermediate features
    input_size = X_train.shape[1]
    teacher_config = config["training"]["teacher"]["mlp"]
    teacher_hidden_sizes = teacher_config["hidden_sizes"]
    teacher_dropout_rate = teacher_config["dropout_rate"]
    teacher_model = MLP(input_size, hidden_sizes=teacher_hidden_sizes, dropout_rate=teacher_dropout_rate).to(device)
    teacher_model.load_state_dict(torch.load(f"{config['data']['models_path']}/mlp_model.pt", map_location=device))
    teacher_model.eval()
    
    student_model = StudentModelWithFeatures(input_size, hidden_sizes).to(device)
    
    # Register hooks for teacher model to extract features
    teacher_features = {}
    
    def get_teacher_features(name):
        def hook(model, input, output):
            teacher_features[name] = output.detach()
        return hook
    
    # Apply hooks to teacher model layers
    # This assumes the teacher model has specific structure - adjust based on actual teacher model
    teacher_model.model[1].register_forward_hook(get_teacher_features('layer1'))
    teacher_model.model[4].register_forward_hook(get_teacher_features('layer2')) 
    
    # Loss functions and optimizer
    criterion_hard = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
    
    # Early stopping variables
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_state = None
    
    # Adaptation layer for feature matching
    adaptation_layer1 = nn.Linear(teacher_hidden_sizes[0], hidden_sizes[0]).to(device)
    adaptation_layer2 = nn.Linear(teacher_hidden_sizes[1], hidden_sizes[1]).to(device)
    
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
        feature_loss = feature_weights[0] * feature_loss1 + feature_weights[1] * feature_loss2
        
        # Weighted combined loss
        loss = (1 - alpha - beta) * hard_loss + alpha * soft_loss + beta * feature_loss
        
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
        student_binary_preds = (student_preds >= 0.5).astype(int)
        
        teacher_preds = torch.sigmoid(teacher_model(X_test_tensor)).cpu().numpy().flatten()
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

    display_biggest_variations(y_test, student_binary_preds, pl_names)
    display_habitability_rankings(y_test, student_binary_preds, pl_names)

    torch.save(student_model.state_dict(), f"{config['data']['models_path']}/feature_distilled_nn_model.pt")
    return student_model

if __name__ == "__main__":
    config = load_config()
    X_train, X_test, y_train, y_test, pl_names = load_and_split_data(config)

    # Train baseline student model 
    train_baseline_student_model(X_train, X_test, y_train, y_test, pl_names)
    
    # Perform knowledge distillation
    logit_distillation(X_train, X_test, y_train, y_test, pl_names)
    # relational_distillation(X_train, X_test, y_train, y_test, pl_names) # No memory AHH
    feature_distillation(X_train, X_test, y_train, y_test, pl_names)
