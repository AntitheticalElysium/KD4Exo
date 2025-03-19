from train_teacher_models import train_mlp train_xgboost, train_random_forest, load_and_split_data
from core_preprocessing import load_config 


def optimize_mlp(X_train, X_test, y_train, y_test, pl_names, config):
    # run the training function until the model has produced a good resulot

    # Get hyperparameters
    hidden_sizes = mlp_config['hidden_sizes']
    dropout_rate = mlp_config['dropout_rate']
    lr = mlp_config['learning_rate']
    weight_decay = mlp_config['weight_decay']
    epochs = mlp_config['epochs']
    patience = mlp_config['patience']
    batch_size = mlp_config['batch_size']

    # Optimize hyperparameters

    # write the changes


if __name__ == "__main__":
    config = load_config()
    X_train, X_test, y_train, y_test, pl_names = load_and_split_data(config)

