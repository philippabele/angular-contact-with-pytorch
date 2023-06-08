from torch import nn
import os

# Global Configuration
config = {
    "folder_path": 'data/ShaftBearing/lifetime-categorization/',
    "input_size": 2,
    "output_size": 10,
    "hidden_size": 200,
    "activation_function": nn.ReLU(),
    "learning_rate": 0.001,
    "epochs": 100,
    "batch_size": 64,
    "prefer_cuda": False,
    "test_size": 0.2,
    "validation_size": 0.1,
}

# Paths
config["train_data_path"] = os.path.join(config["folder_path"], 'train-lt.csv')
config["val_data_path"] = os.path.join(config["folder_path"], 'val-lt.csv')
config["test_data_path"] = os.path.join(config["folder_path"], 'test-lt.csv')
config["model_path"] = os.path.join(config["folder_path"], 'model-lt.pt')
config["scaler_path"] = os.path.join(config["folder_path"], 'scaler-lt.pt')
config["bins_path"] = os.path.join(config["folder_path"], 'bins-lt.joblib')
config["lambda_path"] = os.path.join(config["folder_path"], 'lambda-lt.joblib')
