from torch import nn
import os

# Global Configuration
config = {
    "folder_path": 'data/shaftbearing/categorize_lt/',
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
config["train_data_path"] = os.path.join(config["folder_path"], 'train.csv')
config["val_data_path"] = os.path.join(config["folder_path"], 'val.csv')
config["test_data_path"] = os.path.join(config["folder_path"], 'test.csv')
config["model_path"] = os.path.join(config["folder_path"], 'model.pt')
config["scaler_path"] = os.path.join(config["folder_path"], 'scaler.pt')
config["bins_path"] = os.path.join(config["folder_path"], 'bins.joblib')
config["lambda_path"] = os.path.join(config["folder_path"], 'lambda.joblib')
config["dataset_path"] = os.path.join(config["folder_path"], 'data_gen.csv')
