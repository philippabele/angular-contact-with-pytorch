import numpy as np
from torch import nn
import torch
from torch.utils.data import DataLoader
from itertools import product
from torch.utils.tensorboard import SummaryWriter
from generate_dataset import generate_dataset
from split_dataset import split_and_transform_dataset
from train_model import split_train_features_targets, train_model
from test_model import split_test_features_targets, test_model
from model import LifetimeModel, BearingDataset
from config import config

fr_range = (200, 4000)
n_range = (100, 3500)
fr_interval = 100
n_interval = 100
output_path = config["dataset_path"]
output_sizes = [5, 10, 20, 50, 100]
hidden_sizes = [20, 50, 100, 200, 500]
activation_functions = [nn.ReLU(), nn.Tanh(), nn.Sigmoid()]
batch_sizes = [32, 64, 128, 256]
prefer_cuda = False

hyperparameters = ['output_size', 'hidden_size', 'activation_function', 'batch_size']
parameter_combinations_list = list(product(output_sizes, hidden_sizes, activation_functions, batch_sizes))


def average_mean_squared_error(true, pred, output_size):
    squared_errors = (np.array(true) - np.array(pred)) ** 2
    average_mse = np.mean(squared_errors)
    normalized_mse = average_mse / output_size
    return normalized_mse


# Dictionary to store sum of mse and count for each hyperparameter value
sum_mse = {}
count = {}

total_combinations = len(parameter_combinations_list)
parameter_combinations = iter(parameter_combinations_list)

device = torch.device("cuda" if torch.cuda.is_available() and prefer_cuda else "cpu")
print(f'Using device: {device}')
dataset = generate_dataset(fr_range, n_range, fr_interval, n_interval, output_path)
writer = SummaryWriter('runs/categorize/lifetime/grid_search')

for index, (output_size, hidden_size, activation_function, batch_size) in enumerate(parameter_combinations, start=1):
    print(f'Iteration {index}/{total_combinations}, Output Size: {output_size}, Hidden Size: {hidden_size}, Activation Function: {activation_function}, Batch Size: {batch_size}')
    data = dataset.copy()
    config["output_size"] = output_size
    config["hidden_size"] = hidden_size
    config["activation_function"] = activation_function
    config["batch_size"] = batch_size

    model = LifetimeModel(config["input_size"], config["hidden_size"], config["output_size"], config["activation_function"])
    model.to(device)

    train_data, val_data, test_data = split_and_transform_dataset(data)
    train_features, train_targets, val_features, val_targets = split_train_features_targets(train_data, val_data)
    train_dataset = BearingDataset(train_features, train_targets)
    val_dataset = BearingDataset(val_features, val_targets)
    train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=True, pin_memory=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    loss_fn = nn.CrossEntropyLoss()

    train_model(train_dataloader, val_dataloader, model, loss_fn, optimizer, config["epochs"], device)

    test_features, test_targets = split_test_features_targets(test_data)
    test_dataset = BearingDataset(test_features, test_targets)
    test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=True, pin_memory=True)

    mse = test_model(model, test_dataloader, average_mean_squared_error, output_size)
    test_score = 1/mse
    print('Average Mean Squared Error: ', mse, ' -> Score: ', test_score, '\n')

    hparams = {
        'output_size': output_size,
        'hidden_size': hidden_size,
        'activation_function': str(activation_function),
        'batch_size': batch_size
    }
    
    metrics = {'score': test_score}
    writer.add_hparams(hparams, metrics)

    # Storing the sum of average mse and count for each hyperparameter value
    hyperparameter_values = {'output_size': output_size, 'hidden_size': hidden_size, 'activation_function': str(activation_function), 'batch_size': batch_size}
    
    for hyperparam in hyperparameters:
        value = hyperparameter_values[hyperparam]
        
        if hyperparam not in sum_mse:
            sum_mse[hyperparam] = {}
            count[hyperparam] = {}

        sum_mse[hyperparam][value] = sum_mse[hyperparam].get(value, 0) + test_score
        count[hyperparam][value] = count[hyperparam].get(value, 0) + 1

# Calculating average mse for each hyperparameter value and sorting them
average_mse_by_hyperparameter = {}
for hyperparam in sum_mse:
    average_mse_by_hyperparameter[hyperparam] = {k: round(v / count[hyperparam][k], 3) for k, v in sum_mse[hyperparam].items()}
    average_mse_by_hyperparameter[hyperparam] = sorted(average_mse_by_hyperparameter[hyperparam].items(), key=lambda item: item[1], reverse=True)

# Output the sorted average mean squared errors for each hyperparameter value
for hyperparam, averages in average_mse_by_hyperparameter.items():
    print(f"Averages for {hyperparam}: {averages}")

writer.close()
