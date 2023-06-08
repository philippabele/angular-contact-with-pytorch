import os
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
from model import LifetimeModel, BearingDataset
from config import config

def load_data(train_data_path, val_data_path):
    print('Loading data for training and evaluation from csv files...')
    train_data = pd.read_csv(train_data_path)
    val_data = pd.read_csv(val_data_path)

    train_features, train_targets = train_data[['Fr', 'n']], train_data['Lifetime']
    val_features, val_targets = val_data[['Fr', 'n']], val_data['Lifetime']

    return train_features.values, train_targets.values, val_features.values, val_targets.values

def train_model(train_dataloader, val_dataloader, model, loss_fn, optimizer, epochs, device):
    start_time = time.time()
    for epoch in range(epochs):
        for inputs, targets in train_dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            # Forward pass
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validate the model
        if epoch % 10 == 0:
            val_loss = 0
            val_acc = 0
            with torch.no_grad():
                for inputs, targets in val_dataloader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    val_loss += loss_fn(outputs, targets)
                    val_acc += (predicted == targets).sum().item()

            val_loss /= len(val_dataloader)
            val_acc /= len(val_targets)
            print(f'Epoch {epoch}, Loss: {loss.item(): .3f}, Validation Loss: {val_loss.item(): .3f}, Validation Accuracy: {val_acc: .3f}')
    training_time = time.time() - start_time
    print(f'Training took {training_time:.2f} seconds')

# Load data
train_features, train_targets, val_features, val_targets = load_data(config["train_data_path"], config["val_data_path"])

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() and config["prefer_cuda"] else "cpu")
print(f'Using device: {device}')

# Initialize model
model = LifetimeModel(config["input_size"], config["hidden_size"], config["output_size"], config["activation_function"])
if config["load_model"] and os.path.exists(config["model_path"]):
    print('Loading model from file...')
    model.load_state_dict(torch.load(config["model_path"]))
model.to(device)

train_dataset = BearingDataset(train_features, train_targets)
val_dataset = BearingDataset(val_features, val_targets)

# Creating PyTorch DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=True, pin_memory=True)

# Optimizer and Loss Function
optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
loss_fn = nn.CrossEntropyLoss()

train_model(train_dataloader, val_dataloader, model, loss_fn, optimizer, config["epochs"], device)

# Save the model after training
if config["save_model"]:
    torch.save(model.state_dict(), config["model_path"])
