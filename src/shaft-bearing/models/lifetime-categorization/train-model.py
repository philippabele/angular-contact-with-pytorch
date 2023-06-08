import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import pandas as pd
from model import LifetimeModel, BearingDataset

# Paths
folder_path = 'data/ShaftBearing/lifetime-categorization/'
train_data_path = folder_path + 'train-lt.csv'
val_data_path = folder_path + 'val-lt.csv'
model_path = folder_path + 'model-lt.pt'

# Configurations
load_model = False
save_model = True
input_size = 2
output_size = 10
hidden_size = 50
activation_function = nn.ReLU()
learning_rate = 0.001
epochs = 100
batch_size = 64

def load_data(train_data_path, val_data_path):
    print('Loading data for training and evaluation from csv files...')
    train_data = pd.read_csv(train_data_path)
    val_data = pd.read_csv(val_data_path)

    train_features, train_targets = train_data[['Fr', 'n']], train_data['Lifetime']
    val_features, val_targets = val_data[['Fr', 'n']], val_data['Lifetime']

    return train_features.values, train_targets.values, val_features.values, val_targets.values

def train_model(train_dataloader, val_dataloader, model, loss_fn, optimizer, epochs):
    for epoch in range(epochs):
        for inputs, targets in train_dataloader:
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
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, 1)
                    val_loss += loss_fn(outputs, targets)
                    val_acc += (predicted == targets).sum().item()

            val_loss /= len(val_dataloader)
            val_acc /= len(val_targets)
            print(f'Epoch {epoch}, Loss: {loss.item()}, Validation Loss: {val_loss.item()}, Validation Accuracy: {val_acc}')


# Load data
train_features, train_targets, val_features, val_targets = load_data(train_data_path, val_data_path)

# Initialize model
model = LifetimeModel(input_size, hidden_size, output_size, activation_function)
if load_model and os.path.exists(model_path):
    print('Loading model from file...')
    model.load_state_dict(torch.load(model_path))

train_dataset = BearingDataset(train_features, train_targets)
val_dataset = BearingDataset(val_features, val_targets)

# Creating PyTorch DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Optimizer and Loss Function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

train_model(train_dataloader, val_dataloader, model, loss_fn, optimizer, epochs)

# Save the model after training
if save_model:
    torch.save(model.state_dict(), model_path)
