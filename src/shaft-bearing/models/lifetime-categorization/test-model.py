import torch
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader
from model import LifetimeModel

# Paths
folder_path = 'data/ShaftBearing/lifetime-categorization/'
model_path = folder_path + 'model-lt.pt'
test_data_path = folder_path + 'test-lt.csv'

# Configuration
input_size = 2
output_size = 10
hidden_size = 50
activation_function = nn.ReLU()
batch_size = 64

# Load the test dataset
test_data = pd.read_csv(test_data_path)

# Separate features and targets
test_features = test_data[['Fr', 'n']].values
test_targets = test_data['Lifetime'].values

# Load the model
model = LifetimeModel(input_size, hidden_size, output_size, activation_function)
model.load_state_dict(torch.load(model_path))

# Create PyTorch Dataset for test data
class BearingDataset(Dataset):
    def __init__(self, features, targets):
        self.data = torch.tensor(features, dtype=torch.float32)
        self.target = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

# Create DataLoader for the test dataset
test_dataset = BearingDataset(test_features, test_targets)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Test the model
model.eval()
test_acc = 0
with torch.no_grad():
    for inputs, targets in test_dataloader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        test_acc += (predicted == targets).sum().item()

test_acc /= len(test_data)
print(f'Test Accuracy: {test_acc}')
