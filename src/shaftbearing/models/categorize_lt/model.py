import torch
from torch import nn
from torch.utils.data import Dataset

class LifetimeModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation_function):
        super(LifetimeModel, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.activation_function = activation_function
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output = self.linear1(x)
        output = self.activation_function(output)
        output = self.linear2(output)
        return output

class BearingDataset(Dataset):
    def __init__(self, features, targets):
        self.data = torch.tensor(features, dtype=torch.float32)
        self.target = torch.tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]
