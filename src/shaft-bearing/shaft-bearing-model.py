from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter

# Hyperparameters
learning_rate = 1e-5
batch_size = 64
epochs = 10000
test_size = 0.01
log_count = 10

load_model = False
save_model = True
train_model = True
eval_model = True

# Define custom dataset
class ShaftBearingDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return torch.Tensor(self.X[idx]), torch.Tensor([self.y[idx]])

# Load data
data = pd.read_csv('data/ShaftBearing/data.csv')
X = data[['Fr', 'n']].values
y = data['Lifetime'].values

# Split into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Standardize the features
X_scaler = StandardScaler()
X_train = X_scaler.fit_transform(X_train)
X_test = X_scaler.transform(X_test)

# Standardize the labels
y_scaler = StandardScaler()
y_train = y_scaler.fit_transform(y_train.reshape(-1,1)).flatten()
y_test = y_scaler.transform(y_test.reshape(-1,1)).flatten()

# Create datasets
train_data = ShaftBearingDataset(X_train, y_train)
test_data = ShaftBearingDataset(X_test, y_test)

print(f'Train data: {len(train_data)} samples')
print(f'Test data: {len(test_data)} samples')

# Create dataloaders
dataloader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Define model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(2, 1)  # Two input features: Fr and n

    def forward(self, x):
        out = self.linear(x)
        return out

model = LinearRegression().to(device)
if load_model:
    try:
        model.load_state_dict(torch.load("data/ShaftBearing/model.pth"))
        print("Loaded model from disk.")
    except:
        print("No model found, training from scratch.")

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Set up TensorBoard writer
writer = SummaryWriter('runs/ShaftBearing')
writer_no_outliers = SummaryWriter('runs/ShaftBearing_no_outliers')

loss_values = []

# Training loop
def train(dataloader, model, criterion, optimizer, epochs):
    for epoch in range(epochs):
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_values.append(loss.item())
        writer.add_scalar('train/loss/all', loss.item(), epoch)
        if (epoch+1) % log_count == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

    # If loss_value is not mor than 2 std away from the mean, add it to the no_outliers loss
    mean_loss = np.mean(loss_values)
    std_loss = np.std(loss_values)
    for i, loss_value in enumerate(loss_values):
        if np.abs(loss_value - mean_loss) <= 1*std_loss:
            print(loss_value)
            writer_no_outliers.add_scalar('train/loss/no_outliers', loss_value, i)


def eval(dataloader, model, criterion):
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        print(f'Loss: {loss.item()}')

if train_model:
    train(dataloader, model, criterion, optimizer, epochs)
    writer.close()
    writer_no_outliers.close()
    if save_model:
        torch.save(model.state_dict(), 'data/ShaftBearing/model.pth')
        print('Saved model state as data/ShaftBearing/model.pth.')

if eval_model:
    eval(test_loader, model, criterion)
