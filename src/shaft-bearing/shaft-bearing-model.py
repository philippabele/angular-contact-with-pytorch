import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Hyperparameters
learning_rate = 1e-2
batch_size = 64
epochs = 100
test_size = 0.01

load_model = True
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

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create datasets
train_data = ShaftBearingDataset(X_train, y_train)
test_data = ShaftBearingDataset(X_test, y_test)

print(f'Train data: {len(train_data)} samples')
print(f'Test data: {len(test_data)} samples')

# for i in range(len(test_data)):
#     print(test_data[i])

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

# Training loop
def train(dataloader, model, criterion, optimizer, epochs):
    for epoch in range(epochs):
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # print inputs, labels and outputs
            if (epoch+1) % 10 == 0 and i == 0:  # print for the first batch of every 10th epoch
                print("Inputs: ", inputs)
                print("Labels: ", labels)
                print("Outputs: ", outputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


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
    if save_model:
        torch.save(model.state_dict(), 'data/ShaftBearing/model.pth')
        print('Saved model state as data/ShaftBearing/model.pth.')

if eval_model:
    eval(test_loader, model, criterion)
