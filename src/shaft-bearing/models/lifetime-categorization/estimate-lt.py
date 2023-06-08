import numpy as np
import torch
import joblib
from scipy.stats import boxcox
from model import LifetimeModel
from torch import nn
import pandas as pd

# Paths
folder_path = 'data/ShaftBearing/lifetime-categorization/'
model_path = folder_path + 'model-lt.pt'
scaler_path = folder_path + 'scaler-lt.pt'
bins_path = folder_path + 'bins.joblib'
lambda_path = folder_path + 'lambda.joblib'

# Load bins
bins = joblib.load(bins_path)
fitted_lambda = joblib.load(lambda_path)

# Configuration
input_size = 2
output_size = 10
hidden_size = 50
activation_function = nn.ReLU()

# Load the model
model = LifetimeModel(input_size, hidden_size, output_size, activation_function)
model.load_state_dict(torch.load(model_path))

# Load the scaler
scaler = torch.load(scaler_path)

# User Input
Fr = float(input("Enter Fr value: "))
n = float(input("Enter n value: "))

# Scaling input features
input_data = pd.DataFrame({'Fr': [Fr], 'n': [n]})
scaled_features = scaler.transform(input_data)

# Predict the lifetime category
model.eval()
with torch.no_grad():
    inputs = torch.tensor(scaled_features, dtype=torch.float32)
    outputs = model(inputs)
    _, predicted_category = torch.max(outputs, 1)
    predicted_category = predicted_category.item()

# Calculate the actual lifetime in hours
Lifetime = 4.13786 * 10**17 * Fr ** (-10/3) * n ** (-1.0)

# Calculate the actual lifetime bin
if Lifetime > 0:
    lifetime_transformed = boxcox(np.array([Lifetime]), fitted_lambda)
    actual_bin = np.digitize(lifetime_transformed, bins) - 1
else:
    actual_bin = None

print(f"Predicted Lifetime Bin: {predicted_category}")
if actual_bin is not None:
    print(f"Actual Lifetime Bin: {actual_bin[0]}")
else:
    print("Actual Lifetime Bin cannot be calculated.")
print(f"Actual Lifetime in hours: {Lifetime}")
