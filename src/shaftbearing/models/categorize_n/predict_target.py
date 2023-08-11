import numpy as np
import torch
import joblib
import pandas as pd
from model import LifetimeModel
from config import config

# Load bins
bins = joblib.load(config["bins_path"])

# Load the model
model = LifetimeModel(config["input_size"], config["hidden_size"], config["output_size"], config["activation_function"])
model.load_state_dict(torch.load(config["model_path"]))

# Load the scaler
scaler = torch.load(config["scaler_path"])

# User Input
Fr = float(input("Enter Fr value: "))
Lifetime = float(input("Enter Lifetime value: "))

# Scaling input features
input_data = pd.DataFrame({'Fr': [Fr], 'Lifetime': [Lifetime]})
scaled_features = scaler.transform(input_data)

# Predict the n bin
model.eval()
with torch.no_grad():
    inputs = torch.tensor(scaled_features, dtype=torch.float32)
    outputs = model(inputs)
    _, predicted_category = torch.max(outputs, 1)
    predicted_category = predicted_category.item()

# Calculate the actual n value
n = (4.1378625767 * 10**17 * Fr ** (-10/3) / Lifetime) ** -1.0

# Calculate the actual n bin
n_transformed = np.digitize(np.array([n]), bins) - 1

print(f"Predicted n Bin: {predicted_category}")
print(f"Actual n Bin: {n_transformed[0]}")
print(f"Computed n value: {n}")
