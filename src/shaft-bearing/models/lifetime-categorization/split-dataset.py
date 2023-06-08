import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import boxcox
from sklearn.preprocessing import StandardScaler
import torch
import joblib
from config import config

print('Loading data from csv file...')
data = pd.read_csv(os.path.join(config["folder_path"], 'data-lt.csv'))

# Box-Cox transformation
data['Lifetime'], fitted_lambda = boxcox(data['Lifetime'])

# Bin 'Lifetime' into categories using quantiles
data['Lifetime'], bins = pd.qcut(data['Lifetime'], q=config["output_size"], labels=False, retbins=True, duplicates='drop')
joblib.dump(bins, config["bins_path"])
joblib.dump(fitted_lambda, config["lambda_path"])

# Remove rows with missing 'Lifetime' values
data = data[~data['Lifetime'].isnull()]

# Split data into train, validation and test sets
train_data, test_data = train_test_split(data, test_size=config["test_size"], random_state=42)
train_data, val_data = train_test_split(train_data, test_size=config["validation_size"], random_state=42)

scaler = StandardScaler()   
scaler.fit(train_data[['Fr', 'n']])
torch.save(scaler, config["scaler_path"])

# Normalization
train_data[['Fr', 'n']] = scaler.transform(train_data[['Fr', 'n']])
val_data[['Fr', 'n']] = scaler.transform(val_data[['Fr', 'n']])
test_data[['Fr', 'n']] = scaler.transform(test_data[['Fr', 'n']])

# Save data
train_data.to_csv(config["train_data_path"], index=False)
val_data.to_csv(config["val_data_path"], index=False)
test_data.to_csv(config["test_data_path"], index=False)
print('Data saved to csv files at: ' + config["folder_path"])
