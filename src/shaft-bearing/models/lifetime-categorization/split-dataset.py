import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import boxcox
from sklearn.preprocessing import StandardScaler
import torch
import joblib

# Paths
bearing_path = 'data/ShaftBearing/'
folder_path = bearing_path + 'lifetime-categorization/'

data_path = bearing_path + 'data.csv'
train_data_path = folder_path + 'train-lt.csv'
val_data_path = folder_path + 'val-lt.csv'
test_data_path = folder_path + 'test-lt.csv'
scaler_path = folder_path + 'scaler-lt.pt'
bins_path = folder_path + 'bins.joblib'
lambda_path = folder_path + 'lambda.joblib'

# Configurations
Lifetime_bins = 10
test_size = 0.2
validation_size = 0.1

print('Loading data from csv file...')
data = pd.read_csv(data_path)

# Box-Cox transformation
data['Lifetime'], fitted_lambda = boxcox(data['Lifetime'])

# Bin 'Lifetime' into categories using quantiles
data['Lifetime'], bins = pd.qcut(data['Lifetime'], q=Lifetime_bins, labels=False, retbins=True, duplicates='drop')
joblib.dump(bins, bins_path)
joblib.dump(fitted_lambda, lambda_path)

# Remove rows with missing 'Lifetime' values
data = data[~data['Lifetime'].isnull()]

# Split data into train, validation and test sets
train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
train_data, val_data = train_test_split(train_data, test_size=validation_size, random_state=42)

scaler = StandardScaler()   
scaler.fit(train_data[['Fr', 'n']])
torch.save(scaler, scaler_path)

# Normalization
train_data[['Fr', 'n']] = scaler.transform(train_data[['Fr', 'n']])
val_data[['Fr', 'n']] = scaler.transform(val_data[['Fr', 'n']])
test_data[['Fr', 'n']] = scaler.transform(test_data[['Fr', 'n']])

# Save data
train_data.to_csv(train_data_path, index=False)
val_data.to_csv(val_data_path, index=False)
test_data.to_csv(test_data_path, index=False)
print('Data saved to csv files at: ' + folder_path)
