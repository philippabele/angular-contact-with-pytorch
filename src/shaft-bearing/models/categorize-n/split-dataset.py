import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import boxcox
from sklearn.preprocessing import StandardScaler
import torch
import joblib
from config import config

print('Loading data from csv file...')
data = pd.read_csv(config["dataset_path"])

# Box-Cox transformation on 'Lifetime'
data['Lifetime'], fitted_lambda = boxcox(data['Lifetime'])
joblib.dump(fitted_lambda, config["lambda_path"])

# Bin 'n' into categories using quantiles
n_bins, bins = pd.qcut(data['n'], q=config["output_size"], labels=False, retbins=True, duplicates='drop')
data['n'] = n_bins
joblib.dump(bins, config["bins_path"])

# Remove rows with missing 'n' values
data = data[~data['n'].isnull()]

# Split data into train, validation, and test sets
train_data, test_data = train_test_split(data, test_size=config["test_size"], random_state=42)
train_data, val_data = train_test_split(train_data, test_size=config["validation_size"], random_state=42)

scaler = StandardScaler()
scaler.fit(train_data[['Fr', 'Lifetime']])
torch.save(scaler, config["scaler_path"])

# Normalization
train_data[['Fr', 'Lifetime']] = scaler.transform(train_data[['Fr', 'Lifetime']])
val_data[['Fr', 'Lifetime']] = scaler.transform(val_data[['Fr', 'Lifetime']])
test_data[['Fr', 'Lifetime']] = scaler.transform(test_data[['Fr', 'Lifetime']])

# Reorder the columns so that 'n' is the last column
cols = ['Fr', 'Lifetime', 'n']
train_data = train_data[cols]
val_data = val_data[cols]
test_data = test_data[cols]

# Save data
train_data.to_csv(config["train_data_path"], index=False)
val_data.to_csv(config["val_data_path"], index=False)
test_data.to_csv(config["test_data_path"], index=False)
print('Data saved to csv files at: ' + config["folder_path"])
