import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import boxcox
from sklearn.preprocessing import StandardScaler
import torch
import joblib
from config import config

# Box-Cox transformation
def apply_boxcox(data, column, save_to_file=False):
    data[column], fitted_lambda = boxcox(data[column])
    if save_to_file:
        joblib.dump(fitted_lambda, config["lambda_path"])
    return data


# Bin target into categories using quantiles
def apply_binning(data, column, save_to_file=False):
    data[column], bins = pd.qcut(data[column], q=config["output_size"], labels=False, retbins=True, duplicates='drop')
    if save_to_file: 
        joblib.dump(bins, config["bins_path"])
    return data


# Split data into train, validation and test sets
def split_data(data):
    train_data, test_data = train_test_split(data, test_size=config["test_size"], random_state=42)
    train_data, val_data = train_test_split(train_data, test_size=config["validation_size"], random_state=42)
    return train_data, val_data, test_data


def normalize_data(features, train_data, val_data=None, test_data=None):
    scaler = StandardScaler()   
    scaler.fit(train_data[features])
    torch.save(scaler, config["scaler_path"])

    train_data[features] = scaler.transform(train_data[features])
    val_data[features] = scaler.transform(val_data[features])
    test_data[features] = scaler.transform(test_data[features])
    return train_data, val_data, test_data


def save_data(train_data=None, val_data=None, test_data=None):
    train_data.to_csv(config["train_data_path"], index=False)
    val_data.to_csv(config["val_data_path"], index=False)
    test_data.to_csv(config["test_data_path"], index=False)
    print('Data saved to csv files at: ' + config["folder_path"])


def split_and_transform_dataset(data, save_to_file=False):
    apply_boxcox(data, 'Lifetime', save_to_file=save_to_file)
    apply_binning(data, 'Lifetime', save_to_file=save_to_file)
    train_data, val_data, test_data = split_data(data)
    normalize_data(['Fr', 'n'], train_data, val_data, test_data)
    if save_to_file:
        save_data(train_data, val_data, test_data)
    return train_data, val_data, test_data

if __name__ == '__main__':
    print('Loading data from csv file...')
    data = pd.read_csv(config["dataset_path"])
    split_and_transform_dataset(data, save_to_file=True)
