import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from model import LifetimeModel, BearingDataset
from config import config

def load_test_data(test_data_path):
    test_data = pd.read_csv(test_data_path)
    return test_data


def split_test_features_targets(test_data):
    test_features = test_data[['Fr', 'n']].values
    test_targets = test_data['Lifetime'].values
    return test_features, test_targets


def load_model():
    model = LifetimeModel(config["input_size"], config["hidden_size"], config["output_size"], config["activation_function"])
    model.load_state_dict(torch.load(config["model_path"]))
    return model


def test_model(model, test_dataloader, evaluation_function):
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.numpy())
            all_targets.extend(targets.numpy())
    
    score = evaluation_function(all_targets, all_predictions)
    return score


def test_model_matrix(model, test_dataloader):
    print('Testing model...')
    model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.numpy())
            all_targets.extend(targets.numpy())

    # Create confusion matrix
    cf_matrix = confusion_matrix(all_targets, all_predictions)

    # Calculate total accuracy
    num_correct = np.sum(np.diag(cf_matrix))
    total_samples = np.sum(cf_matrix)
    total_accuracy = num_correct / total_samples
    print(f'Total Accuracy: {total_accuracy:.3f} ({num_correct} out of {total_samples} correct)')

    # Plot confusion matrix
    print('PLOOOT')
    plt.figure(figsize=(10, 7))
    sns.heatmap(cf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


# test_data = load_test_data(config['test_data_path'])
# test_features, test_targets = split_test_features_targets(test_data)
# model = load_model()

# # Create DataLoader for the test dataset
# test_dataset = BearingDataset(test_features, test_targets)
# test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

# test_model_matrix(model, test_dataloader)
