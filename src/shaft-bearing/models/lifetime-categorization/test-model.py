import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from torch import nn
from torch.utils.data import Dataset, DataLoader
from model import LifetimeModel, BearingDataset
from sklearn.metrics import confusion_matrix
from config import config

# Load the test dataset
test_data = pd.read_csv(config['test_data_path'])

# Separate features and targets
test_features = test_data[['Fr', 'n']].values
test_targets = test_data['Lifetime'].values

# Load the model
model = LifetimeModel(config["input_size"], config["hidden_size"], config["output_size"], config["activation_function"])
model.load_state_dict(torch.load(config["model_path"]))

# Create DataLoader for the test dataset
test_dataset = BearingDataset(test_features, test_targets)
test_dataloader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

# Test the model
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
plt.figure(figsize=(10, 7))
sns.heatmap(cf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
