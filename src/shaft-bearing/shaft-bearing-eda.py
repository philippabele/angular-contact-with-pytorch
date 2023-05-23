import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_csv('data/shaft-bearing.csv')

# Print an overview of the data
print(df.head())
print(df.describe())

# Create a figure and a grid of subplots
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(20, 30))

# Histograms of all columns
for i, column in enumerate(df.columns):
    df[column].hist(bins=50, ax=axes[i, 0])
    axes[i, 0].set_title(f'Histogram of {column}')

# Boxplots of all columns
for i, column in enumerate(df.columns):
    df[column].plot(kind='box', ax=axes[i, 1])
    axes[i, 1].set_title(f'Boxplot of {column}')

# Compute the correlation matrix
corr_matrix = df.corr()

# Generate a heatmap of the correlation matrix
sns.heatmap(corr_matrix, annot=True, ax=axes[3, 0])
axes[3, 0].set_title('Correlation Heatmap')

# Scatter plot of first two columns
axes[4, 0].scatter(df.iloc[:, 0], df.iloc[:, 1])
axes[4, 0].set_title(f'Scatter Plot: {df.columns[0]} vs {df.columns[1]}')

# Scatter plot of last two columns
axes[4, 1].scatter(df.iloc[:, 1], df.iloc[:, 2])
axes[4, 1].set_title(f'Scatter Plot: {df.columns[1]} vs {df.columns[2]}')

sns.pairplot(df)

# Show the plot
plt.tight_layout(pad=5.0)
plt.show()
