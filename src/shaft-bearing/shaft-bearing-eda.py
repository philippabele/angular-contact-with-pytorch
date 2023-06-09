import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data
data = pd.read_csv('data/ShaftBearing/data.csv')

# Transform 'Lifetime' column using Logarithm and add it as a new column
data['log(Lifetime)'] = np.log(data['Lifetime'])

# Basic information about the data
print('\nFirst few rows of the data:')
print(data.head())

print('\nBasic Summary Statistics:')
print(data.describe())

print('\nMissing Values:')
print(data.isnull().sum())

# Distribution Plots (logarithm transformed Lifetime)
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for i, column in enumerate(data.columns):
    if column == 'Lifetime':
        sns.histplot(data['log(Lifetime)'], ax=ax[i%3])
        ax[i%3].set_title('Logarithmic Scale for ' + column)
    elif column != 'log(Lifetime)':
        n_bins = data[column].nunique()
        sns.histplot(data[column], bins=n_bins, ax=ax[i%3])
        ax[i%3].set_title('Distribution of ' + column)
fig.suptitle('Distribution Plots')
fig.savefig('docs/assets/bearings-eda/histogram.png', dpi=300)

# Boxplots
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for i, column in enumerate(data.columns):
    if column != 'log(Lifetime)':
        sns.boxplot(x=data[column], ax=ax[i%3])
        ax[i%3].set_title('Boxplot of ' + column)
fig.suptitle('Boxplots')
fig.savefig('docs/assets/bearings-eda/boxplot.png', dpi=300)

# Pairplot
data_pair = data.copy()
pair = sns.pairplot(data_pair, hue='log(Lifetime)', palette='crest')
plt.subplots_adjust(top=0.95)
plt.suptitle('Pairplot of Fr, n, Lifetime and interaction features')
pair.savefig('docs/assets/bearings-eda/pairplot.png', dpi=300)

# Heatmap of Correlations
corr = data.corr()
print('\nCorrelation Matrix:')
print(corr)

fig = plt.figure()
sns.heatmap(corr, annot=True)
plt.title('Heatmap of Correlations')
fig.savefig('docs/assets/bearings-eda/correlation-matrix.png', dpi=300)

# 3D Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(data['Fr'], data['n'], data['Lifetime'])
ax.set_xlabel('Fr')
ax.set_ylabel('n')
ax.set_zlabel('Lifetime')
plt.title('3D Plot of Fr, n and Lifetime')
fig.savefig('docs/assets/bearings-eda/3dplot.png', dpi=300)

# 3D Plot of logarithm transformed Lifetime
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(data['Fr'], data['n'], data['log(Lifetime)'])
ax.set_xlabel('Fr')
ax.set_ylabel('n')
ax.set_zlabel('log(Lifetime)')
plt.title('3D Plot of Fr, n and log(Lifetime)')
fig.savefig('docs/assets/bearings-eda/3dplot-log.png', dpi=300)

# Estimate Lifetime Function and calculate the relative error
data['LifetimeFunc'] = 4.1378625767 * 10**17 * (data['Fr']) ** (-10/3) * (data['n']) ** (-1.0)
data['RelativeError'] = (data['Lifetime'] / data['LifetimeFunc']) - 1
print('\Relative Error:')
print(data['RelativeError'].describe())

# 3D Plot of Relative Error
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(data['Fr'], data['n'], data['RelativeError'])
ax.set_xlabel('Fr')
ax.set_ylabel('n')
ax.set_zlabel('Relative Error')
plt.title('3D Plot of Fr, n and Relative Error')
fig.savefig('docs/assets/bearings-eda/3dplot-error.png', dpi=300)

plt.show()
