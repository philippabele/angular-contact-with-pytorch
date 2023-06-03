import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the data
data = pd.read_csv('data/ShaftBearing/data.csv')

# Basic information about the data
print('\nFirst few rows of the data:')
print(data.head())

print('\nBasic Summary Statistics:')
print(data.describe())

print('\nMissing Values:')
print(data.isnull().sum())

# Distribution Plots (Logarithmic Scale for Lifetime)
fig1, ax1 = plt.subplots(1, 3, figsize=(15, 5))
for i, column in enumerate(data.columns):
    if column == 'Lifetime' or column == 'Fr*n' or column == 'Fr/n' or column == 'LifetimeFunc':
        # Create a temporary variable for Lifetime, replacing 0 with a very small number
        temp_column = data[column].replace(0, 0.0001)
        sns.histplot(np.log(temp_column), ax=ax1[i%3])
        ax1[i%3].set_title('Logarithmic Scale for ' + column)
    else:
        n_bins = data[column].nunique()
        sns.histplot(data[column], bins=n_bins, ax=ax1[i%3])
        ax1[i%3].set_title('Distribution of ' + column)
fig1.suptitle('Distribution Plots')
fig1.savefig('docs/assets/bearings-eda/distribution-plots.png')

# Boxplots
fig4, ax4 = plt.subplots(1, 3, figsize=(15, 5))
for i, column in enumerate(data.columns):
    sns.boxplot(x=data[column], ax=ax4[i%3])
    ax4[i%3].set_title('Boxplot of ' + column)
fig4.suptitle('Boxplots')
fig4.savefig('docs/assets/bearings-eda/boxplots.png')

# Pairplot
data_pair = data.copy()
data_pair['Lifetime_log'] = np.log(data_pair['Lifetime'])
pair = sns.pairplot(data_pair, hue='Lifetime_log', palette='crest')
plt.subplots_adjust(top=0.95)
plt.suptitle('Pairplot of Fr, n, Lifetime and interaction features')
pair.savefig('docs/assets/bearings-eda/pairplot.png')

# Heatmap of Correlations
corr = data.corr()
print('\nCorrelation Matrix:')
print(corr)

fig2 = plt.figure()
sns.heatmap(corr, annot=True)
plt.title('Heatmap of Correlations')
fig2.savefig('docs/assets/bearings-eda/correlation-matrix.png')

# 3D Plot
fig3 = plt.figure()
ax3 = fig3.add_subplot(111, projection = '3d')
ax3.scatter(data['Fr'], data['n'], data['Lifetime'])
ax3.set_xlabel('Fr')
ax3.set_ylabel('n')
ax3.set_zlabel('Lifetime')
plt.title('3D Plot of Fr, n and Lifetime')
fig3.savefig('docs/assets/bearings-eda/3dplot.png')

# Estimate Lifetime Function and calculate Residuals
data['LifetimeFunc'] = 4.13786 * 10**17 * (data['Fr']) ** (-10/3) * (data['n']) ** (-1.0)
data['Residuals (% -1)'] = (data['Lifetime'] / data['LifetimeFunc']) - 1
print('\nResiduals:')
print(data['Residuals (% -1)'].describe())

# 3D Plot of Residuals
fig32 = plt.figure()
ax32 = fig32.add_subplot(111, projection = '3d')
ax32.scatter(data['Fr'], data['n'], data['Residuals (% -1)'])
ax32.set_xlabel('Fr')
ax32.set_ylabel('n')
ax32.set_zlabel('Residuals')
plt.title('3D Plot of Fr, n and Residuals')
fig32.savefig('docs/assets/bearings-eda/3dplot-residuals.png')

plt.show()
