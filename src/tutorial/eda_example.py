import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use('pgf')
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

data = sns.load_dataset('iris')

# Show data distribution to identify any potential skewness or patterns
print(data.describe())
data.hist()
plt.savefig('docs/assets/eda/histogram.pgf')

# Boxplot to identify outliers and compare the distribution of the data between different groups or categories
data.plot(kind='box')
plt.savefig('docs/assets/eda/boxplot.pgf')

# Pairplot
sns.pairplot(data, hue='species')
plt.savefig('docs/assets/eda/pairplot.pgf')

# Correlation and covariance
fig, ax = plt.subplots(1, 2, figsize=(6, 3))
sns.heatmap(data.corr(), annot=True, ax=ax[0])
ax[0].set_title('Correlation Heatmap')
sns.heatmap(data.cov(), annot=True, ax=ax[1])
ax[1].set_title('Covariance Heatmap')
plt.savefig('docs/assets/eda/cor-cov.pgf')

# Cross-correlation
fig, ax = plt.subplots(1, 2, figsize=(6, 3))
ax[0].xcorr(data['sepal_length'], data['sepal_width'], usevlines=True, maxlags=50, normed=True, lw=2)
ax[0].grid(True)
ax[0].set_title('Cross-correlation between sepal length and width')
ax[1].xcorr(data['petal_length'], data['petal_width'], usevlines=True, maxlags=50, normed=True, lw=2)
ax[1].grid(True)
ax[1].set_title('Cross-correlation between petal length and width')
plt.savefig('docs/assets/eda/cross-cor.pgf')
