# Ensemble + EDA

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

# Load Dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print("Shape:",df.shape)
print(df.head())
print(df.info())

# Basic Statistics
print(df.describe())

# Missing Values
print(df.isnull().sum())

# Target Distribution
sns.countplot(x=df['target'])
plt.title("Target Distribution (0 - Malignant, 1 - Benign)")
plt.savefig("target_distribution_plot.jpg")

# Histograms
df.hist(figsize=(18, 15))
plt.tight_layout()      # Adjusts spacing between plots
plt.savefig("histogram_plot.png")

# Correlation Heatmap
plt.figure(figsize=(12, 9))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
