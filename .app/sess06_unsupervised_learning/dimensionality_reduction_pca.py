# Python file to demonstrate feature reduction of 'income' and 'spending score'
# on a dataset using Principal Component Analysis  (PCA)

# Import the required modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA

# Generate a synthetic customer dataset with income and spending
np.random.seed(42) # For reproducibility
X,_ = make_blobs(n_samples=300, centers=4, cluster_std=1.2, random_state=42)
df = pd.DataFrame(X, columns=['Income', 'Spending Score'])

# Standardise the data
df_standardised = (df - df.mean())/df.std()

# Apply PCA to reduce to 2 components
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_standardised)
df_pca = pd.DataFrame(df_pca, columns=['Economic Component 1', 'Economic Component 2'])

# Explain the variance
explained_variance = pca.explained_variance_ratio_
print(f"Explained variance by Component: {explained_variance}")
print(f"Cumulative explained variance by Component: {explained_variance.cumsum()}")

# Plot the PCA-transformed data
plt.figure(figsize=(10,8))
plt.scatter(df_pca['Economic Component 1'], df_pca['Economic Component 2'],c='blue',marker='o')
plt.title("PCA on Customer Dataset")
plt.xlabel("Economic Component 1")
plt.ylabel("Economic Component 2")
plt.show()