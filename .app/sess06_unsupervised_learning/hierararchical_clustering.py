"""
======================================================================================================================
Python script to demonstrate hierarchical clustering
======================================================================================================================


Requirements
-------------------------
pip install matplotlib numpy pandas scipy scikit-learn

Author: Karanja Mwaniki
Date: 11 September 2026
"""
# ======================================================================================================================
# SECTION 0: Import the required modules
# ======================================================================================================================
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

# ======================================================================================================================
# SECTION 1: Generate and display some rows from the dataset
# ======================================================================================================================
# Generate a synthetic customer dataset with income and spending score
data = {
    'Income': np.random.randint(20000,100000, 100),
    'Spending Score': np.random.randint(1,50, 100)
}

# Create a dataframe from the data and display the first 10 rows
df = pd.DataFrame(data)
print(f"The first 10 rows of the dataframe:\n{df.head(10)}")

# ======================================================================================================================
# SECTION 2: Visualise the dataset and apply agglomerative clustering
# ======================================================================================================================
# Visualise the entire dataset
plt.figure(figsize=(10,8))
plt.scatter(df['Income'], df['Spending Score'])
plt.title('Synthetic Dataset: Income vs Spending Score')
plt.xlabel('Income')
plt.ylabel('Spending Score')
plt.show()

# Create a dendogram
plt.figure(figsize=(10,8))
dendogram = sch.dendrogram(sch.linkage(df, method='ward'))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Samples')
plt.ylabel('Euclidian Distances')
plt.show()

# Apply Agglomerative clustering
hc = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
df['Cluster'] = hc.fit_predict(df)

# Plot the Clusters
plt.figure(figsize=(10,8))
plt.scatter(df['Income'], df['Spending Score'], c=df['Cluster'], cmap='viridis')
plt.title('Hierarchical Clustering: Income vs. Spending Score')
plt.xlabel('Income')
plt.ylabel('Spending Score')
plt.show()

