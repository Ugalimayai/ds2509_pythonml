"""
======================================================================================================================
Python script to demonstrate K-means algorithm
======================================================================================================================
This script demonstrates the K-means clustering algorithm to determine customer segmentation based on income
and spending score

Requirements
-------------------------
pip install matplotlib pandas scikit-learn

Author: Karanja Mwaniki
Date: 11 September 2026
"""
# ======================================================================================================================
# SECTION 0: Import the required modules
# ======================================================================================================================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# ======================================================================================================================
# SECTION 1: Generate & display some rows from the dataset
# ======================================================================================================================
# Generate a synthetic dataset
X,y = make_blobs(n_samples=1000, centers=3, cluster_std=0.5, random_state=42)

# Convert the dataset into a pandas dataframe
df = pd.DataFrame(X, columns=['Income', 'Spending Score'])

# Display the first 10 rows of the database
print(f"The first 10 rows of the dataframe:\n{df.head(10)}")

# ======================================================================================================================
# SECTION 0: Visualise and evaluate the model
# ======================================================================================================================
# Visualise the dataset
plt.scatter(df['Income'], df['Spending Score'])
plt.title("Synthetic Dataset: Income vs Spending Score")
plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.show()

# Apply K-means
kmeans = KMeans(n_clusters=3)
kmeans.fit(df)

# Add cluster labels to the dataframe
df['Cluster'] = kmeans.labels_

# Plot the clusters
plt.scatter(df['Income'], df['Spending Score'], c=df['Cluster'], cmap='viridis', marker='o')
plt.title("K-means Clusters")
plt.xlabel("Income")
plt.ylabel("Spending Score")
# Plot the cluster centers
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], marker='x', s=200, color='red', alpha=0.8, label='centroids')
plt.legend()
plt.show()

# Evaluate the model
print(f"Inertia: {kmeans.inertia_}")

# Inertia is the sum of squared distances between each point and its assigned cluster center.
# It helps to assess how well the clusters have been formed, with lower values
# indicating tighter clusters.