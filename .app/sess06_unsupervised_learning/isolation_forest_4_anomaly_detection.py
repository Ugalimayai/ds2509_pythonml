"""
======================================================================================================================
Python script to demonstrate Isolation Forest
======================================================================================================================
This script demonstrates Isolation FOrest to detect anomalies

Requirements
-------------------------
pip install matplotlib numpy pandas scikit-learn

Author: Karanja Mwaniki
Date: 11 September 2026
"""
# ======================================================================================================================
# SECTION 0: Import the required modules
# ======================================================================================================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# ======================================================================================================================
# SECTION 1: Generate & display some rows from the dataset
# ======================================================================================================================
np.random.seed(42)

# Create a normal data point around income ~(30k-70k) and spending score ~(20-80)
income = np.random.normal(50_000, 10_000, 300)
spending_score = np.random.normal(50, 15, 300)
X = np.column_stack((income, spending_score))

# Introduce some anomalies with higher or lower income and spending scores
anomalies = np.array([[100_000,10], [20_000,90],[80_000, 75], [35_000, 5], [60000,95]])
X = np.vstack((X, anomalies))

# Convert the data into a pandas DataFrame
df = pd.DataFrame(X, columns=['Income', 'Spending Score'])

# Display the first 5 records
print(f"The first 5 rows are:\n{df.head(5)}")

# ======================================================================================================================
# SECTION 2: Isolation Forest and visualisation
# ======================================================================================================================
# Apply Isolation Forest
iso_forest = IsolationForest(contamination=.02, random_state=42)
df['Anomaly Score'] = iso_forest.fit_predict(df[['Income', 'Spending Score']])

# Separate normal points and anomalies for visualisation
normal = df[df['Anomaly Score']==1]
anomaly = df[df['Anomaly Score']==-1]

# Plot the results
plt.figure(figsize = (10,8))
plt.scatter(normal['Income'],normal['Spending Score'],
            color='blue', label='Normal',alpha=.6)
plt.scatter(anomaly['Income'],anomaly['Spending Score'],
            color='red', label='Anomaly',marker='x',s=100)
plt.xlabel('Income')
plt.ylabel('Spending Score')
plt.title('Anomaly Detection with Isolation Forest')
plt.legend()
plt.grid(True)
plt.show()
