"""
======================================================================================================================
Python script to demonstrate various data transformation techniques on a sample dataset
======================================================================================================================


Requirements
-------------------------
pip install matplotlib numpy pandas scipy

Author: Karanja Mwaniki
Date: 28 August 2026
"""
# ======================================================================================================================
# SECTION 0: Import the required modules
# ======================================================================================================================
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import boxcox

# ======================================================================================================================
# SECTION 1: Sample dataset with skewed data
# ======================================================================================================================
data = pd.DataFrame(
    {
        'income': [30000, 50000, 200000, 1000000, 60000, 80000, 120000, 15000, 2500000, 100000],
        'transaction_amount': [200, 500, 1000, 25000, 800, 1500, 200, 350, 50000, 1200]
    }
)

# set up the plot
fig, axes = plt.subplots(1, 3, figsize=(12, 8))
# create the first subplot
axes[0].hist(data['income'], bins=10, color='skyblue', edgecolor='black')
axes[0].set_title('Original Income distribution')
#create the second subplot
axes[1].hist(data['transaction_amount'], bins=10, color='salmon', edgecolor='black')
axes[1].set_title('Original Transaction amount distribution')
#Display the original income and transaction amount data
print(f"Original Data:\n{data}")

# ======================================================================================================================
# SECTION 2: Log Transform the 'income' and 'transaction amount'
# ======================================================================================================================
data['log_income'] = np.log(data['income'])
data['log_transaction_amount'] = np.log(data['transaction_amount'])

axes[2].hist(data['log_income'], bins=10, color='lightgreen', edgecolor='black')
axes[2].set_title('Log Transformed Income distribution')
plt.show()

# ======================================================================================================================
# SECTION 3: Square root transformations
# ======================================================================================================================
data['sqrt_income'] = np.sqrt(data['income'])
data['sqrt_transaction_amount'] = np.sqrt(data['transaction_amount'])

# ======================================================================================================================
# SECTION 4: Box-cox transformation (NB: require strictly positive values)
# ======================================================================================================================
data['positive_income'] = data['income'] + 1 # Offset to ensure positivity
data['boxcox_income'], income_lambda = boxcox(data['positive_income'])
print(f"Optimal Lambda for Income Box-Cox transformation: {income_lambda}")

# ======================================================================================================================
# SECTION 5:  Visualisation for all income transformations
# ======================================================================================================================
fix, ax = plt.subplots(2,2, figsize=(15, 10))
ax[0,0].hist(data['income'], bins=10, color='skyblue', edgecolor='black')
ax[0,0].set_title('Original Income')

ax[0,1].hist(data['log_income'], bins=10, color='lightgreen', edgecolor='black')
ax[0,1].set_title('Log Transformed Income')

ax[1,0].hist(data['sqrt_income'], bins=10, color='lightcoral', edgecolor='black')
ax[1,0].set_title('Square-root Income')

ax[1,1].hist(data['boxcox_income'], bins=10, color='lightblue', edgecolor='black')
ax[1,1].set_title('BoxCox Transformed Income')

plt.tight_layout()
plt.show()

# Display the transformed dataset
print(f"\nTransformed Data:\n{data}")
print(data([['income', 'log_income', 'sqrt_income', 'boxcox_income']]))
