"""
======================================================================================================================
Python script to demonstrate handling missing values in data
======================================================================================================================


Requirements
-------------------------
pip install numpy pandas

Author: Karanja Mwaniki
Date: 15 08 2026
"""
# ======================================================================================================================
# SECTION 0: Import the required modules
# ======================================================================================================================
import pandas as pd
import numpy as np

# ======================================================================================================================
# STEP 1: Create a sample dataset
# ======================================================================================================================
data = {
    'Name': ['Abigail', 'Kamau', 'Sharlene', 'Diana', 'Mueni'],
    'Age': [25,np.nan,22,32,np.nan],
    'City': ["Nakuru", "Limuru", np.nan, "Homabay", "Makueni"]
}

# ======================================================================================================================
# STEP 2: Convert the above dataset to a dataframe
# ======================================================================================================================
df = pd.DataFrame(data)

# ======================================================================================================================
# STEP 3: Display the original dataframe with missing values
# ======================================================================================================================
print("Original Dataframe with missing values:".center(50, '-'))
print(df)
print()

# ======================================================================================================================
# STEP 4: Handling the missing data/values
# ======================================================================================================================
# OPTION 1: Drop rows with missing values
df_dropna = df.dropna(how='any')
# Display the dataset after dropping rows with missing values
print("Dataset after dropping missing values:".center(50, '-'))
print(df_dropna)
print()
# OPTION 2. Impute/Fill in the missing values
# Fill in the mean age for missing values in 'Age' column
df.fillna({'Age': df['Age'].mean().round()}, inplace=True)
# Fill in the missing categorical data values 'City' with ffill
df['City'] = df['City'].ffill()

# Display the dataset after filling in the null values
print(f"Dataset after filling missing values:".center(50, '-'))
print(df)
print()
