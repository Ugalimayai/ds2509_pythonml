"""
======================================================================================================================
Python script to detect and handle ouliers
======================================================================================================================

This script demonstrates how to detect and handle ouliers using the IQR(interquartile range)

Requirements
-------------------------
pip install numpy pandas

Author: Karanja Mwaniki
Date: 15 08 2026
"""
# ======================================================================================================================
# STEP 0: Import the required modules
# ======================================================================================================================
import pandas as pd
import numpy as np

# ======================================================================================================================
# STEP 1: Create a simple dataset
# ======================================================================================================================
data = {
    'Name': ['Abigail', 'Kamau', 'Sharlene', 'Diana', 'Mueni', 'Frank', 'Grace'],
    'Age': [5,35,32,29,45, 120, 28], # 5 and 120 are outliers
    'Salary': [50000, 2500, 54000, 52000, 110000, 47000, 51000] # 2500 and 110000 are outliers
}

# ======================================================================================================================
# STEP 2: Convert above dictionary into a dataframe and display it
# ======================================================================================================================
df = pd.DataFrame(data)
print("Original Dataframe".center(50, '-'))
print(df)
print()

# ======================================================================================================================
# STEP 3: Detect 'age' and 'salary' outliers and display them
# ======================================================================================================================
Q1 = df['Age'].quantile(.25) # First quartile (Q1)
Q3 = df['Age'].quantile(.75) # Third quartile (Q3)
IQR = Q3 - Q1 # Get the interquartile range for age
# Define the 'Age' outlier boundaries
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# TODO: Detect outliers using IQR for the 'Salary' column, display and follow similar steps below

# Identify and display the outliers for the 'Age' column
outliers = df[(df['Age'] < lower_bound) | (df['Age'] > upper_bound)]
print("Detected outliers for the 'Age' column".center(50, '-'))
print(outliers)
print()


# ======================================================================================================================
# STEP 4: Handle outliers -> Method i) Remove/drop outliers
# ======================================================================================================================
df_no_age_outliers = df[(df['Age'] >= lower_bound) & (df['Age'] <= upper_bound)]
# Display the dataset after removing outliers for the age column
print("Dataset after dropping outliers for the Age column".center(50, '-'))
print(df_no_age_outliers)


# ======================================================================================================================
# STEP 2: Handle outliers -> Method ii) Cap outliers with boundary values i.e. upper and lower bound values
# ======================================================================================================================
df['Age'] = np.where(df['Age'] < lower_bound, lower_bound, np.where(df['Age'] > upper_bound, upper_bound, df['Age']))
print("Dataset after dropping the outliers for the age column".center(50, '-'))
print(df)
print()
