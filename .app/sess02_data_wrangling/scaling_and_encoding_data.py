"""
======================================================================================================================
Python script to demonstrate scaling and enccoding of data
======================================================================================================================


Requirements
-------------------------
pip install scikit-learn, pandas

Author: Karanja Mwaniki
Date: 27 08 2026
"""
# ======================================================================================================================
# STEP 0: Import the required modules
# ======================================================================================================================
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder, StandardScaler

# ======================================================================================================================
# STEP 1: Create a sample dataset
# ======================================================================================================================
data = {
    'Name': ['Abigail', 'Kamau', 'Sharlene', 'Diana', 'Mueni'],
    'Age': [55,32,22,45,28],
    'Salary': [50000, 54000, 52000, 110000, 51000],
    'Gender': ['Female', 'Male', 'Female', 'Female', 'Female'],
    'City': ["Nakuru", "Limuru", "Kisumu", "Homabay", "Makueni"]
}

# Convert above dataset into pandas dataframe
df = pd.DataFrame(data)

# ======================================================================================================================
# STEP 2: Standardise the Age and Salary
# ======================================================================================================================
scaler = StandardScaler() #uses the default range which is 0-1
df[['Age', 'Salary']] = scaler.fit_transform(df[['Age', 'Salary']])

# ======================================================================================================================
# STEP 3: One-Hot encode the Gender and City
# ======================================================================================================================
encoder = OneHotEncoder(sparse_output=False)
encoded_columns = pd.DataFrame(encoder.fit_transform(df[['Gender', 'City']]),
                               columns=encoder.get_feature_names_out(['Gender', 'City']))

# ======================================================================================================================
# STEP 4: Join/Concatenate the transformed data and display it
# ======================================================================================================================
df = pd.concat([df.drop(columns=['Gender', 'City']), encoded_columns], axis=1)
print(df)
