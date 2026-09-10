"""
======================================================================================================================
Python script to demonstrate working with derived features
======================================================================================================================


Requirements
-------------------------
pip install pandas

Author: Karanja Mwaniki
Date: 28 08 2026
"""
# ======================================================================================================================
# SECTION 0: Import the required modules
# ======================================================================================================================
import pandas as pd

# Sample dataset: Customer transactions with birthdate, purchase date, quantity and product price

data = pd.DataFrame({
   'customer_id': [1, 2, 1, 3, 2, 4, 5, 4, 6],
   'birthdate': ['1990-05-15', '1985-08-20', '1990-05-15', '1992-01-10', '1985-08-20', '1993-07-25', '1980-11-30',
                 '1993-07-25', '2000-09-12'],
   'purchase_date': ['2023-01-01', '2023-02-15', '2023-03-05', '2023-01-20', '2023-04-05', '2023-05-10', '2023-06-25',
                     '2023-07-15', '2023-08-05'],
   'quantity': [3, 1, 2, 5, 2, 1, 3, 4, 2],
   'product_price': [20, 15, 30, 10, 25, 50, 20, 15, 40]
})

# ======================================================================================================================
# SECTION 1: Convert to datetime objects and display the original dataset
# ======================================================================================================================
data['birthdate'] = pd.to_datetime(data['birthdate'])
data['purchase_date'] = pd.to_datetime(data['purchase_date'])

# Display the original dataset
print("Original Dataset")
print("-"*50)
print(data)
print("-"*50)

# ======================================================================================================================
# SECTION 2: Derive the various features
# ======================================================================================================================
# Feature 1: Calculate the age from birthdate
today = pd.to_datetime('today')
data['age'] = data['birthdate'].apply(lambda x:
                                      today.year - x.year -
                                      ((today.month, today.day) < (x.month, x.day)))

# Derived feature 2: Extracting the day of the week and month from the purchase date
data['day_of_week'] = data['purchase_date'].dt.day_name()
data['month_of_year'] = data['purchase_date'].dt.month

# Derived feature 3: Calculating the total sales (quantity * product price)
data['total_sales'] = data['quantity'] * data['product_price']

# Derived feature 4: Calculating cumulative total sales per customer
data['cumulative_sales'] = data.groupby(['customer_id'])['total_sales'].cumsum()

# Derived feature 5:  Average sales per customer
# Calculate average sales for each customer
average_sales_per_customer = data.groupby(['customer_id'])['total_sales'].mean().reset_index()
average_sales_per_customer.rename(columns={'total_sales': 'average_sales'}, inplace=True)

# Display the modified dataset with derived features
print("\nDataset with derived features")
print("-"*50)
print(data)
print("-"*50)
