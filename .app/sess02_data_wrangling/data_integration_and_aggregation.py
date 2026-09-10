"""
======================================================================================================================
Python script to demonstrate data integration and aggregation
======================================================================================================================


Requirements
-------------------------
pip install pandas

Author: Karanja Mwaniki
Date: 27 08 2026
"""
# ======================================================================================================================
# STEP 0: Import the required modules
# ======================================================================================================================
import pandas as pd

# ======================================================================================================================
# STEP 1: Create new datasets to be merged
# ======================================================================================================================
#a) Customer information dataset
customers = {
    'CustomerID' : [1,2,3],
    'Name' : ['Aaron', 'Brenda', 'Charlie']
}

# create the customer information dataframe
df_customers = pd.DataFrame(customers)

#b) Customer transaction dataset
transactions = {
    'CustomerID' : [1,1,2,3,3],
    'TransactionAmount' : [200,150,300,500,250],
    'TransactionDate' : ["2024-01-01","2024-01-15","2024-02-01","2024-03-01","2024-03-10"]
}

# create the customer transaction Dataframe
df_transactions = pd.DataFrame(transactions)

# Display the original datasets
print(f"Customer Information Dataset\n")
print("-"*55)
print(df_customers)
print("-"*55)
print(f"Customer Transaction Dataset\n")
print("-"*55)
print(df_transactions)
print("-"*55)

# ======================================================================================================================
# STEP 2: Integrate/Merge the customer and transaction details on a common field "CustomerID"
# ======================================================================================================================

df = pd.merge(df_customers, df_transactions, on='CustomerID')
print(f"\nMerged customer and transaction details dataset:\n{df}")

# ======================================================================================================================
# STEP 3: Aggregate the data -> Calculate the total amount spent by each customer
# ======================================================================================================================
df_aggregated = df.groupby('CustomerID')['TransactionAmount'].sum().reset_index()

# Display the aggregated dataset
print(f"\nTotal transaction amount per customer: \n{df_aggregated}")