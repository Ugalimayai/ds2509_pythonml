"""
======================================================================================================================
Python script to analyze and predict the unemployment rate based on GDP using the polynomial regression model
======================================================================================================================

Requirements
-------------------------
pip install matplotlib numpy scikit-learn pandas

Author: Karanja Mwaniki
Date: 27 08 2026
"""
# ======================================================================================================================
# SECTION 0: Import the required modules
# ======================================================================================================================
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# ======================================================================================================================
# SECTION 1: Getting, Loading and reshaping data from a csv file
# ======================================================================================================================
# Path to the csv file
file_path = os.path.abspath(os.path.join(os.getcwd(), "..","files","gdp_unemployment.csv"))

# load the data from the csv file
data = pd.read_csv(file_path)

# reshape the data
year = data['Year'].values.reshape(-1,1)
gdp = data['GDP'].values.reshape(-1,1)
unemployment_rate = data['Unemployment Rate'].values.reshape(-1,1)

# ======================================================================================================================
# SECTION 2: Create, Train, Predict and display predicted unemployment rate
# ======================================================================================================================
# Create the polynomial features
poly = PolynomialFeatures(degree = 2) #You can adjust the squared terms(degree) to suit your needs
X_poly = poly.fit_transform(gdp)

# Train/Fit the model
model = LinearRegression()
model.fit(X_poly, unemployment_rate)

# predict the unemployment rate for 2020 given a GDP of $620 Billion
gdp_2020 = np.array([[620]])
gdp_2020_poly = poly.transform(gdp_2020)
predicted_unemployment = model.predict(gdp_2020_poly)

# Display the predicted unemployment rate/percentage
print(f"The predicted unemployment rate for 2020 @ a GDP of $ 620 Billion is: {predicted_unemployment[0][0]:.2f}%.")


# ======================================================================================================================
# SECTION 3: Visualisation
# ======================================================================================================================
# plot the graph
plt.scatter(gdp,unemployment_rate, color='blue', label='Data Points')
gdp_range = np.linspace(290, 650,100).reshape(-1,1)
gdp_range_poly = poly.transform(gdp_range)
predicted_rates = model.predict(gdp_range_poly)
plt.plot(gdp_range, predicted_rates, color='red', label='Polynomial Fit')
plt.scatter(gdp_2020, predicted_unemployment, color='orange', label='2020 Prediction',
            s=100, edgecolors='black')

# plot labels
plt.title('GDP vs. Unemployment Rate')
plt.xlabel('GDP in Billion $')
plt.ylabel('Unemployment Rate (%)')
plt.legend()
plt.grid(True)
plt.show()
