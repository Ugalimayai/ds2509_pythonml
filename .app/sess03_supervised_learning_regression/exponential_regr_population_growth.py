# Python script to demonstrate the use of exponential regression to predict the
# population growth in a country

# Import the required modules
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.optimize import curve_fit

# Get the path to the file with the data to be loaded
file_path = os.path.abspath(os.path.join(os.getcwd(), "..", "files", "population_2009_2019.csv"))

# Load the data from the population csv file
data = pd.read_csv(file_path)

# Extract the years and population values
years = data['Year'].to_numpy()
population = data['Population'].to_numpy()

# Exponential function to calculate the population
def exponential_model(x, a, b):
   return a * np.exp(b * x)

# Log-transform the population for fitting (training the model)
log_population = np.log(population)

# Fit the model to the log-transformed data
initial_guess = [1, 0.1]
params, _ = curve_fit(lambda x, a, b: a + b * x, years, log_population, p0=initial_guess)

# Extract the fitted Parameters
a = np.exp(params[0])
b= params[1]

# Predict the population for the year 2022 using the exponential model
year_to_predict = 2022
predicted_population = exponential_model(year_to_predict, a, b)

# Display the predicted population for 2022
print(f"The predicted population is: {predicted_population:.2f} million people.")

# Visualise/Plot the data
# Generate the x and y values for plotting
x_values = np.linspace(2009,2025,100)
y_values = exponential_model(x_values, a, b)

plt.figure(figsize=(10,8))
plt.scatter(years, population, color='red',label='Actual Data')
plt.plot(x_values,y_values,label="Exponential Fit", color='blue')
plt.scatter(year_to_predict, predicted_population, color='green',label='Predicted Population 2022')
plt.title('Population Growth Prediction')
plt.xlabel('Year')
plt.ylabel('Population (in Millions)')
plt.legend()
plt.grid()
plt.show()