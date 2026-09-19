"""
======================================================================================================================
Python script to demonstrate time series forecasting using exponential smoothing
======================================================================================================================


Requirements
-------------------------
pip install matplotlib numpy pandas statsmodels scikit-learn

Author: Karanja Mwaniki
Date: dd mmmm 2026
"""
# ======================================================================================================================
# SECTION 0: Import the required modules
# ======================================================================================================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing, Holt

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# ======================================================================================================================
# SECTION 1: Generate Synthetic Time Series Data
# ======================================================================================================================
# set a random seed for reproducibility
np.random.seed(42)

# Create a dataset that includes a trend and seasonality for illustration
time_index = pd.date_range(start='2020-01-01', periods=100 ,freq='ME')
seasonal_pattern = np.sin(np.linspace(0,2*np.pi, 12)) # One year seasonality

# Extend seasonal pattern to match the length of the time series
seasonal_pattern_extended = np.tile(seasonal_pattern, int(np.ceil(100/12)))[:100]

data = (50 + np.arange(100) * .5 + seasonal_pattern_extended + np.random.normal(scale=2, size=100))
time_series = pd.Series(data, index=time_index)

# Plot the synthetic data
plt.figure(figsize=(10,5))
plt.plot(time_series, label='Synthetic Time Series Data')
plt.title('Synthetic Time Series Data')
plt.xlabel('Date')
plt.ylabel('Values')
plt.legend()
plt.show()

# ======================================================================================================================
# SECTION 2: Simple Exponential Smoothing
# ======================================================================================================================
# Ideal for time series without trend or seasonality
# Fit/Train the simple exponential smoothing model

ses_model = SimpleExpSmoothing(time_series).fit(smoothing_level=.2, optimized=True)
ses_forecast = ses_model.forecast(steps=12) # predict for the next 12 months

# Plot the SES results
plt.figure(figsize=(10,5))
plt.plot(time_series, label='Original Data')
plt.plot(ses_model.fittedvalues, label='SimpleExpSmoothing Fitted Values', color='green')
plt.plot(ses_forecast, label='SimpleExpSmoothing Forecast', color='red', linestyle='dashed')
plt.title('Simple Exponential Smoothing (SES)')
plt.xlabel("Date")
plt.ylabel("Values")
plt.legend()
plt.show()

# ======================================================================================================================
# SECTION 3: Holt's Linear Trend Model
# ======================================================================================================================
# Useful for data with a trend but no seasonality

# Fit/Train Holt's Linear trend model
holt_model = Holt(time_series).fit(smoothing_level=.8, smoothing_trend=.2, optimized=True)
holt_forecast = holt_model.forecast(steps=12)

# Plot Holt's Linear Trend Model
plt.figure(figsize=(10,5))
plt.plot(time_series, label='Original Data')
plt.plot(holt_model.fittedvalues, label='Holts Linear Fitted Values', color='orange')
plt.plot(holt_forecast, label='Holts Linear Forecast', color='red', linestyle='--')
plt.title('Holts Linear Trend Model')
plt.xlabel("Date")
plt.ylabel("Values")
plt.legend()
plt.show()

# ======================================================================================================================
# SECTION 4: Holt-Winters Seasonal Model
# ======================================================================================================================
# Ideal for data with trend and seasonality
# Using seasonal='add' for additive seasonality, 'mul' for multiplicative seasonality
hw_model = ExponentialSmoothing(time_series, trend='add', seasonal='add', seasonal_periods=12).fit()
hw_forecast = hw_model.forecast(steps=12)

# Plot Holt-Winters seasonal model results
plt.figure(figsize=(12,8))
plt.plot(time_series, label="Original Data")
plt.plot(hw_model.fittedvalues, label= "Holt Winters Fitted values", color='purple')
plt.plot(hw_forecast, label="Holt Winters Forecast", color='red', linestyle='--')
plt.title("Holt Winters Seasonal Model")
plt.xlabel("Date")
plt.ylabel("Values")
plt.legend()
plt.show()

# ======================================================================================================================
# SECTION 5: Evaluate and Interpret the results
# ======================================================================================================================
# Simple evaluation metrics to assess the accuracy of the fitted values

# Define a function for Mean Absolute Error (MAE)
def mean_absolute_error(y_true, y_pred):
   return np.mean(np.abs(y_true - y_pred))

# Calculate the errors for each model
ses_error = mean_absolute_error(time_series, ses_model.fittedvalues)
holt_error = mean_absolute_error(time_series, holt_model.fittedvalues)
hw_error = mean_absolute_error(time_series, hw_model.fittedvalues)

# Display the errors
print("Model Evaluation Metrics (MAE):")
print(f'Simple Exponential Smoothing MAE: {ses_error:.2f}')
print(f"Holt's Linear Trend Model MAE: {holt_error:.2f}")
print(f"Holt-Winters Seasonal Model MAE: {hw_error:.2f}")
