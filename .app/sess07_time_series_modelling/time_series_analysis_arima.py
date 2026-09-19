"""
======================================================================================================================
Python script to demonstrate time series analysis using ARIMA model
======================================================================================================================


Requirements
-------------------------
pip install matplotlib numpy pandas scikit-learn statsmodels (version 0.14.6)

Author: Karanja Mwaniki
Date: 17 September 2026
"""
# ======================================================================================================================
# SECTION 0: Import the required modules
# ======================================================================================================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

#suppress warnings
import warnings

warnings.filterwarnings('ignore')

# ======================================================================================================================
# SECTION 1: Generate the data
# ======================================================================================================================
np.random.seed(42)
date_range = pd.date_range('2024-01-01', periods=365, freq='D')     # Dataset for 1yr (365 days)
trend = np.linspace(10, 20, 365)                            # Linear trend from 10 to 20
seasonal = 5 * np.sin(np.linspace(0, (3 *np.pi), 365))          # Sinusoidal seasonal component
noise = np.random.normal(0,2,365)                                        # Random noise
data = trend + seasonal + noise
time_series = pd.Series(data, index=date_range)

# Display the first 10 entries
print(f"The first 10 entries of the time series are:\n{time_series[:10]}")

# ======================================================================================================================
# SECTION 2: Visualisation and Decomposition
# ======================================================================================================================
# visualise the time series
plt.figure(figsize=(12,8))
plt.plot(time_series, label='Time Series Data')
plt.title('Synthetic Time Series Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

# Decomposition
decomposition = seasonal_decompose(time_series, model='seasonal')
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.figure(figsize=(12,8))
plt.subplot(4,1,1)
plt.plot(time_series, label='Original Data')
plt.legend(loc='upper left')
plt.subplot(4,1,2)
plt.plot(trend, label='Trend')
plt.legend(loc='upper left')
plt.subplot(4,1,3)
plt.plot(seasonal, label='Seasonal')
plt.legend(loc='upper left')
plt.subplot(4,1,4)
plt.plot(residual, label='Residuals')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# ======================================================================================================================
# SECTION 3: ARIMA Model & Forecasting
# ======================================================================================================================
# ARIMA model

train_size = int(len(time_series) * 0.8)
train, test = time_series[:train_size], time_series[train_size:]
model = ARIMA(train, order=(5, 1, 0))
model_fit = model.fit()
print(model_fit.summary())

# Forecasting
forecast = model_fit.forecast(steps=len(test))
plt.figure(figsize=(12,8))
plt.plot(train, label='Train Data')
plt.plot(test.index, test, label='Test Data')
plt.plot(test.index, forecast, label='Forecast Data')
plt.title('ARIMA Forecasting')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

# Calculate and display the Mean Squared Error(MSE)
mse = mean_squared_error(test, forecast)
print(f"Mean Squared Error: {mse:.2f}")

