# Python script to demonstrate several time series analysis techniques on a synthetic time series

# Import the required modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from unicodedata import decomposition

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Random seed value for reproducibility
np.random.seed(42)

# --------------------------------------------------------------------------
# 1. Generate synthetic Time Series Data with a Trend and Seasonality
# --------------------------------------------------------------------------

n_points = 500
time = np.arange(n_points)
trend = time * .1 # Linear trend
seasonality = 10 * np.sin(2 * np.pi * time/12 ) # Seasonal component with 12 cycles(months)
noise = np.random.normal(scale=2, size=n_points) # Some random noise
data = trend + seasonality + noise

# Put the data into a dataframe
df = pd.DataFrame(data,columns=['Value'])
df['Time'] = pd.date_range(start='2023-01-01',periods=n_points,freq='ME')
df.set_index('Time',inplace=True)

# Plot the synthetic data
plt.figure(figsize=(12,8))
plt.plot(df.index, df['Value'],label="Synthetic Time Series")
plt.title("Synthetic Time Series with Trend and Seasonality")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()

# --------------------------------------------------------------------------
# 2. Exploratory Data Analysis (EDA)
# --------------------------------------------------------------------------
# Plot the Rolling mean and Rolling standard deviation
rolling_mean = df['Value'].rolling(window=12).mean()
rolling_std = df['Value'].rolling(window=12).std()

# Visualise the dataframe for the rolling meand and std. dev.
plt.figure(figsize=(12,8))
plt.plot(df['Value'],label="Original Series")
plt.plot(rolling_mean,label="Rolling Mean",color='red')
plt.plot(rolling_std,label="Rolling Std",color='black')
plt.title("Rolling Mean and Standard Deviation")
plt.legend()
plt.show()

# --------------------------------------------------------------------------
# 3. Moving average and smoothing
# --------------------------------------------------------------------------
df['SMA_12'] = df['Value'].rolling(window=12).mean() # Simple Moving Average with a window of 12
df['EMA_12'] = df['Value'].ewm(span=12,adjust=False).mean() # Exponential Moving Avg. with a span of 12

# Visualise the Moving avg. & Smoothing
plt.figure(figsize=(12,8))
plt.plot(df['Value'],label="Original Series")
plt.plot(df['SMA_12'],label="12-Month SMA",color='red')
plt.plot(df['EMA_12'],label="12-Month EMA",color='green')
plt.title("Simple Moving and Exponential Moving Averages")
plt.legend()
plt.plot(df['Value'],label="Original Series")

# --------------------------------------------------------------------------
# 4. Autocorrelation (ACF) and Partial Autocorrelation (PACF) Functions
# --------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12,8))
plot_acf(df['Value'],ax=axes[0],lags=40)
plot_pacf(df['Value'],ax=axes[1],lags=40)
plt.show()

# --------------------------------------------------------------------------
# 5. Stationarity Testing with Augmented Dickey-Fuller Test
# --------------------------------------------------------------------------
def adf_test(timeseries):
   result = adfuller(timeseries)
   print(f"ADF Statistic: {result[0]}")
   print(f"p-value: {result[1]}")
   for key, value in result[4].items():
      print(f"Critical Value ({key}): {value}")

# Perform the ADF test
print("Augmented Dickey-Fuller Test for Stationarity")
adf_test(df['Value'])

# --------------------------------------------------------------------------
# 6. Decomposition of Time Series
# --------------------------------------------------------------------------
decomposition = sm.tsa.seasonal_decompose(df['Value'], model='additive',period=12)
fig = decomposition.plot()
fig.set_size_inches(14,10)
plt.show()

# --------------------------------------------------------------------------
# 7. Fitting an ARIMA Model
# --------------------------------------------------------------------------
# Since data has seasonality, we'll apply differencing
df['Value_diff'] = df['Value'] - df['Value'].shift(1)
df['Value_diff'] = df['Value'].dropna()

# Re-check for stationarity on the differenced data
print("\nADF Test on Differenced Data")
adf_test(df['Value_diff'].dropna())

# Fit the ARIMA model (p=1, d=1, q=1) as an example
model = ARIMA(df['Value'], order=(1,1,1))
arima_result = model.fit()

# Print summary of the ARIMA model
print("\nARIMA Model Summary")
print("-" * 55)
print(arima_result.summary())

# Plot the forecast
forecast_steps = 12
forecast = arima_result.get_forecast(steps=forecast_steps)
forecast_ci = forecast.conf_int()

# Plot the forecasted values
plt.figure(figsize=(12,8))
plt.plot(df['Value'], label="Historical Data")
plt.plot(pd.date_range(df.index[-1],periods=forecast_steps,freq='ME'),
         forecast.predicted_mean,label="Forecast",color='red')
plt.fill_between(pd.date_range(df.index[-1],periods=forecast_steps,freq='ME'),
                 forecast_ci.iloc[:,0],forecast_ci.iloc[:,1],color='pink',alpha=0.3)
plt.title("ARIMA Model Forecast")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()