"""
======================================================================================================================
Python script to demonstrate advanced time series forecasting models using Long Short-Term Memory (LSTM) and
Gate Recurrent Unit Models. (GRU)
======================================================================================================================


Requirements
-------------------------
pip install matplotlib numpy pandas torch scikit-learn tensorflow

Author: Karanja Mwaniki
Date: 18 September 2026
"""
from cProfile import label

# ======================================================================================================================
# SECTION 0: Import the required modules
# ======================================================================================================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# suppress warnings
import warnings
warnings.filterwarnings('ignore')

# ======================================================================================================================
# SECTION 1: Generate the data
# ======================================================================================================================
# Set random seed for reproducibility
np.random.seed(42)

# Function to generate synthetic data
def generate_synthetic_data(n_point=1000):
    time = np.arange(n_point)
    trend = time * 0.05
    seasonality = 10 * np.sin(time/30)
    noise = np.random.normal(scale=2, size=n_point)
    data = trend + seasonality + noise
    return pd.DataFrame({'date':pd.date_range('2020-01-01', periods=n_point, freq='D'),'value':data})

df = generate_synthetic_data()
# Display the first ten entries(days)
print(f"The first 10 entries of the time series are:\n{df.head(10)}")

# ======================================================================================================================
# SECTION 2: Visualisation and preprocessing
# ======================================================================================================================
# plot the synthetic data
plt.figure(figsize=(12,8))
plt.plot(df['date'], df['value'], label='Synthetic Time Series')
plt.title('Synthetic Time Series Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

# Pre-processing
train_size = int(len(df) * 0.8)
train_df, test_df = df[:train_size], df[train_size:]

scaler = MinMaxScaler(feature_range=(0,1))
scaled_train = scaler.fit_transform(train_df['value'].values.reshape(-1,1))
scaled_test = scaler.transform(test_df['value'].values.reshape(-1,1))

# Create time series generators for LSTM and GRU models
look_back = 30 # Use the last 30 days to predict the next days value
train_generator = TimeseriesGenerator(scaled_train, scaled_train, length=look_back, batch_size=32)
test_generator = TimeseriesGenerator(scaled_test, scaled_test, length=look_back, batch_size=32)

# Function to build and train LSTM/GRU models
def build_model(model_type='LSTM'):
    model = Sequential()

    if model_type == 'LSTM':
        model.add(LSTM(50, activation='relu', input_shape=(look_back, 1)))
    elif model_type == 'GRU':
        model.add(GRU(50, activation='relu', input_shape=(look_back, 1)))

    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Build and train the LSTM model
lstm_model = build_model('LSTM')
lstm_model.fit(train_generator, epochs=10, verbose=1)

# Build and train the GRU model
gru_model = build_model('GRU')
gru_model.fit(train_generator, epochs=10, verbose=1)

# Make predictions using the LSTM and GRU models
lstm_predictions = lstm_model.predict(test_generator)
gru_predictions = gru_model.predict(test_generator)

# Inverse scale the predictions and actual values to compare in original scale
lstm_predictions_rescaled = scaler.inverse_transform(lstm_predictions)
gru_predictions_rescaled = scaler.inverse_transform(gru_predictions)
test_actual_rescaled = scaler.inverse_transform(scaled_test[look_back:])

# Evaluate the model
lstm_rmse = np.sqrt(mean_squared_error(test_actual_rescaled, lstm_predictions_rescaled))
gru_rmse = np.sqrt(mean_squared_error(test_actual_rescaled, gru_predictions_rescaled))
lstm_mae = mean_absolute_error(test_actual_rescaled, lstm_predictions_rescaled)
gru_mae = mean_absolute_error(test_actual_rescaled, gru_predictions_rescaled)

print(f"LSTM RMSE: {lstm_rmse:.4f}")
print(f"GRU RMSE: {gru_rmse:.4f}")
print(f"LSTM MAE: {lstm_mae:.4f}")
print(f"GRU MAE: {gru_mae:.4f}")

# Plot the results
plt.figure(figsize=(12,8))
plt.plot(df['date'][train_size + look_back:], test_actual_rescaled, label='Actual Data')
plt.plot(df['date'][train_size + look_back:], lstm_predictions_rescaled, label='LSTM Predictions', linestyle='--')
plt.plot(df['date'][train_size + look_back:], gru_predictions_rescaled, label='GRU Predictions', linestyle='--')

plt.title("LSTM vs GRU Predictions")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.show()
