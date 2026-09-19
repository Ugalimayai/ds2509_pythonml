"""
======================================================================================================================
Python script to demonstrate advanced time series forecasting models using Long Short-Term Memory (LSTM) and
Gate Recurrent Unit Models. (GRU)
======================================================================================================================


Requirements
-------------------------
pip install matplotlib numpy pandas torch scikit-learn

Author: Karanja Mwaniki
Date: dd mmmm 2026
"""
# ======================================================================================================================
# SECTION 0: Import the required modules
# ======================================================================================================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from torch import nn, manual_seed
from torch.utils.data import TensorDataset, DataLoader

# suppress warnings
import warnings
warnings.filterwarnings('ignore')

# ======================================================================================================================
# SECTION 1: Generate the data
# ======================================================================================================================
# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

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
plt.plot(df['date'], df['value'])
plt.title('Synthetic Time Series Data')
plt.xlabel('Date')
plt.ylabel('Value')
plt.show()

# Pre-processing
train_size = int(len(df) * 0.8)
train_df, test_df = df[:train_size], df[train_size:]

scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train_df[['value']])
scaled_test = scaler.transform(test_df[['value']])

# Function to create sequences
def create_sequences(data, look_back=30):
   X, y = [], []
   for i in range(len(data) - look_back):
      X.append(data[i:i + look_back])
      y.append(data[i + look_back])
   return np.array(X), np.array(y)


look_back = 30
X_train, y_train = create_sequences(scaled_train, look_back)
X_test, y_test = create_sequences(scaled_test, look_back)

X_train = torch.tensor(X_train, dtype=torch.float32).view(-1, look_back, 1)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1)
X_test = torch.tensor(X_test, dtype=torch.float32).view(-1, look_back, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)


class RNNModel(nn.Module):
   def __init__(self, rnn_type="LSTM", input_size=1, hidden_size=50):
      super().__init__()
      self.rnn_type = rnn_type
      if rnn_type == "LSTM":
         self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
      else:
         self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
      self.fc = nn.Linear(hidden_size, 1)

   def forward(self, x):
      out, _ = self.rnn(x)
      return self.fc(out[:, -1, :])

def train(model, loader, epochs=10):
   model.train()
   optimizer = torch.optim.Adam(model.parameters())
   loss_fn = nn.MSELoss()
   for _ in range(epochs):
      for xb, yb in loader:
         pred = model(xb).squeeze()
         loss = loss_fn(pred, yb)
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()

def predict(model, X):
   model.eval()
   with torch.no_grad():
      return model(X).squeeze().numpy()

# Train LSTM and GRU models
lstm_model = RNNModel("LSTM")
gru_model = RNNModel("GRU")

train(lstm_model, train_loader)
train(gru_model, train_loader)

# Make predictions
lstm_preds = scaler.inverse_transform(predict(lstm_model, X_test).reshape(-1, 1))
gru_preds = scaler.inverse_transform(predict(gru_model, X_test).reshape(-1, 1))
actual = scaler.inverse_transform(y_test.view(-1, 1).numpy())

# Model Evaluation
print("LSTM RMSE:", np.sqrt(mean_squared_error(actual, lstm_preds)))
print("GRU RMSE:", np.sqrt(mean_squared_error(actual, gru_preds)))
print("LSTM MAE:", mean_absolute_error(actual, lstm_preds))
print("GRU MAE:", mean_absolute_error(actual, gru_preds))

# Plotting
dates = df['date'][train_size + look_back:]
plt.figure(figsize=(12, 5))
plt.plot(dates,actual, label='Actual')
plt.plot(dates,lstm_preds, '--', label='LSTM')
plt.plot(dates,gru_preds, '--', label='GRU')
plt.title("LSTM vs GRU Predictions")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.show()
