# Python file to demonstrate an autoencoder using TensorFlow to detect anomalies in a
# synthetic dataset of daily customer transactions with features representing
# 'Transaction amount' and 'Transaction Frequency'

# Import the required modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

# Generate a synthetic dataset with Transaction Amount and Frequency
np.random.seed(42)
normal_data = np.random.normal([200, 50], scale=[50, 10], size=[1000, 2])  # Normal data
anomalies = np.array([[500, 5], [100, 90], [450, 10]])  # Add some anomalies
data = np.vstack([normal_data, anomalies])

# Create a dataframe for ease of use
df = pd.DataFrame(data, columns=['Transaction Amount', 'Transaction Frequency'])

# Display the first five records
print(f"First five transaction amounts & frequencies:\n{df.head(5)}")

# Build the autoencoder model
encoding_dim = 1  # Compress the data into a 1D latent space

# Encoder
input_data = tf.keras.Input(shape=(2,))
encoded = layers.Dense(2, activation="relu")(input_data)
encoded = layers.Dense(encoding_dim, activation="relu")(encoded)

# Decoder
decoded = layers.Dense(encoding_dim, activation="relu")(encoded)
decoded = layers.Dense(2)(decoded)

# Autoencoder model
autoencoder = models.Model(input_data, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder on normal data only
autoencoder.fit(normal_data, normal_data, epochs=50, batch_size=32, shuffle=True)

# Calculate the reconstruction errors for all data points
reconstructed_data = autoencoder.predict(data)
reconstruction_error = np.mean(np.square(data - reconstructed_data), axis=1)

# set an anomaly threshold (here, we use a percentile of the errors in normal data)
threshold = np.percentile(reconstruction_error[:1000], 95)

# Flag data points with high reconstruction errors as anomalies
anomalies_detected = reconstruction_error > threshold

# Visualise anomalies
plt.figure(figsize=(8, 6))
plt.scatter(df['Transaction Amount'], df['Transaction Frequency'], label='Normal',
            color='blue', alpha=.6)
plt.scatter(df['Transaction Amount'][anomalies_detected],
            df['Transaction Frequency'][anomalies_detected],
            label='Anomaly', color='red', marker='x', s=100)
plt.xlabel('Transaction Amount')
plt.ylabel('Transaction Frequency')
plt.title('Anomaly Detection with Autoencoder')
plt.legend()
plt.grid(True)
plt.show()

