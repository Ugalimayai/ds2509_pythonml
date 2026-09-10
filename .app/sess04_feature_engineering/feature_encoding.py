# Python file to demonstrate one-hot encoding and label encoding on a sample data set

# Import the required modules
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Sample dataset of colours, sizes and cities
data = pd.DataFrame({
    'colour': [
        'Red', 'Blue', 'Green', 'Red', 'Blue',
        'Red', 'Red', 'Red', 'Blue', 'Red', 'Green', 'Red'
    ],
    'size': [
        'Small', 'Medium', 'Large', 'Small', 'Large',
        'Medium', 'Medium', 'Medium', 'Small', 'Large', 'Small', 'Medium'
    ],
    'city': [
        'London', 'Paris', 'London', 'New York', 'Paris',
        'London', 'London', 'London', 'London', 'London', 'New York', 'London'
    ]
})

# Display the original dataset
print("Original Dataset")
print("-" * 55)
print(data)
print("-" * 55)

# Display the value count for each 'colour', 'size' and 'city'
print("\nValue count for each 'colour', 'size' and 'city' ")
print("-" * 55)
print(data['colour'].value_counts())
print(data['size'].value_counts())
print(data['city'].value_counts())
print("-" * 55)

# --- One-Hot Encoding ---
# Apply One-Hot encoding to the 'colour' and 'city' columns
data_one_hot = pd.get_dummies(data, columns=['colour', 'city'], prefix=['colour', 'city'])

# Display the dataset after one-hot encoding it
print("\nData after One-Hot Encoding:")
print("-" * 55)
print(data_one_hot)
print("-" * 55)

# --- Label Encoding ---
# Label Encode the 'size' column as it has an inherent order (small < medium < large)
size_mapping = {'Small': 1, 'Medium': 2, 'Large': 3}  # Define order for size
data['size_encoded'] = data['size'].map(size_mapping)  # Map each category to an integer/number

# Alternatively, LabelEncoder can be used for non-ordinal encoding
label_encoder = LabelEncoder()
data['colour_encoded'] = label_encoder.fit_transform(data['colour'])

# Display the data after label encoding it
print("\nData after Label Encoding:")
print("-" * 55)
print(data[['size', 'size_encoded', 'colour', 'colour_encoded']])
print("-" * 55)

# --- Visualise the Encoded Data ---
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(14, 7))

# Plot one-hot encoded data for colours
one_hot_encoded = pd.get_dummies(data['colour'])
one_hot_encoded.sum().plot(kind='bar', ax=ax[0], color=['blue', 'green', 'red'])
ax[0].set_title("One-Hot Encoded 'Colour' Distribution")
ax[0].set_xlabel("Colours")
ax[0].set_ylabel("Frequency")

# Plot label encoded data for size
ax[1].hist(data['size_encoded'], color=['lightcoral'], edgecolor='black')
ax[1].set_xticks([1,2,3])
ax[1].set_xticklabels(['Small', 'Medium', 'Large'])
ax[1].set_title("Label Encoded 'Size' Distribution")
ax[1].set_xlabel("Size Encoded Values")
ax[1].set_ylabel("Frequency")

plt.tight_layout()
plt.show()


