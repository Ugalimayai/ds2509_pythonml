# Python file to demonstrate feature scaling

# Import the required modules
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Sample dataset
data = pd.DataFrame({
'height_cm': [150, 160, 165, 170, 180,155, 162, 167, 175, 185, 155, 168],
   'weight_kg': [65, 70, 72, 75, 80, 68, 73, 74, 78, 85, 66, 77],
'income': [30000, 50000, 70000, 100000, 120000,35000, 60000, 80000, 110000, 130000, 32000, 105000]
})

# Display the original dataset
print(f"Original Dataset")
print("-" * 55)
print(data)
print("-" * 55)

# Plot the original dataset distribution
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
axes[0].hist(data['height_cm'],bins=5,color='lightblue',edgecolor='black')
axes[0].set_title('Original Height(cm) Distribution')
axes[1].hist(data['weight_kg'],bins=5,color='salmon',edgecolor='black')
axes[1].set_title('Original Weight(kg) Distribution')
axes[2].hist(data['income'],bins=5,color='lightgreen',edgecolor='black')
axes[2].set_title('Original Income(Kes.) Distribution')
plt.show()

# Standardisation using the StandardScaler
scaler = StandardScaler()
data[['height_standardised','weight_standardised','income_standardised']]\
   = scaler.fit_transform(data[['height_cm','weight_kg','income']])

# Min-Max scaling using the MinMaxScaler
min_max_scaler = MinMaxScaler()
data[['height_minmax','weight_minmax','income_minmax']]\
   = min_max_scaler.fit_transform(data[['height_cm','weight_kg','income']])

# Display the scaled data
print("\nData after standardisation".center(60))
print("-" * 60)
print(data[['height_standardised','weight_standardised','income_standardised']])
print("-" * 60)

print("\nData after MinMax scaling".center(60))
print("-" * 60)
print(data[['height_minmax','weight_minmax','income_minmax']])
print("-" * 60)

# Plot the scaled data distributions
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10,10))

# Standardised data
ax[0,0].hist(data['height_standardised'],bins=5,color='lightblue',edgecolor='black')
ax[0,0].set_title('Standardised Height')
ax[0,1].hist(data['weight_standardised'],bins=5,color='salmon',edgecolor='black')
ax[0,1].set_title('Standardised Weight')
ax[0,2].hist(data['income_standardised'],bins=5,color='lightgreen',edgecolor='black')
ax[0,2].set_title('Standardised Income')

# Min-Max data
ax[1,0].hist(data['height_minmax'],bins=5,color='lightblue',edgecolor='black')
ax[1,0].set_title('Min-Max Height')
ax[1,1].hist(data['weight_minmax'],bins=5,color='salmon',edgecolor='black')
ax[1,1].set_title('Min-Max Weight')
ax[1,2].hist(data['income_minmax'],bins=5,color='lightgreen',edgecolor='black')
ax[1,2].set_title('Min-Max Income')

# Specify the layout and display
plt.tight_layout()
plt.show()