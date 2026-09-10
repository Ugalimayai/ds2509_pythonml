"""
======================================================================================================================
Python script to demonstrate K-Nearest Neighbours algorithm
======================================================================================================================
The script uses K-Nearest Neighbours to classify a fruit as either an apple or an orange, or a banana
based on its weight (grams) and colour score

Requirements
-------------------------
pip install matplotlib numpy scikit-learn

Author: Karanja Mwaniki
Date: 10 August 2026
"""
# ======================================================================================================================
# SECTION 0: Import the required modules
# ======================================================================================================================
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import warnings

# Suppress deprecation and future warnings for clean output
warnings.filterwarnings('ignore')

# ======================================================================================================================
# SECTION 1: Sample dataset & corresponding labels
# ======================================================================================================================
# Sample dataset of fruits
# (features: [weight in grams, colour score]), (labels: 0 -> apple, 1 -> orange, 2 -> banana)
X = np.array([
   # Apples
   [150, 0.80], [170, 0.75], [140, 0.85],
   [160, 0.82], [180, 0.78], [145, 0.88], [155, 0.76], [190, 0.80], [135, 0.87], [175, 0.79],

   # Oranges
   [130, 0.60], [120, 0.58], [115, 0.65],
   [140, 0.55], [125, 0.66], [110, 0.62], [100, 0.50], [135, 0.59], [145, 0.68], [105, 0.53],

   # Bananas
   [180, 0.55], [200, 0.50], [220, 0.48],
   [140, 0.42], [135, 0.45], [150, 0.46], [125, 0.51], [110, 0.43], [95, 0.57], [100, 0.49],
])

# Corresponding labels
y = np.array([
   # Apples (label 0)
   0,0,0,0,0,0,0,0,0,0,

   # Oranges (label 1)
   1,1,1,1,1,1,1,1,1,1,

   # Bananas (label 2)
   2,2,2,2,2,2,2,2,2,2,
])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ======================================================================================================================
# SECTION 2: Model initialisation, training, predictions, and evaluation
# ======================================================================================================================
# Initialise the KNN model with K = 5
k = 5
knn = KNeighborsClassifier(n_neighbors=k)

# Train the model
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=["Apple","Orange","Banana"])

# ======================================================================================================================
# SECTION 3: Results Visualisation
# ======================================================================================================================
# Visualise the decision boundaries
# Create mesh grid
x_min, x_max = X[:, 0].min() - 10, X[:, 0].max() + 10
y_min, y_max = X[:, 1].min() - .1, X[:, 1].max() + .10
xx, yy = np.meshgrid(np.arange(x_min, x_max, 1), np.arange(y_min, y_max, 0.01))

# Train again for visualisation  purposes
knn_plot = KNeighborsClassifier(n_neighbors=k)
knn_plot.fit(X_train, y_train)

# Plotting the decision boundary and the data points
plt.figure(figsize=(10, 6))
cmap_light = ListedColormap(['#ffaaaa', '#aaffaa', '#aaaaff'])
cmap_bold = ListedColormap(['#ff0000', '#00ff00', '#0000ff'])

# Predict on the mesh grid
Z = knn_plot.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_light, edgecolors='k', s=50)
plt.xlabel('Weight (grams)')
plt.ylabel('Colour Score')
plt.title(f"KNN Decision Boundary with K={k}")
plt.show()