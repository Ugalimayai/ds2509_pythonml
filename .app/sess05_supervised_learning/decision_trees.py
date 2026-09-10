"""
======================================================================================================================
Python script to demonstrate the use of decision trees for classification
======================================================================================================================


Requirements
-------------------------
pip install matplotlib numpy pandas scikit-learn

Author: Karanja Mwaniki
Date: 04 Sept 2026
"""
# ======================================================================================================================
# SECTION 0: Import the required modules
# ======================================================================================================================
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

# ======================================================================================================================
# SECTION 1: Sample dataset of fruits and their corresponding labels
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
   0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

   # Oranges (label 1)
   1, 1, 1, 1, 1, 1, 1, 1, 1, 1,

   # Bananas (label 2)
   2, 2, 2, 2, 2, 2, 2, 2, 2, 2
])

# ======================================================================================================================
# SECTION 2: Feature definition and data train-test split
# ======================================================================================================================
# Define the features and their target names/labels for clarity
feature_names = ['Weight', 'Color Score']
target_names = ["Apple", "Orange", "Banana"]

#split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ======================================================================================================================
# SECTION 3: Decision Tree model initialisation, training, prediction & evaluation
# ======================================================================================================================
# Initialise the Decision Tree model
tree_clf = DecisionTreeClassifier(criterion='gini', random_state=42, max_depth=3)
# criterion: the rule the decision tree uses to evaluate the quality of a split

# Train the model
tree_clf.fit(X_train, y_train)

# Make the prediction
y_pred = tree_clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=target_names)

# ======================================================================================================================
# SECTION 4: Display and visualization
# ======================================================================================================================
# Display the results
print(f"Accuracy of the Decision Tree Model: {accuracy}")
print(f"Confusion Matrix of the Decision Tree Model:\n {conf_matrix}")
print(f"Classification Report of the Decision Tree Model:\n {class_report}")

# Visualise the decision tree
plt.figure(figsize=(20, 10))
plot_tree(tree_clf, feature_names=feature_names, filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree Structure for Fruit Classification", fontsize=15, pad=20)
plt.tight_layout()
plt.show()