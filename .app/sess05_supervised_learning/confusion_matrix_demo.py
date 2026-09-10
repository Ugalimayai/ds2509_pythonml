# Python file to demonstrate a Confusion matrix in evaluating predictions using
# a synthetic dataset

# Import the required modules
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay,roc_auc_score
from sklearn.model_selection import train_test_split


# 1. Generate a realistic dataset
# Simulate a binary classification problem with imbalance
X, y = make_classification(
   n_samples=1000,      # Total number of samples/observations in the dataset
   n_features=10,       # Total number of input features (columns) for each sample
   n_informative=5,     # Number of features that actually contribute useful information for classification
   n_redundant=2,       # Number of features that are generated as combinations of informative features
   n_classes=2,         # Number of target classes (binary classification: 0 and 1)
   weights=[.7, .3],    # Proportion of samples in each class: 70% class 0 and 30% class 1
   random_state=42,     # Sets a fixed random seed so the same dataset is generated each time
)


# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
   X, y,
   test_size=0.3,       # Allocate 30% of the data to the testing set and 70% to the training set
   stratify=y,          # Preserve the original class distribution in both training and testing sets
   random_state=42     # Sets a fixed random seed so the split is reproducible
)


# 3. Train a model
model = LogisticRegression()
model.fit(X_train, y_train)

# 4. Predictions
# Default predictions (threshold = 0.5)
y_pred = model.predict(X_test)

# Probabilities for threshold tuning
y_probs = model.predict_proba(X_test)[:,1]

# 5. Confusion Matrix
conf_matrix = confusion_matrix(y_test,y_pred)
print(f'Confusion Matrix:\n{conf_matrix}')

# 6. Classification Report
print(f"\nClassification Report:\n{classification_report(y_test,y_pred)}")

# 7. ROC-AUC Score
roc_score = roc_auc_score(y_test,y_probs)
print(f"\nROC AUC score:\n{roc_score:.3f}")

# 8. Visualisation
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix (Default Threshold = .5')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 9. Threshold Tuning
# Adjust the threshold to show impact on confusion matrix
threshold = .3
y_pred_custom = (y_probs >= threshold).astype(int)
conf_matrix_custom = confusion_matrix(y_test,y_pred_custom)

print(f"\nConfusion Matrix (Threshold = {threshold:.2f}):\n{conf_matrix_custom}")
print(f"\nClassification Report (Custom Threshold)\n{classification_report(y_test,y_pred_custom)}")

# Visualise the new/custom confusion matrix
disp_custom = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_custom)
disp_custom.plot(cmap='Oranges')
plt.title(f'Confusion Matrix (Custom Threshold = {threshold})')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()