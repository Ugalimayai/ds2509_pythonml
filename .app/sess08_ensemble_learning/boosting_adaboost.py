"""
======================================================================================================================
Python script to demonstrate ensemble learning using Boosting (Ada Boost)
======================================================================================================================


Requirements
-------------------------
pip install numpy plotly scikit-learn

Author: Karanja Mwaniki
Date: 19 September 2026
"""
# ======================================================================================================================
# SECTION 0: Import the required modules
# ======================================================================================================================
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# suppress warnings
import warnings
warnings.filterwarnings('ignore')

# ======================================================================================================================
# SECTION 1: Create a synthetic dataset
# ======================================================================================================================
X,y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialise the AdaBoost Classifier with Decision Trees as estimators
adaboost_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    random_state=42
)

# Train the model
adaboost_model.fit(X_train, y_train)

# Plot the decision boundary
x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = adaboost_model.predict(np.c_[xx.ravel(), yy.ravel()])

fig = go.Figure()
fig.add_trace(go.Contour(x=np.arange(x_min, x_max, 0.1), y=np.arange(y_min, y_max, 0.1), z=Z, colorscale='RdBu', showscale=True, opacity=0.6))

# Add scatter plot of the training points
fig.add_trace(go.Scatter(x=X_train[y_train==0][:, 0], y=X_train[y_train==0][:, 1], mode="markers", marker=dict(size=5, color="red"), name="Class 0"))
fig.add_trace(go.Scatter(x=X_train[y_train==1][:,0], y=X_train[y_train==1][:,1], mode="markers", marker=dict(size=5, color="blue"), name="Class 1"))

# Update the layout
fig.update_layout(title="Boosting with Adaboost - Decision Boundary", showlegend=True, width=1200, height=800)
fig.show()