# Python file to demonstrate stacking on a synthetic dataset with plotly for visualisation

# -----------------------------------------------------------------------------
# 0. Import the required modules
# -----------------------------------------------------------------------------
import plotly.graph_objects as go
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# -----------------------------------------------------------------------------
# 1. Generate a synthetic dataset for binary classification
# -----------------------------------------------------------------------------
X, y = make_classification(
    n_samples=1000,  # Number of samples
    n_features=20,  # Number of features
    n_informative=15,  # Number of informative features
    n_redundant=5,  # Number of redundant features
    random_state=42  # Seed for reproducibility
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)

# -----------------------------------------------------------------------------
# 2. Define the base models
# -----------------------------------------------------------------------------
base_estimators = [
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),  # Random Forest
    ('knn', KNeighborsClassifier(n_neighbors=5)),  # K-nearest neighbours
    ('svm', SVC(kernel='linear', probability=True, random_state=42)),  # Support vector machines
]

# -----------------------------------------------------------------------------
# 3. Define the meta model
# -----------------------------------------------------------------------------
meta_model = LogisticRegression(random_state=42)

# -----------------------------------------------------------------------------
# 4. Build the stacking classifier
# -----------------------------------------------------------------------------
stacking_model = StackingClassifier(
    estimators=base_estimators,  # List of the base models
    final_estimator=meta_model,  # Meta model
    cv=5  # 5-fold cross validation for training the meta model
)

# -----------------------------------------------------------------------------
# 5. Train the stacking model
# -----------------------------------------------------------------------------
stacking_model.fit(X_train, y_train)

# -----------------------------------------------------------------------------
# 6. Make predictions on the test set
# -----------------------------------------------------------------------------
y_pred = stacking_model.predict(X_test)

# -----------------------------------------------------------------------------
# 7. Evaluate the accuracy of the stacking model
# -----------------------------------------------------------------------------
stacking_accuracy = accuracy_score(y_test, y_pred)

# Training each base model seperately for individual evaluation
# a. Random Forest
rf_model = RandomForestClassifier(n_estimators=50, random_state=42)
rf_model.fit(X_train, y_train)
rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))

# b. K-nearest neighbour
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_accuracy = accuracy_score(y_test, knn_model.predict(X_test))

# c. Support vector machine
svm_model = SVC(kernel='linear', probability=True, random_state=42)
svm_model.fit(X_train, y_train)
svm_accuracy = accuracy_score(y_test, svm_model.predict(X_test))

# Display individual model accuracies alongside the stacking model's accuracy
print(f"Random Forest Accuracy: {rf_accuracy:.2f}")
print(f"K-Nearest Neighbours Accuracy: {knn_accuracy:.2f}")
print(f"Support Vector Machine Accuracy: {svm_accuracy:.2f}")
print(f"Stacking Model(meta-model) Accuracy: {stacking_accuracy:.2f}")

# Plotting the accuracies of each model
accuracies = [rf_accuracy, knn_accuracy, svm_accuracy, stacking_accuracy]
model_names = ["Random Forest", "K-Nearest Neighbours", "Support Vector Machine", "Stacking Model"]

fig = go.Figure(
    data=[
        go.Bar(
            x=model_names,
            y=accuracies,
            text=[f"{acc:.2%}" for acc in accuracies],  # Display accuracy percentages
            textposition="auto",
            marker=dict(color=['#636EFA', '#EF553B', '#00CC96', '#AB63FA'])  # Different colours for clarity
        )
    ]
)

# Update the layout for readability
fig.update_layout(
    title="Model Accuracies - Individual Models vs. Stacking Model",
    xaxis_title="Model",
    yaxis_title="Accuracy",
    yaxis=dict(tickformat=".2%"),
    template="plotly_white"
)
fig.show()

