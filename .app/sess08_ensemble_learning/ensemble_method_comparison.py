# Python script to compare the various ensemble models

# -----------------------------------------------------------------------------
# 0. Import the required modules
# -----------------------------------------------------------------------------
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sess08_ensemble_learning.stacking import stacking_model
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Suppress warnings
import warnings

warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# 1. Generate a synthetic dataset for binary classification
# -----------------------------------------------------------------------------
X, y = make_classification(
    n_samples=1000,  # Number of samples
    n_features=20,  # Number of features
    n_informative=15,  # Number of informative features
    n_redundant=5,  # Number of redundant features
    random_state=42,  # Seed for reproducibility
    n_classes=2  # Number of classes (binary classification)
)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# -----------------------------------------------------------------------------
# 2. Define the 3 ensemble models
# -----------------------------------------------------------------------------
# a. Bagging model using Random Forest
bagging_model = RandomForestClassifier(n_estimators=100, random_state=42)

# b. Boosting Model using Gradient Boosting
boosting_model = GradientBoostingClassifier(n_estimators=100, random_state=42)

# c. Stacking Model using Logistic Regression meta model with diverse base learners
models = {
    'Bagging (Random Forest)': bagging_model,
    'Boosting (Gradient Boosting)': boosting_model,
    'Stacking ': stacking_model
}

results = {}

for name, model in models.items():
    # Train each model
    model.fit(X_train, y_train)

    # Get the prediction on the test set
    y_pred = model.predict(X_test)

    # Calculate the model's evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Store the result
    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Confusion Matrix': conf_matrix
    }

    # Display each model's results
    print(f"Results for {name}:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"Confusion Matrix:\n{conf_matrix}\n")

# -----------------------------------------------------------------------------
# 4. Visualisation of the Results using plotly
# -----------------------------------------------------------------------------
# Convert results dictionary to a Datafram for easier plotting
results_df = pd.DataFrame(results).T

# Bar chart for Model Comparison on Metrics
fig = go.Figure()

# Adding traces for each metric
for metric in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
    fig.add_trace(go.Bar(
        x=results_df.index,
        y=results_df[metric],
        name=metric,
    ))

fig.update_layout(
    title="Comparison of Ensemble Models on Different Metrics",
    xaxis_title="Model",
    yaxis_title="Score",
    barmode='group'
)
fig.show()

# Visualisation of the confusion matrices
for model_name, metrics in results.items():
    cm = metrics['Confusion Matrix']
    cm_fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                       title=f"Confustion Matrix for {model_name}",
                       labels=dict(x="Predicted Label", y="True Label", color="Count"))
    cm_fig.update_xaxes(side="bottom")
    cm_fig.show()

# -----------------------------------------------------------------------------
# 5. Enhanced Comparison Summary with Deeper Interpretation
# -----------------------------------------------------------------------------
print("\n==== Enhanced Comparison Summary ====")

# Calculate baseline accuracy (always predicting with the majority class)
baseline_accuracy = max(y_test.mean(), 1 - y_test.mean())
print(f"\nBaseline Accuracy (majority class): {baseline_accuracy:.2f}")

for name, metrics in results.items():
    print(f"\nModel: {name}")
    print(f" - Accuracy: {metrics['Accuracy']:.2f} (Baseline: {baseline_accuracy:.2f}")
    print(f" - Precision: {metrics['Precision']:.2f}")
    print(f" - Recall: {metrics['Recall']:.2f}")
    print(f" - F1 Score: {metrics['F1 Score']:.2f}")

    # Interpretation of confusion matrix
    tn, fp, fn, tp = metrics['Confusion Matrix'].ravel()
    print(f" - False Positives: {fp} | False Negatives: {fn}")

# Enhanced Interpretation
print("\n==== Detailed Interpretation ====")

# i. Overall performance comparison
best_overall = results_df['Accuracy'].idxmax()
print(f"\n1. Overall  Performance:")
print(f" - {best_overall} performs best overall with {results[best_overall]['Accuracy']:.2f} accuracy.")
print("- All ensemble methods should outperform the baseline accuracy of {:.2f}.".format(baseline_accuracy))

# ii Model-specific strengths
print(f"\n2. Model-Specific Strengths")
for name, metrics in results.items():
    strengths = []
    if metrics['Accuracy'] == results_df['Accuracy'].max():
        strengths.append('Best overall accuracy')
    if metrics['Precision'] == results_df['Precision'].max():
        strengths.append('Best precision (fewest false positives)')
    if metrics['Recall'] == results_df['Recall'].max():
        strengths.append('Best recall (fewest false negatives)')
    if metrics['F1 Score'] == results_df['F1 Score'].max():
        strengths.append('Best balance between precision and recall')

    if strengths:
        print(f" - {name} excels in : {', '.join(strengths)}")

# iii. Practical recommendations
print("\nPractical Recommendations:")
print("- For applications where false positives are costly (e.g., spam filtering):")
best_precision = results_df['Precision'].idxmax()
print(f" -> Choose {best_precision} (Precision: {results[best_precision]['Precision']:.2f})")

print("- For applications where false negatives are costly (e.g., medical diagnosis):")
best_recall = results_df['Recall'].idxmax()
print(f" -> Choose {best_recall} (Recall: {results[best_recall]['Recall']:.2f})")

print("- For balanced performance across metrics (general use cases):")
best_f1 = results_df['F1 Score'].idxmax()
print(f" -> Choose {best_f1} (F1 Score: {results[best_f1]['F1 Score']:.2f})")

# iv. Model characteristics explanation
print("\n4. Why these results? Understanding the models:")
print("- Bagging (Random Forest): Works well with diverse features by building many decorrelated trees.")
print("- Boosting (Gradient Boosting): Often achieves highest accuracy by sequentially correcting errors.")
print(" -Stacking can combine strengths of different models but may be more complex to tune.")

# v. Potential next steps
print("\n5. Potential next steps for improvements:")
print("- Try tuning hyperparameters for each model (especially for the underperforming ones)")
print("- Experiment with different base learners for the stacking model")
print("- Consider feature importance analysis to understand which features drive predictions")
print("- Test with different random seeds to check result stability")
