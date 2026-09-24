"""
======================================================================================================================
Python script to demonstrate ensemble learning using bagging (Bootstrap Aggregating)
======================================================================================================================


Requirements
-------------------------
pip install matplotlib numpy scikit-learn

Author: Karanja Mwaniki
Date: 19 September 2026
"""
# ======================================================================================================================
# SECTION 0: Import the required modules
# ======================================================================================================================
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# suppress warnings
import warnings
warnings.filterwarnings('ignore')

# ======================================================================================================================
# SECTION 1: Create a synthetic dataset
# ======================================================================================================================
# Generate a synthetic binary classification dataset with 2 features which allows for
# easy 2d visualisation of decision boundaries
print("="*62)
print("BAGGING DEMONSTRATION WITH RANDOM FOREST")
print("="*62)

X,y = make_classification(n_samples=100,            # Total number of samples
                          n_features= 2,            # Two features for easy 2D visualisation
                          n_informative= 2,         # Both features are informative for classification
                          n_redundant= 0,           # No redundant features
                          n_clusters_per_class=1,   # Single cluster per class(default)
                          random_state=42           # For reproducibility
                          )

# Split the data into training and testing sets(70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Display the dataset summary
print("="*62)
print("\nDataset Summary")
print(f" - Total samples: {len(X)}")
print(f" - Training samples: {len(X_train)}")
print(f" - Test samples: {len(X_test)}")
print(f" - Number of Features: {X.shape[1]}")
print(f" - Class distribution (train): {np.bincount(y_train)}")

# ======================================================================================================================
# SECTION 2: Train the Random Forest Classifier
# ======================================================================================================================
print("=" * 62)
print("TRAINING RANDOM FOREST ENSEMBLE ")
print("=" * 62)
rf_model = RandomForestClassifier(
   n_estimators=10,  # Number of tree in the ensemble
   max_features='sqrt',  # sqrt(n_features) for classification
   bootstrap=True,  # Use bootstrap sampling for each tree
   oob_score=True,  # Compute out-of-bag (oob) score for validation
   n_jobs=-1,  # Use all available CPU cores
   random_state=42,  # For reproducibility
)

# Train the model
rf_model.fit(X_train, y_train)
print(f"\nRandom Forest Classifier trained with {rf_model.n_estimators} trees")
print(f"Out-of-Bag (OOB) Score: {rf_model.oob_score_:.3f}")
print(" (OOB score is an unbiased estimate of generalised error.)")

# ------------------------------------------------------------------------------------
# 3. Compare Individual Tree Accuracies vs. Ensemble Accuracy
# ------------------------------------------------------------------------------------
# Each tree in the Random Forest is trained on a different bootstrap sample. Each will be
# evaluated individually and compared to the ensemble
print("=" * 60)
print("ACCURACY COMPARISON: INDIVIDUAL TREES vs. ENSEMBLE")
print("=" * 60)

# List to store accuracy for each individual tree
tree_accuracies = []

print("\nIndividual Tree Test Accuracies:")
print('-' * 40)
for n, estimator in enumerate(rf_model.estimators_):
   y_pred_tree = estimator.predict(X_test)
   acc = accuracy_score(y_test, y_pred_tree)
   tree_accuracies.append(acc)
   print(f" Tree {n + 1:2d}: {acc:.3f}")

# Evaluate the ensemble (aggregated predictions)
y_pred_rf = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, y_pred_rf)

# Display ensemble details
print('-' * 40)
print(f"\nRandom Forest Ensemble Accuracy: {rf_acc:.3f}")
print(f"Improvement over average tree: {rf_acc - np.mean(tree_accuracies):.3f}")
print(f"Improvement over best tree: {rf_acc - np.max(tree_accuracies):.3f}")

# Visualise the accuracy comparisn
fig, ax = plt.subplots(figsize=(12, 8))

# Bar chart for individual tree accuracies
bars = ax.bar(
   range(1, len(rf_model.estimators_) + 1),
   tree_accuracies,
   color='skyblue',
   edgecolor='navy',
   alpha=0.8,
   label='Individual Tree Accuracy'
)

# Add a horizontal line for the ensemble's accuracy
ax.axhline(
   y=rf_acc,
   color='red',
   linestyle='--',
   linewidth=2,
   label=f"Ensemble Accuracy: {rf_acc:.3f}"
)

# Add a horizontal line for the average tree's accuracy
avg_tree_ac = np.mean(tree_accuracies)
ax.axhline(
   y=avg_tree_ac,
   color='blue',
   linestyle='-.',
   linewidth=2,
   label=f"Average Tree Accuracy: {avg_tree_ac:.3f}"
)

ax.set_xlabel('Tree Number', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title("Individual Tree Accuracy vs. Ensemble Accuracy", fontsize=14)
ax.set_xticks(range(1, len(tree_accuracies) + 1))
ax.set_ylim([.7, 1.0])
ax.legend(loc='lower right')
ax.grid(axis='y', alpha=.3)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------------
# 4. Visualise Bootstrap Samples for Individual Trees
# ------------------------------------------------------------------------------------
# Each decision tree in the Random Forest is trained on a bootstrap sample:
# a random sample drawn WITH REPLACEMENT from the original training data.
# This typically results in about 63.2% unique samples, with the rest being duplicates.
#
# The out-of-bag (OOB) samples (the ~36.8% not selected) serve as a validation set.
print("=" * 60)
print("BOOTSTRAP SAMPLE VISUALISATION")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Bootstrap Sample for the First 3 Trees', fontsize=14, y=1.02)

for n, estimator in enumerate(rf_model.estimators_[:3]):
   # Reconstruct the bootstrap sample indices using the estimator's random_sate
   # This is a close approximation of sklearns's internal bootstrap process
   rng = np.random.RandomState(estimator.random_state)

   # Sample with replacement (bootstrap)
   n_samples = len(X_train)
   sample_indices = rng.choice(n_samples, size=n_samples, replace=True)

   # Extract the bootstrap sample
   X_sample = X_train[sample_indices]
   y_sample = y_train[sample_indices]

   # Calculate bootstrap sample coverage (unique samples/total)
   n_unique = len(np.unique(sample_indices))
   coverage_pct = (n_unique / n_samples) * 100

   print(f"\nTree {n + 1} Bootstrap Statistics:")
   print(f"  - Unique samples: {n_unique}/{n_samples} ({coverage_pct:.2f}%)")
   print(f"  - Theoretical expectation: ~63.2% unique samples")

   # Plot the original data (faded background)
   axes[n].scatter(
      X_train[:, 0], X_train[:, 1],
      c=y_train,
      cmap='coolwarm',
      alpha=0.2,  # try .15
      s=40,
      label='Original Dataset'
   )

   # Plot the bootstrap sample (highlighted points with edges)
   axes[n].scatter(
      X_sample[:, 0], X_sample[:, 1],
      c=y_sample,
      cmap='coolwarm',
      edgecolor='black',
      linewidth=.5,
      s=50,
      alpha=0.9,
      label='Bootstrap Sample'
   )

   axes[n].set_xlabel('Feature 1', fontsize=10)
   axes[n].set_ylabel('Feature 2', fontsize=10)
   axes[n].set_title(f'Tree {n + 1} Bootstrap\nCoverage: {coverage_pct:.2f}% unique')
   axes[n].legend(loc='upper right', fontsize=9)
   axes[n].set_aspect('equal')

plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------------
# 5. Compare Decision Boundaries
# ------------------------------------------------------------------------------------
# Visualise how each tree partitions the feature space differently, and how the
# ensemble (aggregating votes) produces a smoother, more robust boundary
print("=" * 60)
print("DECISION BOUNDARY VISUALISATION")
print("=" * 60)

# Create a mesh grid for plotting decision boundaries
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(
   np.arange(x_min, x_max, .1),
   np.arange(y_min, y_max, .1)
)

# Create a 3 x 4 grid of subplots ( 3 rows, 4 columns)
fig, axes = plt.subplots(3, 4, figsize=(16, 12))
axes = axes.ravel()  # Flatten for easier indexing

print("\nPlotting decision boundaries for the first 10 trees...")

# Plot decision boundaries for the 1st 10 individual trees
for idx, estimator in enumerate(rf_model.estimators_):
   if idx >= 10:
      break

   # Predict class for each point in the mesh grid
   Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])
   Z = Z.reshape(xx.shape)

   # Plot decision boundary as filled contour
   axes[idx].contourf(xx, yy, Z, alpha=0.4, cmap='Blues', levels=1)

   # Plot the training data points
   axes[idx].scatter(
      X_train[:, 0], X_train[:, 1],
      c=y_train,
      cmap='coolwarm',
      alpha=0.8,
      s=15,
      edgecolor='k',
      linewidth=.3
   )

   axes[idx].set_xlim([x_min, x_max])
   axes[idx].set_ylim([y_min, y_max])
   axes[idx].set_title(f'Tree {idx + 1}', fontsize=11)
   axes[idx].set_aspect('equal')

# Plot the ensemble (Random Forest) decision boundary
print("Plotting Random Forest ensemble boundary...")
Z_ensemble = rf_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z_ensemble = Z_ensemble.reshape(xx.shape)

axes[10].contourf(xx, yy, Z_ensemble, cmap='Blues', alpha=0.4, levels=1)
axes[10].scatter(
   X_train[:, 0], X_train[:, 1],
   c=y_train,
   cmap='coolwarm',
   s=15,
   alpha=0.8,
   edgecolor='k',
   linewidth=0.3
)
axes[10].set_xlim(x_min, x_max)
axes[10].set_ylim(y_min, y_max)
axes[10].set_title('Random Forest (Ensemble)', fontsize=12, fontweight='bold')
axes[10].set_aspect('equal')

# Hide any unused subplots
for idx in range(11, 12):
   axes[idx].axis('off')

fig.suptitle("Decision Boundaries: Individual Trees vs. Ensemble\n"
             "Ensemble boundary is smoother and more robust",
             fontsize=14, y=1.02)
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------------
# 6. Summary Statistics
# ------------------------------------------------------------------------------------
print("=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)

print(f"\nModel Performance:")
print(f"  - Ensemble Test Accuracy:           {rf_acc:.3f}")
print(f"  - Out-of-Bag (OOB) Score:           {rf_model.oob_score_:.3f}")
print(f"  - Mean Tree Accuracy:               {np.mean(tree_accuracies):.3f}")
print(f"  - Std. Dev. of Tree Accuracy:       {np.std(tree_accuracies):.3f}")
print(f"  - Best Tree Accuracy:               {np.max(tree_accuracies):.3f}")
print(f"  - Worst Tree Accuracy:              {np.min(tree_accuracies):.3f}")

print(f"\nBootstrap Insights:")
print(f"  - Each tree trains on a different bootstrap sample")
print(f"  - Approximately 63.2% of original data appears in each sample")
print(f"  - The remaining 36.8% serves as out-of-bag validation data")
print(f"  - Diversity among trees reduces overfitting")

print(f"\nEnsemble Advantages:")
print(f"  ✓ Reduced variance compared to individual trees")
print(f"  ✓ More robust to noise and outliers")
print(f"  ✓ Smooth decision boundary from averaging/voting")
print(f"  ✓ Built-in validation via OOB samples")

print("=" * 60)
print("BAGGING DEMONSTRATION COMPLETE")
print("=" * 60)
