"""
======================================================================================================================
Python script to demonstrate CF (Collaborative Filtering) recommender system with visualisation and
SVD decomposition
======================================================================================================================


Requirements
-------------------------
pip install matplotlib numpy pandas networkx scikit-learn

Author: Karanja Mwaniki
Date: 25 September 2026
"""
# ======================================================================================================================
# SECTION 0: Import the required modules
# ======================================================================================================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from networkx.algorithms import similarity
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

# ======================================================================================================================
# SECTION 1: Generate a synthetic dataset
# ======================================================================================================================
np.random.seed(42)
num_users = 1000
num_items = 500
sparsity = 0.9
ratings = np.random.randint(1, 5, size=(num_users, num_items))
ratings[np.random.rand(num_users, num_items)<sparsity] = 0 # 90% of randomly selected values are set to zero representing missing/unobserved entries.

user_ids = [f"User_{n + 1}" for n in range(num_users)]
item_ids = [f"Item_{n + 1}" for n in range(num_items)]
ratings_df = pd.DataFrame(ratings, index=user_ids, columns=item_ids)

# Display the dataset shape and its sparsity
print(f"Dataset shape: {ratings_df.shape}")
print(f"Sparsity: {(ratings_df==0).mean().mean():.2%}")

# ======================================================================================================================
# SECTION 2: Train Test Split (user-wise)
# ======================================================================================================================

def train_test_split_user_wise(ratings, test_size=0.2):
   """
   Splits a user-item ratings matrix into train and test sets on a per-user basis.

   For each user, a fraction of their rated items (non-zero entries) is randomly
   selected and moved to the test set, while the remaining ratings stay in the train set.

   Args:
       ratings (pd.DataFrame): User-item rating matrix.
       test_size (float): Proportion of each user's rated items to include in the test set.

   Returns:
       train (pd.DataFrame): Training dataset with some ratings removed.
       test (pd.DataFrame): Test dataset containing held-out ratings.
   """
   train = ratings.copy()
   test = pd.DataFrame(np.zeros_like(ratings), index=ratings.index, columns=ratings.columns)

   for user in ratings.index:
      rated_items = ratings.loc[user][ratings.loc[user] > 0].index
      if len(rated_items) > 1:
         test_items = np.random.choice(rated_items, size=int(len(rated_items)*test_size), replace=False)
         test.loc[user, test_items] = ratings.loc[user, test_items]
         train.loc[user, test_items] = 0

   return train, test

train_data, test_data = train_test_split_user_wise(ratings_df)

# ======================================================================================================================
# SECTION 3: User-user similarity
# ======================================================================================================================
def calculate_similarity(matrix, shrinkage=10):
    """
    Computes cosine similarity between users with optional shrinkage normalisation.

    The function converts the input matrix into a sparse format, computes pairwise
    cosine similarity, removes self-similarity, and applies shrinkage to stabilize
    similarity scores.

    Args:
        matrix (np.ndarray): User-item rating matrix.
        shrinkage (float): Regularization term to reduce the impact of small denominators.

    Returns:
        np.ndarray: User-user similarity matrix.
    """
    sparse_matrix = csr_matrix(matrix)
    similarity = cosine_similarity(sparse_matrix)
    np.fill_diagonal(similarity, 0)
    similarity = similarity / (np.sum(np.abs(similarity), axis=1)[:, np.newaxis] + shrinkage)
    return similarity

user_similarity = calculate_similarity(train_data.values)

# ======================================================================================================================
# SECTION 4: User-user prediction functions
# ======================================================================================================================
def predict(ratings, similarity, k=20):
   """
   Predicts ratings for all users using user-based collaborative filtering.

   For each user, the function identifies the top-k most similar users and computes
   a weighted average of their rating deviations from their mean ratings.

   Args:
       ratings (np.ndarray): User-item rating matrix.
       similarity (np.ndarray): User-user similarity matrix.
       k (int): Number of nearest neighbors to consider.

   Returns:
       np.ndarray: Predicted rating matrix, clipped to the range [1, 5].
   """
   mean_user_rating = np.divide(
       ratings.sum(axis=1),
       np.count_nonzero(ratings, axis=1),
       out=np.zeros(ratings.shape[0], dtype=float),
       where=np.count_nonzero(ratings, axis=1) != 0
   )

   ratings_diff = ratings - mean_user_rating[:, np.newaxis]

   # Don't allow unrated items to contribute
   ratings_diff = np.where(ratings > 0, ratings_diff, 0)

   pred = np.zeros_like(ratings, dtype=float)

   for i in range(ratings.shape[0]):
       user_sim = similarity[i]

       top_k_users = np.argsort(user_sim)[-k:]

       user_sim_k = np.zeros_like(user_sim, dtype=float)
       user_sim_k[top_k_users] = user_sim[top_k_users]

       weighted_sum = user_sim_k.dot(ratings_diff)
       sum_sim = np.abs(user_sim_k).sum()

       pred[i] = (
               mean_user_rating[i]
               + weighted_sum / (sum_sim + 1e-9)
       )

   return np.clip(pred, 1, 5)

def predict_single_user(user_index, ratings, similarity, k=20):
   """
   Predicts ratings for a single user using user-based collaborative filtering.

   The prediction is based on the top-k most similar users and their rating deviations.

   Args:
       user_index (int): Index of the target user.
       ratings (np.ndarray): User-item rating matrix.
       similarity (np.ndarray): User-user similarity matrix.
       k (int): Number of nearest neighbors to consider.

   Returns:
       np.ndarray: Predicted ratings for the specified user, clipped to [1, 5].
   """
   user_ratings = ratings[user_index]

   rated_count = np.count_nonzero(user_ratings)

   if rated_count == 0:
       mean_user_rating = 0.0
   else:
       mean_user_rating = user_ratings.sum() / rated_count

   user_means = np.divide(
       ratings.sum(axis=1),
       np.count_nonzero(ratings, axis=1),
       out=np.zeros(ratings.shape[0], dtype=float),
       where=np.count_nonzero(ratings, axis=1) != 0
   )

   ratings_diff = ratings - user_means[:, np.newaxis]

   # Ignore unrated items
   ratings_diff = np.where(ratings > 0, ratings_diff, 0)

   user_sim = similarity[user_index]

   top_k_users = np.argsort(user_sim)[-k:]

   user_sim_k = np.zeros_like(user_sim, dtype=float)
   user_sim_k[top_k_users] = user_sim[top_k_users]

   weighted_sum = user_sim_k.dot(ratings_diff)
   sum_sim = np.abs(user_sim_k).sum()

   pred = mean_user_rating + weighted_sum / (sum_sim + 1e-9)

   return np.clip(pred, 1, 5)

# ======================================================================================================================
# SECTION 5: Generate Predictions
# ======================================================================================================================
train_predictions = predict(train_data.values, user_similarity)

test_predictions = np.zeros_like(test_data.values)

for n, user in enumerate(test_data.index):
    user_idx = train_data.index.get_loc(user)
    test_predictions[n] = predict_single_user(user_idx, train_data.values, user_similarity)

# ======================================================================================================================
# SECTION 6: Evaluation
# ======================================================================================================================
def evaluate(actual, predicted):
    """
    Evaluates prediction performance using MSE, MAE, and coverage.

    Only non-zero entries in the actual matrix are considered for evaluation.

    Args:
        actual (np.ndarray): Ground truth rating matrix.
        predicted (np.ndarray): Predicted rating matrix.

    Returns:
        tuple:
            mse (float): Mean Squared Error.
            mae (float): Mean Absolute Error.
            coverage (float): Fraction of entries evaluated (non-zero in actual).
    """
    mask = actual > 0
    if np.sum(mask) == 0:
        return float("nan"), float("nan"), 0.0
    mse = mean_squared_error(actual[mask], predicted[mask])
    mae = np.mean(np.abs(actual[mask] - predicted[mask]))
    coverage = np.mean(mask)
    return mse, mae, coverage

mse_train, mae_train, cov_train = evaluate(train_data.values, train_predictions)
mse_test, mae_test, cov_test = evaluate(test_data.values, test_predictions)

print("\nTraining Metrics")
print(f"MSE: {mse_train:.3f}, MAE: {mae_train:.3f}, Coverage: {cov_train:.2%}")
print("\nTesting Metrics")
print(f"MSE: {mse_test:.3f}, MAE: {mae_test:.3f}, Coverage: {cov_test:.2%}")

# ======================================================================================================================
# SECTION 7: Visualisation
# ======================================================================================================================
def visualise_results(actual, predicted, title):
    """
    Visualises prediction performance with scatter plot and error distribution.

    Generates:
        1. Scatter plot comparing actual vs predicted ratings.
        2. Histogram of prediction errors.

    Args:
        actual (np.ndarray): Ground truth rating matrix.
        predicted (np.ndarray): Predicted rating matrix.
        title (str): Title prefix for the plots.

    Returns:
        None
    """
    mask = actual > 0
    if np.sum(mask) == 0:
        print(f"No data to visualize for {title}")
        return

    plt.figure(figsize=(12, 6))

    # Scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(actual[mask], predicted[mask], alpha=0.6)
    plt.plot([1, 5], [1, 5], 'r--')
    plt.title(f"{title}\nActual vs Predicted")
    plt.xlabel('Actual Rating')
    plt.ylabel('Predicted Rating')
    plt.grid(True)

    # Error histogram
    plt.subplot(1, 2, 2)
    errors = actual[mask] - predicted[mask]
    plt.hist(errors, bins=20, alpha=0.7)
    plt.title("Prediction Error Distribution")
    plt.xlabel('Error (Actual - Predicted)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


visualise_results(train_data.values, train_predictions, "Training Data")
visualise_results(test_data.values, test_predictions, "Test Data")

# ======================================================================================================================
# SECTION 8: Singular Value Decomposition Model based Decomposition
# ======================================================================================================================
def svd_recommend(train_data, k = 5):
    """
    Generates rating predictions using Singular Value Decomposition (SVD).

    The function factorizes the user-item matrix into latent factors and reconstructs
    the matrix using the top-k singular values.

    Args:
        train_data (np.ndarray): User-item rating matrix.
        k (int): Number of latent factors.

    Returns:
        np.ndarray: Reconstructed rating matrix with predicted values clipped to [1, 5].
    """
    sparse_matrix = csr_matrix(train_data)
    u, s, vt = svds(sparse_matrix, k = k)
    s = s[::-1] # Reverse to descending order
    u = u[:, ::-1]
    vt = vt[::-1, :]
    pred = np.dot(np.dot(u, np.diag(s)), vt)
    return np.clip(pred, 1, 5)

svd_prediction = svd_recommend(train_data.values)
mse_svd, mae_svd, cov_svd = evaluate(train_data.values, svd_prediction)

print("\nSVD (Model-Based) Prediction Results: ")
print(f"- MSE: {mse_svd:.3f} MAE {mae_svd:.3f} Coverage: {cov_svd:.2%}")
