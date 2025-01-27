# Importing necessary modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv('winequality-red.csv')
print("Dataset loaded successfully. First five rows:")
print(df.head())

X = df.drop('quality', axis=1)
y = df['quality']

# Standardizing features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



# PCA
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# Number of principal components retained
print(f'# Principal components retained: {pca.n_components_}')

# Contributions of original features to the retained principal components
components = pd.DataFrame(
    pca.components_,
    columns=X.columns,
    index=[f'Principal Component {i+1}' for i in range(pca.n_components_)]
)
print('\n# Contributions of each original feature to principal components:')
print(components)

# Feature contributions to all principal components
feature_contributions = np.sum(np.abs(pca.components_), axis=0)
sorted_indices = np.argsort(-feature_contributions)

# Separate retained and not retained features
retained_features = [X.columns[i] for i in sorted_indices[:pca.n_components_]]
not_retained_features = [X.columns[i] for i in sorted_indices[pca.n_components_:]]

print('\n# Retained features based on contributions:')
print(retained_features)

print('\n# Not retained features based on contributions:')
print(not_retained_features)

# Define a function for cross-validation with Grid Search
# Define a function for cross-validation with Grid Search
def cross_validate_svm_with_grid_search(X, y, folds, param_grid):
    metrics_without_pca = {"MAE": [], "MSE": [], "R2": []}
    metrics_with_pca = {"MAE": [], "MSE": [], "R2": []}

    y_true_all = []  # To store all the actual values
    y_pred_all_without_pca = []  # To store all predictions without PCA
    y_pred_all_with_pca = []  # To store all predictions with PCA

    for i, (X_train_idx, X_val_idx) in enumerate(folds):
        # Split the data into training and validation sets
        X_train, X_val = X.to_numpy()[X_train_idx], X.to_numpy()[X_val_idx]
        y_train, y_val = y.to_numpy()[X_train_idx], y.to_numpy()[X_val_idx]

        # Scale the data
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Apply PCA
        pca = PCA(n_components=0.95)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_val_pca = pca.transform(X_val_scaled)

        # Grid Search without PCA
        svr = SVR()
        grid_search_without_pca = GridSearchCV(svr, param_grid, cv=3, scoring='neg_mean_squared_error')
        grid_search_without_pca.fit(X_train_scaled, y_train)
        best_model_without_pca = grid_search_without_pca.best_estimator_
        y_pred_without_pca = best_model_without_pca.predict(X_val_scaled)

        # Store metrics without PCA
        metrics_without_pca["MAE"].append(float(mean_absolute_error(y_val, y_pred_without_pca)))
        metrics_without_pca["MSE"].append(float(mean_squared_error(y_val, y_pred_without_pca)))
        metrics_without_pca["R2"].append(float(r2_score(y_val, y_pred_without_pca)))

        # Store actual and predicted values for further visualization
        y_true_all.extend(y_val)
        y_pred_all_without_pca.extend(y_pred_without_pca)

        # Grid Search with PCA
        grid_search_with_pca = GridSearchCV(svr, param_grid, cv=3, scoring='neg_mean_squared_error')
        grid_search_with_pca.fit(X_train_pca, y_train)
        best_model_with_pca = grid_search_with_pca.best_estimator_
        y_pred_with_pca = best_model_with_pca.predict(X_val_pca)

        # Store metrics with PCA
        metrics_with_pca["MAE"].append(float(mean_absolute_error(y_val, y_pred_with_pca)))
        metrics_with_pca["MSE"].append(float(mean_squared_error(y_val, y_pred_with_pca)))
        metrics_with_pca["R2"].append(float(r2_score(y_val, y_pred_with_pca)))

        # Store predictions with PCA
        y_pred_all_with_pca.extend(y_pred_with_pca)

    return metrics_without_pca, metrics_with_pca, y_true_all, y_pred_all_without_pca, y_pred_all_with_pca, grid_search_without_pca.best_params_, grid_search_with_pca.best_params_




# Define hyperparameter grid
param_grid = {
    'C': [0.1, 1, 10],
    'epsilon': [0.1, 0.2, 0.3],
    'kernel': ['linear', 'rbf']
}

# Shuffle the data before splitting
shuffled_indices = np.random.permutation(len(X))
X = X.iloc[shuffled_indices].reset_index(drop=True)
y = y.iloc[shuffled_indices].reset_index(drop=True)

# Create folds
num_folds = 5
fold_size = len(X) // num_folds
folds = [
    (np.concatenate([np.arange(0, i * fold_size), np.arange((i + 1) * fold_size, len(X))]),
     np.arange(i * fold_size, (i + 1) * fold_size))
    for i in range(num_folds)
]
# Perform cross-validation with Grid Search
metrics_without_pca, metrics_with_pca, y_true_all, y_pred_all_without_pca, y_pred_all_with_pca, best_params_without_pca, best_params_with_pca = cross_validate_svm_with_grid_search(
    X, y, folds, param_grid
)

# Visualizations for metrics without PCA and with PCA
# MAE (Mean Absolute Error)
plt.figure(figsize=(10, 6))
plt.plot(metrics_without_pca["MAE"], label='Without PCA', color="orange")
plt.plot(metrics_with_pca["MAE"], label='With PCA', color="blue")
plt.xlabel("Fold")
plt.ylabel("MAE")
plt.title("MAE Comparison (With vs Without PCA)")
plt.legend()
plt.show()

# MSE (Mean Squared Error)
plt.figure(figsize=(10, 6))
plt.plot(metrics_without_pca["MSE"], label='Without PCA', color="orange")
plt.plot(metrics_with_pca["MSE"], label='With PCA', color="blue")
plt.xlabel("Fold")
plt.ylabel("MSE")
plt.title("MSE Comparison (With vs Without PCA)")
plt.legend()
plt.show()

# R2 Score
plt.figure(figsize=(10, 6))
plt.plot(metrics_without_pca["R2"], label='Without PCA', color="orange")
plt.plot(metrics_with_pca["R2"], label='With PCA', color="blue")
plt.xlabel("Fold")
plt.ylabel("R2")
plt.title("R2 Score Comparison (With vs Without PCA)")
plt.legend()
plt.show()


# Print cross-validation results
print("\nBest Parameters Without PCA:", best_params_without_pca)
print("\nCross-Validation Results (Without PCA):")
for metric, values in metrics_without_pca.items():
    print(f"{metric}: {values} (Mean: {np.mean(values):.2f})")

print("\nBest Parameters With PCA:", best_params_with_pca)
print("\nCross-Validation Results (With PCA):")
for metric, values in metrics_with_pca.items():
    print(f"{metric}: {values} (Mean: {np.mean(values):.2f})")

# Feature Importance Bar Plot
feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": feature_contributions
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance["Feature"], feature_importance["Importance"], color="skyblue")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importance in PCA")
plt.gca().invert_yaxis()  # Display the most important feature at the top
plt.show()

# Scatter plot for actual vs predicted values (with PCA)
plt.figure(figsize=(8, 6))
plt.scatter(y_true_all, y_pred_all_with_pca, alpha=0.6, color="orange")
plt.plot([min(y_true_all), max(y_true_all)], [min(y_true_all), max(y_true_all)], 'r--')  # Ideal prediction line
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality (With PCA)")
plt.title("Actual vs Predicted Quality (with PCA)")
plt.show()

# Scatter plot for actual vs predicted values (without PCA)
plt.figure(figsize=(8, 6))
plt.scatter(y_true_all, y_pred_all_without_pca, alpha=0.6, color="blue")
plt.plot([min(y_true_all), max(y_true_all)], [min(y_true_all), max(y_true_all)], 'r--')  # Ideal prediction line
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality (Without PCA)")
plt.title("Actual vs Predicted Quality (without PCA)")
plt.show()

