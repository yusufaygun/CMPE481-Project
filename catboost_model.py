import time
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv('winequality-red.csv')
X = data.drop(columns=['quality'])  # Features
y = data['quality']  # Target variable

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start the timer for hyperparameter tuning
start_time = time.time()

# Prepare data for PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Standardize the features

# Perform PCA to retain 95% of variance
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

# Split PCA-transformed data into training and test sets
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Define a parameter grid for GridSearchCV
param_grid_cb = {
    'iterations': [300, 325, 350, 375, 400, 425],  # Number of boosting rounds
    'depth': [6, 8, 10, 12, 14],           # Tree depth (complexity of trees)
    'learning_rate': [0.02, 0.03, 0.05, 0.07],  # Learning rate for boosting
    'l2_leaf_reg': [1, 2, 3, 5, 7],       # Regularization parameter
    'bagging_temperature': [0.3, 0.5, 0.7, 1.0] # Adding randomness to reduce overfitting
}


# Train CatBoost without PCA
catboost_no_pca = GridSearchCV(
    estimator=CatBoostRegressor(random_state=42, verbose=0),
    param_grid=param_grid_cb,
    scoring='neg_mean_squared_error',  # Optimize for MSE
    cv=5,  # Use 5-fold cross-validation
    verbose=1
)
catboost_no_pca.fit(X_train, y_train)

# Train CatBoost with PCA
catboost_with_pca = GridSearchCV(
    estimator=CatBoostRegressor(random_state=42, verbose=0),
    param_grid=param_grid_cb,
    scoring='neg_mean_squared_error',  # Optimize for MSE
    cv=5,  # Use 5-fold cross-validation
    verbose=1
)
catboost_with_pca.fit(X_train_pca, y_train_pca)

# End the timer for hyperparameter tuning
tuning_end_time = time.time()
print(f"Hyperparameter Tuning Time: {tuning_end_time - start_time:.2f} seconds")

# Print the best parameters for both models
print("Best parameters no pca (CatBoost):", catboost_no_pca.best_params_)
print("Best parameters with pca (CatBoost):", catboost_with_pca.best_params_)

# Predict on the test set without PCA
best_model_no_pca = catboost_no_pca.best_estimator_
y_pred_no_pca = best_model_no_pca.predict(X_test)

# Predict on the test set with PCA
best_model_with_pca = catboost_with_pca.best_estimator_
y_pred_with_pca = best_model_with_pca.predict(X_test_pca)

# Evaluate performance
mse_no_pca = mean_squared_error(y_test, y_pred_no_pca)
mae_no_pca = mean_absolute_error(y_test, y_pred_no_pca)
r2_no_pca = r2_score(y_test, y_pred_no_pca)

mse_with_pca = mean_squared_error(y_test_pca, y_pred_with_pca)
mae_with_pca = mean_absolute_error(y_test_pca, y_pred_with_pca)
r2_with_pca = r2_score(y_test_pca, y_pred_with_pca)

print("\nNo PCA:")
print("Best Model - MSE:", mse_no_pca)
print("Best Model - MAE:", mae_no_pca)
print("Best Model - R²:", r2_no_pca)

print("\nWith PCA:")
print("Best Model - MSE:", mse_with_pca)
print("Best Model - MAE:", mae_with_pca)
print("Best Model - R²:", r2_with_pca)

# Visualize Actual vs Predicted (No PCA)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_no_pca, alpha=0.7, color='blue', label='Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Perfect Prediction')
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")
plt.title(f"CatBoost: Actual vs Predicted (No PCA)\nMSE: {mse_no_pca:.3f}, R²: {r2_no_pca:.3f}")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Visualize Actual vs Predicted (With PCA)
plt.figure(figsize=(10, 6))
plt.scatter(y_test_pca, y_pred_with_pca, alpha=0.7, color='green', label='Predictions')
plt.plot([min(y_test_pca), max(y_test_pca)], [min(y_test_pca), max(y_test_pca)], color='red', linestyle='--', label='Perfect Prediction')
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")
plt.title(f"CatBoost: Actual vs Predicted (With PCA)\nMSE: {mse_with_pca:.3f}, R²: {r2_with_pca:.3f}")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Visualize Residual Errors (No PCA)
residuals_no_pca = y_test - y_pred_no_pca
plt.figure(figsize=(10, 6))
plt.hist(residuals_no_pca, bins=20, edgecolor='k', alpha=0.7, color='blue')
plt.axvline(residuals_no_pca.mean(), color='red', linestyle='--', label='Mean Residual')
plt.title("Residual Error Distribution (No PCA)")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Visualize Residual Errors (With PCA)
residuals_with_pca = y_test_pca - y_pred_with_pca
plt.figure(figsize=(10, 6))
plt.hist(residuals_with_pca, bins=20, edgecolor='k', alpha=0.7, color='green')
plt.axvline(residuals_with_pca.mean(), color='red', linestyle='--', label='Mean Residual')
plt.title("Residual Error Distribution (With PCA)")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

# Feature Importance (No PCA)
feature_importances = pd.Series(best_model_no_pca.get_feature_importance(), index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(14, 8))  # Larger plot for better visibility
colors = ['red' if i < 2 else 'blue' for i in range(len(feature_importances))]
feature_importances.plot(kind='bar', color=colors, edgecolor='k', alpha=0.8)
plt.title("Feature Importance (CatBoost - No PCA)", fontsize=16)
plt.ylabel("Importance Score", fontsize=14)
plt.xlabel("Features", fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=12)  # Rotate and adjust labels
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()  # Ensure labels fit in the plot
plt.show()
