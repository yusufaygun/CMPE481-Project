import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('winequality-red.csv')

# Split features (X) and target variable (y)
X = data.drop(columns=['quality'])  # Input features
y = data['quality']  # Target variable (wine quality)

# Scale the features for standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model without PCA
X_train_no_pca, X_test_no_pca, y_train_no_pca, y_test_no_pca = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model_no_pca = LinearRegression()  # Create the Linear Regression model
model_no_pca.fit(X_train_no_pca, y_train_no_pca)  # Train the model

y_pred_no_pca = model_no_pca.predict(X_test_no_pca)  # Make predictions

# Evaluate model performance without PCA
mse_no_pca = mean_squared_error(y_test_no_pca, y_pred_no_pca)
mae_no_pca = mean_absolute_error(y_test_no_pca, y_pred_no_pca)
r2_no_pca = r2_score(y_test_no_pca, y_pred_no_pca)

print("Without PCA:")
print(f"MSE: {mse_no_pca:.4f}, MAE: {mae_no_pca:.4f}, R²: {r2_no_pca:.4f}")

# Apply PCA for dimensionality reduction
pca = PCA(n_components=0.95)  # Retain 95% variance
X_pca = pca.fit_transform(X_scaled)

# Model with PCA
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(
    X_pca, y, test_size=0.2, random_state=42
)

model_pca = LinearRegression()  # Create the Linear Regression model
model_pca.fit(X_train_pca, y_train_pca)  # Train the model

y_pred_pca = model_pca.predict(X_test_pca)  # Make predictions

# Evaluate model performance with PCA
mse_pca = mean_squared_error(y_test_pca, y_pred_pca)
mae_pca = mean_absolute_error(y_test_pca, y_pred_pca)
r2_pca = r2_score(y_test_pca, y_pred_pca)

print("With PCA:")
print(f"MSE: {mse_pca:.4f}, MAE: {mae_pca:.4f}, R²: {r2_pca:.4f}")

# Scatter plot: Actual vs Predicted (No PCA)
plt.figure(figsize=(10, 6))
plt.scatter(y_test_no_pca, y_pred_no_pca, alpha=0.7, color='blue', label='Predictions')
plt.plot([min(y_test_no_pca), max(y_test_no_pca)], [min(y_test_no_pca), max(y_test_no_pca)],
         color='red', linestyle='--', label='Perfect Prediction')
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")
plt.title("Linear Regression: Actual vs Predicted (No PCA)")
plt.legend()
plt.show()

# Scatter plot: Actual vs Predicted (With PCA)
plt.figure(figsize=(10, 6))
plt.scatter(y_test_pca, y_pred_pca, alpha=0.7, color='green', label='Predictions')
plt.plot([min(y_test_pca), max(y_test_pca)], [min(y_test_pca), max(y_test_pca)],
         color='red', linestyle='--', label='Perfect Prediction')
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")
plt.title("Linear Regression: Actual vs Predicted (With PCA)")
plt.legend()
plt.show()

# Residual Error Histogram (No PCA)
residuals_no_pca = y_test_no_pca - y_pred_no_pca
plt.figure(figsize=(10, 6))
plt.hist(residuals_no_pca, bins=20, edgecolor='k', alpha=0.7, color='blue')
plt.title("Residual Error Distribution (No PCA)")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.axvline(x=0, color='red', linestyle='--', label='Zero Error')
plt.legend()
plt.show()

# Residual Error Histogram (With PCA)
residuals_pca = y_test_pca - y_pred_pca
plt.figure(figsize=(10, 6))
plt.hist(residuals_pca, bins=20, edgecolor='k', alpha=0.7, color='green')
plt.title("Residual Error Distribution (With PCA)")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.axvline(x=0, color='red', linestyle='--', label='Zero Error')
plt.legend()
plt.show()

# Feature Importance (No PCA)
feature_importances_no_pca = pd.Series(
    model_no_pca.coef_, index=X.columns
).sort_values(ascending=False)

# Visualize feature importance for No PCA
plt.figure(figsize=(12, 6))
feature_importances_no_pca.plot(kind='bar', color='blue', edgecolor='black')
plt.title("Feature Importance (Linear Regression - No PCA)")
plt.ylabel("Coefficient Value")
plt.xlabel("Features")
plt.xticks(rotation=45, ha='right')  # Adjust for readability
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
