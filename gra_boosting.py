import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the dataset
file_path = 'winequality-red.csv'
df = pd.read_csv(file_path)

# Define features (X) and target (y)
X = df.drop(columns=["quality"])  # Features
y = df["quality"]                # Target variable

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Without PCA
model_no_pca = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model_no_pca.fit(X_train, y_train)
y_pred_no_pca = model_no_pca.predict(X_test)

# Evaluate without PCA
mae_no_pca = mean_absolute_error(y_test, y_pred_no_pca)
mse_no_pca = mean_squared_error(y_test, y_pred_no_pca)
r2_no_pca = r2_score(y_test, y_pred_no_pca)

print("\nWithout PCA:")
print(f"MAE: {mae_no_pca:.2f}")
print(f"MSE: {mse_no_pca:.2f}")
print(f"R^2: {r2_no_pca:.2f}")

# With PCA
pca = PCA(n_components=5)  # Choose the number of components based on explained variance
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

model_with_pca = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model_with_pca.fit(X_train_pca, y_train)
y_pred_with_pca = model_with_pca.predict(X_test_pca)

# Evaluate with PCA
mae_with_pca = mean_absolute_error(y_test, y_pred_with_pca)
mse_with_pca = mean_squared_error(y_test, y_pred_with_pca)
r2_with_pca = r2_score(y_test, y_pred_with_pca)

print("\nWith PCA:")
print(f"MAE: {mae_with_pca:.2f}")
print(f"MSE: {mse_with_pca:.2f}")
print(f"R^2: {r2_with_pca:.2f}")

# Comparison
print("\nPerformance Comparison:")
print(f"MAE Improvement: {mae_no_pca - mae_with_pca:.2f}")
print(f"MSE Improvement: {mse_no_pca - mse_with_pca:.2f}")
print(f"R^2 Improvement: {r2_with_pca - r2_no_pca:.2f}")

# Plot actual vs predicted values (Without PCA)
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_no_pca, alpha=0.6, color="blue", label="Without PCA")
plt.scatter(y_test, y_pred_with_pca, alpha=0.6, color="orange", label="With PCA")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--', label="Ideal Prediction Line")
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")
plt.title("Actual vs Predicted Quality")
plt.legend()
plt.show()
