from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load raw data
CSV_PATH = 'data/winequality-red.csv'
df = pd.read_csv(CSV_PATH)

FEATURES = ['alcohol', 'sulphates', 'citric acid', 'volatile acidity', 'density', 'chlorides']
TARGET = 'quality'

X = df[FEATURES].values
y = df[TARGET].values

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Fit Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
preds = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, preds)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("Linear Regression Results:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"Mean Absolute Error: {mae:.4f}")
print(f"R^2 Score: {r2:.4f}")

# Cross-validation on full dataset
cv_mse = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_squared_error')
cv_mae = cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_absolute_error')
cv_r2 = cross_val_score(model, X_scaled, y, cv=5, scoring='r2')

print("\nCross-Validation Scores (5-fold):")
print(f"MSE: Mean={-cv_mse.mean():.4f}, Std={cv_mse.std():.4f}")
print(f"MAE: Mean={-cv_mae.mean():.4f}, Std={cv_mae.std():.4f}")
print(f"R^2: Mean={cv_r2.mean():.4f}, Std={cv_r2.std():.4f}")

# Convert predictions to classes for comparison

def score_to_class(score):
    if score <= 4:
        return 'low'
    elif score <= 6:
        return 'medium'
    else:
        return 'high'

true_classes = [score_to_class(score) for score in y_test]
pred_classes = [score_to_class(score) for score in preds]

# Output sample predictions
print("\nSample predictions:")
for i in range(10):
    print(f"Predicted: {preds[i]:.2f} (\"{pred_classes[i]}\") | Actual: {y_test[i]} (\"{true_classes[i]}\")")

# Plot true vs predicted
plt.figure(figsize=(6, 6))
plt.scatter(y_test, preds, alpha=0.6)
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")
plt.title("Linear Regression: Actual vs Predicted")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.tight_layout()
plt.savefig("images/regression_actual_vs_predicted.png")
plt.show()
