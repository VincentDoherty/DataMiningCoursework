from common import load_and_preprocess_data, apply_pca, plot_pca_scatter, cross_validate_model
from sklearn.ensemble import RandomForestClassifier
import numpy as np

CSV_PATH = 'data/winequality-red.csv'
FEATURES = ['alcohol', 'sulphates', 'citric acid', 'volatile acidity', 'density', 'chlorides']

# Load and preprocess
(data, labels, encoder) = load_and_preprocess_data(CSV_PATH, FEATURES)
(X_train, X_test, y_train, y_test) = data

# PCA
X_pca, explained_var = apply_pca(X_train)
plot_pca_scatter(X_pca, y_train)
print("PCA Results:")
print(f"Explained variance by PC1: {explained_var[0]:.4f}")
print(f"Explained variance by PC2: {explained_var[1]:.4f}")
print(f"Total explained variance: {np.sum(explained_var):.4f}")

# Cross-validation
rf = RandomForestClassifier(random_state=42)
mean_acc, std_acc = cross_validate_model(rf, X_train, y_train, cv=5)
print("\n5-Fold Cross-Validation Results:")
print(f"Mean Accuracy: {mean_acc:.4f}")
print(f"Standard Deviation: {std_acc:.4f}")