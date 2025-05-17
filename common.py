# common.py
import os

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Tuple

def quality_to_label(value: int) -> str:
    """
    Convert numeric wine quality score into a categorical label.

    Parameters:
        value (int): Numeric quality score.

    Returns:
        str: 'low', 'medium', or 'high' label based on score.
    """
    if value <= 4:
        return 'low'
    elif value <= 6:
        return 'medium'
    else:
        return 'high'

def load_and_preprocess_data(csv_path: str, selected_features: List[str], test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """
    Load CSV, encode labels, normalize features, and split into train/test sets.

    Parameters:
        csv_path (str): Path to dataset.
        selected_features (List[str]): Features to keep.
        test_size (float): Proportion of test data.
        random_state (int): Seed for reproducibility.

    Returns:
        Tuple: (X_train, X_test, y_train, y_test), original labels, label encoder
    """
    df = pd.read_csv(csv_path)
    df['quality_label'] = df['quality'].apply(quality_to_label)

    label_encoder = LabelEncoder()
    df['quality_encoded'] = label_encoder.fit_transform(df['quality_label'])

    X = df[selected_features]
    y = df['quality_encoded']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return train_test_split(X_scaled, y, test_size=test_size, random_state=random_state), df['quality_label'], label_encoder

def plot_label_distribution(labels: pd.Series):
    """
    Plot class distribution of categorical labels.

    Parameters:
        labels (pd.Series): Target label values.
    """
    sns.countplot(x=labels)
    plt.title('Distribution of Quality Labels')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

def apply_pca(X: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Principal Component Analysis (PCA) to reduce dimensionality.

    Parameters:
        X (np.ndarray): Feature matrix.
        n_components (int): Number of principal components to extract.

    Returns:
        Tuple: Transformed data, explained variance ratio.
    """
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca, pca.explained_variance_ratio_

def plot_pca_scatter(X_pca: np.ndarray, labels: pd.Series):
    """
    Scatter plot of PCA-reduced data colored by class label.

    Parameters:
        X_pca (np.ndarray): PCA-transformed data.
        labels (pd.Series): Corresponding class labels (can be encoded integers).
    """
    plt.figure(figsize=(8, 6))

    # Decode numeric labels if necessary
    if np.issubdtype(labels.dtype, np.integer):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.classes_ = np.array(['low', 'medium', 'high'])  # Ensure correct order
        labels = le.inverse_transform(labels)

    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels, palette='Set1')
    plt.title('PCA of Wine Quality Features')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Quality Label')
    plt.tight_layout()
    plt.savefig('images/pca_scatter_plot.png')
    plt.show()

def cross_validate_model(model, X, y, cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate model using cross-validation.

    Parameters:
        model: Scikit-learn estimator.
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.
        cv (int): Number of folds.

    Returns:
        Tuple: Mean accuracy and standard deviation.
    """
    scores = cross_val_score(model, X, y, cv=cv)
    return scores.mean(), scores.std()

def evaluate_and_plot_confusion(y_true, y_pred, class_names: List[str], title: str = "Confusion Matrix", filename: str = "confusion_matrix.png"):
    """
    Print classification metrics and plot/save confusion matrix heatmap.

    Parameters:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        class_names (List[str]): List of class label names.
        title (str): Plot title.
        filename (str): File to save plot.
    """
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join("images", filename))
    plt.show()

def apply_smote(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE to balance class distribution in training set.

    Parameters:
        X_train (np.ndarray): Feature matrix.
        y_train (np.ndarray): Class labels.
        random_state (int): Seed for reproducibility.

    Returns:
        Tuple: Resampled X_train and y_train.
    """
    sm = SMOTE(random_state=random_state)
    return sm.fit_resample(X_train, y_train)
