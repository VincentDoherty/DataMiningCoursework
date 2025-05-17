from common import load_and_preprocess_data, evaluate_and_plot_confusion
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline
from joblib import dump
import os
import matplotlib.pyplot as plt

# Setup
CSV_PATH = 'data/winequality-red.csv'
FEATURES = ['alcohol', 'sulphates', 'citric acid', 'volatile acidity', 'density', 'chlorides']
os.makedirs('models', exist_ok=True)

# Load data
(data, labels, label_encoder) = load_and_preprocess_data(CSV_PATH, FEATURES)
X_train, X_test, y_train, y_test = data

# Pipelines
pipelines = {
    'Logistic Regression': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=1000))
    ]),
    'Random Forest': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(random_state=42))
    ]),
    'SVM': Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC())
    ])
}

# Hyperparameter grids
param_grids = {
    'Logistic Regression': {
        'clf__C': [0.01, 0.1, 1, 10],
        'clf__penalty': ['l2'],
        'clf__solver': ['lbfgs']
    },
    'Random Forest': {
        'clf__n_estimators': [100, 200],
        'clf__max_depth': [None, 10, 20],
        'clf__min_samples_split': [2, 5]
    },
    'SVM': {
        'clf__C': [0.1, 1, 10],
        'clf__kernel': ['linear', 'rbf']
    }
}

# Model training and evaluation
results = {}

for name in pipelines:
    print(f"\nTuning {name}...")
    grid = GridSearchCV(pipelines[name], param_grids[name], cv=5, scoring='f1_macro')
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    acc = best_model.score(X_test, y_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    results[name] = {'accuracy': acc, 'f1_macro': f1}

    print(f"{name} - Best Parameters: {grid.best_params_}")
    print(f"{name} - Test Accuracy: {acc:.4f}")
    print(f"{name} - Macro F1 Score: {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    # Confusion matrix plot
    evaluate_and_plot_confusion(
        y_test, y_pred, label_encoder.classes_.tolist(),
        title=f"{name} (Tuned) - Confusion Matrix",
        filename=f"{name.lower().replace(' ', '_')}_tuned_conf_matrix.png"
    )

    # Cross-validation
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='f1_macro')
    print(f"{name} CV Macro F1 Score: Mean={cv_scores.mean():.4f}, Std={cv_scores.std():.4f}")

    # Save trained model
    dump(best_model, f"models/{name.lower().replace(' ', '_')}_model.joblib")

    # Feature importance (Random Forest only)
    if name == "Random Forest":
        rf_model = best_model.named_steps["clf"]
        importances = rf_model.feature_importances_

        plt.figure(figsize=(8, 5))
        plt.barh(FEATURES, importances)
        plt.title("Random Forest Feature Importance")
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.savefig("images/rf_feature_importance.png")
        plt.show()

# Summary
print("\nModel Performance Summary:")
for name, metrics in results.items():
    print(f"{name}: Accuracy = {metrics['accuracy']:.4f}, F1 Macro = {metrics['f1_macro']:.4f}")
