from common import load_and_preprocess_data, apply_smote, evaluate_and_plot_confusion
from sklearn.ensemble import RandomForestClassifier

CSV_PATH = 'data/winequality-red.csv'
FEATURES = ['alcohol', 'sulphates', 'citric acid', 'volatile acidity', 'density', 'chlorides']

# Load and preprocess data
(data, labels, label_encoder) = load_and_preprocess_data(CSV_PATH, FEATURES)
X_train, X_test, y_train, y_test = data

# Apply SMOTE to the training set
X_train_res, y_train_res = apply_smote(X_train, y_train)

# Train Random Forest on resampled data
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_res, y_train_res)
y_pred = clf.predict(X_test)

# Evaluation and confusion matrix
print("Random Forest with SMOTE")
evaluate_and_plot_confusion(
    y_test, y_pred, label_encoder.classes_.tolist(),
    title="Random Forest with SMOTE - Confusion Matrix",
    filename="rf_smote_conf_matrix.png"
)