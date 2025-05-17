from common import load_and_preprocess_data, evaluate_and_plot_confusion
from sklearn.naive_bayes import GaussianNB

# Parameters
CSV_PATH = 'data/winequality-red.csv'
FEATURES = ['alcohol', 'sulphates', 'citric acid', 'volatile acidity', 'density', 'chlorides']

# Load and preprocess data
(data, labels, label_encoder) = load_and_preprocess_data(CSV_PATH, FEATURES)
X_train, X_test, y_train, y_test = data

# Train Naïve Bayes classifier
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)

# Evaluate and plot confusion matrix
print("Naïve Bayes Classifier")
evaluate_and_plot_confusion(y_test, y_pred, label_encoder.classes_.tolist(), title="Naïve Bayes Confusion Matrix", filename="naive_bayes_conf_matrix.png")