import pandas as pd
from joblib import load
from sklearn.preprocessing import StandardScaler
from common import load_and_preprocess_data

# Define path to the trained model
MODEL_PATH = 'models/random_forest_model.joblib'
CSV_PATH = 'data/winequality-red.csv'
FEATURES = ['alcohol', 'sulphates', 'citric acid', 'volatile acidity', 'density', 'chlorides']

# Load model
model = load(MODEL_PATH)

# Load and preprocess data
(data, labels, label_encoder) = load_and_preprocess_data(CSV_PATH, FEATURES)
X_train, X_test, y_train, y_test = data

# Predict on test set
y_pred = model.predict(X_test)

# Inverse transform predictions to original labels
predicted_labels = label_encoder.inverse_transform(y_pred)

# Export predictions to CSV
import pandas as pd
df = pd.DataFrame(X_test, columns=FEATURES)
df['predicted_label'] = predicted_labels
df.to_csv("predictions/predicted_quality_output.csv", index=False)
print("Predictions exported to predicted_quality_output.csv")

# Visualize prediction distribution
import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(x=predicted_labels)
plt.title("Distribution of Predicted Wine Quality Labels")
plt.xlabel("Predicted Quality")
plt.ylabel("Count")
plt.savefig("predictions/predicted_distribution.png")
plt.show()

# Display a sample of predictions
print("Sample Predictions:")
for i in range(10):
    original_features = StandardScaler().fit(X_train).transform(X_test)
    for i in range(10):
        features = {FEATURES[j]: round(original_features[i][j], 2) for j in range(len(FEATURES))}
        print(f"Predicted: {predicted_labels[i]}  |  Features: {features}")

