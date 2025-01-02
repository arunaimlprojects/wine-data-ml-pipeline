import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os

# Make sure output directory exists
output_dir = 'data/evaluation_results'
os.makedirs(output_dir, exist_ok=True)

# Load test dataset
test_data = pd.read_csv('data/feature_engineered/test_engineered.csv')
print("Columns in test_data:", test_data.columns)

target_column = test_data.columns[0]
y_test = test_data[target_column]
X_test = test_data.drop(target_column, axis=1)

# Load trained model
model = joblib.load('data/models/random_forest_model.pkl')

# Predictions
y_pred = model.predict(X_test)

# Convert continuous to categorical if needed
if y_test.dtypes != 'int64':
    label_encoder = LabelEncoder()
    y_test = label_encoder.fit_transform(y_test)

if y_pred.dtype != 'int64':
    y_pred = label_encoder.transform(y_pred)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")

# Save metrics to text file
with open(os.path.join(output_dir, 'evaluation_metrics.txt'), 'w') as f:
    f.write(f"Model Accuracy: {accuracy:.4f}\n")
    # You can add more metrics if needed:
    # f.write(f"Other metric: {some_other_metric}\n")
