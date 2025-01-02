import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the model
model = joblib.load('data/models/random_forest_model.pkl')

# Load the test data
test_data = pd.read_csv('data/feature_engineered/test_engineered.csv')

# Separate features and target
X_test = test_data.drop('Wine', axis=1)
y_test = test_data['Wine']

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Model Accuracy on Test Data: {accuracy * 100:.2f}%')

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
