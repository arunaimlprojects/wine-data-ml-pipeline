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

# Classification Report
class_report = classification_report(y_test, y_pred)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Print classification report and confusion matrix
print("\nClassification Report:")
print(class_report)

print("\nConfusion Matrix:")
print(conf_matrix)

# Save the evaluation results to a file
output_file = 'data/evaluation_results/evaluation_metrics.txt'
with open(output_file, 'w') as f:
    f.write(f'Model Accuracy: {accuracy * 100:.2f}%\n\n')
    f.write('Classification Report:\n')
    f.write(class_report)
    f.write('\nConfusion Matrix:\n')
    f.write(str(conf_matrix))

print(f"Evaluation results saved to '{output_file}'")
