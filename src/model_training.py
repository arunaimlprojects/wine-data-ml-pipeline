import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.utils.multiclass import type_of_target

# Load training and testing datasets
train_data = pd.read_csv('data/feature_engineered/train_engineered.csv')
test_data = pd.read_csv('data/feature_engineered/test_engineered.csv')

# Debug: Check column names
print("Columns in train_data:", train_data.columns)
print("Columns in test_data:", test_data.columns)

# Assuming target column is at index 0
y_train = train_data.iloc[:, 0]
X_train = train_data.iloc[:, 1:]
y_test = test_data.iloc[:, 0]
X_test = test_data.iloc[:, 1:]

# Debug: Ensure target is categorical
print("y_train type before conversion:", type_of_target(y_train))
print("y_test type before conversion:", type_of_target(y_test))

# Convert target to categorical if necessary
if type_of_target(y_train) != "multiclass":
    y_train = y_train.astype("int")
if type_of_target(y_test) != "multiclass":
    y_test = y_test.astype("int")

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'data/models/random_forest_model.pkl')
print("Model trained and saved successfully.")

# Evaluate the model
y_pred = model.predict(X_test)
print("y_pred type after prediction:", type_of_target(y_pred))

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
