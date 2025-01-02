import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Ensure 'data/models' folder exists
os.makedirs('data/models', exist_ok=True)

# Load data
train_data = pd.read_csv('data/feature_engineered/train_engineered.csv')
test_data = pd.read_csv('data/feature_engineered/test_engineered.csv')

# Separate features and target
X_train = train_data.drop('Wine', axis=1)
y_train = train_data['Wine']
X_test = test_data.drop('Wine', axis=1)
y_test = test_data['Wine']

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the trained model
model_path = 'data/models/random_forest_model.pkl'
joblib.dump(model, model_path)

print(f'Model trained and saved to {model_path}')
