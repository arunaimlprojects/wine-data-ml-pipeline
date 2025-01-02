import pandas as pd
import os

# Load preprocessed data
train_path = 'data/preprocessed/train_preprocessed.csv'
test_path = 'data/preprocessed/test_preprocessed.csv'

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Step 1: Feature Transformation (e.g., Log Transformation for skewed features)
skewed_features = ['Malic.acid', 'Proline']  # Example features
for feature in skewed_features:
    train_data[feature] = train_data[feature].apply(lambda x: x if x <= 0 else x**0.5)
    test_data[feature] = test_data[feature].apply(lambda x: x if x <= 0 else x**0.5)

# Step 2: Create Interaction Features (e.g., Product of two features)
train_data['Alcohol_Proline'] = train_data['Alcohol'] * train_data['Proline']
test_data['Alcohol_Proline'] = test_data['Alcohol'] * test_data['Proline']

# Step 3: Feature Selection (dropping irrelevant features, if any)
columns_to_drop = ['Ash']  # Example column
train_data.drop(columns=columns_to_drop, axis=1, inplace=True)
test_data.drop(columns=columns_to_drop, axis=1, inplace=True)

# Create the feature engineering folder
os.makedirs('data/feature_engineered', exist_ok=True)

# Save engineered data to the new folder
train_data.to_csv('data/feature_engineered/train_engineered.csv', index=False)
test_data.to_csv('data/feature_engineered/test_engineered.csv', index=False)

print("Feature engineering completed. Engineered data saved in 'data/feature_engineered' folder.")
