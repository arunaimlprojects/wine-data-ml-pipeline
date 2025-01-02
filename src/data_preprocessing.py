import pandas as pd
from sklearn.preprocessing import StandardScaler
import os

# Load train and test data
train_path = 'C:\\Users\\hp\\Desktop\\MLOPS-LEARNING\\wine-data-ml-pipeline\\data\\train.csv'
test_path = 'C:\\Users\\hp\\Desktop\\MLOPS-LEARNING\wine-data-ml-pipeline\\data\\test.csv'

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

# Preprocessing: Handling missing values (if any)
# Fill numeric columns with mean
train_data.fillna(train_data.mean(), inplace=True)
test_data.fillna(test_data.mean(), inplace=True)

# Feature Scaling: Standardization
scaler = StandardScaler()

# Get feature columns (exclude target if present)
feature_columns = train_data.columns.drop('Wine')  # Assuming 'Wine' is the target column

train_data[feature_columns] = scaler.fit_transform(train_data[feature_columns])
test_data[feature_columns] = scaler.transform(test_data[feature_columns])

# Save preprocessed data back to disk
os.makedirs('data/preprocessed', exist_ok=True)

train_data.to_csv('data/preprocessed/train_preprocessed.csv', index=False)
test_data.to_csv('data/preprocessed/test_preprocessed.csv', index=False)

print("Preprocessing completed. Preprocessed data saved in 'data/preprocessed' folder.")