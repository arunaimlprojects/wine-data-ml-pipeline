import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Load the dataset
data = pd.read_csv('C:\\Users\\hp\\Desktop\\MLOPS-LEARNING\\wine-data-ml-pipeline\\my_data\\wine.csv')

# Split the data into train and test
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Create directories for train and test data inside 'data' folder
os.makedirs('data/', exist_ok=True)  # If data folder doesn't exist, it will be created

# Create subfolders for train and test data
os.makedirs('data/data_ingestion', exist_ok=True)
os.makedirs('data/data_ingestion', exist_ok=True)

# Save train and test data to CSV files
train_data.to_csv('data/data_ingestion/train.csv', index=False)  # Save in specific folder
test_data.to_csv('data/data_ingestion/test.csv', index=False)  # Save in specific folder

print("Data ingestion completed. Train and test files saved in respective folders.")
