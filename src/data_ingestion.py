import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging


def load_data(data_url: str):
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(data_url)
        print(f"Data loaded successfully from {data_url}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise



def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str):
    """Save the train and test datasets."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        print(f"Train and test data saved in {raw_data_path}")
    except Exception as e:
        print(f"Error saving data: {e}")
        raise

def main():
    """Main function to run the data ingestion pipeline."""
    try:
        # Parameters (default values if params.yaml not used)
        test_size = 0.2
        data_path = './data'
        data_url = 'https://raw.githubusercontent.com/arunaimlprojects/datasets/refs/heads/main/wine.csv'

        # Load, preprocess, and split data
        df = load_data(data_url)
        #processed_df = preprocess_data(df)
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=2)

        # Save the data
        save_data(train_data, test_data, data_path)
        print("Data ingestion process completed successfully.")
    except Exception as e:
        print(f"Error in main process: {e}")

if __name__ == '__main__':
    main()
