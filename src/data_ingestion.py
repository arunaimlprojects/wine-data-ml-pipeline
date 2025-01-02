import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# Load parameters from params.yaml
with open('params.yaml') as f:
    params = yaml.safe_load(f)

# Fetch 'test_size' parameter
test_size = params['data_ingestion']['test_size']

# Ingestion process
df = pd.read_csv('my_data/wine.csv')

# Split data
X = df.drop('Wine', axis=1)
y = df['Wine']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

# Saving files
X_train.to_csv('data/train_data/train.csv', index=False)
X_test.to_csv('data/test_data/test.csv', index=False)

print('Data ingestion completed.')
