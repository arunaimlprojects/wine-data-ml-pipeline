import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
import yaml

# Load parameters from params.yaml
with open('params.yaml') as f:
    params = yaml.safe_load(f)

# Fetch 'max_features' parameter
max_features = params['feature_engineering']['max_features']

# Load preprocessed data
train_data = pd.read_csv('data/preprocessed/train_preprocessed.csv')
test_data = pd.read_csv('data/preprocessed/test_preprocessed.csv')

# Feature Engineering using SelectKBest
X_train = train_data.drop('Wine', axis=1)
y_train = train_data['Wine']
X_test = test_data.drop('Wine', axis=1)
y_test = test_data['Wine']

# Select best 'max_features'
selector = SelectKBest(f_classif, k=max_features)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Saving engineered data
pd.DataFrame(X_train_selected).to_csv('data/feature_engineered/train_engineered.csv', index=False)
pd.DataFrame(X_test_selected).to_csv('data/feature_engineered/test_engineered.csv', index=False)

print('Feature engineering completed.')
