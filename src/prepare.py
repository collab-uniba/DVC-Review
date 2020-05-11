import pandas as pd
from pathlib import Path
from sklearn.impute import SimpleImputer

# Path of the data folder
data_folder_path = Path('./input')

# Path of the files to read
train_path = data_folder_path / 'train.csv'
test_path = data_folder_path / 'test.csv'

# Read dataset from csv file
train_data = pd.read_csv(train_path, index_col='Id')
test_data = pd.read_csv(test_path, index_col='Id')


# ================ #
# DATA PREPARATION #
# ================ #

# Remove rows with missing target
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)

# Separate target from predictors
y = train_data.SalePrice

# Create a DataFrame called `X` holding the predictive features.
X_full = train_data.drop(['SalePrice'], axis=1)

# To keep things simple, we'll use only numerical predictors
X = X_full.select_dtypes(exclude=['object'])
X_test = test_data.select_dtypes(exclude=['object'])

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                      train_size=0.8,
                                                      test_size=0.2,
                                                      random_state=0)

# Handle Missing Values with Imputation
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Imputation removed column names; I put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns
X_train = imputed_X_train
X_valid = imputed_X_valid

X_train.to_csv(data_folder_path / 'X_train.csv')
y_train.to_csv(data_folder_path / 'y_train.csv')
X_valid.to_csv(data_folder_path / 'X_valid.csv')
y_valid.to_csv(data_folder_path / 'y_valid.csv')
