import pandas as pd
from joblib import dump
from pathlib import Path
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

from utils import get_mae


# Path of the prepared data folder
input_folder_path = Path('./prepared')

# Read training dataset
X_train = pd.read_csv(input_folder_path / 'X_train.csv')
y_train = pd.read_csv(input_folder_path / 'y_train.csv')

# Specify the model
# For the sake of reproducibility, I set the `random_state` argument equal to 0
iowa_model = RandomForestRegressor(random_state=1)

# Then I fit the model to the training data
iowa_model.fit(X_train, y_train)

# Eventually I save the model as a pickle file
Path('./models').mkdir(exist_ok=True)
output_folder_path = Path('./models')
dump(iowa_model, output_folder_path / 'default_decision_tree.joblib')
