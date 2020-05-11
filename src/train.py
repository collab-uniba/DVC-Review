import pandas as pd
from joblib import dump
from pathlib import Path
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

from utils import mkdir_p
from utils import get_mae

# Path of the prepared data folder
input_folder_path = Path('./prepared')

# Read training dataset
X_train = pd.read_csv(input_folder_path / 'X_train.csv')
y_train = pd.read_csv(input_folder_path / 'y_train.csv')
X_valid = pd.read_csv(input_folder_path / 'X_valid.csv')
y_valid = pd.read_csv(input_folder_path / 'y_valid.csv')

candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]

# Loop to find the ideal tree size from candidate_max_leaf_nodes
mae_values = []
for max_leaf_nodes in candidate_max_leaf_nodes:
    mae_values.append(get_mae(max_leaf_nodes, X_train, X_valid, y_train, y_valid))

# Best value of max_leaf_nodes
index_of_minimum_mae = mae_values.index(min(mae_values))
best_tree_size = candidate_max_leaf_nodes[index_of_minimum_mae]

# Specify the model
# For the sake of reproducibility, I set the `random_state` argument equal to 0
iowa_model = DecisionTreeRegressor(random_state=0, max_leaf_nodes=best_tree_size)

# Then I fit the model to the training data
iowa_model.fit(X_train, y_train)

# Eventually I save the model as a pickle file
mkdir_p('./models')
output_folder_path = Path('./models')
dump(iowa_model, output_folder_path / 'default_decision_tree.joblib')
