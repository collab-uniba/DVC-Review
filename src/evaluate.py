from sklearn.metrics import mean_absolute_error
from joblib import load
import pandas as pd
from pathlib import Path


# Path to the prepared data folder
input_folder_path = Path('./prepared')
# Path to the models folder
model_folder_path = Path('./models')
# Path to the metrics folder
metrics_folder_path = Path('./metrics')

# Read validation dataset
X_valid = pd.read_csv(input_folder_path / 'X_valid.csv')
y_valid = pd.read_csv(input_folder_path / 'y_valid.csv')

# Load the model
iowa_model = load(model_folder_path / 'default_decision_tree.joblib')

# Compute predictions using the model
val_predictions = iowa_model.predict(X_valid)

# Compute the MAE value for the model
val_mae = mean_absolute_error(y_valid, val_predictions)

# Write MAE to file

with open(metrics_folder_path / 'mae.metric', 'w') as mae_file:
    mae_file.write(str(val_mae))

print("Evaluation completed.")