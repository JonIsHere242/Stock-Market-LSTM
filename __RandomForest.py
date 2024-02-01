import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import logging
from tqdm import tqdm

logging.basicConfig(filename='Data/ModelLogging/TrainingErrors.log', level=logging.INFO, 
    format='%(asctime)s %(levelname)s:%(message)s')

def load_and_process_data(folder_path, target_column, shift_steps=1):
    all_data = []
    for filename in tqdm(os.listdir(folder_path), desc="Loading Data"):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            data = pd.read_csv(file_path)
            data[target_column] = data[target_column].shift(-shift_steps)
            data = data.dropna()
            all_data.append(data)
    logging.info(f"Loaded {len(all_data)} files from {folder_path}...")
    combined_data = pd.concat(all_data, ignore_index=True)
    X = combined_data.drop(columns=[target_column])
    y = combined_data[target_column]
    return X, y



def predict_with_confidence(model, X):
    """
    Make predictions with a confidence measure.

    Parameters:
    model (RandomForestRegressor): The trained Random Forest model.
    X (array-like): Feature matrix for making predictions.

    Returns:
    array-like: Predictions.
    array-like: Confidence measure of predictions.
    """
    predictions = model.predict(X)

    # Estimate the standard deviation of predictions
    all_tree_predictions = np.array([tree.predict(X) for tree in model.estimators_])
    std_deviation = np.std(all_tree_predictions, axis=0)

    return predictions, std_deviation


def train_random_forest(X, y):
    """
    Train a Random Forest Regressor.

    Parameters:
    X (array-like): Feature matrix.
    y (array-like): Target vector.

    Returns:
    RandomForestRegressor: Trained Random Forest model.
    """
    try:
        rf_model = RandomForestRegressor()
        logging.info("Starting model fit")

        # Convert X to a NumPy array to avoid the warning
        rf_model.fit(X.to_numpy(), y)

        logging.info("Ending model fit")
        return rf_model
    except Exception as e:
        logging.error(f"Exception while fitting model: {e}")
        return None



def enhance_dataset_with_predictions(model, data, feature_columns, target_column):
    """
    Enhance the dataset with predictions from the trained model.

    Parameters:
    model: Trained model (RandomForestRegressor).
    data: Original DataFrame.
    feature_columns: List of columns used as features.
    target_column: The name of the target column.

    Returns:
    DataFrame: Enhanced DataFrame with predictions.
    """
    predictions = model.predict(data[feature_columns].to_numpy())  # Convert to NumPy array here as well
    data['RF_Predictions'] = predictions
    return data    




def main():
    """
    Main function to execute the data loading, processing, and model training.
    """
    config = {
        "folder_path": 'Data/ScaledData',
        "target_column": 'percent_change_Close',
        "output_path": 'Data/EnhancedData/EnhancedDataset.csv'
    }

    # Load and process the data
    X, y = load_and_process_data(config['folder_path'], config['target_column'])

    # Keep a copy of the original data for later use
    original_data = pd.concat([X, y], axis=1)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the random forest model
    model = train_random_forest(X_train, y_train)

    if model:
        logging.info("Model trained successfully")

        # Ensure the output directory exists
        ensure_directory_exists(config['output_path'])

        # Enhance the original dataset with RandomForest predictions
        enhanced_data = enhance_dataset_with_predictions(model, original_data, X.columns.tolist(), config['target_column'])
        enhanced_data.to_csv(config['output_path'], index=False)
        logging.info(f"Enhanced dataset saved to {config['output_path']}")
    else:
        logging.error("Model training failed")

if __name__ == "__main__":
    main()