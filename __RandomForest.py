import os
import numpy as np
import pandas as pd
import time
import traceback
import logging
from sklearn.preprocessing import LabelEncoder
from skmultiflow.trees import HoeffdingTree
from skmultiflow.meta import AdaptiveRandomForestClassifier
from skmultiflow.evaluation import EvaluatePrequential


# Configure logging
logging.basicConfig(filename='Data/ModelLogging/TrainingErrors.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s:%(message)s')

def load_and_process_data(folder_path, target_column, shift_steps=1):
    all_data = []

    # Iterate over files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            data = pd.read_csv(file_path)

            # Shift the target column
            data[target_column] = data[target_column].shift(-shift_steps)

            # Drop the last 'shift_steps' rows where the target is NaN after shifting
            data = data.dropna()

            all_data.append(data)
    logging.info(f"Loaded {len(all_data)} files from {folder_path}...")
    return pd.concat(all_data, ignore_index=True)


def train_model(data, target_column):
    logging.info("Splitting data into features and target...")
    X = data.drop(columns=[target_column])
    y = data[target_column]

    if not np.all(X.applymap(np.isreal).all()):
        logging.warning("Non-numeric values found in features.")
        return None

    if X.isnull().values.any():
        logging.warning("Missing values found in features.")
        return None

    X_array = X.to_numpy()
    y_array = y.to_numpy()

    logging.info(f"Shape of Features (X_array): {X_array.shape}")
    logging.info(f"Shape of Target (y_array): {y_array.shape}")

    label_encoder = LabelEncoder()
    y_array_encoded = label_encoder.fit_transform(y_array)

    logging.info("Initializing the Adaptive Random Forest Classifier...")
    model = AdaptiveRandomForestClassifier()

    logging.info("Training the model...")
    start_time = time.time()

    try:
        model.fit(X_array, y_array_encoded)
    except Exception as e:
        logging.error(f"An error occurred during model training: {e}", exc_info=True)
        return None

    end_time = time.time()
    logging.info(f"Training completed in {end_time - start_time:.2f} seconds.")

    return model


def main():
    folder_path = 'Data/ScaledData'
    target_column = 'percent_change_Close'

    data = load_and_process_data(folder_path, target_column)
    model = train_model(data, target_column)

    if model:
        logging.info("Model trained successfully")
    else:
        logging.error("Model training failed")


if __name__ == "__main__":
    main()
