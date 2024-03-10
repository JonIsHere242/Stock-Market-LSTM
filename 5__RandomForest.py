import os
import time
import numpy as np
import pandas as pd
import argparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_array, check_is_fitted
import logging
from tqdm import tqdm

def setup_logging():
    """Configure logging settings for the application."""
    ##change the format of the log file to include the time of the log
    logging.basicConfig(level=logging.INFO, filename='Data/ModelData/TrainingErrors.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("V================(NEW ENTRY)================V")



def ensure_directory_exists(path):
    """Ensure that the directory exists. Create it if it does not."""
    os.makedirs(path, exist_ok=True)



def load_and_process_data(folder_path, target_column, shift_steps=1, correlation_threshold=0.95):
    """
    Load data from CSV files in a specified folder, process it by shifting the target column, 
    remove highly correlated features, and concatenate all data into a single DataFrame.

    Parameters:
    folder_path (str): Path to the folder containing CSV files.
    target_column (str): Name of the target column to be shifted.
    shift_steps (int): Number of steps to shift the target column.
    correlation_threshold (float): Threshold for removing highly correlated features.

    Returns:
    Tuple[pd.DataFrame, pd.Series]: A tuple containing the feature matrix (X) and target vector (y).
    """
    all_data = []
    total_files = 0
    processed_files = 0
    for filename in tqdm(os.listdir(folder_path), desc="Loading Data"):
        if filename.endswith(".csv"):

            if sum(1 for line in open(os.path.join(folder_path, filename))) < 100:
                logging.warning(f"File {filename} has less than 100 rows of data. Skipping...")
                continue

            total_files += 1
            try:
                file_path = os.path.join(folder_path, filename)
                data = pd.read_csv(file_path)

                if not all(isinstance(x, (int, float)) for x in data[target_column]):
                    raise ValueError(f"Non-numeric data found in {filename}")

                # Shift the target column
                data[target_column] = data[target_column].shift(-shift_steps)
                data = data.dropna()

                # Remove highly correlated features
                correlations = data.corr().abs()
                high_corr_features = [feature for feature in correlations if correlations[feature][target_column] > correlation_threshold and feature != target_column]
                data = data.drop(columns=high_corr_features)

                all_data.append(data)
                processed_files += 1
            except (FileNotFoundError, pd.errors.EmptyDataError, ValueError) as e:
                logging.error(f"Error processing file {filename}: {e}")
                continue

    logging.info(f"Loaded {total_files} files from {folder_path}...")
    logging.info(f"Processed {processed_files} files.")

    if processed_files < total_files:
        percentage_processed = (processed_files / total_files) * 100
        logging.warning(f"{percentage_processed:.2f}% of files processed. Some files may have errors.")
    else:
        logging.info("100% of files processed successfully.")
    
    combined_data = pd.concat(all_data, ignore_index=True)

    logging.info(f"Shape of the combined dataframe: {combined_data.shape}")

    X = combined_data.drop(columns=[target_column])
    y = combined_data[target_column]
    return X, y


def predict_with_confidence(model, X):
    """
    Make predictions with a confidence measure.
    """
    try:
        if not isinstance(model, RandomForestRegressor):
            raise ValueError("Model must be a RandomForestRegressor")
        check_is_fitted(model)
        X = check_array(X)

        logging.info("Starting prediction with confidence measure")
        predictions = model.predict(X)

        # Efficient array handling
        all_tree_predictions = np.array([tree.predict(X) for tree in model.estimators_])
        std_deviation = np.std(all_tree_predictions, axis=0)

        logging.info("Prediction completed")
        return predictions, std_deviation
    except Exception as e:
        logging.error(f"Error in predict_with_confidence: {e}")
        return None, None


def train_random_forest(X, y):
    """
    Train a Random Forest Regressor.
    """
    try:
        X = check_array(X)
        y = np.ravel(y) 

        rf_model = RandomForestRegressor(
            n_estimators=25,        # Adjust the number of trees
            max_depth=None,         # Adjust the maximum depth of the tree
            min_samples_split=2,    # Minimum number of samples required to split a node
            min_samples_leaf=1,     # Minimum number of samples required at each leaf node
            n_jobs=-1,              # Use all available cores
            random_state=None       # Random state for reproducibility
        )

        logging.info("Starting model training")
        start_time = time.time()  # Start time
        rf_model.fit(X, y)
        end_time = time.time()    # End time
        elapsed_time = end_time - start_time  # Calculate elapsed time

        logging.info(f"Model training completed in {elapsed_time:.2f} seconds")
        print(f"Model training completed in {elapsed_time:.2f} seconds")
        return rf_model
    except Exception as e:
        logging.error(f"Error in train_random_forest: {e}")
        return None



def enhance_dataset_with_predictions(model, data, feature_columns, target_column):
    """
    Enhance the dataset with predictions from the trained model.
    """
    try:
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Data must be a pandas DataFrame")
        
        X = data[feature_columns]
        check_is_fitted(model)
        X = check_array(X)

        logging.info("Enhancing dataset with predictions")
        predictions = model.predict(X)
        data['RF_Predictions'] = predictions

        logging.info("Dataset enhancement completed")
        return data
    except Exception as e:
        logging.error(f"Error in enhance_dataset_with_predictions: {e}")
        return data  # Return original data in case of failure



def process_and_save_individual_files(model, folder_path, output_folder, target_column, feature_columns):
    """
    Process individual files, make predictions using the trained model, and save the modified files.
    """
    ensure_directory_exists(output_folder)

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            try:
                file_path = os.path.join(folder_path, filename)
                data = pd.read_csv(file_path)

                X = data[feature_columns]

                predictions = model.predict(X)
                data['RF_Predictions'] = predictions

                output_file_path = os.path.join(output_folder, filename)
                data.to_csv(output_file_path, index=False)
                logging.info(f"Predictions added and saved for file: {filename}")
            except Exception as e:
                logging.error(f"Error processing file {filename}: {e}")




def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the Random Forest model.")
    parser.add_argument("--mode", choices=["train", "predict", "both"], default="both", help="Mode: train = train the model, predict = produce predictions, both = both.")
    return parser.parse_args()
def main():
    args = get_args()
    mode = args.mode

    config = {
        "folder_path": 'Data/ScaledData',
        "target_column": 'percent_change_Close',
        "output_path": 'Data/ModelData/EnhancedDataset.csv',
        "prediction_folder_path": 'Data/ScaledData',
        "output_folder": 'Data/individualPredictedFiles',
        "random_state": 3301
    }

    try:
        model = None

        if mode in ["train", "both"]:
            X, y = load_and_process_data(config['folder_path'], config['target_column'])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=config["random_state"])
            model = train_random_forest(X_train, y_train)

            if model:
                logging.info("Model trained successfully")
                ensure_directory_exists(os.path.dirname(config['output_path']))
                enhanced_data = enhance_dataset_with_predictions(model, pd.concat([X, y], axis=1), X.columns.tolist(), config['target_column'])
                enhanced_data.to_csv(config['output_path'], index=False)
                logging.info(f"Enhanced dataset saved to {config['output_path']}")
            else:
                logging.error("Model training failed")

        if mode in ["predict", "both"]:
            if mode == "predict" and not os.path.exists(config['output_path']):
                raise FileNotFoundError("Model file not found. Please train the model first.")

            # If model is None, it means mode is "predict" and model needs to be loaded here.
            # Example: model = load_model_function(config['output_path'])

            # Ensure that the model is loaded
            if model is None:
                raise Exception("Model is not loaded. Unable to proceed with predictions.")

            process_and_save_individual_files(model, config['prediction_folder_path'], config['output_folder'], config['target_column'], X.columns.tolist())

    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    setup_logging()
    main()