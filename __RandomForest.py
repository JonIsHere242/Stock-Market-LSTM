import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import logging
from tqdm import tqdm


def load_and_process_data(folder_path, target_column, shift_steps=1):
    all_data = []
    for filename in tqdm(os.listdir(folder_path), desc="Loading Data"):
        if filename.endswith(".csv"):
            try:
                file_path = os.path.join(folder_path, filename)
                data = pd.read_csv(file_path)
                if not all(isinstance(x, (int, float)) for x in data[target_column]):
                    raise ValueError(f"Non-numeric data found in {filename}")
                data[target_column] = data[target_column].shift(-shift_steps)
                data = data.dropna()
                all_data.append(data)
                logging.info(f"Processed file: {filename}")
            except Exception as e:
                logging.error(f"Error processing file {filename}: {e}")
                continue
    logging.info(f"Loaded {len(all_data)} files from {folder_path}...")
    combined_data = pd.concat(all_data, ignore_index=True)
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
        y = np.ravel(y)  # Ensuring y is a 1D array

        rf_model = RandomForestRegressor()
        logging.info("Starting model training")

        rf_model.fit(X, y)

        logging.info("Model training completed")
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




def main():
    """
    Main function to execute the data loading, processing, and model training.
    """
    config = {
        "folder_path": 'Data/ScaledData',
        "target_column": 'percent_change_Close',
        "output_path": 'Data/EnhancedData/EnhancedDataset.csv'
    }

    try:


        X, y = load_and_process_data(config['folder_path'], config['target_column'])
        original_data = pd.concat([X, y], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = train_random_forest(X_train, y_train)

        if model:
            logging.info("Model trained successfully")

            ensure_directory_exists(config['output_path'])

            enhanced_data = enhance_dataset_with_predictions(model, original_data, X.columns.tolist(), config['target_column'])
            enhanced_data.to_csv(config['output_path'], index=False)
            logging.info(f"Enhanced dataset saved to {config['output_path']}")
        else:
            logging.error("Model training failed")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
    main()