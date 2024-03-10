import os
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import argparse
import matplotlib.pyplot as plt

import time


##create a custom MAPE function that can calulate errors including 0 values
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Calculate absolute percentage error where y_true is not zero
    absolute_percentage_error = np.abs((y_true - y_pred) / np.where(y_true != 0, y_true, 1))
    # Handle cases where y_true is zero, resulting in zero error
    absolute_percentage_error[np.isnan(absolute_percentage_error)] = 0
    return np.mean(absolute_percentage_error) * 100






config = {
    "n_estimators": 1500,
    "max_depth": 5,
    "min_samples_split": 8,
    "min_samples_leaf": 4,
    "max_features": None,  # Using all features, equivalent to 'auto'
    "train_mode": "Batch", 
    "retrain_threshold": 0.1,
    "target_column": "percent_change_Close",
    # Add more hyperparameters here if needed
}

def get_arguments():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--train_mode", help="Options: Batch, Solo", default="Batch")
    argparser.add_argument("--file_percentage", type=float, help="Percentage of files to use for training", default=100)
    return argparser.parse_args()


def setup_logging(config):
    logging.basicConfig(level=logging.INFO, filename='Data/RFpredictions/model_training.log', filemode='a',
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("V================(NEW ENTRY)================V")
    logging.info(f"Target Column: {config['target_column']}")

def check_nan_inf(data):
    # Filter for numeric columns only
    numeric_data = data.select_dtypes(include=[np.number])

    nan_percentage = numeric_data.isnull().sum().sum() / numeric_data.size * 100
    inf_percentage = np.isinf(numeric_data).sum().sum() / numeric_data.size * 100
    neg_inf_percentage = np.isneginf(numeric_data).sum().sum() / numeric_data.size * 100

    # Check if any values exceed the maximum value for float32
    max_value_exceeded = numeric_data.max().max() > np.finfo(np.float32).max
    max_value_percentage = (numeric_data > np.finfo(np.float32).max).sum().sum() / numeric_data.size * 100

    logging.info(f"Percentage of NaN values: {nan_percentage:.6f}%")
    logging.info(f"Percentage of inf values: {inf_percentage:.6f}%")
    logging.info(f"Percentage of -inf values: {neg_inf_percentage:.6f}%")
    logging.info(f"Percentage of values exceeding float32 max value: {max_value_percentage:.6f}%")





    if nan_percentage > 0:
        logging.error(f"Percentage of NaN values is greater than 0%")
    if inf_percentage > 0:
        logging.error(f"Percentage of inf values is greater than 0%")
    if neg_inf_percentage > 0:
        logging.error(f"Percentage of -inf values is greater than 0%")
    if max_value_exceeded:
        logging.error("Data contains values larger than the maximum value of float32")

    return nan_percentage, inf_percentage, neg_inf_percentage



def load_and_process_data(folder_path, target_column, shift_steps=1, file_percentage=100):
    """
    Load data from CSV files in a specified folder, process it by shifting the target column,
    and concatenate all data into a single DataFrame if in 'Batch' mode, or return a list of DataFrames if in 'Solo' mode.

    Parameters:
    folder_path (str): Path to the folder containing CSV files.
    target_column (str): Name of the target column to be shifted.
    shift_steps (int): Number of steps to shift the target column.
    file_percentage (float): Percentage of files to use for training.

    Returns:
    If 'Batch' mode: pd.DataFrame: A combined DataFrame containing all the data.
    If 'Solo' mode: List[pd.DataFrame]: A list of DataFrames, each representing one file's data.
    """
    LoadTimer = time.time()

    all_data = []
    all_files = os.listdir(folder_path)
    num_files_to_process = int(len(all_files) * (file_percentage / 100))
    selected_files = all_files[:num_files_to_process]

    for filename in selected_files:
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            try:
                data = pd.read_csv(file_path)

                # Shift the target column
                data[target_column] = data[target_column].shift(-shift_steps)
                data = data.dropna()  # Drop rows with NaN values which are a result of shifting

                if config["train_mode"] == "Batch":
                    all_data.append(data)
                elif config["train_mode"] == "Solo":
                    all_data.append((filename, data))  # Store as a tuple (filename, dataframe)

            except Exception as e:
                logging.error(f"Error loading file {filename}: {e}")

    if config["train_mode"] == "Batch":
        combined_data = pd.concat(all_data, ignore_index=True)
        logging.info(f"Data loaded in 'Batch' mode. Total shape: {combined_data.shape}")
        logging.info(f"Data loaded in {round(time.time() - LoadTimer, 1)} seconds")
        return combined_data
    elif config["train_mode"] == "Solo":
        logging.info(f"Data loaded in 'Solo' mode. Total files: {len(all_data)}")
        logging.info(f"Data loaded in {round(time.time() - LoadTimer, 1)} seconds")
        return all_data

# Remember to define your `config` dictionary and `check_nan_inf` function as they are used in this function.
    




def train_model(X, y, config):
    """
    Train a Random Forest model using the given features (X) and target variable (y).

    Parameters:
    X (pd.DataFrame): Features data.
    y (pd.Series): Target variable data.
    config (dict): Configuration parameters.

    Returns:
    RandomForestRegressor: Trained Random Forest model.
    """
    modelTimer = time.time()
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create and train the Random Forest model
    model = RandomForestRegressor(
        n_estimators=config['n_estimators'],
        max_depth=config['max_depth'],
        min_samples_split=config['min_samples_split'],
        min_samples_leaf=config['min_samples_leaf'],
        max_features=config['max_features'],
        n_jobs=-1
    )

    model.fit(X_train, y_train)


    evaluate_model(model, X_test, y_test)


    # Retrain the model if necessary
    retrain_model_if_necessary(model, X, y, config)
    print(f"Model trained in {round(time.time() - modelTimer, 1)} seconds")
    logging.info(f"Model trained in {round(time.time() - modelTimer, 1)} seconds")
    return model

def evaluate_model(model, X_test, y_test):
    # Predict using the model and evaluate the predictions vs the actual values
    predictions = model.predict(X_test)

    # Cap the predictions to a maximum of 100% and a minimum of -100%
    predictions = np.clip(predictions, -1, 1)

    # Calculation of statistics
    mean_predictions = np.mean(predictions)
    std_predictions = np.std(predictions)
    max_prediction = np.max(predictions)
    min_prediction = np.min(predictions)

    mean_actual = np.mean(y_test)
    std_actual = np.std(y_test)
    max_actual = np.max(y_test)
    min_actual = np.min(y_test)

    mean_diff = (mean_predictions - mean_actual) / mean_actual * 100
    std_diff = (std_predictions - std_actual) / std_actual * 100

    nan_inf_count = np.sum(~np.isfinite(predictions))

    # Logging
    logging.info(f"Predictions - Mean: {mean_predictions:.4f}, Std: {std_predictions:.4f}, Max: {max_prediction:.4f}, Min: {min_prediction:.4f}")
    logging.info(f"Actual - Mean: {mean_actual:.4f}, Std: {std_actual:.4f}, Max: {max_actual:.4f}, Min: {min_actual:.4f}")
    logging.info(f"Differences - Mean: {mean_diff:.2f}%, Std: {std_diff:.2f}%")
    logging.info(f"Number of inf/nan predictions: {nan_inf_count}")

    # Model Configuration
    logging.info(f"Model Configuration: n_estimators={model.n_estimators}, max_depth={model.max_depth}, min_samples_split={model.min_samples_split}, min_samples_leaf={model.min_samples_leaf}")

    # Error Metrics
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    mape = mean_absolute_percentage_error(y_test, predictions)
    logging.info(f"Error Metrics: MSE={mse:.4f}, R-squared={r2:.4f}, MAE={mae:.4f}, MAPE={mape:.4f}%")

    # Sign Accuracy
    sign_accuracy = calculate_sign_accuracy(y_test, predictions)
    logging.info(f"Sign Accuracy: {sign_accuracy:.2f}%")

    # Consecutive Correct/Wrong Sign Predictions
    consecutive_correct, consecutive_incorrect = calculate_consecutive_signs(y_test, predictions)
    logging.info(f"Consecutive Correct Sign Predictions: {consecutive_correct}, Consecutive Incorrect Sign Predictions: {consecutive_incorrect}")
    RandomSample(predictions, y_test)
    # Distribution Analysis







#make a new function that picks a random sample from the predictions and the acutuals that is 45 long and then log them and plot them
def RandomSample(predictions, y_test):
    #pick a random sample from the predictions and the acutuals that is 45 long and then log them and plot them
    random_sample = np.random.choice(len(predictions), 45)
    logging.info(f"Random Sample of Predictions: {predictions[random_sample]}")
    logging.info(f"Random Sample of Actuals: {y_test.to_numpy()[random_sample]}")

    plt.plot(predictions[random_sample], label='Predictions')
    plt.plot(y_test.to_numpy()[random_sample], label='Actuals')
    plt.legend()
    plt.show()

    return







def calculate_consecutive_signs(y_actual, y_predicted):
    correct = 0
    incorrect = 0
    max_correct = 0
    max_incorrect = 0

    for actual, predicted in zip(y_actual, y_predicted):
        if np.sign(actual) == np.sign(predicted):
            correct += 1
            max_incorrect = max(max_incorrect, incorrect)
            incorrect = 0
        else:
            incorrect += 1
            max_correct = max(max_correct, correct)
            correct = 0

    # Including the last streak
    max_correct = max(max_correct, correct)
    max_incorrect = max(max_incorrect, incorrect)

    return max_correct, max_incorrect







def calculate_sign_accuracy(y_actual, y_predicted):
    """
    Calculate the percentage of correct sign predictions.

    Parameters:
    y_actual (array-like): The actual target values.
    y_predicted (array-like): The predicted target values.

    Returns:
    float: The percentage of correct sign predictions.
    """
    correct_signs = np.sum(np.sign(y_actual) == np.sign(y_predicted))
    total = len(y_actual)
    accuracy = (correct_signs / total) * 100
    return accuracy




def retrain_model_if_necessary(model, X, y, config):
    # Function to retrain the model based on certain criteria
    pass

def remove_highly_correlated_features(data, target_column, threshold=0.8):
    """
    Removes columns that are highly correlated with the target column from the training data, 
    but keeps them in the DataFrame for later use.

    Parameters:
    data (pd.DataFrame): The input DataFrame.
    target_column (str): The name of the target column.
    threshold (float): The correlation threshold. Columns with a correlation higher than this threshold will be removed.

    Returns:
    pd.DataFrame: DataFrame with highly correlated columns removed from training data.
    list: List of column names that were removed.
    """

    # Exclude non-numeric columns like 'Date'
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

    # Calculate the correlation matrix only for numeric columns
    corr_matrix = data[numeric_cols].corr().abs()

    # Extract the correlations with the target column
    target_corr = corr_matrix[target_column]

    # Find columns with correlation higher than the threshold
    high_corr_features = target_corr[target_corr > threshold].index.tolist()
    high_corr_features.remove(target_column)  # Remove the target column from the list

    # Remove the highly correlated features from the training data
    X_train_reduced = data.drop(high_corr_features, axis=1)

    if len(high_corr_features) > 0:
        logging.info(f"Number of columns removed: {len(high_corr_features)}")
        for col in high_corr_features:
            logging.info(f"{col} Correlation with {target_column}: {target_corr[col]}")
    return X_train_reduced, high_corr_features

def main():
    setup_logging(config)
    if config["train_mode"] == "Batch":
        data = load_and_process_data('Data/ScaledData', config["target_column"])
        # Apply the correlation-based feature removal
        data_reduced, removed_features = remove_highly_correlated_features(data, config["target_column"])
        
        X = data_reduced.drop(config["target_column"], axis=1)
        # Drop 'Date' column if it exists in the DataFrame
        if 'Date' in X.columns:
            X = X.drop("Date", axis=1)
        y = data_reduced[config["target_column"]]
        
        model = train_model(X, y, config)
        # Store the removed features somewhere for later use in the ensemble model
        # For example, you can add them to the config dictionary
        config['removed_features'] = removed_features

    elif config["train_mode"] == "Solo":
        # Handle Solo mode if necessary
        pass




if __name__ == "__main__":
    args = get_arguments()
    config["train_mode"] = args.train_mode
    config["file_percentage"] = args.file_percentage

    main()