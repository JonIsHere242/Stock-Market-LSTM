#!/root/root/miniconda4/envs/tf/bin/python
import argparse
import logging
import os
import re
import time
import random
import sys
from joblib import dump, load
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.utils.class_weight import compute_class_weight

# Model Evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import zscore
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

config = {
    "input_directory": "Data/IndicatorData",
    "log_file": "__model_training.log",
    "file_extension": ".csv",
    "output_directory": "Data/ModelData",
    "output_file_name": "AllData.csv",
    "prediction_output_directory": "Data/RFpredictions",
    "target_column": "percent_change_Close",  # Replace with your actual target column name
    "classifier_model_path": None,
    "regressor_model_path": None,
    "verbose_level": 2,  # Set to 0 for no output, 1 for limited output, 2 for detailed output

    "bootstrap": True,  # Whether bootstrap samples are used when building trees
    "criteria": "entropy",  # Splitting criterion, either "gini" or "entropy"
    "n_estimators": 1024,  # Number of trees in the forest
    "max_depth": 15,   # Maximum depth of the tree
    "min_samples_split": 30,  # Minimum number of samples required to split a node
    "min_samples_leaf": 20,  # Minimum number of samples required at each leaf node
    "max_features": 10,  # Number of features to consider when looking for the best split
    "n_jobs": -1,  # Number of CPU cores to use (-1 for using all available cores)
    "fixed_threshold": 1.5,  # Set to a value for fixed threshold discretization, e.g., "2.0"
}



def setup_logging():
    logging.basicConfig(level=logging.INFO, 
                        filename='Data/RFpredictions/__model_training.log', 
                        filemode='a', 
                        format='%(asctime)s - %(levelname)s - %(message)s')


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runpercent", type=int, default=100, help="Percentage of .csv files for training (1-100)")
    parser.add_argument("--predict", action='store_true', help="Predict using a pretrained model without training")
    parser.add_argument("--model_type", choices=['c', 'r'], required=not ('--predict' in sys.argv),
                        help="Specify 'c' for classifier or 'r' for regressor")
    args = parser.parse_args()
    return args





def convert_dates_to_days(df, date_column, reference_date_str='1990-01-01'):
    reference_date = pd.to_datetime(reference_date_str)
    df[date_column] = pd.to_datetime(df[date_column])
    df[date_column] = (df[date_column] - reference_date).dt.days
    return df

def revert_days_to_dates(days, reference_date_str='1990-01-01'):
    reference_date = pd.to_datetime(reference_date_str)
    return reference_date + pd.to_timedelta(days, unit='D')



def concatenate_and_save_data(files, input_directory, output_directory, output_file_name, shift_size=1, trim_size=2):
    fpSize = 32
    logging.info(f"Number of files being used: {len(files)}")
    all_data_frames = []
    FilesTooSmall = 0
    for file in files:
        file_path = os.path.join(input_directory, file)
        try:
            df = pd.read_csv(file_path)
            df = DataFiltering(df)

            float_cols = df.select_dtypes(include=['float64']).columns
            if fpSize == 32:
                df[float_cols] = df[float_cols].astype('float32')
            elif fpSize == 16:
                df[float_cols] = df[float_cols].astype('float16')
            elif fpSize == 8:
                df[float_cols] = df[float_cols].astype('float8')

            if df.shape[0] < 400:
                FilesTooSmall += 1
                continue
            elif df.empty or df.isna().all().all():
                logging.warning(f"File {file} is empty or all NA, skipping...")
                continue

            df['percent_change_Close'] = df['percent_change_Close'].shift(-shift_size)

            df = df.iloc[trim_size:-trim_size]

            all_data_frames.append(df)
        except Exception as e:
            logging.error(f"Error loading file {file}: {e}")

    if all_data_frames:
        combined_data = pd.concat(all_data_frames, ignore_index=True)
        logging.info("Dataframes concatenated successfully.")
        logging.info(f"Number of files with not enough data: {FilesTooSmall}")
    else:
        logging.warning("No data to combine. Creating an empty DataFrame.")
        combined_data = pd.DataFrame()

    output_path = os.path.join(output_directory, output_file_name)
    combined_data.to_csv(output_path, index=False)
    logging.info(f"Combined data saved to {output_path}")
    logging.info(f"Number of rows in the combined data after processing: {combined_data.shape[0]}")
    return combined_data














def manual_train_test_split(X, y, test_size=0.2):
    split_index = int(len(X) * (1 - test_size))
    X_train = X[:split_index]
    y_train = y[:split_index]
    X_test = X[split_index:]
    y_test = y[split_index:]

    #logg the number of traning features and the names
    logging.info(f"Number of training features: {len(X_train.columns)}")
    
    return X_train, X_test, y_train, y_test




def train_model(X, y, config, model_type):
    X_train, X_test, y_train, y_test = manual_train_test_split(X, y, test_size=0.2)

    if model_type == 'c':
        model = RandomForestClassifier(
            bootstrap=config['bootstrap'],
            criterion=config['criteria'],
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            min_samples_split=config['min_samples_split'],
            min_samples_leaf=config['min_samples_leaf'],
            max_features=config['max_features'],
            n_jobs=config['n_jobs'],
            verbose=config.get('verbose_level', 2),
            oob_score=True
            )
        model.fit(X_train, y_train)
        return model, None  # Only return the model for classifier
    elif model_type == 'r':
        model = RandomForestRegressor(
            n_estimators=config.get("n_estimators", 100),
            max_depth=config.get("max_depth", None),
            random_state=config.get("random_state", 42),
            oob_score=True,
            n_jobs=config['n_jobs'],
            verbose=2
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return model, mse, mae, r2  # Return the model and evaluation metrics



def evaluate_model(model, X, y, config=config):
    y_pred = model.predict(X)
    
    # Filter out instances where the prediction is 0 (unsure)
    confident_indices = y_pred != 0
    y_confident = y[confident_indices]
    y_pred_confident = y_pred[confident_indices]

    if len(y_confident) > 0:
        accuracy = accuracy_score(y_confident, y_pred_confident)
        precision = precision_score(y_confident, y_pred_confident, average='weighted', zero_division=0)
        recall = recall_score(y_confident, y_pred_confident, average='weighted', zero_division=0)
        f1 = f1_score(y_confident, y_pred_confident, average='weighted', zero_division=0)
    else:
        accuracy, precision, recall, f1 = 0, 0, 0, 0

    logging.info(f"Confident Predictions - Accuracy: {accuracy}")
    logging.info(f"Confident Predictions - Precision: {precision}")
    logging.info(f"Confident Predictions - Recall: {recall}")
    logging.info(f"Confident Predictions - F1 Score: {f1}")
    return accuracy, precision, recall, f1






def save_model(model, model_type, X, performance_metric, config):
    if model is not None and performance_metric is not None:
        num_rows = X.shape[0]
        model_name = f"{model_type}_rows_{num_rows}_metric_{performance_metric:.2f}.joblib"
        model_path = os.path.join(config["output_directory"], model_name)
        dump(model, model_path)
        logging.info(f"Model saved to {model_path}")
    else:
        logging.warning("One or more parameters for saving the model are missing or invalid.")





def process_and_concatenate_files(input_directory, output_directory, runpercent, file_extension, output_file_name):
    all_files = [f for f in os.listdir(input_directory) if f.endswith(file_extension)]
    number_of_files_to_process = int(len(all_files) * runpercent / 100)
    logging.info(f"Percentage of files to process: {runpercent}%")
    logging.info(f"Number of files to process: {number_of_files_to_process}")
    selected_files = random.sample(all_files, number_of_files_to_process)

    return concatenate_and_save_data(selected_files, input_directory, output_directory, output_file_name)




def discretize_target(data, column, std_method=True, fixed_threshold=config["fixed_threshold"], window_size=20, std_multiplier=1):
    fixed_threshold = fixed_threshold*0.01
    if std_method:
        rolling_mean = data[column].rolling(window=window_size).mean()
        rolling_std = data[column].rolling(window=window_size).std()

        def classify_change_rolling(x, mean, std):
            if x > mean + std_multiplier * std:
                return 1
            elif x < mean - std_multiplier * std:
                return -1
            else:
                return 0

        discretized_col = data[column].apply(lambda x, i: classify_change_rolling(x, rolling_mean[i], rolling_std[i]), args=(rolling_mean.index,))
    
    else:
        if fixed_threshold is None:
            raise ValueError("Fixed threshold is not set. Please provide a valid fixed_threshold value.")

        def classify_change_fixed(x, threshold):
            if x > threshold:
                return 1
            elif x < -threshold:
                return -1
            else:
                return 0

        discretized_col = data[column].apply(lambda x: classify_change_fixed(x, fixed_threshold))

    data_copy = data.copy()
    data_copy[column] = discretized_col

    return data_copy




##=============================================[Prediction]==================================================##
##=============================================[Prediction]==================================================##
##=============================================[Prediction]==================================================##
##=============================================[Prediction]==================================================##
##=============================================[Prediction]==================================================##

def predict_and_append(file_path, classifier_model, regressor_model, target_column='percent_change_Close'):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    feature_names = [col for col in df.columns if col not in [target_column, 'Date']]
    X = df[feature_names]




    # Classifier Probabilities for three classes (up, down, unsure)
    probs = classifier_model.predict_proba(X)
    df["UpProb"] = probs[:, 0]  # Assuming 0 is the index for upward movement
    df["DownProb"] = probs[:, 1]  # Assuming 1 is the index for downward movement
    df["UnsureProb"] = probs[:, 2]  # Assuming 2 is the index for unsure


    df["MagnitudePrediction"] = regressor_model.predict(X)
    df['Prediction'] = np.argmax(probs, axis=1) - 1  # Shift -1 for ternary (-1, 0, 1)

    output_cols = ['Date', 'MagnitudePrediction', target_column, 'Prediction', 'UpProb', "UnsureProb", 'DownProb'] + feature_names

    if len(df) > 100:
        return df[output_cols]
    else:
        return None



















def optimize_threshold(df, target_column):
    best_threshold = 0
    best_score = 0
    for threshold in np.linspace(start=-3, stop=3, num=100):
        temp_predictions = np.where(df['z_score'] > threshold, 1, 
                                    np.where(df['z_score'] < -threshold, -1, 0))
        score = accuracy_score(df[target_column] > 0, temp_predictions > 0)
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return best_threshold, best_score





def BulkPredictAndAppend(input_directory, classifier_model, regressor_model, config, runpercent):
    os.makedirs(config["prediction_output_directory"], exist_ok=True)
    all_files = [f for f in os.listdir(input_directory) if f.endswith(config["file_extension"]) and f != config["log_file"]]
    number_of_files_to_predict = int(len(all_files) * runpercent / 100)
    logging.info(f"Percentage of files to predict: {runpercent}%")
    logging.info(f"Number of files to predict: {number_of_files_to_predict}")
    selected_files = random.sample(all_files, number_of_files_to_predict)

    total_files = len(selected_files)
    successful_files = 0

    for file in tqdm(selected_files, desc="Predicting Target"):
        file_path = os.path.join(input_directory, file)

        try:
            df = predict_and_append(file_path, classifier_model, regressor_model, config["target_column"])
            # Modify the file name for saving the predictions
            save_filename = file.replace('.csv', '_predictions.csv')
            output_path = os.path.join(config["prediction_output_directory"], save_filename)
            df.to_csv(output_path, index=False)
            successful_files += 1
        except Exception as e:
            logging.error(f"Error processing file {file}: {e}")

    logging.info(f"Total files processed: {total_files}")
    logging.info(f"Total files processed successfully: {successful_files}")
    logging.info(f"Total files processed percent successfully: {successful_files / total_files * 100:.2f}%")





def ConvertDateToNumaric(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Date'] = df['Date'].dt.strftime('%Y%m%d').astype(int)
    return df

def select_best_model(model_directory, model_type):
    best_model = None
    best_performance = -float('inf')
    best_rows = -1

    model_pattern = re.compile(rf"{model_type}_rows_(\d+)_metric_([-+]?[0-9]*\.?[0-9]+)\.joblib")

    for filename in os.listdir(model_directory):
        match = model_pattern.match(filename)
        if match:
            rows, performance = map(float, match.groups())
            if (performance > best_performance) or (performance == best_performance and rows > best_rows):
                best_performance = performance
                best_rows = rows
                best_model = filename

    if best_model:
        return os.path.join(model_directory, best_model)
    else:
        return None



def DataFiltering(df):
    df = df[df['Volume'] > 100]
    df = df[df['Close'] > 0.01]
    df = df[df['percent_change_Close'] < 10]
    df = df[abs(df['percent_change_Close']) > 0.0001]
    return df

def TrinerizeClassifierInput(value):
    threshold_up = 0.01  # define threshold_up
    threshold_down = -0.01  # define threshold_down
    if value > threshold_up:
        return 1  # Upward movement
    elif value < threshold_down:
        return -1  # Downward movement
    else:
        return 0  # No significant movement



def main():
    setup_logging()
    args = get_arguments()

    if args.predict:
        classifier_model_path = config.get("classifier_model_path", None)
        regressor_model_path = config.get("regressor_model_path", None)

        ##get the output directory for the predictions
        os.makedirs(config["prediction_output_directory"], exist_ok=True)
        ##remove any files that end in csv in the output directory
        for file in os.listdir(config["prediction_output_directory"]):
            if file.endswith('.csv'):
                os.remove(os.path.join(config["prediction_output_directory"], file))



        # Dynamically select the best models if paths are set to None
        if classifier_model_path is None:
            classifier_model_path = select_best_model(config["output_directory"], "c")
        if regressor_model_path is None:
            regressor_model_path = select_best_model(config["output_directory"], "r")

        if classifier_model_path and regressor_model_path:
            classifier_model = load(classifier_model_path)
            regressor_model = load(regressor_model_path)
            feature_names_path = "Data/ModelData/important_features.csv"  # or the appropriate path
            BulkPredictAndAppend(config["input_directory"], classifier_model, regressor_model, config, args.runpercent)
        else:
            logging.error("Could not find suitable models for prediction.")

    else:
        logging.info("========== Preprocessing data and training model ==========")
        data = process_and_concatenate_files(config["input_directory"], config["output_directory"], args.runpercent, config["file_extension"], config["output_file_name"])

        data['Date'] = pd.to_datetime(data['Date'])
        data = data.drop(columns=['Date'])

        X = data.drop(columns=[config["target_column"]])
        y = data[config["target_column"]]

        if args.model_type == 'c':
            y = y.apply(TrinerizeClassifierInput)


            model, _ = train_model(X, y, config, args.model_type)
            performance_metric = evaluate_model(model, X, y)
            performance_metric = performance_metric[0]  # Assuming using accuracy
        elif args.model_type == 'r':
            y = np.abs(y)
            y = np.where(y > 10, 10, y)

            model, mse, mae, r2 = train_model(X, y, config, args.model_type)
            performance_metric = mse  # or any other chosen metric

        model_type = "classifier" if args.model_type == 'c' else "regressor"
        save_model(model, args.model_type, X, performance_metric, config)
        logging.info(f"Model training completed for {model_type}")

if __name__ == "__main__":
    main()
