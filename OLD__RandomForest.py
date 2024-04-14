# Basic Utilities
import os
import time
import logging
import argparse
from joblib import dump, load
import random
import argparse
from tqdm import tqdm
# Data Handling
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Data Preprocessing
from sklearn.decomposition import PCA

# Feature Selection
from sklearn.feature_selection import SelectKBest, f_classif

# Machine Learning Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.class_weight import compute_class_weight

# Model Evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import zscore
##import all the regressor metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



"""
This script is designed for training and predicting with a Random Forest classifier on financial market data. It efficiently handles large datasets, utilizing advanced machine learning techniques for analysis and prediction.

Features:
- Trains a Random Forest classifier using financial data from CSV files.
- Supports feature importance analysis and correlation-based feature removal.
- Evaluates model performance with accuracy, precision, recall, and F1 score.
- Predicts new data using a pre-trained Random Forest model.
- Handles data preprocessing, including outliers and missing value treatment.

Usage:
- For training: Automatically processes data from CSV files in a specified directory, training a Random Forest model.
- For prediction: Uses a pre-trained model to predict on new data.
- Command-Line Arguments:
    --runpercent: Specifies the percentage of CSV files to use for training (1-100). Default is 100 (all files).
    --predict: Use this flag to run the script in prediction mode with a pre-trained model.

Examples:
- Train with 50% of available data: `python 5__RandomForest.py --runpercent 50`
- Predict using a pre-trained model: `python 5__RandomForest.py --predict`

Notes:
- Ensure the input directory contains properly formatted CSV files with necessary columns.
- Adjust the script's 'config' dictionary to change input/output paths and model parameters.
- The script logs detailed information for process monitoring and debugging.
"""

config = {
    "input_directory": "Data/IndicatorData",
    "log_file": "__model_training.log",
    "file_extension": ".csv",
    "output_directory": "Data/ModelData",
    "output_file_name": "AllData.csv",
    "prediction_output_directory": "Data/RFpredictions",
    "target_column": "percent_change_Close",  # Replace with your actual target column name
    "classifier_model_path": "Data/ModelData/classifier_rows_456856_metric_0.57.joblib",
    "regressor_model_path": "Data/ModelData/regressor_rows_444687_metric_-4.40.joblib",
    "verbose_level": 2,  # Set to 0 for no output, 1 for limited output, 2 for detailed output




    "bootstrap": True,  # Whether bootstrap samples are used when building trees
    "criteria": "entropy",  # Splitting criterion, either "gini" or "entropy"
    "n_estimators": 150,  # Number of trees in the forest
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
    parser.add_argument("--runpercent", type=int, default=100, 
                        help="Randomly select a percentage of .csv files for training (between 1 and 100)")
    parser.add_argument("--predict", action='store_true', 
                        help="Predict using a pretrained model without training a new one")
    args = parser.parse_args()
    if not 1 <= args.runpercent <= 100:
        raise ValueError("runpercent argument must be between 1 and 100")
    return args



def apply_pca(data, n_components=None):
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data)
    return pd.DataFrame(data=principal_components), pca.explained_variance_ratio_


def convert_dates_to_days(df, date_column, reference_date_str='1990-01-01'):
    reference_date = pd.to_datetime(reference_date_str)
    df[date_column] = pd.to_datetime(df[date_column])
    df[date_column] = (df[date_column] - reference_date).dt.days
    return df

def revert_days_to_dates(days, reference_date_str='1990-01-01'):
    reference_date = pd.to_datetime(reference_date_str)
    return reference_date + pd.to_timedelta(days, unit='D')



def concatenate_and_save_data(files, input_directory, output_directory, output_file_name, shift_size=1, trim_size=2):
    logging.info(f"Number of files being used: {len(files)}")
    all_data_frames = []
    
    for file in files:
        file_path = os.path.join(input_directory, file)
        try:
            df = pd.read_csv(file_path)

            if df.shape[0] < 400:
                logging.warning(f"File {file} has less than 400 rows, skipping...")
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
    else:
        logging.warning("No data to combine. Creating an empty DataFrame.")
        combined_data = pd.DataFrame()

    output_path = os.path.join(output_directory, output_file_name)
    combined_data.to_csv(output_path, index=False)
    logging.info(f"Combined data saved to {output_path}")

    logging.info(f"Number of rows in the combined data after processing: {combined_data.shape[0]}")

    return combined_data





def train_random_forest(X, y, config):
    # Calculate class weights
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weights_dict = {i: weight for i, weight in zip(np.unique(y), class_weights)}
    # Create and train the Random Forest model with class weights
    model = RandomForestClassifier(
        bootstrap=config['bootstrap'],
        criterion=config['criteria'],
        n_estimators=config['n_estimators'],
        max_depth=config['max_depth'],
        min_samples_split=config['min_samples_split'],
        min_samples_leaf=config['min_samples_leaf'],
        max_features=config['max_features'],
        n_jobs=config['n_jobs'],
        class_weight=class_weights_dict,  # Set class weights
        verbose=config.get('verbose_level', 2),
        oob_score=True
    )
    model.fit(X, y)
    return model





def get_feature_importances(X, y, config):
    n_estimators_quick = 50  # Ensure at least 10 estimators

    #remove any non-numeric columns
    X = X.select_dtypes(include=[np.number])



    quick_model = RandomForestClassifier(
        n_estimators=n_estimators_quick,
        max_depth=config['max_depth'],
        min_samples_split=config['min_samples_split'],
        min_samples_leaf=config['min_samples_leaf'],
        max_features=config['max_features'],
        n_jobs=config['n_jobs']
    )
    quick_model.fit(X, y)
    importances = quick_model.feature_importances_
    feature_importances = dict(zip(X.columns, importances))
    feature_importances = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    return feature_importances





def remove_highly_correlated_features(data, target_column, threshold=0.75):
    binary_cols = data.columns[data.nunique() == 2]
    numeric_cols = data.select_dtypes(include=[np.number]).drop(columns=binary_cols).columns
    corr_matrix = data[numeric_cols].corr().abs()
    high_corr_features = corr_matrix[target_column][corr_matrix[target_column] > threshold].index.tolist()
    if target_column in high_corr_features:
        high_corr_features.remove(target_column)
    data = data.drop(columns=high_corr_features, errors='ignore')
    logging.info(f"Number of features removed: {len(high_corr_features)}")
    logging.info(f"Highly correlated features removed: {high_corr_features}")
    return data, high_corr_features


 


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






def save_model_and_features(model, model_type, X, performance_metric, important_features, config):
    if model is not None and performance_metric is not None and important_features:
        # Only save the model if the performance metric is above a certain threshold
        
        num_rows = X.shape[0]
        model_name = f"{model_type}_rows_{num_rows}_metric_{performance_metric:.2f}.joblib"
        model_path = os.path.join(config["output_directory"], model_name)
        dump(model, model_path)
        logging.info(f"Model saved to {model_path}")
        # Save the important features
        important_features_path = os.path.join(config["output_directory"], "important_features.csv")
        pd.Series(list(important_features)).to_csv(important_features_path, index=False, header=False)
        logging.info(f"Important features saved to {important_features_path}")


        
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


def predict_and_append(file_path, classifier_model, regressor_model, feature_names_path, target_column='percent_change_Close', debug_mode=True):
    # Load feature names used in the trained model
    feature_names = pd.read_csv(feature_names_path, header=None).iloc[:, 0].tolist()
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])

    # Keep only the features used during model training
    df = df[feature_names + [target_column, 'Date']]
    X = df[feature_names]

    missing_features = set(feature_names) - set(X.columns)
    additional_features = set(X.columns) - set(feature_names)
    if missing_features:
        logging.error(f"Missing features: {missing_features}")
    if additional_features:
        logging.error(f"Additional features: {additional_features}")
    if missing_features or additional_features:
        raise ValueError("Feature mismatch detected.")




    if debug_mode:
        logging.debug(f"Debug Mode ON: Prediction feature names at prediction: {feature_names}")
    else:
        logging.info(f"Prediction feature names at prediction: {feature_names}")

    # Classifier Probabilities
    classifier_probabilities = classifier_model.predict_proba(X)
    df["UpProb"] = np.round(classifier_probabilities[:, 1], 4)  # Assuming 1 is the positive class
    df["DownProb"] = np.round(classifier_probabilities[:, 0], 4)

    if not debug_mode:
        # Regressor Predictions
        regressor_predictions = regressor_model.predict(X)
        df["MagnitudePrediction"] = regressor_predictions
        logging.info(f"Predictions made for file {regressor_predictions}")

    # Additional Calculations
    df["ProbDiff"] = df["UpProb"] - df["DownProb"]
    df["ProbSum"] = df["UpProb"] + df["DownProb"]
    df["ProbRatio"] = df["UpProb"] / df["DownProb"]

    df['z_score'] = zscore(df['ProbDiff'])
    
    # Optimize z-score threshold
    best_threshold, best_score = optimize_threshold(df, target_column)

    df['Prediction'] = np.where(df['z_score'] > best_threshold, 1, np.where(df['z_score'] < -best_threshold, -1, 0))
    df['ZscoreThreshold'] = best_threshold

    # Rearrange columns
    output_cols = ['Date', target_column, 'Prediction', 'UpProb', 'DownProb', 'ProbDiff', 'ProbSum', 'ProbRatio', 'z_score']
    if not debug_mode:
        output_cols.insert(3, 'MagnitudePrediction')  # Insert this column only in non-debug mode
    df = df[output_cols + feature_names]

    return df

def optimize_threshold(df, target_column):
    best_threshold = 0
    best_score = 0
    for threshold in np.linspace(start=-3, stop=3, num=100):
        temp_predictions = np.where(df['z_score'] > threshold, 1, np.where(df['z_score'] < -threshold, -1, 0))
        score = accuracy_score(df[target_column] > 0, temp_predictions > 0)
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return best_threshold, best_score





def train_random_forest_regressor(X, y, config):
    logging.info("Starting Random Forest Regressor training...")
    start_time = time.time()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.get("test_size", 0.2), random_state=config.get("random_state", 42))

    regressor = RandomForestRegressor(
        n_estimators=config.get("n_estimators", 100),
        max_depth=config.get("max_depth", None),
        random_state=config.get("random_state", 42),
        oob_score=True,
        n_jobs=config.get("n_jobs", -1),
        verbose=2  # Set verbose level to 2 for more detailed output
    )
    
    logging.info("Fitting the model...")
    regressor.fit(X_train, y_train)

    logging.info("Model fitting complete. Making predictions and calculating performance metrics...")
    y_pred = regressor.predict(X_test)


    ##if the trarget is below 2% in abs terms then dont count it for the accuracy tests 
    y_test = np.where(np.abs(y_test) < 0.02, 0, y_test)
    y_pred = np.where(np.abs(y_pred) < 0.02, 0, y_pred)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    end_time = time.time()
    logging.info(f"Random Forest Regressor - MSE: {mse:.2f}, MAE: {mae:.2f}, R-squared: {r2:.2f}")
    logging.info(f"Training completed in {end_time - start_time:.2f} seconds")

    return regressor, mse, mae, r2




def BulkPredictAndAppend(input_directory, classifier_model, regressor_model, feature_names_path, config, runpercent):
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
            df = predict_and_append(file_path, classifier_model, regressor_model, feature_names_path, config["target_column"])
            output_path = os.path.join(config["prediction_output_directory"], file)
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






def main2():
    setup_logging()
    args = get_arguments()

    if args.predict:
        logging.info("========== Predicting using pretrained models ==========")
        classifier_model_path = config["classifier_model_path"]
        regressor_model_path = config["regressor_model_path"]
        feature_names_path = "Data/ModelData/important_features.csv"
        
        classifier_model = load(classifier_model_path)
        regressor_model = load(regressor_model_path)
        
        BulkPredictAndAppend(config["input_directory"], classifier_model, regressor_model, feature_names_path, config, args.runpercent)
        return

    else:
        logging.info("==========XX Preprocessing data...XX==========")
        essential_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

        data = process_and_concatenate_files(config["input_directory"], config["output_directory"], args.runpercent, config["file_extension"], config["output_file_name"])
        
        data['Date'] = pd.to_datetime(data['Date'])
        data = data.drop(columns=['Date'])

        X = data.drop(columns=[config["target_column"]])
        y_classifier = np.where(data[config["target_column"]] > 0, 1, 0)  # Discretize target for classification
        y_regressor = data[config["target_column"]].copy()  # Continuous target for regression

        # Get important features for classifier
        feature_importances = get_feature_importances(X, y_classifier, config)
        importance_values = [importance for _, importance in feature_importances]
        threshold = np.percentile(importance_values, 25)
        important_features = set(feature for feature, importance in feature_importances if importance >= threshold) | set(essential_cols)
        X = X[list(important_features)]
        logging.info(f"Prediction feature names at training: {important_features}")
        # Classifier Training
        logging.info("========== Training the classifier model ==========")
        classifier = train_random_forest(X, y_classifier, config)
        accuracy, precision, recall, f1 = evaluate_model(classifier, X, y_classifier)
        save_model_and_features(classifier, "classifier", X, accuracy, important_features, config)
        

        # Regressor Training
        logging.info("========== Training the regressor model ==========")
        regressor, mse, mae, r2 = train_random_forest_regressor(X, y_regressor, config)
        save_model_and_features(regressor, "regressor", X, r2, important_features, config)






if __name__ == "__main__":
    main2()
