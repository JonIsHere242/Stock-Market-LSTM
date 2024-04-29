import os
import random
import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score
from joblib import dump
from joblib import load

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

config = {
    "input_directory": "Data/IndicatorData",
    "output_directory": "Data/ModelData",
    "output_file_name": "AllData.csv",
    "file_selection_percentage": 20,  # Use 100% for all files, modify as needed
    "target_column": "percent_change_Close",
    "shift_size": 1,  # Modify as needed for different forecast horizons
    "trim_size": 2,  # Rows to trim from start and end of each file
    "redo_data": False,  # Set to True to force data reprocessing, False to use existing data file if it exists
    "n_estimators": 512,  # Number of trees in the forest
    "max_depth": None,  # Maximum depth of each tree
    "random_state": 3301,  # Random state for reproducibility
    "verbose": 2
}

def PrepareData():
    input_directory = config["input_directory"]
    output_directory = config["output_directory"]
    output_file_name = config["output_file_name"]
    file_selection_percentage = config["file_selection_percentage"]
    target_column = config["target_column"]
    shift_size = config["shift_size"]
    trim_size = config["trim_size"]

    # Check for existing data file and decide whether to reprocess
    output_path = os.path.join(output_directory, output_file_name)
    if not config["redo_data"] and os.path.exists(output_path):
        logging.info("Existing data file found and reuse enabled. Loading data...")
        return pd.read_csv(output_path)

    logging.info("No existing data file found or reprocessing required. Generating new data...")
    files = os.listdir(input_directory)
    num_files_to_select = int(len(files) * (file_selection_percentage / 100))
    selected_files = random.sample(files, num_files_to_select)

    all_data_frames = []
    for file in selected_files:
        file_path = os.path.join(input_directory, file)
        try:
            df = pd.read_csv(file_path)
            df[target_column] = df[target_column].shift(-shift_size)  # Shift for future prediction
            df = df.iloc[trim_size:-trim_size]  # Trim rows
            df.fillna(0, inplace=True)
            all_data_frames.append(df)
        except Exception as e:
            logging.error(f"Error processing file {file}: {e}")

    if all_data_frames:
        combined_data = pd.concat(all_data_frames, ignore_index=True)
        combined_data = combined_data.sample(frac=1).reset_index(drop=True)  # Shuffle data
        combined_data.to_csv(output_path, index=False)
        logging.info(f"Data concatenated, shuffled, and saved to {output_file_name}.")
        return combined_data
    else:
        logging.warning("No data frames to concatenate. Check your file selection and formats.")
        return pd.DataFrame()  # Return an empty DataFrame if no data to process






def discretize_target(val):
    if val > 0:
        return 1
    else:
        return 0
    



def train_random_forest(data, config=config):
    X = data.drop(config["target_column"], axis=1)
    X = X.select_dtypes(include=[np.number])
    y = data[config["target_column"]].apply(discretize_target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=config["random_state"])
    model = RandomForestClassifier(n_estimators=config["n_estimators"], max_depth=config["max_depth"],
                                   random_state=config["random_state"], verbose=config["verbose"], n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy:", accuracy)

    # Save the model and feature names
    model_filename = f"Model_{f1:.2f}.joblib"
    model_path = os.path.join(config["output_directory"], model_filename)
    dump({'model': model, 'features': X_train.columns.tolist()}, model_path)
    logging.info(f"Model and features saved to {model_path}")
    return model, model_path



def predict_new_data(model_path, input_directory, config=config):
    output_directory = "Data/RFpredictions"

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Clear existing CSV files in the output directory
    for file in os.listdir(output_directory):
        if file.endswith(".csv"):
            os.remove(os.path.join(output_directory, file))

    # Load the model and features
    loaded = load(model_path)
    model = loaded['model']
    train_features = loaded['features']

    # Prediction logic
    for file in os.listdir(input_directory):
        if file.endswith(".csv"):
            file_path = os.path.join(input_directory, file)
            try:
                df = pd.read_csv(file_path)

                df_processed = df[train_features].select_dtypes(include=[np.number])

                if df_processed.isna().any().any():
                    logging.warning(f"Missing values found in {file}, filling with zeros.")
                    df_processed.fillna(0, inplace=True)

                predictions = model.predict_proba(df_processed)
                df['UpProb'] = predictions[:, 1]  # Assuming class 1 is 'Up'
                df['DownProb'] = predictions[:, 0]  # Assuming class 0 is 'Down'

                output_path = os.path.join(output_directory, file)
                df.to_csv(output_path, index=False)
                logging.info(f"Predictions made for file: {file}")
            except Exception as e:
                logging.error(f"Error making predictions for file {file}: {e}")






def main():
    data = PrepareData()
    if not data.empty:
        model_files = [f for f in os.listdir(config["output_directory"]) if f.endswith('.joblib')]
        if model_files:
            model_path = os.path.join(config["output_directory"], model_files[0])
            predict_new_data(model_path, config["input_directory"])
        else:
            model, model_path = train_random_forest(data)
            predict_new_data(model_path, config["input_directory"])

if __name__ == "__main__":
    main()
