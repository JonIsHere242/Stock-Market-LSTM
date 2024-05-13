import os
import random
import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
from joblib import dump, load
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

argparser = argparse.ArgumentParser()
argparser.add_argument("--runpercent", type=int, default=10, help="Percentage of files to process.")
argparser.add_argument("--clear", action='store_true', help="Flag to clear the model and data directories.")
argparser.add_argument("--predict", type=int, default=0, help="Specify the number of day shifts to use for predictions. If 0, no predictions are made.")
args = argparser.parse_args()

config = {
    "input_directory": "Data/IndicatorData",
    "model_output_directory": "Data/ModelData/TimeShiftedModels",
    "data_output_directory": "Data/ModelData/TimeShiftedTrainingData",
    "prediction_output_directory": "Data/RFpredictions",
    "file_selection_percentage": args.runpercent,  # Updated to use command line argument
    "target_column": "percent_change_Close",
    "n_estimators": 256,
    "max_depth": 25,
    "random_state": 3301,
    "verbose": 2
}

def predict_new_data(num_models, config=config):
    input_directory = config["input_directory"]
    model_output_directory = config["model_output_directory"]
    output_directory = config["prediction_output_directory"]

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    for f in os.listdir(output_directory):
        if f.endswith(".csv"):
            os.remove(os.path.join(output_directory, f))

    models = []
    # Load the models and prepare for prediction
    for i in range(1, num_models + 1):
        model_files = [f for f in os.listdir(model_output_directory) if f.startswith(f"Model_shift_{i}_")]
        if model_files:
            model_file = os.path.join(model_output_directory, model_files[0])
            try:
                loaded_model = load(model_file)
                models.append(loaded_model)
                logging.info(f"Loaded model {model_file}")
            except Exception as e:
                logging.error(f"Error loading model {model_file}: {e}")
        else:
            logging.error(f"No model files found for shift {i}")
    # Prediction logic
    for file in os.listdir(input_directory):
        if file.endswith(".csv"):
            file_path = os.path.join(input_directory, file)
            try:
                df = pd.read_csv(file_path)
                # Process each model
                for i, model in enumerate(models, start=1):
                    features = model['features']  # Assuming 'features' are correctly saved in the model
                    df_processed = df[features].select_dtypes(include=[np.number])

                    if df_processed.isna().any().any():
                        df_processed.fillna(0, inplace=True)
                        logging.warning(f"Missing values found in {file}, filling with zeros.")

                    predictions = model['model'].predict_proba(df_processed)
                    df[f'UpProb_Shift_{i}'] = predictions[:, 1]  # Assuming class 1 is 'Up'
                    df[f'DownProb_Shift_{i}'] = predictions[:, 0]  # Assuming class 0 is 'Down'

                df['Weighted_UpProb'] = (df['UpProb_Shift_1'] * 0.8) + (df['UpProb_Shift_2'] * 0.1) + (df['UpProb_Shift_3'] * 0.05) + (df['UpProb_Shift_4'] * 0.035) + (df['UpProb_Shift_5'] * 0.015)

                output_path = os.path.join(output_directory, file)
                df.to_csv(output_path, index=False)
                logging.info(f"Predictions made and saved for file: {file}")
            except Exception as e:
                logging.error(f"Error making predictions for file {file}: {e}")

def prepare_and_train_models(shift_sizes=[1, 2, 3, 4, 5]):
    input_directory = config["input_directory"]
    model_output_directory = config["model_output_directory"]
    data_output_directory = config["data_output_directory"]

    # Ensure output directories exist
    os.makedirs(model_output_directory, exist_ok=True)
    os.makedirs(data_output_directory, exist_ok=True)

    # Clear existing training data and models if the flag is set
    if args.clear:
        for folder in [model_output_directory, data_output_directory]:
            for f in os.listdir(folder):
                os.remove(os.path.join(folder, f))

    files = os.listdir(input_directory)
    num_files_to_select = int(len(files) * (config["file_selection_percentage"] / 100))
    selected_files = random.sample(files, num_files_to_select)

    for shift_size in shift_sizes:
        all_data_frames = []
        trim_size = shift_size + 1  # Ensuring no empty rows in the training data due to shifting

        for file in selected_files:
            file_path = os.path.join(input_directory, file)
            try:
                df = pd.read_csv(file_path)
                df[config["target_column"]] = df[config["target_column"]].shift(-shift_size)
                df = df.iloc[trim_size:-trim_size]
                df.fillna(0, inplace=True)
                all_data_frames.append(df)
            except Exception as e:
                logging.error(f"Error processing file {file}: {e}")

        if all_data_frames:
            combined_data = pd.concat(all_data_frames, ignore_index=True)
            combined_data = combined_data.sample(frac=1).reset_index(drop=True)
            data_file_path = os.path.join(data_output_directory, f"training_data_shift_{shift_size}.csv")
            combined_data.to_csv(data_file_path, index=False)
            # Train and save model
            model, model_path = train_random_forest(combined_data, shift_size)
            logging.info(f"Model trained for shift size {shift_size} and saved to {model_path}")

def train_random_forest(data, shift_size):
    X = data.drop(config["target_column"], axis=1)
    X = X.select_dtypes(include=[np.number])
    y = data[config["target_column"]].apply(lambda val: 1 if val > 0 else 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=config["random_state"])
    model = RandomForestClassifier(n_estimators=config["n_estimators"], max_depth=config["max_depth"],
                                   random_state=config["random_state"], verbose=config["verbose"], n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy:", accuracy)

    model_filename = f"Model_shift_{shift_size}_{f1:.2f}.joblib"
    model_path = os.path.join(config["model_output_directory"], model_filename)
    # Save the model and feature names as a dictionary
    dump({'model': model, 'features': X_train.columns.tolist()}, model_path)
    logging.info(f"Model and features saved to {model_path}")
    return model, model_path


if __name__ == "__main__":
    
    if args.predict > 0:
        predict_new_data(args.predict)
    else:
        prepare_and_train_models()


