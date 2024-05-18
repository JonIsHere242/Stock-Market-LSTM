import os
import random
import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from joblib import dump, load
import argparse
from sklearn.metrics import precision_recall_curve

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')



argparser = argparse.ArgumentParser()
argparser.add_argument("--runpercent", type=int, default=10, help="Percentage of files to process.")
argparser.add_argument("--clear", action='store_true', help="Flag to clear the model and data directories.")
argparser.add_argument("--predict", action='store_true', help="Flag to predict new data.")
argparser.add_argument("--threshold", type=float, default=0.51, help="Threshold for making positive predictions.")
argparser.add_argument("--reuse", action='store_true', help="Flag to reuse existing training data if available.")
args = argparser.parse_args()



config = {
    "input_directory": "Data/IndicatorData",
    "model_output_directory": "Data/ModelData",
    "data_output_directory": "Data/ModelData/TrainingData",
    "prediction_output_directory": "Data/RFpredictions",
    "feature_importance_output": "Data/ModelData/FeatureImportances/feature_importance.csv",
    "file_selection_percentage": args.runpercent,  # Updated to use command line argument
    "target_column": "percent_change_Close",
    "n_estimators": 1024,
    "max_depth": 20,
    "random_state": 3301,
    "verbose": 2
}




def predict_new_data(model_path, threshold, config=config):
    input_directory = config["input_directory"]
    output_directory = config["prediction_output_directory"]

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    for f in os.listdir(output_directory):
        if f.endswith(".csv"):
            os.remove(os.path.join(output_directory, f))

    try:
        model_data = load(model_path)
        model = model_data['model']
        features = model_data['features']
        logging.info(f"Loaded model {model_path}")
    except Exception as e:
        logging.error(f"Error loading model {model_path}: {e}")
        return

    # Prediction logic
    for file in os.listdir(input_directory):
        if file.endswith(".csv"):
            file_path = os.path.join(input_directory, file)
            try:
                df = pd.read_csv(file_path)
                df_processed = df[features].select_dtypes(include=[np.number])

                if df_processed.isna().any().any():
                    df_processed.fillna(0, inplace=True)
                    logging.warning(f"Missing values found in {file}, filling with zeros.")

                # Predict probabilities
                probabilities = model.predict_proba(df_processed)[:, 1]  # Probability of class 1 (Up)

                # Apply threshold
                predictions = (probabilities >= threshold).astype(int)
                df['UpPrediction'] = predictions
                df['UpProbability'] = probabilities

                output_path = os.path.join(output_directory, file)
                df.to_csv(output_path, index=False)
                logging.info(f"Predictions made and saved for file: {file}")
            except Exception as e:
                logging.error(f"Error making predictions for file {file}: {e}")




def prepare_and_train_model(target_precision):
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

    training_data_path = os.path.join(data_output_directory, "training_data.csv")

    if args.reuse and os.path.exists(training_data_path):
        logging.info("Reusing existing training data.")
        combined_data = pd.read_csv(training_data_path)
    else:
        files = os.listdir(input_directory)
        num_files_to_select = int(len(files) * (config["file_selection_percentage"] / 100))
        selected_files = random.sample(files, num_files_to_select)

        all_data_frames = []

        for file in selected_files:
            file_path = os.path.join(input_directory, file)
            try:
                df = pd.read_csv(file_path)

                if len(df) < 50:
                    continue

                df[config["target_column"]] = df[config["target_column"]].shift(-1)
                df = df.iloc[5:-5]
                df.fillna(0, inplace=True)
                all_data_frames.append(df)
            except Exception as e:
                logging.error(f"Error processing file {file}: {e}")

        if all_data_frames:
            combined_data = pd.concat(all_data_frames, ignore_index=True)
            combined_data = combined_data.sample(frac=1).reset_index(drop=True)
            combined_data.to_csv(training_data_path, index=False)
            logging.info(f"Training data saved to {training_data_path}")

    if not combined_data.empty:
        model, model_path = train_random_forest_with_threshold(combined_data, target_precision)
        logging.info(f"Model trained and saved to {model_path}")

def find_threshold_for_target_precision(y_true, y_scores, target_precision=0.75):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    for precision, recall, threshold in zip(precisions, recalls, thresholds):
        if precision >= target_precision:
            return threshold, precision, recall
    return thresholds[-1], precisions[-1], recalls[-1]

def train_random_forest_with_threshold(data, target_precision):
    X = data.drop(config["target_column"], axis=1)
    X = X.select_dtypes(include=[np.number])
    y = data[config["target_column"]].apply(lambda val: 1 if val > 0 else 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=config["random_state"])
    model = RandomForestClassifier(n_estimators=config["n_estimators"], max_depth=config["max_depth"],
                                   random_state=config["random_state"], verbose=config["verbose"], n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Find optimal threshold for target precision
    optimal_threshold, optimal_precision, optimal_recall = find_threshold_for_target_precision(y_test, y_pred_proba, target_precision)
    print(f"Optimal threshold for {target_precision*100:.1f}% precision: {optimal_threshold:.2f}")
    print(f"Precision at optimal threshold: {optimal_precision:.2f}")
    print(f"Recall at optimal threshold: {optimal_recall:.2f}")

    # Apply optimal threshold
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)

    f1 = f1_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')

    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)

    # Save feature importances
    feature_importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    })

    feature_importances = feature_importances.sort_values(by='importance', ascending=False)

    os.makedirs(os.path.dirname(config["feature_importance_output"]), exist_ok=True)
    feature_importances.to_csv(config["feature_importance_output"], index=False)

    model_filename = f"Model_{f1:.2f}.joblib"
    model_path = os.path.join(config["model_output_directory"], model_filename)
    # Save the model and feature names as a dictionary
    dump({'model': model, 'features': X_train.columns.tolist()}, model_path)

    logging.info(f"Model and features saved to {model_path}")
    return model, model_path

if __name__ == "__main__":
    target_precision = 0.51  # 60% precision for upward movements

    if args.predict:
        model_files = [f for f in os.listdir(config["model_output_directory"]) if f.endswith(".joblib")]
        if model_files:
            model_path = os.path.join(config["model_output_directory"], model_files[0])
            predict_new_data(model_path, args.threshold)
        else:
            logging.error("No model files found for prediction.")
    else:
        prepare_and_train_model(target_precision)
