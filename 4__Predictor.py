import os
import random
import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc, brier_score_loss, precision_recall_curve, roc_auc_score, average_precision_score
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
from joblib import dump, load
import argparse
from imblearn.over_sampling import RandomOverSampler

import time

from tqdm import tqdm
from joblib import parallel_backend
from contextlib import redirect_stdout, redirect_stderr
import io
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

##logging file is called "Data\RFpredictions\__model_training.log"
logging.basicConfig(filename='Data/RFpredictions/__model_training.log', level=logging.INFO, format='%(asctime)s - %(message)s')

argparser = argparse.ArgumentParser()
argparser.add_argument("--runpercent", type=int, default=50, help="Percentage of files to process.")
argparser.add_argument("--clear", action='store_true', help="Flag to clear the model and data directories.")
argparser.add_argument("--predict", action='store_true', help="Flag to predict new data.")
argparser.add_argument("--reuse", action='store_true', help="Flag to reuse existing training data if available.")
argparser.add_argument("--feature_cut", type=float, default=0, help="Percentage of least important features to cut (0-100).")
args = argparser.parse_args()

RUNNINGCONFIG = {
    "input_directory": "Data/IndicatorData",
    "model_output_directory": "Data/ModelData",
    "data_output_directory": "Data/ModelData/TrainingData",
    "prediction_output_directory": "Data/RFpredictions",
    "feature_importance_output": "Data/ModelData/FeatureImportances/feature_importance.parquet",
    "feature_importance_input": "Data/ModelData/FeatureImportances/feature_importance.parquet",


    "feature_cut_percentage": args.feature_cut,
    "perm_feature_importance_output": "Data/ModelData/FeatureImportances/perm_feature_importance.parquet",
    "file_selection_percentage": args.runpercent,
    "target_column": "percent_change_Close",
}

RFCONFIG = {
    "n_estimators": 64,  
    "criterion": "entropy",
    "max_depth": 15,  
    "min_samples_split": 10,  
    "min_samples_leaf": 5,  
    "min_weight_fraction_leaf": 0,
    "max_features": 0.10,  
    "max_leaf_nodes": None,
    "min_impurity_decrease": 0,
    "bootstrap": True,
    "oob_score": True,  
    "random_state": 69,
    "verbose": 2,
    "n_jobs": -1,
    "warm_start": False,
    "class_weight": {0: 0.5, 1: 2.5},  
    "ccp_alpha": 0,  
    "max_samples": None
}





def load_and_filter_features(config):
    feature_importance_path = config['feature_importance_input']
    cut_percentage = config['feature_cut_percentage']
    
    if not os.path.exists(feature_importance_path):
        logging.warning(f"Feature importance file not found: {feature_importance_path}")
        return None
    
    df = pd.read_parquet(feature_importance_path)
    
    # Remove essential columns from feature importance ranking
    essential_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    df_filtered = df[~df['feature'].isin(essential_columns)]
    
    df_sorted = df_filtered.sort_values('importance', ascending=False)
    
    if cut_percentage > 0:
        n_keep = int(len(df_sorted) * (1 - cut_percentage / 100))
        df_kept = df_sorted.head(n_keep)
        logging.info(f"Keeping top {100 - cut_percentage}% of features: {n_keep} out of {len(df_sorted)} (excluding essential columns)")
    else:
        df_kept = df_sorted
        logging.info(f"Using all {len(df_sorted)} features (excluding essential columns)")
    
    # Add essential columns back to the list of features to keep
    features_to_keep = df_kept['feature'].tolist() + essential_columns
    
    return features_to_keep


def prepare_training_data(input_directory, output_directory, file_selection_percentage, target_column, reuse, date_column, selected_features=None):
    output_file = os.path.join(output_directory, 'training_data.parquet')
    if reuse and os.path.exists(output_file):
        logging.info("Reusing existing training data.")
        print("Reusing existing training data.")
        return pd.read_parquet(output_file)
    
    logging.info("Preparing new training data.")
    all_files = [f for f in os.listdir(input_directory) if f.endswith('.parquet')]
    selected_files = random.sample(all_files, int(len(all_files) * file_selection_percentage / 100))
    
    if os.path.exists(output_file):
        os.remove(output_file) 
    pbar = tqdm(total=len(selected_files), desc="Processing files")
    
    all_data = []
    
    for file in selected_files:
        df = pd.read_parquet(os.path.join(input_directory, file))
        if df.shape[0] > 50 and target_column in df.columns and date_column in df.columns:
            df[date_column] = pd.to_datetime(df[date_column])
            df[target_column] = df[target_column].shift(-1)
            df = df.iloc[2:-2]
            df = df.dropna(subset=[target_column])
            df = df[(df[target_column] <= 10000) & (df[target_column] >= -10000)]
            all_data.append(df)
        pbar.update(1)
    pbar.close()
    combined_df = pd.concat(all_data)
    grouped = combined_df.groupby(date_column)
    shuffled_groups = [group.sample(frac=1).reset_index(drop=True) for _, group in grouped]



    
    final_df = pd.concat(shuffled_groups).reset_index(drop=True)
    
    essential_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', target_column]
    
    if selected_features:
        features_to_keep = [col for col in selected_features if col in final_df.columns]
        features_to_keep = list(set(features_to_keep + essential_columns))
        final_df = final_df[features_to_keep]
        logging.info(f"Kept {len(features_to_keep)} features after selection (including essential columns)")
    


    ##convert all numaric columns to fp 32 to save space
    final_df = final_df.astype({col: np.float32 for col in final_df.select_dtypes(include=[np.number]).columns})
    ##if there are any all int columns convert them to int 16 to save space
    final_df = final_df.astype({col: np.int32 for col in final_df.select_dtypes(include=[np.int64]).columns})
    ##round the numaric columns down to 5 decimal places
    final_df = final_df.round(5)

    final_df.to_parquet(output_file, index=False)
    return final_df






def train_precision_focused_random_forest(training_data, config, target_precision=0.70, prediction_percentage=0.05):
    logging.info("Training Precision-Focused Random Forest model with Random Over-Sampling.")
    
    model_output_path = os.path.join(config['model_output_directory'], 'random_forest_model.joblib')
    if os.path.exists(model_output_path):
        os.remove(model_output_path)

    # Remove any rows that have nan values
    training_data = training_data.dropna()

    X = training_data.drop(columns=[config['target_column']])
    y = training_data[config['target_column']]
    y = y.apply(lambda x: 1 if x > 0.0001 else 0)

    datetime_columns = X.select_dtypes(include=['datetime64']).columns
    X = X.drop(columns=datetime_columns)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3301)

    # Apply Random Over-Sampling
    ros = RandomOverSampler(random_state=3301)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

    logging.info(f"Original training set shape: {X_train.shape}")
    logging.info(f"Resampled training set shape: {X_train_resampled.shape}")

    clf = RandomForestClassifier(**RFCONFIG)
    clf.fit(X_train_resampled, y_train_resampled)
    
    y_scores = clf.predict_proba(X_test)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)
    
    target_precision_index = np.argmin(np.abs(precisions - target_precision))
    optimal_threshold = thresholds[target_precision_index]
    
    sorted_scores = np.sort(y_scores)[::-1]
    prediction_threshold = sorted_scores[int(len(sorted_scores) * prediction_percentage)]
    final_threshold = max(optimal_threshold, prediction_threshold)
    
    logging.info(f"Optimal threshold for {target_precision:.2f} precision: {optimal_threshold:.4f}")
    logging.info(f"Adjusted threshold for {prediction_percentage:.2%} predictions: {final_threshold:.4f}")
    
    y_pred = (y_scores >= final_threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision (class 1): {precision:.4f}")
    logging.info(f"Recall (class 1): {recall:.4f}")
    logging.info(f"F1 Score (class 1): {f1:.4f}")
    logging.info(f"Percentage of positive predictions: {y_pred.mean():.2%}")
    
    print(classification_report(y_test, y_pred, zero_division=0))

    # Additional detailed reporting
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    logging.info("\nDetailed Class 1 (Upward) Prediction Analysis:")
    logging.info(f"True Positives: {tp}")
    logging.info(f"False Positives: {fp}")
    logging.info(f"False Negatives: {fn}")
    logging.info(f"True Negatives: {tn}")
    
    positive_rate = (tp + fp) / len(y_test)
    logging.info(f"Positive Prediction Rate: {positive_rate:.4f}")
    
    if tp + fp > 0:
        precision = tp / (tp + fp)
        logging.info(f"Precision of Positive Predictions: {precision:.4f}")
    
    if tp + fn > 0:
        recall = tp / (tp + fn)
        logging.info(f"Recall of Positive Class: {recall:.4f}")
    
    # ROC curve
    plt.figure(figsize=(10, 6))
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(config['model_output_directory'], 'roc_curve.png'))
    plt.close()
    
    # Precision-Recall curve
    plt.figure(figsize=(10, 6))
    plt.plot(recalls, precisions, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.axhline(y=target_precision, color='r', linestyle='--', label=f'Target Precision ({target_precision:.2f})')
    plt.axvline(x=recall, color='g', linestyle='--', label=f'Actual Recall ({recall:.4f})')
    plt.legend()
    plt.savefig(os.path.join(config['model_output_directory'], 'precision_recall_curve.png'))
    plt.close()
    
    # Calibration curve
    plt.figure(figsize=(10, 6))
    fraction_of_positives, mean_predicted_value = calibration_curve(y_test, y_scores, n_bins=10)
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Random Forest")
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    plt.xlabel("Mean predicted value")
    plt.ylabel("Fraction of positives")
    plt.title("Calibration curve")
    plt.legend()
    plt.savefig(os.path.join(config['model_output_directory'], 'calibration_curve.png'))
    plt.close()
    
    brier_score = brier_score_loss(y_test, y_scores)
    logging.info(f"Brier Score: {brier_score:.4f}")

    # Feature importance analysis
    feature_importances = pd.DataFrame({
        'feature': X_train.columns,
        'importance': clf.feature_importances_
    }).sort_values(by='importance', ascending=False)
    
    feature_importances['importance'] = feature_importances['importance'].round(4)
    
    logging.info("\nTop 10 Most Important Features:")
    print(feature_importances.head(50))
    
    # Plot feature importances
    plt.figure(figsize=(12, 6))
    plt.bar(feature_importances['feature'][:20], feature_importances['importance'][:20])
    plt.xticks(rotation=90)
    plt.title('Top 20 Feature Importances')
    plt.tight_layout()
    plt.savefig(os.path.join(config['model_output_directory'], 'feature_importances.png'))
    plt.close()
    
    # Save outputs
    dump(clf, model_output_path)
    logging.info(f"Model saved to {model_output_path}")
    
    threshold_path = os.path.join(config['model_output_directory'], 'prediction_threshold.txt')
    with open(threshold_path, 'w') as f:
        f.write(str(final_threshold))
    logging.info(f"Prediction threshold saved to {threshold_path}")
    
    feature_importance_output_path = os.path.join(config['feature_importance_output'])
    feature_importances.to_parquet(feature_importance_output_path, index=False)

    return clf, final_threshold





def predict_and_save(input_directory, model_path, output_directory, target_column, date_column, confidence_threshold_pos, selected_features=None):
    logging.info("Loading the trained model.")
    
    for file in os.listdir(output_directory):
        if file.endswith('.parquet'):
            os.remove(os.path.join(output_directory, file))
    
    clf = load(model_path)
    
    model_features = clf.feature_names_in_
    
    all_files = [f for f in os.listdir(input_directory) if f.endswith('.parquet')]
    
    pbar = tqdm(total=len(all_files), desc="Processing files", ncols=100)
    
    joblib_logger = logging.getLogger('joblib')
    joblib_logger.setLevel(logging.ERROR)  
    
    null_io = io.StringIO()
    
    for file in all_files:
        df = pd.read_parquet(os.path.join(input_directory, file))
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Ensure we only use features that were present during training
        X = df[model_features]
        
        with parallel_backend('threading', n_jobs=-1):
            with redirect_stdout(null_io), redirect_stderr(null_io):
                y_pred_proba = clf.predict_proba(X)
        
        df['UpProbability'] = y_pred_proba[:, 1]
        df['UpPrediction'] = (df['UpProbability'] >= confidence_threshold_pos).astype(int)
        
        # Keep only necessary columns
        columns_to_keep = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'UpPrediction', 'UpProbability']
        if "Distance to Support (%)" in df.columns:
            columns_to_keep.append("Distance to Support (%)")
        if "Distance to Resistance (%)" in df.columns:
            columns_to_keep.append("Distance to Resistance (%)")
        if "percent_change_Close" in df.columns:
            columns_to_keep.append("percent_change_Close")
        
        df = df[columns_to_keep]
        df = df.round(5)
        
        # Convert numeric columns to float32
        float_columns = ['Open', 'High', 'Low', 'Close', 'UpProbability']
        if "Distance to Support (%)" in df.columns:
            float_columns.append("Distance to Support (%)")
        if "Distance to Resistance (%)" in df.columns:
            float_columns.append("Distance to Resistance (%)")
        if "percent_change_Close" in df.columns:
            float_columns.append("percent_change_Close")
        
        df[float_columns] = df[float_columns].astype(np.float32)
        df['UpPrediction'] = df['UpPrediction'].astype(np.int32)
        df['Volume'] = df['Volume'].astype(np.int32)

        output_file_path = os.path.join(output_directory, file)
        df.to_parquet(output_file_path, index=False)
        
        pbar.update(1)
    
    pbar.close()
    logging.info(f"Predictions saved to {output_directory}")






def main():
    selected_features = load_and_filter_features(RUNNINGCONFIG)
    
    if not args.predict:
        training_data = prepare_training_data(
            input_directory=RUNNINGCONFIG['input_directory'],
            output_directory=RUNNINGCONFIG['data_output_directory'],
            file_selection_percentage=RUNNINGCONFIG['file_selection_percentage'],
            target_column=RUNNINGCONFIG['target_column'],
            reuse=args.reuse,
            date_column='Date',
            selected_features=selected_features
        )
        logging.info("Data preparation complete.")
        
        train_precision_focused_random_forest(training_data, RUNNINGCONFIG)
    else:
        # Load the threshold
        threshold_path = os.path.join(RUNNINGCONFIG['model_output_directory'], 'prediction_threshold.txt')
        with open(threshold_path, 'r') as f:
            threshold = float(f.read().strip())
        
        predict_and_save(
            input_directory=RUNNINGCONFIG['input_directory'],
            model_path=os.path.join(RUNNINGCONFIG['model_output_directory'], 'random_forest_model.joblib'),
            output_directory=RUNNINGCONFIG['prediction_output_directory'],
            target_column=RUNNINGCONFIG['target_column'],
            date_column='Date',
            confidence_threshold_pos=threshold,
            selected_features=selected_features
        )


if __name__ == "__main__":
    main()