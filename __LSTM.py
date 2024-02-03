# Standard library imports
import os
import time
import psutil
import logging
import argparse
from datetime import datetime

# Data processing and mathematical operations
import numpy as np
import pandas as pd

# Deep Learning libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import MeanAbsolutePercentageError

# Machine Learning utilities
from sklearn.model_selection import train_test_split

# Visualization tools
import plotly.graph_objects as go
import plotly.express as px


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



config = {
    "directory_path": 'Data/individualPredictedFiles',  # Default directory containing files
    "target_column": "percent_change_Close",           # Replace with your actual target column name
    "sequence_length": 50,
    "model_save_path": "D:/LSTM/Stock-Market-LSTM/Data/ModelData/lstm_model.h5",
    "lstm_predicted_data_path": "Data/LSTMpredictedData",
    "Lstm_logging_path": "Data/ModelData/LSTMTrainingErrors.log"

}


log_file_path = 'Data/LSTMpredictedData/LSTMTrainingErrors.log'

# Create directories if they don't exist
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

# Setup logging
logging.basicConfig(filename=log_file_path, filemode='a', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)




def get_args():
    parser = argparse.ArgumentParser(description="Run LSTM model on stock market data.")
    parser.add_argument("--file", nargs='*', help="Path to a specific file to process", default=None)
    return parser.parse_args()


def add_jitter(data, jitter_amount=0.00001):
    """Add a small random amount to the data to avoid zero values."""
    jitter = np.random.uniform(-jitter_amount, jitter_amount, data.shape)
    return data + jitter

def load_data(file_path):
    """Load the dataset from a CSV file and drop rows with NaN values."""
    data = pd.read_csv(file_path)
    data.dropna(inplace=True)  # Drop rows with NaN values
    return data

def create_sequences(data, target_column, sequence_length=50):
    """Create sequences from the dataset suitable for LSTM training, predicting the next value."""
    sequences = []
    output = []
    for i in range(len(data) - sequence_length):
        sequences.append(data.iloc[i:(i + sequence_length)].to_numpy())  # Convert to numpy array
        output.append(data.iloc[i + sequence_length][target_column])
    output_array = np.array(output)  # Convert output to numpy array
    return np.array(sequences), add_jitter(output_array)  # Add jitter to output





##==================== LSTM Model ====================##
##==================== LSTM Model ====================##
##==================== LSTM Model ====================##
##==================== LSTM Model ====================##
##==================== LSTM Model ====================##


def create_lstm_model(input_shape):
    logging.info("V==================== Creating LSTM Model ====================V")
    logging.info(f"Input Shape: {input_shape}")

    model = Sequential()

    # First LSTM layer with input_shape
    model.add(LSTM(input_shape[1]*4, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    logging.info(f"Added LSTM layer with {input_shape[1]*4} units and input shape {input_shape}")

    # Subsequent layers without input_shape
    lstm_layers = [(input_shape[1]*3, 0.2), (input_shape[1]*2, 0.2), (input_shape[1], 0.2)]
    for units, dropout_rate in lstm_layers:
        model.add(LSTM(units, return_sequences=True))
        model.add(Dropout(dropout_rate))
        logging.info(f"Added LSTM layer with {units} units")

    # Final LSTM layer without return_sequences
    model.add(LSTM(input_shape[1], return_sequences=False))
    model.add(Dropout(0.2))
    logging.info(f"Added final LSTM layer with {input_shape[1]} units")

    # Dense layers
    model.add(Dense(round(input_shape[1]/2), activation='leaky_relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1))
    logging.info("Added Dense layers")

    # Hyperparameters
    learning_rate = 0.40
    clipnorm = 0.82
    logging.info(f"Hyperparameters - Learning Rate: {learning_rate}, Clipnorm: {clipnorm}")

    # Compile settings
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=clipnorm)
    model.compile(optimizer=optimizer, loss='mean_absolute_percentage_error', metrics=[tf.keras.metrics.MeanAbsolutePercentageError()])
    logging.info("Model compilation complete.")

    return model




def log_statistics(model, X_sample, start_time, predictions, y_test, analysis_results, logger):
    """
    Log extensive statistical metrics for the given model, predictions, and actual values.

    :param model: The trained Keras model.
    :param X_sample: A sample batch from the input data for activation analysis.
    :param start_time: The start time of the model training.
    :param predictions: The predictions made by the model.
    :param y_test: The actual values to compare against the predictions.
    :param analysis_results: Results from the analysis of predictions.
    :param logger: Logger object for logging.
    """
    # Calculate Errors
    errors = predictions.flatten() - y_test
    avg_error = np.mean(errors)
    max_error = np.max(errors)
    min_error = np.min(errors)
    max_real_value = np.max(y_test)
    max_predicted_value = np.max(predictions)

    # Statistical Metrics
    real_mean = np.mean(y_test)
    real_median = np.median(y_test)
    real_std = np.std(y_test)
    pred_mean = np.mean(predictions)
    pred_median = np.median(predictions)
    pred_std = np.std(predictions)

    # Log model summary
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    summary_str = "\n".join(model_summary)
    logger.info(f"Model Summary:\n{summary_str}")

    # Log data characteristics (assuming X_sample is a DataFrame)
    skewness = X_sample.skew()
    kurtosis = X_sample.kurtosis()
    logger.info(f"Data Skewness:\n{skewness}")
    logger.info(f"Data Kurtosis:\n{kurtosis}")

    # Log prediction distribution
    logger.info(f"Prediction Distribution: Mean: {np.mean(predictions)}, Std: {np.std(predictions)}")

    # Log computation time and memory usage
    end_time = time.time()
    memory_usage = psutil.Process().memory_info().rss / (1024 * 1024)  # in MB
    logger.info(f"Memory Usage: {memory_usage} MB")
    logger.info(f"Computation Time: {end_time - start_time} seconds")

    # Log the standard statistics
    logger.info(f"Minimum Error: {analysis_results['min_error']}")
    logger.info(f"Predictions form a straight line: {analysis_results['straight_line']}")
    logger.info(f"Low Error Streaks: {analysis_results['low_error_streaks']}")
    logger.info(f"Average Error: {avg_error}")
    logger.info(f"Max Error: {max_error}")
    logger.info(f"Min Error: {min_error}")
    logger.info(f"Max Real Value: {max_real_value}")
    logger.info(f"Max Predicted Value: {max_predicted_value}")
    logger.info(f"Real Mean: {real_mean}, Real Median: {real_median}, Real Std Dev: {real_std}")
    logger.info(f"Predicted Mean: {pred_mean}, Predicted Median: {pred_median}, Predicted Std Dev: {pred_std}")





def analyze_predictions(predictions, y_test, low_error_threshold=0.02):
    """
    Analyze the predictions to detect if they form a straight line and calculate additional metrics.
    
    :param predictions: Array of model predictions
    :param y_test: Array of actual values
    :param low_error_threshold: Threshold below which the error is considered low
    :return: A dictionary of analysis results
    """
    results = {}

    # Calculate errors
    errors = np.abs(predictions.flatten() - y_test)
    min_error = np.min(errors)
    results['min_error'] = min_error


    ##calculate a straight line by checking if the predicted value has a streak of 5 or more that are the same in a row
    straight_line = False
    current_streak = 0
    for i in range(len(predictions) - 1):
        if predictions[i] == predictions[i + 1]:
            current_streak += 1
        else:
            current_streak = 0
        if current_streak >= 5:
            straight_line = True
            break
    results['straight_line'] = straight_line



    # Calculate low error streaks
    low_errors = errors < low_error_threshold
    streaks = []
    current_streak = 0
    for is_low_error in low_errors:
        if is_low_error:
            current_streak += 1
        elif current_streak > 0:
            streaks.append(current_streak)
            current_streak = 0
    if current_streak > 0:
        streaks.append(current_streak)
    results['low_error_streaks'] = streaks

    return results





def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    try:
        early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
        history = model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2, callbacks=[early_stopping])

        epochs_trained = len(history.history['loss'])
        logging.info(f'Training completed. Number of epochs trained: {epochs_trained}')

        predictions = model.predict(X_test)
        mape = MeanAbsolutePercentageError()
        mape.update_state(y_test, predictions)
        mape_value = mape.result().numpy()
        logging.info(f'MAPE: {mape_value}')
        print(f'MAPE: {mape_value}')

        return history, mape_value, epochs_trained
    except Exception as e:
        logging.error(f'Error during training and evaluation: {e}')
        raise





def plot_loss(history, title, window_size=5):
    training_loss = pd.Series(history.history['loss']).rolling(window=window_size).mean()
    validation_loss = pd.Series(history.history['val_loss']).rolling(window=window_size).mean()

    if training_loss.dropna().iloc[-1] < 50:  # Plot only if final average training loss is less than 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=training_loss, mode='lines', name='Smoothed Training Loss'))
        fig.add_trace(go.Scatter(y=validation_loss, mode='lines', name='Smoothed Validation Loss'))
        fig.update_layout(title=title, xaxis_title='Epoch', yaxis_title='Loss')
        fig.show()


def transform_target(data, target_column, shift_value=0.00000001):
    """Transform the target variable to avoid zero values."""
    data[target_column] += shift_value
    return data

def save_and_plot_data_with_predictions(data, predictions, file_path, config, sequence_length):
    # Calculate the starting index for appending predictions
    start_idx = sequence_length

    # Append predictions to the dataframe
    num_predictions = len(predictions)
    data['LSTM_Predictions'] = np.nan
    data.loc[start_idx:start_idx + num_predictions - 1, 'LSTM_Predictions'] = predictions.flatten()

    # Calculate error
    data['Prediction_Error'] = data[config['target_column']] - data['LSTM_Predictions']

    # Save the dataframe
    save_path = os.path.join(config['lstm_predicted_data_path'], os.path.basename(file_path))
    data.to_csv(save_path, index=False)

    # Filter to rows where all values are available
    plot_data = data.dropna(subset=['LSTM_Predictions', config['target_column'], 'Prediction_Error'])

    # Plotting predictions, actual values, and errors
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data[config['target_column']], mode='lines', name='Actual Value'))
    fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['LSTM_Predictions'], mode='lines', name='Predictions'))
    fig.add_trace(go.Scatter(x=plot_data.index, y=plot_data['Prediction_Error'], mode='lines', name='Prediction Error'))
    fig.update_layout(title=f'Predictions and Actual Values for {os.path.basename(file_path)}', xaxis_title='Index', yaxis_title='Value')
    fig.show()



def main():
    args = get_args()

    # If args.file is None, use all files in the directory; otherwise, use the specified files
    file_paths = args.file if args.file else [os.path.join(config['directory_path'], f) for f in os.listdir(config['directory_path']) if f.endswith('.csv')]

    for file_path in file_paths:
        data = load_data(file_path)
        data = transform_target(data, config['target_column'])
        X, y = create_sequences(data, config['target_column'], config['sequence_length'])


        print(f"Shape of X: {X.shape}")  # Debugging line
        print(f"Shape of y: {y.shape}")  # Debugging line

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3301)

        model = create_lstm_model((config["sequence_length"], X.shape[2]))  # Example with learning rate of 0.01
        training_history, mape_value, epochs_trained = train_and_evaluate_model(model, X_train, y_train, X_test, y_test)

        predictions = model.predict(X_test)
        analysis_results = analyze_predictions(predictions, y_test)

        log_statistics(model, data.sample(1), time.time(), predictions, y_test, analysis_results, logging)

        plot_loss(training_history, f"Training and Validation Loss for {os.path.basename(file_path)}")
        save_and_plot_data_with_predictions(data, predictions, file_path, config, config['sequence_length'])


if __name__ == "__main__":
    main()


