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
from scipy.stats import linregress

# Deep Learning libraries
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import MeanAbsolutePercentageError
from tensorflow.keras.callbacks import ModelCheckpoint
##import monte carlo dropout

# Machine Learning utilities
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Visualization tools
import plotly.graph_objects as go
import plotly.express as px


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'




class MonteCarloDropout(Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)

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


def add_jitter(data, jitter_amount=0.0000001):
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

    model = Sequential()

    # First LSTM layer with input_shape
    model.add(LSTM(input_shape[1]*4, return_sequences=True, input_shape=input_shape))
    model.add(MonteCarloDropout(0.3))
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
    learning_rate = 0.01
    clipnorm = 0.95
    logging.info(f"Hyperparameters - Learning Rate: {learning_rate}, Clipnorm: {clipnorm}")

    # Compile settings
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=clipnorm)
    model.compile(optimizer=optimizer, loss='mean_absolute_percentage_error', metrics=[tf.keras.metrics.MeanAbsolutePercentageError()])
    logging.info("Model compilation complete.")

    return model

def predict_with_uncertainty(f_model, X_test, n_iter=3):
    predictions = np.array([f_model.predict(X_test) for _ in range(n_iter)])
    mean_predictions = np.mean(predictions, axis=0)
    std_predictions = np.std(predictions, axis=0)
    return mean_predictions, std_predictions







def calculate_mape(y_true, y_pred):
    """
    Calculate the Mean Absolute Percentage Error (MAPE).
    :param y_true: Actual values.
    :param y_pred: Predicted values.
    :return: MAPE value.
    """
    mape = MeanAbsolutePercentageError()
    mape.update_state(y_true, y_pred)
    return mape.result().numpy()


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, batch_size=32, epochs=100, patience=3, verbose=1, additional_callbacks=[]):
    try:
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
            ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)
        ] + additional_callbacks

        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=verbose, callbacks=callbacks)


        epochs_trained = len(history.history['loss'])
        val_loss = history.history['val_loss'][-1]  # Last validation loss

        return history, epochs_trained, val_loss
    except Exception as e:
        logging.error(f'Error during training and evaluation: {e}')
        raise




def transform_target(data, target_column, shift_value=0.00000001):
    """Transform the target variable to avoid zero values."""
    data[target_column] += shift_value
    return data

##==================== MAIN FUNCTION ====================##
##==================== MAIN FUNCTION ====================##
##==================== MAIN FUNCTION ====================##
##==================== MAIN FUNCTION ====================##
##==================== MAIN FUNCTION ====================##


def main():
    args = get_args()

    # If args.file is None, use all files in the directory; otherwise, use the specified files
    file_paths = args.file if args.file else [os.path.join(config['directory_path'], f) for f in os.listdir(config['directory_path']) if f.endswith('.csv')]

    for file_path in file_paths:
        data = load_data(file_path)
        data = transform_target(data, config['target_column'])
        X, y = create_sequences(data, config['target_column'], config['sequence_length'])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3301)
        model = create_lstm_model((config["sequence_length"], X.shape[2]))
        training_history, epochs_trained, val_loss = train_and_evaluate_model(model, X_train, y_train, X_test, y_test)
        logging.info(f"Training on {os.path.basename(file_path)} complete. Epochs trained: {epochs_trained}, Validation Loss: {val_loss}")

        ##get the percentage of the files in the file paths folder to say "has completed 47% of files trained"
        percentage_completed = (file_paths.index(file_path) / len(file_paths)) * 100
        logging.info(f"Percentage of files trained: {percentage_completed:.2f}%")

        mean_predictions, std_predictions = predict_with_uncertainty(model, X_test, n_iter=1)
        logging.info(f"Standard Deviation of Predictions: {np.mean(std_predictions)}")

        if np.mean(std_predictions) < 0.1 and np.mean(mean_predictions) > 0.1:
            logging.warning("Model is overconfident")

        predictions = mean_predictions.flatten()
        predictions = predictions.flatten()  # Flatten the predictions

        # Before adjusting predictions
        predictions = np.nan_to_num(predictions)
        PreAjustedMape = calculate_mape(y_test, predictions)
        logging.info(f"MAPE before adjustments: {PreAjustedMape:.2f}%")
        
        # Adjusting predictions
        PostAjustedMape = calculate_mape(y_test, predictions)
        logging.info(f"MAPE after adjustments: {PostAjustedMape:.2f}%")
        
        # Improvement calculation
        AjustedImprovment = PreAjustedMape - PostAjustedMape
        logging.info(f"Improvement in MAPE due to adjustments: {AjustedImprovment:.2f}%")
        

if __name__ == "__main__":
    main()
