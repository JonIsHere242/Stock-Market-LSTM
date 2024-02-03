# Data processing and mathematical operations
import numpy as np
import pandas as pd

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
##import train test split
##import mape metric
from sklearn.model_selection import train_test_split
# Metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.metrics import MeanAbsolutePercentageError
import plotly.graph_objects as go
import plotly.express as px

import argparse
import os



import logging
from datetime import datetime

# Setup logging
logging.basicConfig(filename='Data/ModelData/LSTMTrainingErrors.log', filemode='a', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)




config = {
    "directory_path": 'Data/individualPredictedFiles',  # Default directory containing files
    "target_column": "percent_change_Close",           # Replace with your actual target column name
    "sequence_length": 50,
    "model_save_path": "D:/LSTM/Stock-Market-LSTM/Data/ModelData/lstm_model.h5",
    "lstm_predicted_data_path": "Data/LSTMpredictedData",
    "Lstm_logging_path": "Data/ModelData/LSTMTrainingErrors.log"

}






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

# LSTM model creation
def create_lstm_model(input_shape, learning_rate=0.01):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=16))
    model.add(Dropout(0.1))
    model.add(Dense(units=1))

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_absolute_percentage_error')  # Using MAPE as loss function
    return model





def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    try:
        early_stopping = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
        history = model.fit(X_train, y_train, epochs=100, batch_size=8, validation_split=0.2, callbacks=[early_stopping])

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


def transform_target(data, target_column, shift_value=0.0001):
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

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = create_lstm_model((config["sequence_length"], X.shape[2]), learning_rate=0.01)  # Example with learning rate of 0.01
        training_history, mape_value, epochs_trained = train_and_evaluate_model(model, X_train, y_train, X_test, y_test)

        predictions = model.predict(X_test)

        plot_loss(training_history, f"Training and Validation Loss for {os.path.basename(file_path)}")
        save_and_plot_data_with_predictions(data, predictions, file_path, config, config['sequence_length'])


if __name__ == "__main__":
    main()


