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









##==================== FIX PREDICTIONS ====================##
##==================== FIX PREDICTIONS ====================##
##==================== FIX PREDICTIONS ====================##
##==================== FIX PREDICTIONS ====================##


def adjust_predictions(predictions, data, window_size=10):
    """
    Adjust predictions based on specific criteria, ensuring predictions do not exceed the largest value in the real values.
    """
    adjusted_predictions = predictions.copy()

    # 1. Limit Predictions Based on Extreme Values
    rolling_window = data.rolling(window=window_size)
    rolling_max = rolling_window.quantile(0.9).to_numpy()  # Top 10% value
    rolling_min = rolling_window.quantile(0.1).to_numpy()  # Bottom 10% value
    limit_threshold = 0.85 * ((rolling_max + rolling_min) / 2)

    # 2. Slope Direction Change Detection
    def calculate_trend_line(data, window_size):
        x = np.arange(window_size)
        y = data[-window_size:]
        slope, intercept = np.polyfit(x, y, 1)
        return slope * (window_size - 1) + intercept

    # Find the absolute maximum value in the data
    absolute_max_value = max(abs(data.max()), abs(data.min()))

    for i in range(window_size, len(predictions)):
        # Limit based on extreme values
        adjusted_predictions[i] = min(adjusted_predictions[i], limit_threshold[i])

        # Slope direction changes
        slope_changes = np.sign(np.diff(data[i-window_size+1:i+1])).sum()
        if abs(slope_changes) <= 5:  # if slope direction changes frequently
            adjusted_predictions[i] = calculate_trend_line(data[i-5:i], 5)

        # Ensure predictions do not exceed the absolute maximum value in the data
        adjusted_predictions[i] = max(min(adjusted_predictions[i], absolute_max_value), -absolute_max_value)

    # Replace NaNs with the mean of the last 10 predictions
    adjusted_predictions = np.where(np.isnan(adjusted_predictions),
                                    np.nanmean(adjusted_predictions[np.maximum(i-10, 0):i]),
                                    adjusted_predictions)

    return adjusted_predictions














def log_statistics(model, X_sample, start_time, predictions, y_test, analysis_results, logger, accuracy_window=3, close_threshold=0.250):
    """
    Log extensive statistical metrics for the given model, predictions, and actual values, including
    accuracy after a close prediction.

    :param model: The trained Keras model.
    :param X_sample: A sample batch from the input data for activation analysis.
    :param start_time: The start time of the model training.
    :param predictions: The predictions made by the model.
    :param y_test: The actual values to compare against the predictions.
    :param analysis_results: Results from the analysis of predictions.
    :param logger: Logger object for logging.
    :param accuracy_window: The number of predictions to consider after a close prediction.
    :param close_threshold: The threshold to consider a prediction as close.
    """

    ##log the number of predicions that are nan values as a percentage of the total number of values in the predictions
    nan_values = np.isnan(predictions)
    nan_percentage = (np.sum(nan_values) / len(predictions)) * 100
    logger.info(f"Percentage of NaN values in predictions: {nan_percentage:.2f}%")


    ##replace any nan values with the mean of the last 10 predictions
    for i in range(len(predictions)):
        if np.isnan(predictions[i]):
            predictions[i] = np.mean(predictions[i-10:i])

    ##now get the percentage of nan values in the predictions
    nan_values = np.isnan(predictions)
    nan_percentage = (np.sum(nan_values) / len(predictions)) * 100
    logger.info(f"Percentage of NaN values in predictions after replacement: {nan_percentage:.2f}%")



    # Correctness of Slope
    slope_correct_count = 0
    direction_correct_count = 0
    for i in range(1, len(predictions)):
        real_slope = y_test[i] - y_test[i - 1]
        pred_slope = predictions[i] - predictions[i - 1]
        
        # Checking slope direction correctness
        if np.sign(real_slope) == np.sign(pred_slope):
            slope_correct_count += 1

        # Checking direction correctness
        if np.sign(y_test[i]) == np.sign(predictions[i]):
            direction_correct_count += 1

    slope_correctness = (slope_correct_count / (len(predictions) - 1)) * 100
    direction_correctness = (direction_correct_count / len(predictions)) * 100

    logger.info(f"Slope correctness: {slope_correctness:.2f}%")
    logger.info(f"Direction correctness: {direction_correctness:.2f}%")


    # Tracking accuracy after a close prediction
    close_accuracy_count = 0
    total_close_predictions = 0

    for i in range(len(predictions) - accuracy_window):
        if abs(predictions[i] - y_test[i]) <= close_threshold:
            total_close_predictions += 1
            if all(abs(predictions[i + j] - y_test[i + j]) <= close_threshold for j in range(1, accuracy_window + 1)):
                close_accuracy_count += 1

    close_accuracy = (close_accuracy_count / total_close_predictions) if total_close_predictions > 0 else 0
    logger.info(f"Accuracy after close prediction (next {accuracy_window} predictions): {close_accuracy * 100:.2f}%")


    ##log if the predictions have more than 20 values that are the exact same not necessarily in a row
    unique_values = np.unique(predictions)
    same_value_count = 0
    for value in unique_values:
        if np.sum(predictions == value) > 20:
            same_value_count += 1
    logger.info(f"Number of unique values in predictions: {len(unique_values)}")
    logger.info(f"Number of values that are the same in predictions: {same_value_count}")


    mse = mean_squared_error(y_test, predictions)
    r_squared = r2_score(y_test, predictions)

    logger.info(f"Mean Squared Error: {mse}")
    logger.info(f"R-squared: {r_squared}")

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
    logger.info(f"Average Error: {avg_error}")
    logger.info(f"Max Error: {max_error}")
    logger.info(f"Min Error: {min_error}")
    logger.info(f"Max Real Value: {max_real_value}")
    logger.info(f"Max Predicted Value: {max_predicted_value}")
    logger.info(f"Real Mean: {real_mean}, Real Median: {real_median}, Real Std Dev: {real_std}")
    logger.info(f"Predicted Mean: {pred_mean}, Predicted Median: {pred_median}, Predicted Std Dev: {pred_std}")





def analyze_predictions(predictions, y_test, low_error_threshold=0.05):
    """
    Analyze the adjusted predictions to detect if they form a straight line and calculate additional metrics.
    
    param adjusted_predictions: Array of adjusted model predictions
    param y_test: Array of actual values
    return: A dictionary of analysis results
    """


    if len(predictions.shape) > 1:
        predictions = predictions.flatten()

    # Ensure y_test is a numpy array and flatten it
    y_test = np.array(y_test).flatten()

    # Check if shapes are compatible
    if predictions.shape != y_test.shape:
        raise ValueError(f"Shape mismatch: predictions shape {predictions.shape}, y_test shape {y_test.shape}")

    # Flatten adjusted predictions if necessary
    if len(predictions.shape) > 1:
        predictions = predictions.flatten()

    # Ensure that y_test is a numpy array and has the same shape as adjusted_predictions
    y_test = np.array(y_test).flatten()
    if predictions.shape != y_test.shape:
        raise ValueError(f"Shape mismatch: adjusted_predictions shape {predictions.shape}, y_test shape {y_test.shape}")

    


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


    #calculate the number and size of streaks
    low_errors = errors < low_error_threshold
    streaks = []
    current_streak = 0
    streak_sizes = {}

    for is_low_error in low_errors:
        if is_low_error:
            current_streak += 1
        elif current_streak > 0:
            streaks.append(current_streak)
            streak_sizes[current_streak] = streak_sizes.get(current_streak, 0) + 1
            current_streak = 0

    if current_streak > 0:
        streaks.append(current_streak)
        streak_sizes[current_streak] = streak_sizes.get(current_streak, 0) + 1

    results['low_error_streaks'] = streaks

    # Logging streak sizes
    for size, count in streak_sizes.items():
        logging.info(f"{count} of streaks of size {size}")




    return results



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

    # Update this trace to have a dashed green line
    fig.add_trace(go.Scatter(
        x=plot_data.index, 
        y=plot_data['Prediction_Error'], 
        mode='lines', 
        name='Prediction Error',
        line=dict(color='green', dash='dash')  # Dashed green line
    ))

    fig.update_layout(title=f'Predictions and Actual Values for {os.path.basename(file_path)}', xaxis_title='Index', yaxis_title='Value')
    fig.show()


def plot_cumulative_performance(trade_results, initial_investment=1000, smp500_annual_growth=0.07, trading_days_per_year=280):
    """
    Plot the cumulative performance of the trading strategy compared to S&P 500 and target CAGR lines.

    :param trade_results: Array of trade profit/loss results.
    :param initial_investment: Initial investment amount.
    :param smp500_annual_growth: Annual growth rate of S&P 500 (7% standard).
    :param trading_days_per_year: Number of trading days in a year.
    """
    # Calculate cumulative P/L for the trading strategy
    cumulative_pl_strategy = [initial_investment]
    for pl in trade_results:
        cumulative_pl_strategy.append(cumulative_pl_strategy[-1] * (1 + pl))

    x = np.array(range(len(cumulative_pl_strategy)))
    y = np.array(cumulative_pl_strategy)
    slope, intercept, _, _, _ = linregress(x, y)
    line = slope * x + intercept


    # Create the plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=line, mode='lines', name='Line of Best Fit', line=dict(color='green', dash='dash')))



    # Add traces for the trading strategy and S&P 500
    fig.add_trace(go.Scatter(x=list(range(len(cumulative_pl_strategy))), y=cumulative_pl_strategy, mode='lines', name='Trading Strategy'))

    # Add target CAGR lines
    target_cagrs = [0.07, 0.10, 0.20, 0.30]  # 10%, 20%, and 30% CAGR
    target_colors = ['black', 'red', 'orange', 'green']  # Colors for each target line
    for cagr, color in zip(target_cagrs, target_colors):
        target_values = [initial_investment * ((1 + cagr) ** (i/trading_days_per_year)) for i in range(len(cumulative_pl_strategy))]
        fig.add_trace(go.Scatter(x=list(range(len(target_values))), y=target_values, mode='lines', name=f'{int(cagr*100)}% CAGR', line=dict(dash='dash', color=color)))

    # Update layout
    fig.update_layout(title='Cumulative Performance: Trading Strategy vs S&P 500 and Target CAGRs',
                      xaxis_title='Trading Days',
                      yaxis_title='Cumulative Value',
                      yaxis=dict(type='log'),  # Use log scale for better comparison
                      xaxis=dict(tickvals=[i * trading_days_per_year for i in range((len(cumulative_pl_strategy) - 1) // trading_days_per_year + 1)],
                                 ticktext=[f'Year {i+1}' for i in range((len(cumulative_pl_strategy) - 1) // trading_days_per_year + 1)]))

    fig.show()





















##==================== SIMULATE TRADING ====================##
##==================== SIMULATE TRADING ====================##
##==================== SIMULATE TRADING ====================##
##==================== SIMULATE TRADING ====================##
##==================== SIMULATE TRADING ====================##





def simulate_trading(predictions, actual_values, close_threshold=0.10, take_profit_factor=0.85, stop_loss_factor=0.25):

    """
    Simulate trading with detailed logging and modified take profit strategy.

    :param predictions: Array of percentage change predictions.
    :param actual_values: Array of actual percentage change.
    :param close_threshold: Threshold for entering trades.
    :param take_profit_factor: Factor of predicted change for taking profit.
    :param stop_loss_factor: Factor of predicted change for stopping loss.
    """

    global take_profit_hits, stop_loss_hits, end_of_day_exits
    total_trades = 0
    total_profit_loss = 0
    stop_loss_hits = 0
    take_profit_hits = 0
    end_of_day_exits = 0
    trade_results = []
    take_profit_pl = []
    stop_loss_pl = []
    end_of_day_pl = []

    for i in range(len(predictions) - 1):
        predicted_move = predictions[i + 1]
        actual_move = actual_values[i + 1]
        is_close_to_actual = np.sign(predicted_move) == np.sign(actual_move) and \
                             abs(predicted_move - actual_move) / abs(actual_move) <= close_threshold

        if is_close_to_actual:
            total_trades += 1
            take_profit_target = take_profit_factor * abs(predicted_move)
            stop_loss_target = stop_loss_factor * abs(predicted_move)

            # Long position if predicted move is positive, else short position
            if predicted_move > 0:
                trade_profit_loss = calculate_profit_loss(actual_move, take_profit_target, stop_loss_target, is_long=True)
            else:
                trade_profit_loss = calculate_profit_loss(-actual_move, take_profit_target, stop_loss_target, is_long=False)
            

            if trade_profit_loss != 0:
                trade_profit_loss*=0.01
            


            # Classify trade and append to respective lists
            classify_and_log_trade(trade_profit_loss, take_profit_target, stop_loss_target, take_profit_pl, stop_loss_pl, end_of_day_pl)

            # Update total profit/loss
            total_profit_loss += trade_profit_loss
            trade_results.append(trade_profit_loss)

    # Logging and calculating final statistics
    log_final_statistics(total_trades, total_profit_loss, stop_loss_hits, take_profit_hits, end_of_day_exits, take_profit_pl, stop_loss_pl, end_of_day_pl, predictions)

    return trade_results, total_profit_loss

def calculate_profit_loss(actual_move, take_profit_target, stop_loss_target, is_long=True):
    """
    Calculate profit or loss for a trade based on actual move, take profit and stop loss targets.

    :param actual_move: The actual percentage change.
    :param take_profit_target: The take profit target.
    :param stop_loss_target: The stop loss target.
    :param is_long: Boolean indicating if the position is long.
    :return: Profit or loss of the trade.
    """
    if is_long:
        if actual_move >= take_profit_target:
            return 0.8 * take_profit_target + 0.2 * actual_move  # Partial take profit and end of day exit
        elif actual_move <= -stop_loss_target:
            return -stop_loss_target  # Stop loss hit
        else:
            return actual_move  # End of day exit
    else:
        if actual_move <= -take_profit_target:
            return 0.8 * take_profit_target + 0.2 * (-actual_move)  # Partial take profit and end of day exit
        elif actual_move >= stop_loss_target:
            return -stop_loss_target  # Stop loss hit
        else:
            return -actual_move  # End of day exit






def classify_and_log_trade(trade_profit_loss, take_profit_target, stop_loss_target, take_profit_pl, stop_loss_pl, end_of_day_pl):
    """
    Classify the trade based on its outcome and log the results.

    :param trade_profit_loss: The profit or loss of the trade.
    :param take_profit_target: The take profit target.
    :param stop_loss_target: The stop loss target.
    :param take_profit_pl: List to store take profit trade results.
    :param stop_loss_pl: List to store stop loss trade results.
    :param end_of_day_pl: List to store end of day exit trade results.
    """
    global take_profit_hits, stop_loss_hits, end_of_day_exits

    if abs(trade_profit_loss) >= take_profit_target * 0.8:
        take_profit_hits += 1
        take_profit_pl.append(trade_profit_loss)
        logging.info("Exiting Trade: Partial Take Profit Hit, Remaining Exited End of Day")
    elif abs(trade_profit_loss) >= stop_loss_target:
        stop_loss_hits += 1
        stop_loss_pl.append(trade_profit_loss)
        logging.info("Exiting Trade: Stop Loss Hit")
    else:
        end_of_day_exits += 1
        end_of_day_pl.append(trade_profit_loss)
        logging.info("Exiting Trade: End of Day")

def log_final_statistics(total_trades, total_profit_loss, stop_loss_hits, take_profit_hits, end_of_day_exits, take_profit_pl, stop_loss_pl, end_of_day_pl, predictions):
    """
    Log the final statistics of the trading simulation, focusing on average percentage gain/loss per trade.

    :param total_trades: Total number of trades.
    :param total_profit_loss: Total profit or loss.
    :param stop_loss_hits: Number of stop loss hits.
    :param take_profit_hits: Number of take profit hits.
    :param end_of_day_exits: Number of end of day exits.
    :param take_profit_pl: List of take profit trade results (as percentages).
    :param stop_loss_pl: List of stop loss trade results (as percentages).
    :param end_of_day_pl: List of end of day exit trade results (as percentages).
    :param predictions: Array of predictions for calculating P/L per year.
    """

    if total_trades == 0:
        logging.warning("No trades were made. Skipping final statistics.")
        return





    logging.info(f"Total Trades: {total_trades}")
    logging.info(f"Overall P/L: {total_profit_loss}")
    logging.info(f"Stop Loss Hits: {stop_loss_hits} ({stop_loss_hits/total_trades*100:.2f}%)")
    logging.info(f"Take Profit Hits: {take_profit_hits} ({take_profit_hits/total_trades*100:.2f}%)")
    logging.info(f"End of Day Exits: {end_of_day_exits} ({end_of_day_exits/total_trades*100:.2f}%)")
    
    # Calculate and log average percentage gain/loss
    avg_pct_gain_take_profit = np.mean(take_profit_pl) * 100 if take_profit_pl else 0
    avg_pct_loss_stop_loss = np.mean(stop_loss_pl) * 100 if stop_loss_pl else 0
    avg_pct_gain_end_of_day = np.mean(end_of_day_pl) * 100 if end_of_day_pl else 0

    logging.info(f"Average Percentage Gain for Take Profit Trades: {avg_pct_gain_take_profit:.2f}%")
    logging.info(f"Average Percentage Loss for Stop Loss Trades: {avg_pct_loss_stop_loss:.2f}%")
    logging.info(f"Average Percentage Gain for End of Day Exits: {avg_pct_gain_end_of_day:.2f}%")

    trading_days = len(predictions) / 280
    pl_per_year = total_profit_loss / trading_days
    logging.info(f"Overall P/L per year: {pl_per_year}")

    if pl_per_year < 0.07:
        logging.warning("P/L per year is less than the S&P 500 average of 7%")
    if pl_per_year < 0.30:
        logging.warning("P/L per year is less than the target of 30%")









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

        ##if the predictions are a straight line with more than 5 values with almost the same value than just continue
        analysis_results = analyze_predictions(predictions, y_test)
        if analysis_results['straight_line']:
            logging.warning("Predictions form a straight line. Skipping further analysis.")
            continue



        # Before adjusting predictions
        predictions = np.nan_to_num(predictions)
        PreAjustedMape = calculate_mape(y_test, predictions)
        logging.info(f"MAPE before adjustments: {PreAjustedMape:.2f}%")
        
        # Adjusting predictions
        predictions = adjust_predictions(predictions, data[config['target_column']], window_size=20)
        PostAjustedMape = calculate_mape(y_test, predictions)
        logging.info(f"MAPE after adjustments: {PostAjustedMape:.2f}%")
        
        # Improvement calculation
        AjustedImprovment = PreAjustedMape - PostAjustedMape
        logging.info(f"Improvement in MAPE due to adjustments: {AjustedImprovment:.2f}%")
        

        # Use adjusted_predictions instead of predictions for further analysis and plotting
        analysis_results = analyze_predictions(predictions, y_test)
        log_statistics(model, data.sample(1), time.time(), predictions, y_test, analysis_results, logging)


        trade_results, overall_pl = simulate_trading(predictions, y_test)

        if overall_pl < 0.07:
            logging.warning("Overall P/L is less than the S&P 500 average of 7%")
        else :
            logging.info("Overall P/L is greater than the S&P 500 average of 7%")
            save_and_plot_data_with_predictions(data, predictions, file_path, config, config['sequence_length'])
            plot_cumulative_performance(trade_results)




if __name__ == "__main__":
    main()


