# ====== Basic Utilities ======
import logging
import numpy as np
import os 
# ====== Machine Learning Utilities ======
from sklearn.metrics import mean_absolute_percentage_error

#======= Data Preprocessing Utilities ======
import pandas as pd
from scipy.stats import linregress
from scipy.signal import find_peaks



# ====== Trading/Backtesting Libraries ======
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

# ====== Data Visualization Utilities ======
import seaborn as sns
import plotly.graph_objects as go





log_file_path = "Data/LSTMpredictedData/Prediction.log"
logging.basicConfig(filename=log_file_path, filemode='a', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)


def calculate_best_fit_slope(data, window_size):
    # Find peaks and troughs
    peaks, _ = find_peaks(data, distance=window_size)
    troughs, _ = find_peaks(-data, distance=window_size)

    # Combine and sort indices
    extrema_indices = np.sort(np.concatenate((peaks, troughs)))

    # Calculate the line of best fit
    if len(extrema_indices) >= 2:
        slope, intercept, _, _, _ = linregress(extrema_indices, data[extrema_indices])
        return slope
    else:
        return 0  # No clear trend

def adjust_predictions(predictions, data, base_window_size=5, trend_factor=0.1, anomaly_threshold=2.5):
    """
    Adjust predictions using a rolling window approach. The window size is dynamic, 
    based on recent volatility, and adjustments consider trends and potential anomalies.

    Parameters:
    predictions (np.array): Array of predicted values.
    data (np.array): Array of actual historical values.
    base_window_size (int): Base size of the rolling window.
    trend_factor (float): Factor to adjust predictions based on the trend slope.
    anomaly_threshold (float): Z-score threshold for detecting anomalies.

    Returns:
    np.array: Adjusted predictions.
    """
    valid_data = data[~np.isnan(data)]

    if len(valid_data) < base_window_size:
        logging.warning("Insufficient data for making adjustments.")
        return predictions

    adjusted_predictions = predictions.copy()
    rolling_window_size = base_window_size
    mean_data = np.mean(valid_data)
    std_data = np.std(valid_data)

    for i in range(len(predictions)):
        # Adjust the rolling window size based on recent volatility
        if i >= base_window_size:
            recent_std = np.std(valid_data[i-base_window_size:i])
            rolling_window_size = min(len(valid_data), max(base_window_size, int(base_window_size * (1 + recent_std))))

        # Calculate trend slope in the rolling window
        if i >= rolling_window_size:
            rolling_window_data = valid_data[i-rolling_window_size:i]
            slope = calculate_best_fit_slope(rolling_window_data, rolling_window_size)

            # Calculate dynamic bounds based on recent data
            cap_positive = mean_data + anomaly_threshold * std_data
            cap_negative = mean_data - anomaly_threshold * std_data

            # Adjust prediction with trend factor and cap with dynamic bounds
            trend_adjustment = slope * trend_factor
            adjusted_prediction = predictions[i] + trend_adjustment
            adjusted_predictions[i] = min(max(adjusted_prediction, cap_negative), cap_positive)

            # Check for anomalies and adjust
            if abs(adjusted_predictions[i] - mean_data) > anomaly_threshold * std_data:
                adjusted_predictions[i] = mean_data + np.sign(adjusted_predictions[i] - mean_data) * anomaly_threshold * std_data

    return adjusted_predictions




def calculate_mean_percentage_error(target_data, predictions):
    if len(target_data) != len(predictions):
        logging.warning("Target data and predictions arrays have different lengths.")
        return np.nan
    
    valid_indices = ~np.isnan(predictions)
    filtered_target_data = target_data[valid_indices]
    filtered_predictions = predictions[valid_indices]

    if len(filtered_target_data) == 0 or len(filtered_predictions) == 0:
        logging.warning("No valid data available for error calculation. Number of valid data points: {}".format(len(filtered_predictions)))
        return np.nan

    percentage_errors = np.abs((filtered_target_data - filtered_predictions) / filtered_target_data) * 100
    mean_error = np.mean(percentage_errors)

    if np.isnan(mean_error):
        logging.warning("Predictions are Nan")
    return mean_error


def calculate_average_days_between_trades(actual_values, predictions):
    trade_dates = []
    successful_trade = False

    for i in range(1, len(predictions)):  # Start from the second element
        if abs((predictions[i-1] - actual_values[i-1]) / actual_values[i-1]) <= 0.1:
            successful_trade = True

        if successful_trade:
            trade_dates.append(i)  # Assuming each index represents a day
            successful_trade = False

    if len(trade_dates) > 1:
        intervals = [trade_dates[i] - trade_dates[i-1] for i in range(1, len(trade_dates))]
        average_interval = sum(intervals) / len(intervals)
        return average_interval
    else:
        return None



def log_trade_outcomes_and_percentages(actual_values, predictions):
    successful_trade = False
    trade_wins = 0
    trade_losses = 0
    total_trades = 0
    total_win_percentage = 0
    total_loss_percentage = 0
    trade_durations = []
    trade_entries_exits = []

    for i in range(1, len(predictions)):  
        if abs((predictions[i-1] - actual_values[i-1]) / actual_values[i-1]) <= 0.1:
            successful_trade = True
            trade_start = i - 1  # Record trade start

        if successful_trade:
            predicted_change = predictions[i] - predictions[i-1]
            actual_change = actual_values[i] - actual_values[i-1]

            if (predicted_change > 0 and actual_change > 0) or (predicted_change < 0 and actual_change < 0):
                trade_wins += 1
                win_percentage = abs(actual_change / actual_values[i-1])
                total_win_percentage += win_percentage
            else:
                trade_losses += 1
                loss_percentage = abs(actual_change / actual_values[i-1])
                total_loss_percentage += loss_percentage

            total_trades += 1
            trade_durations.append(i - trade_start)
            trade_entries_exits.append((trade_start, i))
            successful_trade = False  

    if total_trades > 0:
        win_percentage = (trade_wins / total_trades) * 100
        mean_win_percentage = total_win_percentage / trade_wins if trade_wins > 0 else 0
        mean_loss_percentage = total_loss_percentage / trade_losses if trade_losses > 0 else 0

        # Logging detailed information
        logging.info(f"Total Trades: {total_trades}, Wins: {trade_wins}, Losses: {trade_losses}")
        logging.info(f"Win Percentage: {win_percentage:.2f}%, Mean Win Percentage: {mean_win_percentage:.2f}%, Mean Loss Percentage: {mean_loss_percentage:.2f}%")
        logging.info(f"Average Trade Duration: {np.mean(trade_durations)} periods")
        logging.info(f"Trade Entry and Exit Points: {trade_entries_exits}")
    else:
        logging.info("No trades were executed.")





class MyTradingStrategy(Strategy):
    def init(self):
        # Initialization logic
        self.predicted_values = self.data.predicted  # Assuming 'predicted' is a column in your data
        self.real_values = self.data.Close  # Assuming 'Close' is the real closing value

    def next(self):
        # Logic for each step in the backtesting
        if len(self.predicted_values) < 2:
            return  # Not enough data for comparison

        # Calculate the accuracy of the previous prediction
        previous_prediction_accuracy = abs(self.predicted_values[-2] - self.real_values[-2]) / self.real_values[-2]

        # Check if the previous prediction was accurate within 10%
        if previous_prediction_accuracy <= 0.1:
            # Calculate the accuracy of the current prediction
            current_prediction_accuracy = abs(self.predicted_values[-1] - self.real_values[-1]) / self.real_values[-1]

            # Make a trade if the current prediction is also accurate within 10%
            if current_prediction_accuracy <= 0.1:
                # Implement the logic for trading. For example:
                # Buy if the prediction is positive, sell if negative
                if self.predicted_values[-1] > 0:
                    self.buy()
                else:
                    self.sell()


def evaluate_trading_model(win_percentage, mean_win_percentage, mean_loss_percentage, average_days_between_trades, min_win_percentage=50, max_loss_percentage=2):
    """
    Evaluates the trading model based on specified criteria including win percentage and loss percentage bounds.

    Parameters:
    win_percentage (float): The win percentage of the model.
    mean_win_percentage (float): The average percentage of winning trades.
    mean_loss_percentage (float): The average percentage of losing trades.
    average_days_between_trades (float): The average number of days between trades.
    min_win_percentage (float): Minimum win percentage for the model to be considered good.
    max_loss_percentage (float): Maximum loss percentage for the model to be considered good.

    Returns:
    bool: True if the model meets the criteria, False otherwise.
    """
    # Criteria
    sufficient_win_rate = win_percentage >= min_win_percentage
    acceptable_losses = mean_loss_percentage <= max_loss_percentage
    profitable = mean_win_percentage > mean_loss_percentage
    frequent_trades = average_days_between_trades <= 60

    # Evaluate criteria
    if sufficient_win_rate and acceptable_losses and profitable and frequent_trades:
        return True
    else:
        return False


def main():
    directory_path = 'Data/LSTMpredictedData'

    target_column = 'percent_change_Close'
    prediction_column = 'LSTM_Predictions'

    # Setup logging
    logging.basicConfig(filename='processing_log.log', filemode='a', 
                        format='%(asctime)s - %(levelname)s - %(message)s', 
                        level=logging.INFO)
    File_processed = 0

    # Iterate over files in the directory
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)

        try:
            if filename.endswith('.csv'):  # Check for CSV files
                File_processed += 1
                data = pd.read_csv(file_path)

                if target_column in data.columns and prediction_column in data.columns:
                    # Extract predictions and target data
                    predictions = data[prediction_column].values
                    target_data = data[target_column].values

                    # Apply adjustments
                    adjusted_predictions = adjust_predictions(predictions, target_data)
                    log_trade_outcomes_and_percentages(target_data, adjusted_predictions)


                    win_percentage = 60  # Example value
                    mean_win_percentage = 1.0  # Example value
                    mean_loss_percentage = 1.0  # Example value
                    average_days_between_trades = 44.54  # Example value



                    average_days_between_trades = calculate_average_days_between_trades(target_data, predictions)
                    if average_days_between_trades is not None:
                        logging.info(f"Average number of days between trades: {average_days_between_trades:.2f} days")
                        meets_criteria = evaluate_trading_model(win_percentage, mean_win_percentage, mean_loss_percentage, average_days_between_trades, min_win_percentage=55, max_loss_percentage=1.5)
                        if meets_criteria:
                            logging.info(f"The trading model for {filename} meets the criteria.")
                        else:
                            logging.info(f"The trading model for {filename} does not meet the criteria.")
                    else:
                        logging.info("Not enough trades to calculate average interval.")
                    # Error calculation before and after adjustment
                    error_before = calculate_mean_percentage_error(target_data, predictions)
                    error_after = calculate_mean_percentage_error(target_data, adjusted_predictions)

                    logging.info(f"Mean percentage error before adjustment: {error_before:.2f}%")
                    logging.info(f"Mean percentage error after adjustment: {error_after:.2f}%")
                    ##log the difference in the mean percentage error before and after adjustment
                    logging.info(f"Difference in mean percentage error: {error_before - error_after:.2f}%")
                    #if the error is not imporving then add a general error message saying it is not improving the model
                    if error_before - error_after < 0:
                        logging.info(f"Error is not improving the model")

                    # Update DataFrame with adjusted predictions
                    data['Adjusted_LSTM_Predictions'] = adjusted_predictions

                    # Save the modified DataFrame back to CSV
                    data.to_csv(file_path, index=False)
                    logging.info(f"File processed and saved: {file_path}")
                else:
                    logging.warning(f"Required columns not found in {file_path}")
            else:
                if filename.endswith('.log'):
                    continue
                logging.warning(f"Skipped non-CSV file: {file_path}")

        except Exception as e:
            logging.info(f"Error processing file {file_path}: {e}")

        # Calculate and log the percentage of files processed
        percentage = round((File_processed / len(os.listdir(directory_path))) * 100, 2)
        logging.info(f"Percentage of files processed: {percentage}%")

    logging.info("Processing complete.")

if __name__ == "__main__":
    main()
