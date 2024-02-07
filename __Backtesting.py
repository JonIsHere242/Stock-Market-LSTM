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

def adjust_predictions(predictions, data, window_size=5):
    """
    Further refined adjustment of predictions considering smoothing, dynamic thresholds, recent error trends, 
    and conditional adjustments based on model confidence (if available).
    """
    valid_data = data[~np.isnan(data)]

    if len(valid_data) < window_size:
        logging.warning("Insufficient data for making adjustments.")
        return predictions

    # Basic stats
    long_term_mean = np.mean(valid_data)
    recent_mean = np.mean(valid_data[-window_size:])
    recent_std = np.std(valid_data[-window_size:])
    max_abs_value = np.max(np.abs(valid_data))

    adjusted_predictions = predictions.copy()

    # Dynamic Threshold for Mean Reversion
    dynamic_threshold = recent_std * 1.5  # Adjust multiplier as needed
    if abs(recent_mean - long_term_mean) > dynamic_threshold:
        reversion_factor = 0.03  # Adjust as needed
        adjusted_predictions += reversion_factor * (long_term_mean - recent_mean)

    # Limiting Extreme Predictions Dynamically
    dynamic_max_abs_value = max_abs_value * 1.1  # Adjust multiplier as needed
    adjusted_predictions = np.clip(adjusted_predictions, -dynamic_max_abs_value, dynamic_max_abs_value)

    # Volatility Scaling
    volatility_factor = max(0, 1 - recent_std / np.std(valid_data))
    adjusted_predictions *= volatility_factor

    # Conditional Adjustments (if model confidence data is available)
    # This would depend on additional data from your model (like confidence intervals)

    ##log the number of times the prediction is within 10% of the actual value in absolute terms
    logging.info(f"Prediction is within 10%: {np.sum(np.abs(predictions) <= 0.1 * np.abs(valid_data))}")



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




def main():
    directory_path = 'Data/LSTMpredictedData'
    window_size = 10
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


                    ##log the shape of the ajusted predictions 
                    logging.info(f"Shape of the adjusted predictions: {adjusted_predictions.shape}")

                    # Error calculation before and after adjustment
                    error_before = calculate_mean_percentage_error(target_data, predictions)
                    error_after = calculate_mean_percentage_error(target_data, adjusted_predictions)
                    logging.info(f"Error before adjustment: {error_before}")
                    logging.info(f"Error after adjustment: {error_after}")

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
