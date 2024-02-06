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
    Adjust predictions based on specific criteria.
    """

    data = data[~np.isnan(data)]
    if len(data) < window_size:
        logging.warning("Insufficient data for making adjustments.")
        return predictions


    adjusted_predictions = predictions.copy()
    long_term_mean = np.mean(data)
    recent_mean = np.mean(data[-window_size:])
    recent_std = np.std(data[-window_size:])
    trend_adjustment_factor = 0.001

    # Extreme Value Normalization
    lower_quantile = np.percentile(adjusted_predictions, 5)
    upper_quantile = np.percentile(adjusted_predictions, 95)
    adjusted_predictions = np.clip(adjusted_predictions, lower_quantile, upper_quantile)

    # Momentum-Based Adjustment
    if np.sign(recent_mean - long_term_mean) == np.sign(adjusted_predictions[-1]):
        momentum_factor = 0.05  # Adjust this factor as needed
        adjusted_predictions *= (1 + momentum_factor * np.sign(recent_mean - long_term_mean))

    # Mean Reversion Adjustment
    if abs(recent_mean - long_term_mean) > recent_std:  # Significant deviation from the mean
        reversion_factor = 0.03  # Adjust this factor as needed
        adjusted_predictions += reversion_factor * (long_term_mean - recent_mean)

    # Volatility Scaling
    volatility_factor = max(0, 1 - recent_std / np.std(data))  # Scaling down if recent volatility is high
    adjusted_predictions *= volatility_factor

    # Trend Alignment Adjustment
    recent_trend = np.sign(recent_mean - np.mean(data[-2 * window_size:-window_size]))
    adjusted_predictions += trend_adjustment_factor * recent_trend * adjusted_predictions



    trend_slope = calculate_best_fit_slope(data, window_size)
    trend_adjustment = 0.002  # Adjust this factor as needed
    if trend_slope > 0:  # Positive trend
        adjusted_predictions += trend_adjustment * np.abs(adjusted_predictions)
    elif trend_slope < 0:  # Negative trend
        adjusted_predictions -= trend_adjustment * np.abs(adjusted_predictions)

    return adjusted_predictions



##make a new function that will get the amount of error in the ppredictions vs the acutal data






def calculate_mean_percentage_error(target_data, predictions):
    """
    Calculate the mean percentage error between the target data and predictions,
    considering only the non-NaN values in the predictions.
    """
    # Ensure that target_data and predictions have the same length
    if len(target_data) != len(predictions):
        logging.warning("Target data and predictions arrays have different lengths.")
        return np.nan

    # Filter out NaN values from predictions and corresponding values in target_data
    valid_indices = ~np.isnan(predictions)
    filtered_target_data = target_data[valid_indices]
    filtered_predictions = predictions[valid_indices]

    # Check if there's enough data for error calculation
    if len(filtered_target_data) == 0 or len(filtered_predictions) == 0:
        logging.warning("No valid data available for error calculation.")
        return np.nan

    # Calculate mean percentage error
    percentage_errors = np.abs((filtered_target_data - filtered_predictions) / filtered_target_data) * 100
    mean_error = np.mean(percentage_errors)

    return mean_error




class MyTradingStrategy(Strategy):
    """
    Define your trading strategy class here inheriting from Strategy.
    """
    def init(self):
        # Initialization logic
        pass

    def next(self):
        # Logic for each step in the backtesting
        pass



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

                    # Calculate the percentage of NaN values
                    percentage_of_nan_in_target = np.mean(data[target_column].isna()) * 100
                    percentage_of_nan_in_prediction = np.mean(data[prediction_column].isna()) * 100
                    logging.info(f"Percentage of Nan values in the target column: {percentage_of_nan_in_target}%")
                    logging.info(f"Percentage of Nan values in the predicted column: {percentage_of_nan_in_prediction}%")

                    # Apply adjustments
                    adjusted_predictions = adjust_predictions(predictions, target_data, window_size)

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
            logging.error(f"Error processing file {file_path}: {e}")

        # Calculate and log the percentage of files processed
        percentage = round((File_processed / len(os.listdir(directory_path))) * 100, 2)
        logging.info(f"Percentage of files processed: {percentage}%")

    logging.info("Processing complete.")

if __name__ == "__main__":
    main()
