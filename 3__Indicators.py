#!/root/root/miniconda4/envs/tf/bin/python
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import time
import scipy.stats as stats
from scipy.stats import linregress
import logging
import argparse
import warnings
import traceback




"""
This script processes financial market data to calculate various technical indicators and saves the processed data in CSV or Parquet formats. It is designed to handle large datasets efficiently by utilizing multiple cores of the processor.

Features:
- Processes data files (CSV or Parquet) containing market data like Open, High, Low, Close, Volume, etc.
- Calculates a comprehensive set of technical indicators including moving averages, ATR, VWAP, RSI, and apoxx 50 others.
- Implements anomaly detection by squashing outliers and interpolating missing data.
- Optimized for performance with multithreading and conditional execution based on processor core count.
- Robust error handling and logging to track process efficiency and diagnose issues.
- Flexibility in processing multiple files from a specified directory and saving them in a desired format.

Usage:
- The script reads market data files from a specified input directory, processes each file to compute various indicators, and writes the enhanced data to an output directory.
- The file format (CSV or Parquet) is preserved during the process.
- Log information about the process, including processing time and any encountered errors, is saved in a log file.
- It can be customized for different sets of indicators or processing logic based on specific needs.

Example:
- To process all files in the 'Data/PriceData' directory and save the processed files in the 'Data/IndicatorData' directory, simply run the script. The process is automated and requires no additional command-line arguments.

Notes:
- Ensure that the input directory contains valid market data files in supported formats.
- Modify the CONFIG dictionary to specify paths for input/output directories and log file.
- The script utilizes multithreading for faster processing, making it suitable for large datasets.
- Check the log file for detailed information about the processing time and potential errors.
"""

logging.basicConfig(filename='Data/IndicatorData/_IndicatorData.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

CONFIG = {
    'input_directory': 'Data/PriceData',
    'output_directory': 'Data/IndicatorData',
    'log_file': 'Data/IndicatorData/_IndicatorData.log',
    'log_lines_to_read': 500,
    'core_count_division': True, 
}



# Suppress only the relevant warnings from numpy operations:
warnings.filterwarnings('ignore', 'invalid value encountered in subtract', RuntimeWarning)
warnings.filterwarnings('ignore', 'invalid value encountered in reduce', RuntimeWarning)

##===========================(Indicators)===========================##
##===========================(Indicators)===========================##
##===========================(Indicators)===========================##

def squash_col_outliers(df, col_name=None, num_std_dev=3):
    columns_to_process = [col_name] if col_name is not None else df.select_dtypes(include=['float64']).columns
    for col in columns_to_process:
        if col not in df.columns or df[col].dtype != 'float64':
            continue  # Skip non-existent or non-float columns
        mean = df[df[col] != 0][col].mean()
        std_dev = df[df[col] != 0][col].std()
        lower_bound = mean - num_std_dev * std_dev
        upper_bound = mean + num_std_dev * std_dev
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    return df


def interpolate_columns(df, max_gap_fill=50):
    """
    Interpolates columns in the DataFrame, limiting the maximum gap to be filled.

    Args:
    df (pd.DataFrame): DataFrame containing the data.
    max_gap_fill (int): Maximum number of consecutive NaNs to fill.

    Returns:
    pd.DataFrame: DataFrame with interpolated columns.
    """



    for column in df.columns:
        if not np.issubdtype(df[column].dtype, np.number):
            continue
        consec_nan_count = df[column].isna().astype(int).groupby(df[column].notna().astype(int).cumsum()).cumsum()
        mask = consec_nan_count <= max_gap_fill
        df.loc[mask, column] = df.loc[mask, column].interpolate(method='linear')
        df[column] = df[column].ffill()
    return df





def find_best_fit_line(x, y):
    """
    Compute the slope and intercept for the line of best fit.
    """
    try:
        slope, intercept, _, _, _ = linregress(x, y)
        return slope, intercept
    except ValueError:  # Handle any mathematical errors
        return np.nan, np.nan
    


def find_levels(df, window_size):
    """
    Function to find the top 10 highest and lowest levels in a rolling window and calculate the distance
    of the current closing price from the line of best fit for each.
    """
    def calculate_level_distance(rolling_window):
        # Handle incomplete window
        if len(rolling_window) < window_size:
            return np.nan, np.nan

        # Find top 10 highest and lowest levels
        high_levels = np.argsort(rolling_window)[-10:]
        low_levels = np.argsort(rolling_window)[:10]

        # Calculate lines of best fit
        high_slope, high_intercept = find_best_fit_line(high_levels, rolling_window[high_levels])
        low_slope, low_intercept = find_best_fit_line(low_levels, rolling_window[low_levels])

        # Current position (last index in the rolling window)
        current_position = len(rolling_window) - 1

        # Calculate distance from high and low lines
        high_line_value = high_slope * current_position + high_intercept
        low_line_value = low_slope * current_position + low_intercept

        # Return the distance from high and low lines separately
        distance_from_high = abs(rolling_window[-1] - high_line_value)
        distance_from_low = abs(rolling_window[-1] - low_line_value)

        return distance_from_high, distance_from_low

    # Apply the function to the rolling window of the closing price
    distances = df['Close'].rolling(window=window_size).apply(calculate_level_distance, raw=True)

    # Split the tuple results into separate columns
    df['Distance_From_High'] = [dist[0] for dist in distances]
    df['Distance_From_Low'] = [dist[1] for dist in distances]

    return df


def calculate_apz(data, ema_period=20, atr_period=14, atr_multiplier=None, volatility_factor=0.1):
    # Calculate EMA
    data['EMA'] = data['Close'].ewm(span=ema_period, adjust=False).mean()
    close = data['Close']
    # Calculate ATR
    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close'].shift()).abs()
    low_close = (data['Low'] - data['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['ATR'] = tr.rolling(window=atr_period).mean()

    # Dynamic ATR Multiplier
    if atr_multiplier is None:
        atr_multiplier = volatility_factor * data['Close'].rolling(window=atr_period).std()

    # Calculate APZ Bounds
    data['APZ_Upper'] = data['EMA'] + (data['ATR'] * atr_multiplier)
    data['APZ_Lower'] = data['EMA'] - (data['ATR'] * atr_multiplier)

    # Optional: Capping extreme values (implementation depends on specific requirements)
    data['APZ_Upper%'] = ((data['APZ_Upper'] - close) / close) * 100
    data['APZ_Lower%'] = ((close - data['APZ_Lower']) / close) * 100
    ##drop the upper and lower apz columns
    data = data.drop(columns=['APZ_Upper', 'APZ_Lower'])
    return data




def calculate_vama(df, price_col='Close', min_period=10, max_period=30):
    # Replace infinite values with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Calculate normalized volatility
    volatility = df[price_col].pct_change().rolling(window=21).std()
    normalized_vol = (volatility - volatility.min()) / (volatility.max() - volatility.min())

    # Determine window size (ensure it's within a reasonable range and is an integer)
    window_size = (normalized_vol * (max_period - min_period) + min_period).clip(min_period, max_period).round().astype(int)

    # Calculate VAMA with error handling
    try:
        vama = df[price_col].rolling(window=window_size, min_periods=min_period).mean()
    except Exception as e:
        logging.error(f"Error in VAMA calculation: {e}")
        vama = pd.Series(index=df.index)

    return vama





def add_vama_changes(df, vama, periods):
    for period in periods:
        label = f'VAMA_pct_change_{period}_days'
        df[label] = vama.pct_change(periods=period)

    return df

def calculate_cumulative_streaks(df, window_size=20):
    # Calculate daily percentage change
    pct_change = df['Close'].pct_change() * 100
    close = df['Close']
    # Identify where the sign changes (positive to negative and vice versa)
    sign_change = pct_change.mul(pct_change.shift(1)) <= 0

    # Create streak counter that resets at every sign change
    streak_counter = sign_change.cumsum()

    # Group by streaks and sum the percentage changes within each streak
    streak_sums = pct_change.groupby(streak_counter).cumsum()

    # Assign positive and negative streaks
    df['positive_streak_cumulative'] = np.where(pct_change > 0, streak_sums, 0)
    df['negative_streak_cumulative'] = np.where(pct_change < 0, streak_sums, 0)

    # Apply rolling window
    df['positive_streak_cumulative'] = df['positive_streak_cumulative'].rolling(window=window_size, min_periods=1).max()
    df['negative_streak_cumulative'] = df['negative_streak_cumulative'].rolling(window=window_size, min_periods=1).min()
 

    return df




def AtrVolume(df):
    df['ATR_std'] = df['ATR'].rolling(window=60).std()
    df['Volume_std'] = df['Volume'].rolling(window=60).std()
    ATR_threshold_multiplier = 2  # Example value, can be adjusted
    Volume_threshold_multiplier = 2  # Example value, can be adjusted
    df['ATR_trigger'] = df['ATR'] > (df['ATR_std'] * ATR_threshold_multiplier)
    df['Volume_trigger'] = df['Volume'] > (df['Volume_std'] * Volume_threshold_multiplier)
    df['Oscillator'] = (df['ATR_trigger'] & df['Volume_trigger']).astype(int)
    reset_points = df['Oscillator'].diff().eq(-1).cumsum()
    # Calculate the trigger counter
    df['Trigger_Counter'] = df['Oscillator'].groupby(reset_points).cumsum()
    return df


def ATR_Based_Adaptive_Trend_Channels(df):
    # 200-day Moving Average
    df['200MA'] = df['Close'].rolling(window=200).mean()

    # 14-day average ATR
    df['14Day_Avg_ATR'] = df['ATR'].rolling(window=14).mean()

    # Upper and Lower Bands
    df['Upper_Band'] = df['200MA'] + df['14Day_Avg_ATR']
    df['Lower_Band'] = df['200MA'] - df['14Day_Avg_ATR']

    # Percentage Deviation from Bands
    df['Pct_Deviation_Upper'] = np.where(df['Close'] > df['Upper_Band'],
                                         (df['Close'] - df['Upper_Band']) / df['Upper_Band'] * 100,
                                         -((df['Upper_Band'] - df['Close']) / df['Close']) * 100)

    df['Pct_Deviation_Lower'] = np.where(df['Close'] < df['Lower_Band'],
                                         -((df['Close'] - df['Lower_Band']) / df['Lower_Band']) * 100,
                                         ((df['Lower_Band'] - df['Close']) / df['Close']) * 100)
    ##drop the intermediate columns used to calculate the bands
    df = df.drop(columns=['200MA', '14Day_Avg_ATR', 'Upper_Band', 'Lower_Band'])
    return df




def imminent_channel_breakout(df, ma_period=200, atr_period=14):
    df['200MA'] = df['Close'].rolling(window=ma_period).mean()
    df['14DayATR'] = df['ATR'].rolling(window=atr_period).mean()
    df['Upper_Band'] = df['200MA'] + df['14DayATR']
    df['Lower_Band'] = df['200MA'] - df['14DayATR']
    df['Upper_Band_Slope'] = df['Upper_Band'].diff() / atr_period
    df['Lower_Band_Slope'] = df['Lower_Band'].diff() / atr_period
    df['Close_Momentum'] = df['Close'].diff()
    def calculate_breakout_score(row):
        upper_proximity = 100 * (1 - min(row['Close'] / row['Upper_Band'], 1))
        lower_proximity = 100 * (1 - min(row['Lower_Band'] / row['Close'], 1))
        slope_strength = 50 * (abs(row['Upper_Band_Slope']) + abs(row['Lower_Band_Slope']))
        if row['14DayATR'] == 0:
            momentum_strength = 0
        else:
            momentum_strength = 50 * abs(row['Close_Momentum']) / row['14DayATR']
        return min(100, upper_proximity * 0.4 + lower_proximity * 0.4 + slope_strength * 0.1 + momentum_strength * 0.1)
    df['Breakout_Score'] = df.apply(calculate_breakout_score, axis=1)
    df.drop(['Upper_Band_Slope', 'Lower_Band_Slope', 'Close_Momentum', '14DayATR'], axis=1, inplace=True)
    return df





def indicators(df):
    df['Close'] = df['Close'].ffill()
    df['High'] = df['High'].ffill()
    df['Low'] = df['Low'].ffill()
    df['Volume'] = df['Volume'].ffill()


    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    calculate_apz(df)




    calculate_cumulative_streaks(df)
    df = ATR_Based_Adaptive_Trend_Channels(df)
    df = imminent_channel_breakout(df)

    close_shift_1 = close.shift(1)
    true_range = np.maximum(high - low, np.maximum(np.abs(high - close_shift_1), np.abs(close_shift_1 - low)))
    window = 14
    rolled = np.lib.stride_tricks.sliding_window_view(true_range, window)
    mean_rolled = np.mean(rolled, axis=1)
    df['ATR'] = np.pad(mean_rolled, (window-1, 0), 'constant', constant_values=(np.nan,))
    df['ATR%'] = (df['ATR'] / close) * 100
    df['ATR%_change'] = df['ATR%'].pct_change()
    df = AtrVolume(df)
    df['200ma'] = close.rolling(window=200).mean()
    df['14ma'] = close.rolling(window=14).mean()
    df['14ma%'] = ((close - df['14ma']) / df['14ma']) * 100
    df['200ma%'] = ((close - df['200ma']) / df['200ma']) * 100
    df['14ma-200ma'] = df['14ma'] - df['200ma']
    df['14ma%_change'] = df['14ma%'].pct_change()
    df['14ma%_count'] = df['14ma%'].gt(0).rolling(window=14).sum()


    df['200ma%_count'] = df['200ma%'].gt(0).rolling(window=200).sum()



    df['14ma_crossover'] = (close > df['14ma'])
    df['200ma_crossover'] = (close > df['200ma'])
    df['200DAY_ATR'] = df['200ma'] + df['ATR']



    df['200DAY_ATR-'] = df['200ma'] - df['ATR']
    df['200DAY_ATR%'] = df['200DAY_ATR'] / close
    df['200DAY_ATR-%'] = df['200DAY_ATR-'] / close
    df['percent_from_high'] = ((close - close.cummax()) / close.cummax()) * 100
    df['new_high'] = (close == close.cummax())
    df['days_since_high'] = (~df['new_high']).cumsum() - (~df['new_high']).cumsum().where(df['new_high']).ffill().fillna(0)
    df['percent_range'] = (high - low) / close * 100
    typical_price = (high + low + close) / 3
    df['VWAP'] = (typical_price * volume).rolling(window=14).sum() / volume.rolling(window=14).sum()
    df['VWAP_std14'] = df['VWAP'].rolling(window=14).std()
    df['VWAP_std200'] = df['VWAP'].rolling(window=20).std()
    df['VWAP%'] = ((close - df['VWAP']) / df['VWAP']) * 100
    df['VWAP%_from_high'] = ((df['VWAP'] - close.cummax()) / close.cummax()) * 100
    obv_condition = df['Close'] > close_shift_1
    df['OBV'] = np.where(obv_condition, volume, -volume).cumsum()

    # Start of Weighted Close Price Change Velocity
    close = df['Close']
    window = 10
    price_change = close.diff()
    price_change = close.diff().fillna(0)
    weights = np.linspace(1, 0, window)
    weights /= np.sum(weights)
    weighted_velocity = price_change.rolling(window=window).apply(lambda x: np.dot(x, weights), raw=True)
    df['Weighted_Close_Change_Velocity'] = weighted_velocity
    # End of Weighted Close Price Change Velocity

    close_shift_1 = close.shift(1)
    low_shift_1 = low.shift(1)
    high_shift_1 = high.shift(1)
    pct_change_close = close.pct_change()

    # Consolidating rolling operations
    rolling_20 = df['Close'].rolling(window=20)
    rolling_14 = df['Close'].rolling(window=14)
    rolling_7 = df['Close'].rolling(window=7)
    rolling_30 = df['Close'].rolling(window=30)
    
    df['percent_change_Close'] = pct_change_close
    df['pct_change_std'] = rolling_20.std()

    # Replacing loop for shifts with direct assignments
    df['percent_change_Close_lag_1'] = pct_change_close.shift(1)  # Lag by 1 period
    df['percent_change_Close_lag_3'] = pct_change_close.shift(3)  # Lag by 3 periods
    df['percent_change_Close_lag_5'] = pct_change_close.shift(5)  # Lag by 5 periods
    df['percent_change_Close_lag_10'] = pct_change_close.shift(10)  # Lag by 10 periods


    df['percent_change_Close_7'] = rolling_7.mean()
    df['percent_change_Close_30'] = rolling_30.mean()
    df['percent_change_Close>std'] = (pct_change_close > df['pct_change_std']).astype(int)
    df['percent_change_Close<std'] = (pct_change_close < df['pct_change_std']).astype(int)

    # Merged rolling std calculations
    df['pct_change_std_rolling'] = rolling_14.mean()
    df['pct_change_std_rolling'] = rolling_20.mean()

    threshold_multiplier = 0.65
    abnormal_pct_change_threshold = rolling_20.mean() + threshold_multiplier * df['pct_change_std']
    df['days_since_abnormal_pct_change'] = (pct_change_close > abnormal_pct_change_threshold).cumsum()



    # VWAP Divergence
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    vwap = (typical_price * df['Volume']).cumsum() / df['Volume'].cumsum()
    df['VWAP_Divergence'] = df['Close'] - vwap

    # RSI Calculation
    delta = df['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14).mean()
    avg_loss = pd.Series(loss).rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI_overbought'] = (df['RSI'] > 70).astype(int)
    df['RSI_oversold'] = (df['RSI'] < 30).astype(int)
    df['RSI_very_overbought'] = (df['RSI'] > 90).astype(int)
    df['RSI_very_oversold'] = (df['RSI'] < 10).astype(int)



    # Elder’s Force Index
    df['EFI'] = df['Close'].diff() * df['Volume']



    ##direction flipper
    df['direction_flipper'] = (pct_change_close > 0).astype(int)
    df['direction_flipper_count5'] = df['direction_flipper'].rolling(window=5).sum()
    df['direction_flipper_count_10'] = df['direction_flipper'].rolling(window=10).sum()
    df['direction_flipper_count_14'] = df['direction_flipper'].rolling(window=14).sum()
    ##drop the direction flipper too correlated
    df = df.drop(columns=['direction_flipper'])



    # ATR calculation streamlined
    df['ATR'] = rolling_14.apply(lambda x: np.mean(np.abs(np.diff(x))))
    keltner_central = df['Close'].ewm(span=20).mean()
    keltner_range = df['ATR'] * 1.5
    df['KC_UPPER%'] = ((keltner_central + keltner_range) - df['Close']) / df['Close'] * 100
    df['KC_LOWER%'] = (df['Close'] - (keltner_central - keltner_range)) / df['Close'] * 100

    # Streak calculations streamlined
    positive_streak = (pct_change_close > 0).astype(int).cumsum()
    negative_streak = (pct_change_close < 0).astype(int).cumsum()
    df['positive_streak'] = positive_streak - positive_streak.where(pct_change_close <= 0).ffill().fillna(0)
    df['negative_streak'] = negative_streak - negative_streak.where(pct_change_close >= 0).ffill().fillna(0)

    ##get the maximum postitoive value in a rolling window of 60
    df['positive_streak_max'] = df['positive_streak'].rolling(window=60).max()
    df['negative_streak_max'] = df['negative_streak'].rolling(window=60).max()



    # Gap calculations streamlined
    gap_threshold_percent = 0.5 if pct_change_close.std() > 1 else 0
    df['is_gap_up'] = (df['Low'] - high_shift_1) / close_shift_1 * 100 > gap_threshold_percent
    df['is_gap_down'] = (df['High'] - low_shift_1) / close_shift_1 * 100 < -gap_threshold_percent
    df['move_from_gap%'] = np.where(df['is_gap_up'], (df['Close'] - low_shift_1) / low_shift_1 * 100,
                                    np.where(df['is_gap_down'], (df['Close'] - high_shift_1) / high_shift_1 * 100, 0))
    df['VPT'] = df['Volume'] * ((df['Close'] - close_shift_1) / close_shift_1)

    # Dropping unnecessary columns
    columns_to_drop = ['Adj Close','ATZ_Upper','ATZ_Lower','VWAP', '200DAY_ATR-', '200DAY_ATR', 'ATR', 'OBV', '200ma', '14ma']
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df = interpolate_columns(df, max_gap_fill=50)  # updated argument name
    df = df.iloc[200:]
    df = df.drop(columns_to_drop, axis=1)
    df = df.round(8)
    return df



def clean_and_interpolate_data(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        df[col] = df[col].interpolate(method='linear', limit_direction='forward', axis=0)
    df.ffill(inplace=True)
    return df




def DataQualityCheck(df):
    """
    This function checks for signs of reverse stock splits and ensures that the data is recent.
    It compares the average prices at the beginning and end of the data series for reverse splits.
    It also checks that the most recent data point is no older than 365 days to ensure relevancy.

    Args:
    df (pd.DataFrame): DataFrame containing stock price data with a 'Close' and 'Date' column.

    Returns:
    pd.DataFrame or None: Processed DataFrame or None if data is unreliable or outdated.
    """
    if df.empty or len(df) < 201:
        logging.error("DataFrame is empty or too short to process.")
        return None

    if df['Date'].dtype != 'datetime64[ns]':
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
    if df['Date'].iloc[-1] < pd.Timestamp.now() - pd.DateOffset(days=365):
        logging.error(f"Data is outdated. Last date in dataset: {df['Date'].iloc[-1]}")
        return None

    sample_size = max(int(len(df) * 0.02), 1)  # Ensure at least one sample is taken
    start_mean = df['Close'].head(sample_size).mean()
    end_mean = df['Close'].tail(sample_size).mean()

    if start_mean > 3000 or (start_mean / max(end_mean, 1e-10) > 20):
        logging.error(f"Signs of reverse stock splits detected or high initial prices: Start mean: {start_mean}, End mean: {end_mean}")
        return None

    return df








##===========================(File Processing)===========================##
##===========================(File Processing)===========================##
##===========================(File Processing)===========================##
def process_file(file_path, output_dir):
    if file_path.endswith('.csv'):
        Timer = time.time()
        df = pd.read_csv(file_path)
    elif file_path.endswith('.parquet'):
        Timer = time.time()
        df = pd.read_parquet(file_path)
    else:
        return f"Skipped {file_path}: Unsupported file format"

    try:
        df = DataQualityCheck(df)
        if df is not None:
            df = indicators(df)
            df = squash_col_outliers(df, num_std_dev=3)
            df = clean_and_interpolate_data(df)

            filename = os.path.basename(file_path)
            new_file_path = os.path.join(output_dir, filename.replace('.csv', '_modified.csv').replace('.parquet', '_modified.parquet'))

            if file_path.endswith('.csv'):
                df.to_csv(new_file_path, index=False)
            else:
                df.to_parquet(new_file_path, index=False)
        
    except Exception as e:
        # Log the error and the traceback
        error_message = f"Error processing {file_path}: {e}"
        logging.error(error_message)
        logging.error("Traceback details:")
        traceback_info = traceback.format_exc()
        logging.error(traceback_info)
        return error_message






def process_files(input_dir, output_dir):
    """
    Processes all files in the specified input directory and saves the processed files in the output directory.
    """
    file_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.csv') or f.endswith('.parquet')]
    with ThreadPoolExecutor() as executor:
        list(executor.map(lambda file: process_file(file, output_dir), file_paths))

def clear_output_directory(output_dir):
    """
    Remove existing files in the output directory.
    """
    for file in os.listdir(output_dir):
        if file.endswith('.csv') or file.endswith('.parquet'):
            os.remove(os.path.join(output_dir, file))

def calculate_average_processing_time(log_file, core_count_division):
    """
    Calculate and log the average time taken to process each file.
    """
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()[-CONFIG['log_lines_to_read']:]
        time_taken = [float(line.split(':')[-1].strip().split(' ')[0]) for line in lines if 'Time taken to process' in line]
        average_time = sum(time_taken) / len(time_taken)
        if core_count_division:
            average_time /= os.cpu_count()
        return average_time
    except Exception as e:
        logging.error(f"Error reading log file: {e}")
        raise





##===========================(Main)===========================##
##===========================(Main)===========================##
##===========================(Main)===========================##
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process financial market data files.")
    parser.add_argument('--runpercent', type=int, default=100, help="Percentage of files to process from the input directory.")
    args = parser.parse_args()

    run_percent = args.runpercent
    os.makedirs(CONFIG['output_directory'], exist_ok=True)
    clear_output_directory(CONFIG['output_directory'])

    # Get all file paths
    file_paths = [os.path.join(CONFIG['input_directory'], f) for f in os.listdir(CONFIG['input_directory']) if f.endswith('.csv') or f.endswith('.parquet')]

    # Determine the number of files to process based on the percentage
    files_to_process = file_paths[:int(len(file_paths) * (run_percent / 100))]

    # Process the files using ThreadPoolExecutor
    with ThreadPoolExecutor() as executor:
        list(executor.map(lambda file: process_file(file, CONFIG['output_directory']), files_to_process))

    try:
        average_time_taken = calculate_average_processing_time(CONFIG['log_file'], CONFIG['core_count_division'])
        logging.info(f"Average time taken to process each file: {average_time_taken:.2f} seconds")
    except Exception as e:
        print(f"Error calculating average processing time: {e}")