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
from pykalman import KalmanFilter
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde, skew, kurtosis
from scipy.signal import argrelextrema
import cProfile
import pstats
from concurrent.futures import ProcessPoolExecutor


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

warnings.filterwarnings('ignore', 'invalid value encountered in subtract', RuntimeWarning)
warnings.filterwarnings('ignore', 'invalid value encountered in reduce', RuntimeWarning)

##===========================(Indicators)===========================##
##===========================(Indicators)===========================##
##===========================(Indicators)===========================##

def squash_col_outliers(df, col_name=None, num_std_dev=3):
    if col_name:
        columns_to_process = [col_name]
    else:
        columns_to_process = df.select_dtypes(include=['float64']).columns

    # Loop through each column
    for col in columns_to_process:
        if col not in df.columns or df[col].dtype != 'float64':
            continue
        rolled_means = df[col][df[col] != 0].rolling(window=282, min_periods=1).mean()
        rolled_stds = df[col][df[col] != 0].rolling(window=282, min_periods=1).std()
        lower_bounds = rolled_means - num_std_dev * rolled_stds
        upper_bounds = rolled_means + num_std_dev * rolled_stds
        df[col] = df[col].clip(lower=lower_bounds, upper=upper_bounds)
    return df


def interpolate_columns(df, max_gap_fill=50):
    for column in df.columns:
        if not np.issubdtype(df[column].dtype, np.number):
            continue
        consec_nan_count = df[column].isna().astype(int).groupby(df[column].notna().astype(int).cumsum()).cumsum()
        mask = consec_nan_count <= max_gap_fill
        df.loc[mask, column] = df.loc[mask, column].interpolate(method='linear')
        df[column] = df[column].ffill()
    return df


def find_best_fit_line(x, y):
    try:
        slope, intercept, _, _, _ = linregress(x, y)
        return slope, intercept
    except ValueError:  # Handle any mathematical errors
        return np.nan, np.nan
    

def find_levels(df, window_size):

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

    distances = df['Close'].rolling(window=window_size).apply(calculate_level_distance, raw=True)
    df['Distance_From_High'] = [dist[0] for dist in distances]
    df['Distance_From_Low'] = [dist[1] for dist in distances]
    return df




def detect_peaks_and_valleys(df, column_name, prominence=1):
    # Initialize Peaks and Valleys columns dynamically based on the input column name
    peaks_column = column_name + '_Peaks'
    valleys_column = column_name + '_Valleys'
    
    df[peaks_column] = np.nan
    df[valleys_column] = np.nan

    # Detect peaks with prominence
    peaks, properties = find_peaks(df[column_name], prominence=prominence)
    df.loc[peaks, peaks_column] = df[column_name][peaks]

    # Detect valleys by inverting the data and using the same prominence
    valleys, properties = find_peaks(-df[column_name], prominence=prominence)
    df.loc[valleys, valleys_column] = df[column_name][valleys]

    return df








def hurst_exponent(time_series):
    """ Returns the Hurst Exponent of the time series """
    lags = range(2, 100)
    tau = [np.std(np.subtract(time_series[lag:], time_series[:-lag])) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0] * 2.0



def rolling_hurst_exponent(series, window_size):
    """ Apply Hurst exponent calculation over a rolling window with raw=True for performance """
    # First, drop any NaNs from the series to avoid issues in the window
    series_clean = series.dropna()

    # Define a function to apply to each window, now assuming no NaNs
    def hurst_window(window):
        return hurst_exponent(window)

    # Apply rolling function with raw=True
    return series_clean.rolling(window=window_size).apply(hurst_window, raw=True)








def calculate_parabolic_SAR(df):
    high = df['High']
    low = df['Low']
    close = df['Close']

    # Initialize the SAR with the first row's low value
    sar = low[0]
    # Initial values for High and Low points
    ep = high[0]
    af = 0.02
    sar_values = [sar]

    for i in range(1, len(df)):
        sar = sar + af * (ep - sar)
        if close[i] > close[i - 1]:
            af = min(af + 0.02, 0.2)
        else:
            af = 0.02

        if close[i] > close[i - 1]:
            ep = max(high[i], ep)
        else:
            ep = min(low[i], ep)

        sar = min(sar, low[i], low[i - 1]) if close[i] > close[i - 1] else max(sar, high[i], high[i - 1])
        sar_values.append(sar)

    df['Parabolic_SAR'] = sar_values
    return df



def calculate_apz(data, ema_period=20, atr_period=14, atr_multiplier=None, volatility_factor=0.1):
    data['EMA'] = data['Close'].ewm(span=ema_period, adjust=False).mean()
    close = data['Close']
    high_low = data['High'] - data['Low']
    high_close = (data['High'] - data['Close'].shift()).abs()
    low_close = (data['Low'] - data['Close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data['ATR'] = tr.rolling(window=atr_period).mean()

    if atr_multiplier is None:
        atr_multiplier = volatility_factor * data['Close'].rolling(window=atr_period).std()

    data['APZ_Upper'] = data['EMA'] + (data['ATR'] * atr_multiplier)
    data['APZ_Lower'] = data['EMA'] - (data['ATR'] * atr_multiplier)

    data['APZ_Upper%'] = ((data['APZ_Upper'] - close) / close) * 100
    data['APZ_Lower%'] = ((close - data['APZ_Lower']) / close) * 100
    ##drop the upper and lower apz columns
    data = data.drop(columns=['APZ_Upper', 'APZ_Lower'])
    return data


def calculate_vama(df, price_col='Close', min_period=10, max_period=30):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    volatility = df[price_col].pct_change().rolling(window=21).std()
    normalized_vol = (volatility - volatility.min()) / (volatility.max() - volatility.min())
    window_size = (normalized_vol * (max_period - min_period) + min_period).clip(min_period, max_period).round().astype(int)
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
    pct_change = df['Close'].pct_change() * 100
    close = df['Close']
    sign_change = pct_change.mul(pct_change.shift(1)) <= 0
    streak_counter = sign_change.cumsum()
    streak_sums = pct_change.groupby(streak_counter).cumsum()
    df['positive_streak_cumulative'] = np.where(pct_change > 0, streak_sums, 0)
    df['negative_streak_cumulative'] = np.where(pct_change < 0, streak_sums, 0)
    df['positive_streak_cumulative'] = df['positive_streak_cumulative'].rolling(window=window_size, min_periods=1).max()
    df['negative_streak_cumulative'] = df['negative_streak_cumulative'].rolling(window=window_size, min_periods=1).min()
    return df



def compute_VPT(df):
    # Pre-compute shifted columns
    close_shift_1 = df['Close'].shift(1)
    low_shift_1 = df['Low'].shift(1)
    high_shift_1 = df['High'].shift(1)

    # Calculate conditions
    is_gap_up = (df['Low'] - high_shift_1) / high_shift_1 * 100 > 0.5  # Assuming gap_threshold_percent is 0.5
    is_gap_down = (df['High'] - low_shift_1) / close_shift_1 * 100 < -0.5

    # Calculate new columns
    move_from_gap_percent = np.where(is_gap_up, (df['Close'] - low_shift_1) / low_shift_1 * 100, 0)
    VPT = df['Volume'] * ((df['Close'] - close_shift_1) / close_shift_1)

    # Store calculations in a temporary DataFrame
    temp_df = pd.DataFrame({
        'is_gap_up': is_gap_up,
        'is_gap_down': is_gap_down,
        'move_from_gap%': move_from_gap_percent,
        'VPT': VPT
    }, index=df.index)

    # Concatenate this DataFrame with the original one
    df = pd.concat([df, temp_df], axis=1)
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
    df['14Day_Avg_ATR'] = df['ATR'].rolling(window=14).mean()
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

def calculate_rsi_divergence(df):
    # Calculate necessary differences and shifts
    price_diff = df['Close'].diff()
    RSI_diff = df['RSI'].diff()
    price_shift = df['Close'].shift(-1)
    RSI_shift = df['RSI'].shift(-1)

    # Identify the local minima and maxima for price and RSI using vectorized operations
    price_is_low = (df['Close'] < np.minimum(price_shift, price_diff))
    RSI_is_low = (df['RSI'] < np.minimum(RSI_shift, RSI_diff))
    price_is_high = (df['Close'] > np.maximum(price_shift, price_diff))
    RSI_is_high = (df['RSI'] > np.maximum(RSI_shift, RSI_diff))

    # Calculate Bullish and Bearish Divergences
    bullish_divergence = np.where(price_is_low & ~RSI_is_low, 1, 0)
    bearish_divergence = np.where(price_is_high & ~RSI_is_high, 1, 0)

    # Combine divergences into a single column with categorical labeling
    RSI_divergence = np.where(bullish_divergence, 'Bullish', np.where(bearish_divergence, 'Bearish', 'None'))

    # Combine all new data into a DataFrame
    temp_df = pd.DataFrame({
        'RSI_divergence': RSI_divergence
    }, index=df.index)

    # Concatenate this DataFrame with the original one
    df = pd.concat([df, temp_df], axis=1)
    return df



def calculate_pivot_points(df):
    # Calculate the pivot point and related resistance and support levels
    Pivot_Point = (df['High'] + df['Low'] + df['Close']) / 3
    R1 = 2 * Pivot_Point - df['Low']  # Resistance 1
    S1 = 2 * Pivot_Point - df['High']  # Support 1
    R2 = Pivot_Point + (df['High'] - df['Low'])  # Resistance 2
    S2 = Pivot_Point - (df['High'] - df['Low'])  # Support 2
    R3 = df['High'] + 2 * (Pivot_Point - df['Low'])  # Resistance 3
    S3 = df['Low'] - 2 * (df['High'] - Pivot_Point)  # Support 3

    # Create a temporary DataFrame with new values
    pivot_data = pd.DataFrame({
        'Pivot_Point': Pivot_Point,
        'R1': R1,
        'S1': S1,
        'R2': R2,
        'S2': S2,
        'R3': R3,
        'S3': S3
    }, index=df.index)

    # Concatenate this DataFrame with the original DataFrame
    df = pd.concat([df, pivot_data], axis=1)
    return df





def calculate_fibonacci_retracement(df):
    Levels = [0.236, 0.382, 0.5, 0.618, 0.786]
    for level in Levels:
        df[f'Fib_Retracement_{level}'] = df['Close'].rolling(window=60).mean() * level
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
    Timer = time.time()
    df['Close'] = df['Close'].ffill()
    df['High'] = df['High'].ffill()
    df['Low'] = df['Low'].ffill()
    df['Volume'] = df['Volume'].ffill()

    df['Close'] = df['Close'].astype(np.float32)
    df['High'] = df['High'].astype(np.float32)
    df['Low'] = df['Low'].astype(np.float32)
    df['Open'] = df['Open'].astype(np.float32)
    df['Volume'] = df['Volume'].astype(np.float32)

    high = df['High']
    low = df['Low']
    close = df['Close']
    volume = df['Volume']

    # Calculate RSI first to ensure it exists for divergence calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0))
    loss = (-delta.where(delta < 0, 0))
    avg_gain = gain.rolling(window=14, min_periods=1).mean() # min_periods=1 makes sure we have at least one value
    avg_loss = loss.rolling(window=14, min_periods=1).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Now we can safely calculate RSI Divergence
    if 'RSI' in df.columns:
        df = calculate_rsi_divergence(df)
    else:
        logging.warning("RSI column not found, skipping RSI divergence calculation")

    calculate_apz(df)
    df = calculate_parabolic_SAR(df)
    df = calculate_cumulative_streaks(df)
    df = ATR_Based_Adaptive_Trend_Channels(df)
    df = imminent_channel_breakout(df)
    df = calculate_pivot_points(df)
    df = calculate_fibonacci_retracement(df)



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

    ##get the std of the 14ma from the 200ma



    df['SMA_200'] = close.rolling(window=200).mean()
    df['SMA_14'] = close.rolling(window=14).mean()
    df['std_14'] = close.rolling(window=14).std()
    df['Std_Devs_from_SMA'] = (df['SMA_200'] - df['SMA_14']) / df['std_14']





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
    window = 10
    price_change = close.diff()
    price_change = close.diff().fillna(0)
    weights = np.linspace(1, 0, window)
    weights /= np.sum(weights)
    weighted_velocity = price_change.rolling(window=window).apply(lambda x: np.dot(x, weights), raw=True)
    df['Weighted_Close_Change_Velocity'] = weighted_velocity
    # End of Weighted Close Price Change Velocity
    pct_change_close = close.pct_change()

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

    df['EFI'] = df['Close'].diff() * df['Volume']

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

    kf = KalmanFilter(transition_matrices=[1],
                      observation_matrices=[1],
                      initial_state_mean=df['Close'].values[0],
                      initial_state_covariance=1,
                      observation_covariance=1,
                      transition_covariance=.01)
    df['Kalman'] = kf.filter(df['Close'].values)[0]

    window = 140
    df['minima'] = df.iloc[argrelextrema(df['Kalman'].values, np.less_equal, order=window)[0]]['Kalman']
    df['maxima'] = df.iloc[argrelextrema(df['Kalman'].values, np.greater_equal, order=window)[0]]['Kalman']
    df['minima'] = df['minima'].ffill()
    df['maxima'] = df['maxima'].ffill()

    conversion_window = 5
    # Ensure index handling within bounds and prevent KeyError
    for index in range(conversion_window, len(df)):
        # Using df.iloc for position-based indexing which is generally safer within loops
        if (df['Kalman'].iloc[index-conversion_window:index+1] > df['maxima'].iloc[index]).all():
            df.at[index, 'minima'] = df.at[index, 'maxima']
            df.at[index, 'maxima'] = np.nan  # Clear the old maxima
        if (df['Kalman'].iloc[index-conversion_window:index+1] < df['minima'].iloc[index]).all():
            df.at[index, 'maxima'] = df.at[index, 'minima']
            df.at[index, 'minima'] = np.nan  # Clear the old minima
    df['Distance to Support (%)'] = (df['Close'] - df['minima']) / df['minima'] * 100
    df['Distance to Resistance (%)'] = (df['maxima'] - df['Close']) / df['Close'] * 100
    df_temp = pd.DataFrame(index=df.index)
    df_temp['kalman_diff'] = (df['Kalman'] - df['Close'].rolling(window=200).mean()) / df['Close'].rolling(window=200).mean() * 100
    df = pd.concat([df, df_temp], axis=1)

    df['bullish_fractal'] = 0
    df['bearish_fractal'] = 0
    bullish_idx = (df['Low'].shift(2) > df['Low'].shift(1)) & \
                  (df['Low'].shift(1) > df['Low']) & \
                  (df['Low'].shift(-1) > df['Low']) & \
                  (df['Low'].shift(-2) > df['Low'].shift(-1))
    bearish_idx = (df['High'].shift(2) < df['High'].shift(1)) & \
                  (df['High'].shift(1) < df['High']) & \
                  (df['High'].shift(-1) < df['High']) & \
                  (df['High'].shift(-2) < df['High'].shift(-1))
    df.loc[bullish_idx, 'bullish_fractal'] = 1
    df.loc[bearish_idx, 'bearish_fractal'] = 1


    df['Distance to Support (%)'] = (df['Close'] - df['minima'].shift(1)) / df['minima'].shift(1) * 100
    df['Distance to Resistance (%)'] = (df['maxima'].shift(1) - df['Close']) / df['Close'] * 100
    df['MA_200'] = df['Close'].rolling(window=200).mean()
    df['Perc_Diff'] = (df['Kalman'] - df['MA_200']) / df['MA_200'] * 100
    std_perc_diff = df['Perc_Diff'].std()
    df['skew'] = df['Perc_Diff'].rolling(window=200).apply(lambda x: skew(x))
    df['kurtosis'] = df['Perc_Diff'].rolling(window=200).apply(lambda x: kurtosis(x))
    df['mean'] = df['Perc_Diff'].rolling(window=200).apply(lambda x: np.mean(x))
    df['std'] = df['Perc_Diff'].rolling(window=200).apply(lambda x: np.std(x))

    epsilon = 0.001  # Small perturbation factor
    df['Perturbed_Kalman'] = df['Kalman'] * (1 + epsilon)
    df['Divergence'] = np.abs(df['Perturbed_Kalman'] - df['Kalman'])
    df['Log_Divergence'] = np.log(df['Divergence'] + np.finfo(float).eps)  # Adding eps to handle log(0)
    df['Lyapunov_Exponent'] = df['Log_Divergence'].diff() / np.log(1 + epsilon)
    window_size = 14  # Adjust window size as needed
    df['Lyapunov_Exponent_MA'] = df['Lyapunov_Exponent'].rolling(window=window_size).mean()
    df = detect_peaks_and_valleys(df, 'Lyapunov_Exponent_MA')

    columns_to_drop = ['Adj Close','ATZ_Upper','ATZ_Lower','VWAP', '200DAY_ATR-', '200DAY_ATR', 'ATR', 'OBV', '200ma', '14ma']
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df = interpolate_columns(df, max_gap_fill=50)  # updated argument name
    df = df.iloc[200:]
    df = df.drop(columns_to_drop, axis=1)
    df = df.round(8)
    print(f"Indicators calculated in {time.time() - Timer:.2f} seconds")
    return df




















##===========================(File Processing)===========================##
##===========================(File Processing)===========================##
##===========================(File Processing)===========================##

def clean_and_interpolate_data(df):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        df[col] = df[col].interpolate(method='linear', limit_direction='forward', axis=0)
    df.ffill(inplace=True)
    return df

def validate_columns(df, required_columns):
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        logging.error(f"Missing columns: {missing_columns}")
        return False
    return True


def DataQualityCheck(df):
    if df.empty or len(df) < 201:
        logging.error("DataFrame is empty or too short to process.")
        return None
    
    ##if more than 1/3 of the df has a close price below 1 then skip it 
    if len(df[df['Close'] < 1]) > len(df) / 3:
        logging.error("More than 1/3 of the data has a close price below 1. Skipping the data.")
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


def process_file(file_path, output_dir):
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        else:
            logging.error(f"Skipped {file_path}: Unsupported file format")
            return
        
        if not validate_columns(df, ['Close', 'High', 'Low', 'Volume']):
            logging.error(f"File {file_path} does not contain all required columns.")
            return

        df = DataQualityCheck(df)
        if df is None:
            logging.error(f"Data quality check failed for {file_path}.")
            return

        df = indicators(df)
        df = clean_and_interpolate_data(df)
        SaveData(df, file_path, output_dir)

    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}")
        traceback_info = traceback.format_exc()
        logging.error(traceback_info)






def SaveData(df, file_path, output_dir):
    file_name = os.path.basename(file_path)
    output_file = os.path.join(output_dir, file_name)
    if file_path.endswith('.csv'):
        df.to_csv(output_file, index=False)
    elif file_path.endswith('.parquet'):
        df.to_parquet(output_file, index=False)
    logging.info(f"Processed {file_path} and saved to {output_file}")





def clear_output_directory(output_dir):
    """
    Remove existing files in the output directory.
    """
    for file in os.listdir(output_dir):
        if file.endswith('.csv') or file.endswith('.parquet'):
            os.remove(os.path.join(output_dir, file))






##lets make a function that converts the csv files in the 





##===========================(Main Function)===========================##
##===========================(Main Function)===========================##
##===========================(Main Function)===========================##

def process_file_wrapper(file_path):
    return process_file(file_path, CONFIG['output_directory'])

def process_data_files(run_percent):
    # Set up profiling

    print(f"Processing {run_percent}% of files from {CONFIG['input_directory']}")
    StartTimer = time.time()
    os.makedirs(CONFIG['output_directory'], exist_ok=True)
    clear_output_directory(CONFIG['output_directory'])

    file_paths = [os.path.join(CONFIG['input_directory'], f) for f in os.listdir(CONFIG['input_directory']) if f.endswith('.csv') or f.endswith('.parquet')]
    files_to_process = file_paths[:int(len(file_paths) * (run_percent / 100))]

    with ProcessPoolExecutor() as executor:
        list(executor.map(process_file_wrapper, files_to_process))

    print(f"Processed {len(files_to_process)} files in {time.time() - StartTimer:.2f} seconds")
    print(f'Averaging {round((time.time() - StartTimer) / len(files_to_process), 2)} seconds per file.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process financial market data files.")
    parser.add_argument('--runpercent', type=int, default=100, help="Percentage of files to process from the input directory.")
    args = parser.parse_args()

    process_data_files(args.runpercent)

