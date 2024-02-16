import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import time
import scipy.stats as stats
from scipy.stats import linregress
import logging

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


def interpolate_columns(df, Interpolation_Window=200):
    for column in df.columns:
        non_nan_percentage = df[column].notna().mean()
        if non_nan_percentage > 0.8:
            df.loc[:Interpolation_Window, column] = df.loc[:Interpolation_Window, column].bfill()
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





def indicators(df):
    df['Close'] = df['Close'].bfill()
    df['High'] = df['High'].bfill()
    df['Low'] = df['Low'].bfill()
    df['Volume'] = df['Volume'].bfill()


    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
    calculate_apz(df)




    calculate_cumulative_streaks(df)


    ##df['VAMA'] = calculate_vama(df)
    ##periods = [63, 10, 3]  # Approx 3 months, 2 weeks, 3 days in trading days
    ##df = add_vama_changes(df, df['VAMA'], periods)
    ##df.drop(columns=['VAMA'], inplace=True)


    ##Renable this when not testing
    ##window_size = 30  # Example window size
    ##df = find_levels(df, window_size)

    close_shift_1 = close.shift(1)
    true_range = np.maximum(high - low, np.maximum(np.abs(high - close_shift_1), np.abs(close_shift_1 - low)))
    window = 14
    rolled = np.lib.stride_tricks.sliding_window_view(true_range, window)
    mean_rolled = np.mean(rolled, axis=1)
    df['ATR'] = np.pad(mean_rolled, (window-1, 0), 'constant', constant_values=(np.nan,))
    df['ATR%'] = (df['ATR'] / close) * 100
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
    columns_to_drop = ['Open', 'High', 'Low', 'Adj Close','ATZ_Upper','ATZ_Lower','VWAP', '200DAY_ATR-', '200DAY_ATR', 'Volume', 'ATR', 'OBV', '200ma', '14ma']
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df = interpolate_columns(df, Interpolation_Window=210)
    df = df.replace({True: 1, False: 0})
    df = df.iloc[200:]
    df = df.drop(columns_to_drop, axis=1)
    df = df.round(8)
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

        df = indicators(df)
        df = squash_col_outliers(df, num_std_dev=3)    
        ##log the time taken to process the file here
        filename = os.path.basename(file_path)
        new_file_path = os.path.join(output_dir, filename.replace('.csv', '_modified.csv').replace('.parquet', '_modified.parquet'))

        if file_path.endswith('.csv'):
            df.to_csv(new_file_path, index=False)
        else:
            df.to_parquet(new_file_path, index=False)
        
        logging.info(f"Time taken to process {file_path}: {time.time() - Timer:.2f} seconds")
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        return f"Error processing {file_path}: {e}"






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
    os.makedirs(CONFIG['output_directory'], exist_ok=True)
    clear_output_directory(CONFIG['output_directory'])
    process_files(CONFIG['input_directory'], CONFIG['output_directory'])

    try:
        average_time_taken = calculate_average_processing_time(CONFIG['log_file'], CONFIG['core_count_division'])
        logging.info(f"Average time taken to process each file: {average_time_taken:.2f} seconds")
    except Exception as e:
        print(f"Error calculating average processing time: {e}")
