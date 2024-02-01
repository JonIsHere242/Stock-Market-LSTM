import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import time
import logging
logging.basicConfig(filename='Data/IndicatorData/_IndicatorData.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


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

def indicators(df):
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']
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



    close_shift_1 = close.shift(1)
    low_shift_1 = low.shift(1)
    high_shift_1 = high.shift(1)
    pct_change_close = close.pct_change()

    # Consolidating rolling operations
    rolling_20 = df['Close'].rolling(window=20)
    rolling_14 = df['Close'].rolling(window=14)
    rolling_7 = df['Close'].rolling(window=7)
    rolling_30 = df['Close'].rolling(window=30)
    rolling_500 = df['Close'].rolling(window=500)
    
    df['percent_change_Close'] = pct_change_close
    df['pct_change_std'] = rolling_20.std()

    # Replacing loop for shifts with direct assignments
    df['percent_change_Close_lag_1'] = pct_change_close.shift(1)  # Lag by 1 period
    df['percent_change_Close_lag_2'] = pct_change_close.shift(2)  # Lag by 2 periods
    df['percent_change_Close_lag_3'] = pct_change_close.shift(3)  # Lag by 3 periods


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

    # RSI Divergence
    # Note: Actual divergence calculation may depend on further rules (e.g., comparing local maxima of price with RSI)
    df['RSI_Divergence'] = df['Close'] - df['RSI']

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

    # RSI and Bollinger Bands calculations
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = (-delta).clip(lower=0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    rolling_mean = rolling_20.mean()
    rolling_std = rolling_20.std()
    df['B%'] = (df['Close'] - (rolling_mean - 2 * rolling_std)) / (4 * rolling_std)

    # Streak calculations streamlined
    positive_streak = (pct_change_close > 0).astype(int).cumsum()
    negative_streak = (pct_change_close < 0).astype(int).cumsum()
    df['positive_streak'] = positive_streak - positive_streak.where(pct_change_close <= 0).ffill().fillna(0)
    df['negative_streak'] = negative_streak - negative_streak.where(pct_change_close >= 0).ffill().fillna(0)

    # Gap calculations streamlined
    gap_threshold_percent = 0.5 if pct_change_close.std() > 1 else 0
    df['is_gap_up'] = (df['Low'] - high_shift_1) / close_shift_1 * 100 > gap_threshold_percent
    df['is_gap_down'] = (df['High'] - low_shift_1) / close_shift_1 * 100 < -gap_threshold_percent
    df['move_from_gap%'] = np.where(df['is_gap_up'], (df['Close'] - low_shift_1) / low_shift_1 * 100,
                                    np.where(df['is_gap_down'], (df['Close'] - high_shift_1) / high_shift_1 * 100, 0))
    df['VPT'] = df['Volume'] * ((df['Close'] - close_shift_1) / close_shift_1)

    # Dropping unnecessary columns
    columns_to_drop = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'VWAP', '200DAY_ATR-', '200DAY_ATR', 'Close_change', 'Volume', 'ATR', 'OBV', '200ma', '14ma']
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
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

    ##multithreaded processing of each df here
    ##start a timer here called timer_filename

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




if __name__ == "__main__":
    INPUT_DIRECTORY = 'Data/PriceData'  # Directory containing the original files
    OUTPUT_DIRECTORY = 'Data/IndicatorData'  # Directory where modified files will be saved

    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    process_files(INPUT_DIRECTORY, OUTPUT_DIRECTORY)
    
    LOG_FILE = 'Data/IndicatorData/_IndicatorData.log'
    try:
        with open(LOG_FILE, 'r') as f:
            lines = f.readlines()[-500:]
        time_taken = []
        for line in lines:
            if 'Time taken to process' in line:
                time_taken.append(float(line.split(':')[-1].strip().split(' ')[0]))
        average_time_taken = sum(time_taken)/len(time_taken)

        ##dvide the average time take by the core count used so that we get a real average time taken
        average_time_taken = average_time_taken / os.cpu_count()

        print(f"Average time taken to process each file: {average_time_taken:.2f} seconds")
    except Exception as e:
        print(f"Error reading log file: {e}")
        logging.error(f"Error reading log file: {e}")
        pass

