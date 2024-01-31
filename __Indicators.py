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

