import pandas as pd
import os
from sklearn.preprocessing import RobustScaler, MinMaxScaler
import multiprocessing
import logging
import glob
import argparse


# Logging configuration

logging.basicConfig(
    filename='Data/ScaledData/_ScalingErrors.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

def squash_col_outliers(df, col_name=None, min_quantile=0.01, max_quantile=0.99):
    """
    Squashes outliers in a DataFrame based on quantile thresholds.

    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    col_name (str, optional): Specific column name to process; if None, process all float64 columns.
    min_quantile (float): The lower quantile threshold.
    max_quantile (float): The upper quantile threshold.

    Returns:
    pd.DataFrame: The DataFrame with outliers squashed.
    """
    # Check if col_name is None and process accordingly
    if col_name is None:
        for col in df.columns:
            if df[col].dtype == 'float64':

                q_lo = df[col].quantile(min_quantile)
                q_hi = df[col].quantile(max_quantile)

                
                # Applying the quantile thresholds
                df.loc[df[col] >= q_hi, col] = q_hi
                df.loc[df[col] <= q_lo, col] = q_lo

    else:
        logging.warning("Column name provided is not None, but function is not set up to handle specific columns.")
    return df




def scale_data_with_rolling_window(df, window_size_func):
    """
    Scales numeric data in the DataFrame using a rolling window adjusted for variability.

    Parameters:
    df (pd.DataFrame): The DataFrame to scale.
    window_size_func (function): Function to determine the window size based on column statistics.

    Returns:
    pd.DataFrame: Scaled DataFrame.
    """
    df_scaled = pd.DataFrame(index=df.index)
    for col in df.columns:
        if col == 'Date' or not pd.api.types.is_numeric_dtype(df[col]):
            continue  # Skip non-numeric columns and the Date column

        window_size = window_size_func(df[col])

        # Apply scaling on a rolling basis
        scaled_values = []
        for i in range(len(df)):
            if i < window_size - 1:
                # Not enough data for a full window, use available data
                window_data = df[col].iloc[:i+1]
            else:
                window_data = df[col].iloc[i-window_size+1:i+1]

            # Scale the data in the current window
            scaler = RobustScaler()
            scaled_window = scaler.fit_transform(window_data.values.reshape(-1, 1)).flatten()
            scaled_values.append(scaled_window[-1])  # Append the last scaled value

        df_scaled[col] = scaled_values

    return df_scaled






def process_file(args):
    """
    Process a single file - read, clean, interpolate, scale, and save the data.

    Parameters:
    args (tuple): Contains file_path and output_dir.
    """
    file_path, output_dir = args
    # Detect file format
    if file_path.endswith('.csv'):
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            logging.error(f"Error reading {file_path}: {e}")
            return
    elif file_path.endswith('.parquet'):
        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            logging.error(f"Error reading {file_path}: {e}")
            return
    else:
        logging.warning(f"Unsupported file format for {file_path}. Skipping.")
        return
    

    ## calculate the percentage of the file that is NaN and log it
    percent_missing = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100


    if percent_missing > 0:
        logging.info(f"Percentage of missing data in {file_path}: {percent_missing:.2f}%")

    ##============(Data Interpolation, NaN filling, Scaling)=============#
    ##============(Data Interpolation, NaN filling, Scaling)=============#
    ##============(Data Interpolation, NaN filling, Scaling)=============#


    df = squash_col_outliers(df)


    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        try:
            # Ensure there are enough non-NaN data points for spline interpolation
            if df[col].count() >= 5:
                df[col].interpolate(method='spline', order=3, inplace=True)
            else:
                raise ValueError("Not enough data points for spline interpolation.")
        except Exception as e:
            logging.warning(f"Spline interpolation failed for column {col} in file {file_path}: {e}")
            # Use backward fill and forward fill for NaN values
            df[col].bfill(inplace=True)
            df[col].ffill(inplace=True)

    df = scale_data_with_rolling_window(df, window_size_based_on_stats)
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].round(8).astype('float16')

    #save file if the percent nan is low enough
    if percent_missing < 0.01:

        # Save processed file
        file_name = os.path.basename(file_path)
        output_path = os.path.join(output_dir, file_name)
        if file_path.endswith('.csv'):
            df.to_csv(output_path, index=False)
        elif file_path.endswith('.parquet'):
            df.to_parquet(output_path)
    else:
        logging.warning(f"Percentage of missing data in {file_path} is too high. File not saved.")


def window_size_based_on_stats(df_col, small_window=10, medium_window=20, large_window=30):
    # Calculate statistical measures for the column
    std_dev = df_col.std()
    percentile_range = df_col.quantile(0.99) - df_col.quantile(0.01)

    # Define thresholds based on statistical measures
    high_volatility_threshold = 20
    medium_volatility_threshold = 15 

    # Determine window size based on volatility
    if std_dev > high_volatility_threshold or percentile_range > high_volatility_threshold:
        return small_window  # smaller window for high volatility
    elif std_dev > medium_volatility_threshold or percentile_range > medium_volatility_threshold:
        return medium_window
    else:
        return large_window






def main(input_dir, output_dir, percent_to_run):
    """
    Main function to process a given percentage of files in a directory.

    Parameters:
    input_dir (str): Directory containing files to process.
    output_dir (str): Directory to save processed files.
    percent_to_run (int): Percentage of files from the input directory to process.
    """

    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        # Delete existing CSV and Parquet files in the output directory
        files_to_remove = glob.glob(os.path.join(output_dir, '*.csv')) + \
                          glob.glob(os.path.join(output_dir, '*.parquet'))
        for file in files_to_remove:
            try:
                os.remove(file)
            except Exception as e:
                logging.error(f"Error removing file {file}: {e}")

    all_files = os.listdir(input_dir)
    num_files_to_process = int(len(all_files) * (percent_to_run / 100)) if percent_to_run else len(all_files)

    pool = multiprocessing.Pool(multiprocessing.cpu_count())  # Create a pool of worker processes
    tasks = [(os.path.join(input_dir, file), output_dir) for file in all_files[:num_files_to_process]]
    pool.map(process_file, tasks)  # Process each file in the pool

    pool.close()
    pool.join()





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process a percentage of files from the input directory.")
    parser.add_argument('--percenttorun', type=int, help='Percentage of files to process', default=100)
    args = parser.parse_args()

    input_dir = 'Data/IndicatorData'
    output_dir = 'Data/ScaledData'
    main(input_dir, output_dir, args.percenttorun)