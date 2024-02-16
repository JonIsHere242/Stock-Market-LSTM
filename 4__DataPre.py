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


def scale_data(df):
    """
    Scales numeric data in the DataFrame using appropriate scalers.

    Parameters:
    df (pd.DataFrame): The DataFrame to scale.

    Returns:
    tuple: Scaled DataFrame and dictionary of scalers used.
    """


    df.ffill(inplace=True)
    df.bfill(inplace=True)
    scalers = {}
    df_scaled = pd.DataFrame(index=df.index) 
    for col in df.columns:
        if col == 'Date':
            continue

        if col == 'Close':
            continue

        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)
            scaler = MinMaxScaler()
        else:
            df[col] = df[col] + 0.00001  # Add a small amount to avoid division by zero
            scaler = RobustScaler()
        df_scaled[col] = scaler.fit_transform(df[[col]])  # Scale each column individually
        scalers[f'scaler_{col}'] = scaler  # Store each scaler with a unique name
    return df_scaled, scalers


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

    # Spline interpolate
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col] = df[col].interpolate(method='spline', order=3)





    # Fill remaining NaNs
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    df, _ = scale_data(df)
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
                logging.info(f"Removed existing file: {file}")
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