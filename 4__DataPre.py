import pandas as pd
import numpy as np
import os
import glob
import argparse
import logging
from sklearn.preprocessing import RobustScaler
import multiprocessing



from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import plotly.graph_objects as go
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant


# Setup logging
logging.basicConfig(
    filename='Data/ScaledData/_ScalingErrors.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

# Argument parsing
parser = argparse.ArgumentParser(description='Process files for scaling and cleaning.')
parser.add_argument('--percentrun', type=int, default=100, help='Percentage of files to process')
parser.add_argument('--input', type=str, help='Input directory path')
parser.add_argument('--output', type=str, help='Output directory path')
parser.add_argument('--singlefile', type=str, help='Path to a single file for processing')
parser.add_argument('--amountscaled', action='store_true', help='Prints the amount scaled as a percentage')
args = parser.parse_args()

# Outlier squashing function
def outlyerSquasher(df, percentile1=0.999, percentile2=0.001):
    """
    Squashes outliers to specific percentiles in a DataFrame.
    """
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        lower_quantile = df[col].quantile(percentile2)  # Get the lower quantile
        upper_quantile = df[col].quantile(percentile1)  # Get the upper quantile
        df[col] = df[col].clip(lower=lower_quantile, upper=upper_quantile)
    return df


# Configuration for input/output directory
CONFIG = {
    "input_dir": args.input or "Data/IndicatorData",
    "output_dir": args.output or "Data/ScaledData"
}




##define a scaling function that only scales numaric colculmns
def scale_data(df):



    # Separate numeric and non-numeric data
    df_numeric = df.select_dtypes(include=['float64', 'int64'])
    df_non_numeric = df.select_dtypes(exclude=['float64', 'int64'])




    # Check if df_numeric is empty
    if df_numeric.empty:
        logging.warning(f"DataFrame is empty after selecting numeric columns. Original DF shape: {df.shape}")
        return df
    # Scale the numeric data
    scaler = RobustScaler()
    scaled_values = scaler.fit_transform(df_numeric)
    df_scaled = pd.DataFrame(scaled_values, columns=df_numeric.columns)

    # Reintegrate non-numeric data
    df_final = pd.concat([df_non_numeric, df_scaled], axis=1)

    # Change all numeric columns to float32
    for col in df_final.select_dtypes(include=['float64', 'int64']):
        df_final[col] = df_final[col].astype('float32')

    return df_final




def dataValidation(df, file_path):
    ##ensure the df is longer than 500 rows 
    if len(df) < 500:
        logging.warning(f"{file_path} has too few rows. Original DF shape: {df.shape}")
        return False
    
    ##ensure the df has more than 10 columns
    if len(df.columns) < 10:
        logging.warning(f"{file_path} has too few columns. Original DF shape: {df.shape}")
        return False
    
    ##ensure the start and end of the df have unique values 
    if df.iloc[0].equals(df.iloc[-1]):
        logging.warning(f"{file_path} has the same start and end values. Original DF shape: {df.shape}")
        return False
    ##ensure the df has less then 10% of the values as nan o inf or -inf
    if df.isna().sum().sum() > len(df)*0.1:
        logging.warning(f"{file_path} has too many NaN values. Original DF shape: {df.shape}")
        return False
    
    if df.isin([np.inf, -np.inf]).sum().sum() > len(df)*0.1:
        logging.warning(f"{file_path} has too many inf values. Original DF shape: {df.shape}")
        return False





def has_trend(series, threshold=0.2):
    if len(series.dropna()) < 2:  # Ensure there are enough data points
        return False
    x = np.arange(len(series))
    x = add_constant(x)
    model = OLS(series, x).fit()
    slope = model.params[1]
    return abs(slope) > threshold

def detrend_with_poly(data, degree=2):
    y = data.values
    x = np.arange(len(y)).reshape(-1, 1)
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(x)
    model = LinearRegression().fit(X_poly, y)
    trend = model.predict(X_poly)
    return y - trend

def detrend_data(df):
    for col in df.select_dtypes(include=['float64', 'int64']):
        if has_trend(df[col]):
            df[col] = detrend_with_poly(df[col])
    return df

def process_file(file_path):
    # Read the file
    df = pd.read_csv(file_path)
    if dataValidation(df, file_path) == False:
        return
    df = outlyerSquasher(df)  # Squash outliers

    df = detrend_data(df)  # Detrend data
    df = scale_data(df)  # Scale data

    df.to_csv(os.path.join(CONFIG["output_dir"], os.path.basename(file_path)), index=False)



# Main function
def main():

    ##for all the files in the output that end in csv delete them 
    files = glob.glob('Data/ScaledData/*.csv')
    for f in files:
        os.remove(f)


    ##get all the files in the input directory that end in csv 
    input_files = glob.glob(os.path.join(CONFIG["input_dir"], "*.csv"))


    num_files_to_process = int(len(input_files) * (args.percentrun / 100))

    # Create a pool of worker processes
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # Map the file processing function to each file
        pool.map(process_file, input_files[:num_files_to_process])

if __name__ == "__main__":
    main()
