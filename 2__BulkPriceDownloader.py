#!/root/root/miniconda4/envs/tf/bin/python
import yfinance as yf
import pandas as pd
import os
import logging
import time
import argparse
from datetime import datetime
import glob
from datetime import timedelta

"""
This script is used for downloading stock data from Yahoo Finance.

Command Line Arguments:
- --NumberOfFiles: Specifies the maximum number of ticker files to download.
- --PercentDownload: Specifies the percentage of total tickers for which data should be downloaded.
- --ClearOldData: Clears any existing CSV or Parquet files in the output directory before downloading new data.

Usage Examples:
- To download data for 30 specific tickers: python 2__BulkPriceDownloader.py --NumberOfFiles 30
- To download data for 2% of the total number of tickers: pyhton 2__BulkPriceDownloader.py --PercentDownload 2
- To clear old data and download for 2% of tickers: python 2__BulkPriceDownloader.py --ClearOldData --PercentDownload 2
"""


DATA_DIRECTORY = 'Data/PriceData'
TICKERS_CIK_DIRECTORY = 'Data/TickerCikData'
LOG_DIRECTORY = 'Data/PriceData'
LOG_FILE = "Data/PriceData/_Price_Data_download.log"
START_DATE = '1995-01-01'
RATE_LIMIT = 1.0  # seconds between downloads

os.makedirs(DATA_DIRECTORY, exist_ok=True)
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def is_data_up_to_date(file_path, days_threshold=100):
    try:
        data = pd.read_csv(file_path, parse_dates=['Date'])
        last_date = data['Date'].max()
        return (datetime.now() - last_date) <= timedelta(days=days_threshold)
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")
        return False



def find_latest_ticker_cik_file(directory):
    files = glob.glob(os.path.join(directory, 'TickerCIKs_*.csv'))  # Corrected this line
    if not files:
        return None
    return max(files, key=os.path.getmtime)



def clear_old_data(data_dir):
    for file in os.listdir(data_dir):
        if file.endswith('.csv') or file.endswith('.parquet'):
            os.remove(os.path.join(data_dir, file))
    logging.info("Cleared old data files.")





def fetch_and_save_stock_data(tickers, data_dir, start_date=START_DATE, max_downloads=None, rate_limit=RATE_LIMIT):
    download_count = 0
    wait_time = rate_limit * 1.05 if max_downloads is not None else rate_limit * 1.10
    max_downloads = max_downloads or len(tickers)

    for ticker in tickers:
        if download_count >= max_downloads:
            break

        file_path = os.path.join(data_dir, f"{ticker}.csv")

        # Check if existing data is up to date
        if os.path.exists(file_path) and is_data_up_to_date(file_path):
            continue

        try:
            start_download_time = time.time()
            stock_data = yf.download(ticker, start=start_date, progress=False)

            if stock_data.empty:
                logging.warning(f"Stock data file is empty: {ticker}")
                continue

            if len(stock_data) < 201:
                logging.warning(f"Stock data file has less than 100 records: {ticker}")
                continue

            stock_data.to_csv(file_path)
            ##logging.info(f"Downloaded data for {ticker}.")

            download_count += 1
            end_download_time = time.time()
            elapsed_time = end_download_time - start_download_time
            adjusted_wait_time = max(wait_time - elapsed_time, 0)
            time.sleep(adjusted_wait_time)

        except Exception as e:
            logging.error(f"Error downloading data for {ticker}: {e}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download stock data from Yahoo Finance.")
    parser.add_argument("--NumberOfFiles", type=int, help="Maximum number of ticker files to download.")
    parser.add_argument("--PercentDownload", type=float, help="Percentage of total tickers to download.")
    parser.add_argument("--ClearOldData", action='store_true', help="Clears any existing data in the output directory.")

    args = parser.parse_args()
    max_files = None
    TotalTimer = time.time()
    try:
        TICKERS_CIK_FILE = find_latest_ticker_cik_file(TICKERS_CIK_DIRECTORY)
        if TICKERS_CIK_FILE is None:
            logging.error("No Ticker CIK file found.")
            exit(1)

        if args.ClearOldData:
            clear_old_data(DATA_DIRECTORY)

        df = pd.read_csv(TICKERS_CIK_FILE)
        tickers = df['ticker'].unique().tolist()

        if args.NumberOfFiles:
            max_files = args.NumberOfFiles
        elif args.PercentDownload:
            max_files = int(len(tickers) * (args.PercentDownload / 100))

        fetch_and_save_stock_data(tickers, DATA_DIRECTORY, max_downloads=max_files, start_date=START_DATE, rate_limit=RATE_LIMIT)
        logging.info("Bulk historical data has been downloaded.")
        filecounter = 0
        ##add something that chescks the number of files in the output directory and devides it by the number of tickers oin teh file to get a percentage of the tickers downloaded and then log the result 
        for file in os.listdir(DATA_DIRECTORY):
            if file.endswith('.csv'):
                filecounter += 1
        logging.info(f"Number of files downloaded: {filecounter}")
        logging.info(f"Percentage of tickers downloaded: {filecounter/len(tickers)*100}%")
    except Exception as e:
        logging.error(f"Error: {e}")
    ##log the total time rounded to 2 decimal places
    TotalTimer = time.time() - TotalTimer
    logging.info(f"Total time taken: {TotalTimer:.2f} seconds")