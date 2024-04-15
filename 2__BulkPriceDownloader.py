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
import re


"""
This script downloads stock data from Yahoo Finance based on the presence of ticker files in the output directory.
"""
FINAL_DATA_DIRECTORY = "Data/RFpredictions"
DATA_DIRECTORY = 'Data/PriceData'
TICKERS_CIK_DIRECTORY = 'Data/TickerCikData'
LOG_FILE = "Data/PriceData/_Price_Data_download.log"
START_DATE = '2010-01-01'
RATE_LIMIT = 1.0  # seconds between downloads

os.makedirs(DATA_DIRECTORY, exist_ok=True)
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_existing_tickers(data_dir):
    """Extract ticker symbols from existing data files in the directory using regex."""
    pattern = re.compile(r"^(.*?)_modified_predictions\.csv$")
    tickers = []
    for file in os.listdir(data_dir):
        match = pattern.match(file)
        if match:
            tickers.append(match.group(1))
    return tickers
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
    wait_time = rate_limit
    for ticker in tickers:
        if max_downloads and download_count >= max_downloads:
            break
        file_path = os.path.join(data_dir, f"{ticker}.csv")
        if not os.path.exists(file_path):
            try:
                stock_data = yf.download(ticker, start=start_date, progress=False)
                if not stock_data.empty:
                    stock_data.to_csv(file_path)
                    logging.info(f"Downloaded data for {ticker}.")
                    download_count += 1
                time.sleep(wait_time)
            except Exception as e:
                logging.error(f"Error downloading data for {ticker}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download stock data from Yahoo Finance.")
    parser.add_argument("--NumberOfFiles", type=int, help="Maximum number of ticker files to download.")
    parser.add_argument("--PercentDownload", type=float, help="Percentage of total tickers to download.")
    parser.add_argument("--ClearOldData", action='store_true', help="Clears any existing data in the output directory.")
    parser.add_argument("--RefreshMode", action='store_true', help="Only refresh data for tickers already present in the output directory.")

    args = parser.parse_args()

    if args.ClearOldData:
        clear_old_data(DATA_DIRECTORY)

    tickers = get_existing_tickers(FINAL_DATA_DIRECTORY) if args.RefreshMode else pd.read_csv(find_latest_ticker_cik_file(TICKERS_CIK_DIRECTORY))['ticker'].tolist()
    max_files = args.NumberOfFiles or int(len(tickers) * (args.PercentDownload / 100)) if args.PercentDownload else None

    fetch_and_save_stock_data(tickers, DATA_DIRECTORY, max_downloads=max_files, start_date=START_DATE, rate_limit=RATE_LIMIT)

    logging.info("Data download completed.")
