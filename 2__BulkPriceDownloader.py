#!/root/root/miniconda4/envs/tf/bin/python
import yfinance as yf
import pandas as pd
import os
import logging
import time
import argparse
from datetime import datetime
import glob
import re
from tqdm import tqdm

"""
This script downloads stock data from Yahoo Finance based on the presence of ticker files in the output directory.
"""
FINAL_DATA_DIRECTORY = "Data/RFpredictions"
DATA_DIRECTORY = 'Data/PriceData'
TICKERS_CIK_DIRECTORY = 'Data/TickerCikData'
LOG_FILE = "Data/PriceData/_Price_Data_download.log"
START_DATE = '2020-01-01'
RATE_LIMIT = 1.0  # seconds between downloads

os.makedirs(DATA_DIRECTORY, exist_ok=True)
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_existing_tickers(data_dir):
    """Extract ticker symbols from existing data files in the directory using regex."""
    pattern = re.compile(r"^(.*?)\.parquet$")
    return [match.group(1) for file in os.listdir(data_dir) if (match := pattern.match(file))]

def find_latest_ticker_cik_file(directory):
    files = glob.glob(os.path.join(directory, 'TickerCIKs_*.parquet'))
    return max(files, key=os.path.getmtime) if files else None

def clear_old_data(data_dir):
    for file in os.listdir(data_dir):
        if file.endswith('.csv') or file.endswith('.parquet'):
            os.remove(os.path.join(data_dir, file))
    logging.info("Cleared old data files.")

def fetch_and_save_stock_data(tickers, data_dir, start_date=START_DATE, max_downloads=None, rate_limit=RATE_LIMIT):
    download_count = 0
    wait_time = rate_limit
    start_time = time.time()
    existing_files = set(os.listdir(data_dir))
    
    progress_bar = tqdm(total=max_downloads or len(tickers), desc="Downloading stock data", unit="ticker")
    
    for ticker in tickers:
        if max_downloads and download_count >= max_downloads:
            break
        file_name = f"{ticker}.parquet"
        file_path = os.path.join(data_dir, file_name)
        if file_name not in existing_files:
            try:
                stock_data = yf.download(ticker, start=start_date, progress=False)
                if not stock_data.empty and len(stock_data) >= 282 and stock_data['Close'].max() <= 100000000000:
                    stock_data = stock_data.round(5)
                    stock_data.to_parquet(file_path)
                    download_count += 1
                    if download_count % 100 == 0:
                        logging.info(f"Downloaded data for {download_count} tickers taking {time.time() - start_time:.2f} seconds")
                else:
                    logging.warning(f"Skipping {ticker}: insufficient data or abnormal price.")
                time.sleep(wait_time)
            except Exception as e:
                logging.error(f"Error downloading data for {ticker}: {e}")
        progress_bar.update(1)
        progress_bar.refresh()  # Force refresh the progress bar

    progress_bar.close()
    logging.info(f"Total download time: {time.time() - start_time:.2f} seconds")
    return download_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download stock data from Yahoo Finance.")
    parser.add_argument("--NumberOfFiles", type=int, help="Maximum number of ticker files to download.")
    parser.add_argument("--PercentDownload", type=float, help="Percentage of total tickers to download.")
    parser.add_argument("--ClearOldData", action='store_true', help="Clears any existing data in the output directory.")
    parser.add_argument("--RefreshMode", action='store_true', help="Only refresh data for tickers already present in the output directory.")

    args = parser.parse_args()
    if args.ClearOldData:
        clear_old_data(DATA_DIRECTORY)

    if args.RefreshMode:
        tickers = get_existing_tickers(FINAL_DATA_DIRECTORY)
        max_files = len(tickers)
    else:
        all_tickers = pd.read_parquet(find_latest_ticker_cik_file(TICKERS_CIK_DIRECTORY))
        nyse_nasdaq_tickers = all_tickers[all_tickers['exchange'].isin(['NYSE', 'NASDAQ'])]['ticker'].tolist()
        tickers = nyse_nasdaq_tickers
        max_files = len(nyse_nasdaq_tickers)

    if args.PercentDownload:
        max_files = int(max_files * (args.PercentDownload / 100))
    if args.NumberOfFiles:
        max_files = min(args.NumberOfFiles, max_files)

    downloaded = fetch_and_save_stock_data(tickers, DATA_DIRECTORY, max_downloads=max_files, start_date=START_DATE, rate_limit=RATE_LIMIT)

    logging.info(f"Data download completed. Downloaded {downloaded} new files.")
    print(f"Downloaded {downloaded} new files.")