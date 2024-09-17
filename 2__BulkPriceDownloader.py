#!/root/root/miniconda4/envs/tf/bin/python
import yfinance as yf
import pandas as pd
import os
import logging
import time
from datetime import datetime, timedelta
import argparse
import glob
import re
from tqdm import tqdm

"""
This script downloads stock data from Yahoo Finance based on the presence of ticker files in the output directory.
"""

FINAL_DATA_DIRECTORY = "Data/RFpredictions"
DATA_DIRECTORY = 'Data/PriceData'
TICKERS_CIK_DIRECTORY = 'Data/TickerCikData'
LOG_FILE = "data/logging/2__BulkPriceDownloader.log"
RATE_LIMIT = 1.0  # seconds between downloads
START_DATE = "2020-01-01"
os.makedirs(DATA_DIRECTORY, exist_ok=True)

def setup_logging():
    """Set up logging configuration."""
    log_dir = os.path.dirname(LOG_FILE)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logging.basicConfig(filename=LOG_FILE, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

def get_existing_tickers(data_dir):
    """Extract ticker symbols from existing data files in the directory using regex."""
    pattern = re.compile(r"^(.*?)\.parquet$")
    tickers = []
    for file in os.listdir(data_dir):
        match = pattern.match(file)
        if match:
            tickers.append(match.group(1))
    return tickers

def find_latest_ticker_cik_file(directory):
    files = glob.glob(os.path.join(directory, 'TickerCIKs_*.parquet'))
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def clear_old_data(data_dir):
    for file in os.listdir(data_dir):
        if file.endswith('.csv') or file.endswith('.parquet'):
            os.remove(os.path.join(data_dir, file))
    logging.info("Cleared old data files.")

def fetch_and_save_stock_data(tickers, data_dir, max_downloads=None, start_date=None, end_date=None, rate_limit=RATE_LIMIT):
    download_count = 0
    wait_time = rate_limit
    Timer = time.time()
    
    for ticker in tqdm(tickers, desc="Downloading stock data", unit="ticker"):
        if max_downloads and download_count >= max_downloads:
            break
        
        file_path = os.path.join(data_dir, f"{ticker}.parquet")
        
        # Skip download if data already exists
        if not os.path.exists(file_path):
            try:
                # Fetch daily OHLCV data from the past 5 years
                stock_data = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
                
                if not stock_data.empty:
                    # Add checks for the validity of data (optional)
                    if len(stock_data) < 252:  # Skip tickers with very short histories
                        logging.warning(f"Data for {ticker} is too short. Skipping.")
                        continue
                    
                    stock_data = stock_data.round(5)
                    
                    # Save data to parquet
                    stock_data.to_parquet(file_path)
                    download_count += 1
                    
                    if download_count % 100 == 0:
                        logging.info(f"Downloaded data for {download_count} tickers, taking {time.time() - Timer} seconds.")
                
                # Enforce rate limit between downloads
                time.sleep(wait_time)
            
            except Exception as e:
                logging.error(f"Error downloading data for {ticker}: {e}")
                continue

if __name__ == "__main__":
    setup_logging()
    
    parser = argparse.ArgumentParser(description="Download stock data from Yahoo Finance.")
    parser.add_argument("--NumberOfFiles", type=int, help="Maximum number of ticker files to download.")
    parser.add_argument("--PercentDownload", type=float, help="Percentage of total tickers to download.")
    parser.add_argument("--ClearOldData", action='store_true', help="Clears any existing data in the output directory.")
    parser.add_argument("--RefreshMode", action='store_true', help="Only refresh data for tickers already present in the output directory.")

    args = parser.parse_args()
    if args.ClearOldData:
        clear_old_data(DATA_DIRECTORY)

    tickers = get_existing_tickers(FINAL_DATA_DIRECTORY) if args.RefreshMode else pd.read_parquet(find_latest_ticker_cik_file(TICKERS_CIK_DIRECTORY))['ticker'].tolist()
    max_files = args.NumberOfFiles or int(len(tickers) * (args.PercentDownload / 100)) if args.PercentDownload else None
    
    # Download the stock data, using start_date and end_date for more flexibility
    fetch_and_save_stock_data(
        tickers,
        DATA_DIRECTORY,
        max_downloads=max_files,
        start_date=START_DATE,
        end_date=datetime.now().strftime('%Y-%m-%d'),
        rate_limit=RATE_LIMIT
    )

    logging.info("Data download completed.")
