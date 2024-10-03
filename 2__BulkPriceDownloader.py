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
from zoneinfo import ZoneInfo  # Requires Python 3.9+
import pytz  # Add this line
import sys

"""
This script downloads stock data from Yahoo Finance based on the presence of ticker files in the output directory.
It supports both initial downloads (ColdStart) and refreshing existing data by appending the latest missing data.
"""

# Directory and File Configurations
FINAL_DATA_DIRECTORY = "Data\RFpredictions"
DATA_DIRECTORY = 'Data/PriceData'
TICKERS_CIK_DIRECTORY = 'Data/TickerCikData'
LOG_FILE = "Data\logging\2__BulkPriceDownloader.log"
RATE_LIMIT = 1.0  # seconds between downloads
START_DATE = "2020-01-01"

# Ensure necessary directories exist
os.makedirs(DATA_DIRECTORY, exist_ok=True)
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.DEBUG,  # Change to DEBUG for more detailed logs
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.debug(f"Data directory: {os.path.abspath(DATA_DIRECTORY)}")
    logging.debug(f"Ticker CIK directory: {os.path.abspath(TICKERS_CIK_DIRECTORY)}")






def get_existing_tickers(data_dir):
    """
    Extract ticker symbols from existing data files in the RFpredictions directory.
    This assumes that each file is named as {ticker}.parquet.
    """
    pattern = re.compile(r"^(.*?)\.parquet$")
    tickers = []
    for file in os.listdir(data_dir):
        match = pattern.match(file)
        if match:
            tickers.append(match.group(1))
    return tickers

def find_latest_ticker_cik_file(directory):
    """Find the latest TickerCIKs file based on modification time."""
    files = glob.glob(os.path.join(directory, 'TickerCIKs_*.parquet'))
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def clear_old_data(data_dir):
    """Remove all .csv and .parquet files from the data directory."""
    removed_files = 0
    for file in os.listdir(data_dir):
        if file.endswith('.csv') or file.endswith('.parquet'):
            os.remove(os.path.join(data_dir, file))
            removed_files += 1
    logging.info(f"Cleared {removed_files} old data files from {data_dir}.")




def get_last_trading_day(reference_datetime=None):
    """
    Returns the last trading day based on the current time in US/Eastern timezone.
    If the current time is after 4 PM ET on a weekday, returns today.
    Otherwise, returns the most recent previous trading day.
    """
    eastern = pytz.timezone('US/Eastern')
    
    if reference_datetime is None:
        # Get current time directly in US/Eastern timezone
        reference_datetime = datetime.now(eastern)
    else:
        # Ensure the reference_datetime is timezone-aware
        if reference_datetime.tzinfo is None:
            reference_datetime = eastern.localize(reference_datetime)
        else:
            reference_datetime = reference_datetime.astimezone(eastern)

    weekday = reference_datetime.weekday()  # Monday=0, Sunday=6
    current_time = reference_datetime.time()
    
    # Define market close time
    market_close_time = datetime.strptime("16:00", "%H:%M").time()
    
    logging.debug(f"Reference datetime (US/Eastern): {reference_datetime}")
    logging.debug(f"Weekday: {weekday}, Current Time: {current_time}")
    logging.debug(f"Market Close Time: {market_close_time}")

    if weekday < 5 and current_time >= market_close_time:
        last_trading_day = reference_datetime.date()
    else:
        # Adjust to previous day if it's weekend or before market close
        last_trading_day = reference_datetime.date() - timedelta(days=1)
        
        # If the adjusted day is Saturday or Sunday, move back to Friday
        while last_trading_day.weekday() >= 5:
            last_trading_day -= timedelta(days=1)
    
    logging.debug(f"Determined last trading day as: {last_trading_day}")
    return last_trading_day









def fetch_and_save_stock_data(tickers, data_dir, start_date=None, end_date=None, rate_limit=RATE_LIMIT, refresh=False):
    """
    Fetches stock data for given tickers and saves them as parquet files.
    Validates that the data matches the most common end date after the first 10 files are downloaded.
    """
    download_count = 0
    wait_time = rate_limit
    Timer = time.time()
    expected_latest_date = get_last_trading_day()

    # Adjust end_date to be one day after the expected latest trading day
    adjusted_end_date = expected_latest_date + timedelta(days=1)

    logging.info(f"Fetching data from {start_date} to {adjusted_end_date}")

    # Track the end dates of the first 10 files to establish a baseline
    end_dates = []
    end_date_sample_size = 10  # Number of files to sample before enforcing the end date check
    common_end_date = None  # Most common end date in the first 10 files

    for ticker in tqdm(tickers, desc="Downloading stock data", unit="ticker"):
        file_path = os.path.join(data_dir, f"{ticker}.parquet")
        
        try:
            if refresh and os.path.exists(file_path):
                # Read existing data to find the latest date
                existing_df = pd.read_parquet(file_path)
                if isinstance(existing_df.index, pd.DatetimeIndex):
                    latest_date = existing_df.index.max().date()
                elif 'Date' in existing_df.columns:
                    existing_df['Date'] = pd.to_datetime(existing_df['Date'])
                    latest_date = existing_df['Date'].max().date()
                else:
                    logging.warning(f"File {ticker}.parquet does not have a 'Date' column or index. Skipping refresh.")
                    continue

                # Determine the start date for new data
                new_start_date = latest_date + timedelta(days=1)
                if new_start_date > expected_latest_date:
                    # No new data to fetch
                    logging.info(f"{ticker} is already up-to-date.")
                    continue
            else:
                # Initial download
                new_start_date = start_date

            # Fetch data from Yahoo Finance using the adjusted end date
            stock_data = yf.download(
                ticker, 
                start=new_start_date, 
                end=adjusted_end_date.strftime('%Y-%m-%d'), 
                interval="1d", 
                progress=False
            )

            if stock_data.empty:
                logging.warning(f"No new data found for {ticker}.")
                continue

            # Check if the file has a valid end date after sampling enough files
            fetched_end_date = stock_data.index.max().date()

            if download_count < end_date_sample_size:
                # Add to the initial end date sample
                end_dates.append(fetched_end_date)
                if len(end_dates) == end_date_sample_size:
                    # Determine the most common end date after sampling enough files
                    common_end_date = max(set(end_dates), key=end_dates.count)
                    logging.info(f"Most common end date after sampling: {common_end_date}")
            elif fetched_end_date != common_end_date:
                logging.warning(f"{ticker} has a different end date ({fetched_end_date}) than the most common ({common_end_date}). Skipping.")
                continue

            # Ensure the data has sufficient historical coverage (at least 400 days)
            fetched_start_date = stock_data.index.min().date()
            days_of_data = (fetched_end_date - fetched_start_date).days
            if days_of_data < 400:
                logging.warning(f"{ticker} has less than 400 days of data ({days_of_data} days). Skipping.")
                continue

            stock_data = stock_data.round(5)

            if refresh and os.path.exists(file_path):
                # Append new data to existing data
                stock_data.index = pd.to_datetime(stock_data.index)
                updated_df = pd.concat([existing_df, stock_data])
                updated_df = updated_df[~updated_df.index.duplicated(keep='last')]
                updated_df = updated_df.sort_index()
                # Save back to parquet
                updated_df.to_parquet(file_path)
                logging.info(f"Appended new data for {ticker} up to {stock_data.index.max().date()}.")
            else:
                # Save new data to parquet
                stock_data.index.name = 'Date'
                stock_data.to_parquet(file_path)
                logging.info(f"Downloaded data for {ticker} up to {stock_data.index.max().date()}.")

            download_count += 1

            if download_count % 100 == 0:
                logging.info(f"Downloaded data for {download_count} tickers, taking {time.time() - Timer:.2f} seconds.")

        except Exception as e:
            logging.error(f"Error downloading data for {ticker}: {e}")
            continue

        finally:
            # Enforce rate limit between downloads
            logging.debug(f"Sleeping for {wait_time} seconds to respect rate limit.")
            time.sleep(wait_time)

    logging.info(f"Total tickers processed: {download_count}")






if __name__ == "__main__":
    setup_logging()
    
    logging.info(f"Python Version: {sys.version}")
    logging.info(f"Timezone: {time.tzname}")

    parser = argparse.ArgumentParser(description="Download or refresh stock data from Yahoo Finance.")
    parser.add_argument("--ClearOldData", action='store_true', help="Clears any existing data in the output directory.")
    parser.add_argument("--RefreshMode", action='store_true', help="Refresh existing data by appending the latest missing data.")
    parser.add_argument("--ColdStart", action='store_true', help="Initial download of all tickers from the CIK file.")
    parser.add_argument("--TickerListFile", help="Path to a file containing a list of tickers to download.")
    
    args = parser.parse_args()
    
    if args.ClearOldData:
        clear_old_data(DATA_DIRECTORY)
    
    if args.RefreshMode and args.ColdStart:
        logging.error("Cannot use both --RefreshMode and --ColdStart simultaneously. Exiting.")
        exit(1)
    
    if args.TickerListFile:
        if not os.path.exists(args.TickerListFile):
            logging.error(f"Ticker list file {args.TickerListFile} does not exist. Exiting.")
            exit(1)
        with open(args.TickerListFile, 'r') as f:
            tickers = [line.strip() for line in f if line.strip()]
        logging.info(f"Loaded {len(tickers)} tickers from {args.TickerListFile}.")
    elif args.RefreshMode:
        logging.info("Running in Refresh Mode: Refreshing data for tickers from the latest TickerCIKs file.")
        ticker_cik_file = find_latest_ticker_cik_file(TICKERS_CIK_DIRECTORY)
        if ticker_cik_file is None:
            logging.error("No TickerCIKs file found. Exiting.")
            exit(1)
        tickers_df = pd.read_parquet(ticker_cik_file)
        if 'ticker' not in tickers_df.columns:
            logging.error("The TickerCIKs file does not contain a 'ticker' column. Exiting.")
            exit(1)
        tickers = tickers_df['ticker'].dropna().unique().tolist()
    elif args.ColdStart:
        logging.info("Running in ColdStart Mode: Downloading data for all tickers from the CIK file.")
        ticker_cik_file = find_latest_ticker_cik_file(TICKERS_CIK_DIRECTORY)
        if ticker_cik_file is None:
            logging.error("No TickerCIKs file found. Exiting.")
            exit(1)
        tickers_df = pd.read_parquet(ticker_cik_file)
        if 'ticker' not in tickers_df.columns:
            logging.error("The TickerCIKs file does not contain a 'ticker' column. Exiting.")
            exit(1)
        tickers = tickers_df['ticker'].dropna().unique().tolist()
    else:
        logging.error("No mode selected. Please use either --RefreshMode or --ColdStart.")
        parser.print_help()
        exit(1)
    
    # Define end_date as the expected latest trading day
    expected_latest_date = get_last_trading_day()
    
    # Download or refresh the stock data
    fetch_and_save_stock_data(
        tickers,
        DATA_DIRECTORY,
        start_date=START_DATE,
        end_date=expected_latest_date.strftime('%Y-%m-%d'),
        rate_limit=RATE_LIMIT,
        refresh=args.RefreshMode
    )
    
    logging.info("Data download and refresh process completed.")
