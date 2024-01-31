import yfinance as yf
import pandas as pd
import os
import logging
import time
from datetime import datetime

# Configuration
DATA_DIRECTORY = 'Data/PriceData'
TICKERS_CIK_FILE = 'Data/TickerCikData/TickerCIKs.csv'
LOG_DIRECTORY = 'Data/PriceData'
LOG_FILE = "Data/PriceData/_Price_Data_download.log"
START_DATE = '1990-01-01'
MAX_DOWNLOADS = 20
RATE_LIMIT = 1.0  # seconds between downloads 5% is added when MAX_DOWNLOADS is not None, 15% when running full script

os.makedirs(DATA_DIRECTORY, exist_ok=True)
os.makedirs(LOG_DIRECTORY, exist_ok=True)
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_and_save_stock_data(tickers, data_dir, start_date=START_DATE, max_downloads=None, rate_limit=1.0):
    """
    Fetches and saves historical stock data for a list of tickers from Yahoo Finance.

    Args:
        tickers (list): List of ticker symbols.
        data_dir (str): Directory where the stock data files will be saved.
        start_date (str): Start date for historical data in 'YYYY-MM-DD' format.
        max_downloads (int, optional): Maximum number of tickers to download data for.
        rate_limit (float): Base time in seconds to wait between downloads.
    """
    download_count = 0
    wait_time = rate_limit * 1.05 if max_downloads is not None else rate_limit * 1.15
    max_downloads = max_downloads or len(tickers)
    for ticker in tickers:
        if download_count >= max_downloads:
            logging.info("Maximum number of downloads reached.")
            total_tickers = len(tickers)
            percent_of_total_tickers = (len(os.listdir(data_dir))/total_tickers)*100
            logging.info(f"Total number of tickers: {percent_of_total_tickers}")

            break

        file_path = os.path.join(data_dir, f"{ticker}.csv")

        if os.path.exists(file_path):
            continue

        try:
            start_download_time = time.time()
            stock_data = yf.download(ticker, start=start_date, progress=False)

            if stock_data.empty:
                logging.warning(f"No data found for {ticker}")
                continue

            stock_data.to_csv(file_path)
            logging.info(f"Data for {ticker} downloaded and saved to {file_path}")

            download_count += 1
            end_download_time = time.time()
            elapsed_time = end_download_time - start_download_time
            adjusted_wait_time = max(wait_time - elapsed_time, 0)
            logging.info(f"Waiting {adjusted_wait_time:.2f} seconds before next download.")
            time.sleep(adjusted_wait_time)

        except Exception as e:
            logging.error(f"Error downloading data for {ticker}: {e}")

if __name__ == "__main__":
    try:
        df = pd.read_csv(TICKERS_CIK_FILE)
        tickers = df['ticker'].unique().tolist()
        fetch_and_save_stock_data(tickers, DATA_DIRECTORY, max_downloads=MAX_DOWNLOADS,start_date=START_DATE, rate_limit=RATE_LIMIT)
    except Exception as e:
        logging.error(f"Error reading tickers file: {e}")
