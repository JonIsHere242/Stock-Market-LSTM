#!/root/root/miniconda4/envs/tf/bin/python
import datetime
import logging
import os
import requests
import pandas as pd
import argparse
import sys
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

CONFIG = {
    "url": "https://www.sec.gov/files/company_tickers_exchange.json",
    "parquet_file_path": "Data/TickerCikData/TickerCIKs_{date}.parquet",
    "log_file": "Data/TickerCikData/TickerCIK.log",
    "user_agent": "MarketAnalysis Masamunex9000@gmail.com"
}

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(filename=CONFIG["log_file"], level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

def setup_args():
    parser = argparse.ArgumentParser(description="Download and convert Ticker CIK data.")
    parser.add_argument("--ImmediateDownload", action='store_true', 
                        help="Download the file immediately without waiting for the scheduled time.")
    args = parser.parse_args()
    return args

def download_and_convert_ticker_cik_file():
    try:
        # Insert the current date in the file paths
        current_date = datetime.datetime.now().strftime("%Y%m%d")
        parquet_file_path = CONFIG["parquet_file_path"].format(date=current_date)
        session = requests.Session()
        headers = {
            'User-Agent': CONFIG["user_agent"],
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'www.sec.gov'
        }

        response = session.get(CONFIG["url"], headers=headers)
        response.raise_for_status()

        json_data = response.json()
        df = pd.DataFrame(json_data['data'], columns=json_data['fields'])
        df.to_parquet(parquet_file_path, index=False)

        logging.info("File downloaded and saved successfully in Parquet format.")
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP error occurred: {e}")
    except Exception as e:
        logging.error(f"Error occurred: {e}")

def is_file_recent(file_path):
    """
    Check if the file was modified today.

    Args:
        file_path (str): Path of the file to check.

    Returns:
        bool: True if the file was modified today, False otherwise.
    """
    if os.path.exists(file_path):
        last_modified = datetime.datetime.fromtimestamp(os.path.getmtime(file_path)).date()
        return last_modified == datetime.date.today()
    return False

if __name__ == "__main__":
    setup_logging()
    args = setup_args()

    if args.ImmediateDownload:
        download_and_convert_ticker_cik_file()
