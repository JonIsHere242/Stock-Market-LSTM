import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import datetime
import logging
import pandas as pd
import argparse
import os
import sys


"""
This script downloads and converts Ticker and CIK (Central Index Key) data from the U.S. Securities and Exchange Commission (SEC) website. 
The data is saved in both CSV and Parquet formats. The script is designed to be run either on a schedule or immediately based on a command-line argument.

Command-line Arguments:
    --immediate_download: If this argument is provided, the script will download the file immediately without waiting for the scheduled time. 
    This is useful for cases where an immediate update of the data is required, bypassing any scheduled checks or conditions.

Use Case:
    This script is ideal for scenarios where up-to-date Ticker and CIK data is required. It can be scheduled to run at specific intervals (e.g., weekly) 
    to ensure the data is current. The immediate download option provides flexibility for ad-hoc updates outside the regular schedule.

Example:
    To run the script immediately, use the following command:
    python 1__TickerDownloader.py --immediate_download
"""


CONFIG = {
    "url": "https://www.sec.gov/files/company_tickers_exchange.json",
    "csv_file_path": "Data/TickerCikData/TickerCIKs.csv",
    "parquet_file_path": "Data/TickerCikData/TickerCIKs.parquet",
    "log_file": "Data/TickerCikData/TickerCIK.log",
    "user_agent": "PersonalTesting Masamunex9000@gmail.com"
}

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(filename=CONFIG["log_file"], level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')

def download_and_convert_ticker_cik_file():
    """
    Download ticker and CIK data from SEC and save in both CSV and Parquet formats.
    """
    try:
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))

        headers = {
            'User-Agent': CONFIG["user_agent"],
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'www.sec.gov'
        }

        response = session.get(CONFIG["url"], headers=headers)
        response.raise_for_status()

        json_data = response.json()
        df = pd.DataFrame(json_data['data'], columns=json_data['fields'])
        df.to_csv(CONFIG["csv_file_path"], index=False)
        df.to_parquet(CONFIG["parquet_file_path"], index=False)

        logging.info("File downloaded and saved successfully in CSV and Parquet formats.")
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

    parser = argparse.ArgumentParser(description="Download and convert Ticker CIK data.")
    parser.add_argument("--immediate_download", action='store_true', 
                        help="Download the file immediately without waiting for the scheduled time.")
    args = parser.parse_args()

    if args.immediate_download or datetime.date.today().weekday() == 5:  # 5 represents Saturday
        if is_file_recent(CONFIG["csv_file_path"]):
            time_since_last_mod = datetime.datetime.now() - datetime.datetime.fromtimestamp(os.path.getmtime(CONFIG["csv_file_path"]))
            logging.info(f"File already downloaded today. Exiting...File last modified {time_since_last_mod} ago")
            sys.exit(0)

        download_and_convert_ticker_cik_file()
