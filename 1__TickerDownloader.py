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

Notes:
    The correct API useage includes user_agent as "{reason for use} {email}".
    url": "https://www.sec.gov/files/company_tickers_exchange.json" may be subject to change in the future
"""


CONFIG = {
    "url": "https://www.sec.gov/files/company_tickers_exchange.json",
    "csv_file_path": "Data/TickerCikData/TickerCIKs_{date}.csv",
    "parquet_file_path": "Data/TickerCikData/TickerCIKs_{date}.parquet",
    "log_file": "Data/TickerCikData/TickerCIK.log",
    "user_agent": "PersonalTesting Masamunex9000@gmail.com"
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
    """
    Download ticker and CIK data from SEC and save in both CSV.
    """
    try:
        # Insert the current date in the file paths
        current_date = datetime.datetime.now().strftime("%Y%m%d")
        csv_file_path = CONFIG["csv_file_path"].format(date=current_date)
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
        df.to_csv(csv_file_path, index=False)

        logging.info("File downloaded and saved successfully in CSV.")
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
