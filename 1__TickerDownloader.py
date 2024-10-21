#!/root/root/miniconda4/envs/tf/bin/python
import datetime
import logging
import os
import requests
import pandas as pd
import argparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import sys

CONFIG = {
    "url": "https://www.sec.gov/files/company_tickers_exchange.json",
    "parquet_file_path": "Data/TickerCikData/TickerCIKs_{date}.parquet",
    "log_file": "data/logging/1__TickerDownloader.log",
    "user_agent": "MarketAnalysis Masamunex9000@gmail.com"
}

def setup_logging(verbose=False):
    log_dir = os.path.dirname(CONFIG["log_file"])
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(filename=CONFIG["log_file"], level=level, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    # Also log to console
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    logging.getLogger('').addHandler(console)

def setup_args():
    parser = argparse.ArgumentParser(description="Download and convert Ticker CIK data.")
    parser.add_argument("--ImmediateDownload", action='store_true', 
                        help="Download the file immediately without waiting for the scheduled time.")
    parser.add_argument("-v", "--verbose", action='store_true', 
                        help="Increase output verbosity")
    args = parser.parse_args()
    return args

def download_and_convert_ticker_cik_file():
    try:
        logging.debug("Starting download_and_convert_ticker_cik_file function")
        current_date = datetime.datetime.now().strftime("%Y%m%d")
        parquet_file_path = CONFIG["parquet_file_path"].format(date=current_date)
        logging.debug(f"Parquet file path: {parquet_file_path}")

        session = requests.Session()
        retry = Retry(total=5, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        
        headers = {
            'User-Agent': CONFIG["user_agent"],
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'www.sec.gov'
        }
        logging.debug(f"Making request to {CONFIG['url']}")
        response = session.get(CONFIG["url"], headers=headers, timeout=30)
        response.raise_for_status()
        logging.debug("Request successful")

        json_data = response.json()
        logging.debug(f"Received JSON data with {len(json_data['data'])} entries")
        df = pd.DataFrame(json_data['data'], columns=json_data['fields'])
        logging.debug(f"Created DataFrame with shape {df.shape}")

        df = df[df['exchange'].notna()]
        df = df[~df['exchange'].isin(['OTC', 'CBOE'])]
        logging.debug(f"Filtered DataFrame, new shape: {df.shape}")

        os.makedirs(os.path.dirname(parquet_file_path), exist_ok=True)
        df.to_parquet(parquet_file_path, index=False)
        logging.info(f"File saved successfully: {parquet_file_path}")

    except requests.exceptions.RequestException as e:
        logging.error(f"Request error occurred: {e}")
        raise
    except Exception as e:
        logging.error(f"Error occurred: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    args = setup_args()
    setup_logging(args.verbose)

    logging.info("Script started")
    if args.ImmediateDownload:
        logging.info("Immediate download requested")
        download_and_convert_ticker_cik_file()
    else:
        logging.info("No immediate download requested")
    logging.info("Script completed")