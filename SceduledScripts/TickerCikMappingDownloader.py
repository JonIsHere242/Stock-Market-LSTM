import requests
import datetime
import logging
import pandas as pd
import json
import argparse

# Setting up logging
log_file = "Data/TickerCikData/testing.log"
logging.basicConfig(filename=log_file, level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# User-Agent String
user_agent = 'PersonalTesting Masamunex9000@gmail.com'

def download_and_convert_ticker_cik_file(url, csv_save_path, parquet_save_path):
    try:
        headers = {
            'User-Agent': user_agent,
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'www.sec.gov'
        }
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            json_data = response.json()

            # Converting JSON to DataFrame
            df = pd.DataFrame(json_data['data'], columns=json_data['fields'])

            # Save to CSV
            df.to_csv(csv_save_path, index=False)

            # Save to Parquet
            df.to_parquet(parquet_save_path, index=False)

            logging.info("File downloaded and saved successfully in CSV and Parquet formats.")
        else:
            logging.error(f"Failed to download file: Status code {response.status_code}")
    except Exception as e:
        logging.error(f"Error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--immediate_download", action='store_true', 
                        help="Download the file immediately without waiting for the scheduled time")
    args = parser.parse_args()

    today = datetime.date.today()
    # Check if immediate download is requested or if today is Saturday
    if args.immediate_download or today.weekday() == 5:  # 5 represents Saturday
        url = "https://www.sec.gov/files/company_tickers_exchange.json"  # Adjusted the URL
        csv_save_path = "Data/TickerCikData/TickerCIKs.csv"  # Path for the CSV file
        parquet_save_path = "Data/TickerCikData/TickerCIKs.parquet"  # Path for the Parquet file
        download_and_convert_ticker_cik_file(url, csv_save_path, parquet_save_path)
