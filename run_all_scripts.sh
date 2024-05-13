#!/bin/bash
# Stop on any error
set -e
set -x  # Enable printing of commands and their arguments as they are executed

# Define the base path
BASE_PATH="/root/Stock-Market-LSTM"

echo "Starting the stock market data download and processing sequence..."
cd $BASE_PATH  # Change directory to the base path to ensure relative paths work


echo "Running Ticker Downloader..."
/root/root/miniconda4/envs/tf/bin/python ./1__TickerDownloader.py --ImmediateDownload
echo "Ticker Downloader completed. Waiting 10 seconds before next step..."
sleep 10  

echo "Running Bulk Price Downloader..."
/root/root/miniconda4/envs/tf/bin/python ./2__BulkPriceDownloader.py --ClearOldData --PercentDownload 40
echo "Bulk Price Downloader completed. Waiting approximately 1 minute before next step..."
sleep 60  

echo "Running Indicators script..."
/root/root/miniconda4/envs/tf/bin/python ./3__Indicators.py
echo "Indicators script completed. Waiting approximately 1 minute before next step..."
sleep 60  

echo "Running Predictor script..."
/root/root/miniconda4/envs/tf/bin/python ./4__Predictor.py --predict 5
echo "Predictor script completed. Waiting approximately 1 minute before next step..."
sleep 60  

