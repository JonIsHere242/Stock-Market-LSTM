#!/bin/bash
# Stop on any error
set -e

echo "Starting the stock market data download and processing sequence..."

##Run Python scripts sequentially
echo "Running Ticker Downloader..."
/root/root/miniconda4/envs/tf/bin/python /root/Stock-Market-LSTM/1__TickerDownloader.py --ImmediateDownload
echo "Ticker Downloader completed. Waiting 10 seconds before next step..."
sleep 10  # Waits for 60 seconds to ensure the previous task completes

echo "Running Bulk Price Downloader..."
/root/root/miniconda4/envs/tf/bin/python /root/Stock-Market-LSTM/2__BulkPriceDownloader.py --ClearOldData --PercentDownload 0.5
echo "Bulk Price Downloader completed. Waiting approximately 1 minute before next step..."
sleep 60  # Waits for 13000 seconds

echo "Running Indicators script..."
/root/root/miniconda4/envs/tf/bin/python /root/Stock-Market-LSTM/3__Indicators.py
echo "Indicators script completed. Waiting approximately 1 minute before next step..."
sleep 60  # Waits for 13000 seconds

echo "Running Predictor script..."
/root/root/miniconda4/envs/tf/bin/python /root/Stock-Market-LSTM/4__Predictor.py --runpercent 100 --predict
echo "Predictor script completed. Waiting approximately 1 minute before next step..."
sleep 60  # Waits for 1000 seconds

echo "Running Backtesting script..."
/root/root/miniconda4/envs/tf/bin/python /root/Stock-Market-LSTM/6__Backtesting.py --RandomFilePercent 100
echo "Daily Backtesting script completed"


echo "All tasks completed successfully!"

