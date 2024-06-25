# Define the base path
$basePath = "C:\apitesting\Stock-Market-LSTM"

# Define the log file path
$logFile = "$basePath\run_all_scripts.log"

# Start logging
Start-Transcript -Path $logFile -Append

# Change directory to the base path to ensure relative paths work
cd $basePath

Write-Host "Starting the stock market data download and processing sequence..."

Write-Host "Running Ticker Downloader..."
C:\Users\Masam\miniconda3\envs\tf\python.exe .\1__TickerDownloader.py --ImmediateDownload
Write-Host "Ticker Downloader completed. Waiting before next step..."
Start-Sleep -Seconds 5

Write-Host "Running Bulk Price Downloader..."
C:\Users\Masam\miniconda3\envs\tf\python.exe .\2__BulkPriceDownloader.py --ClearOldData --RefreshMode
Write-Host "Bulk Price Downloader completed. Waiting before next step..."
Start-Sleep -Seconds 30

Write-Host "Running Indicators script..."
C:\Users\Masam\miniconda3\envs\tf\python.exe .\3__Indicators.py
Write-Host "Indicators script completed. Waiting before next step..."
Start-Sleep -Seconds 30

Write-Host "Running Predictor script..."
C:\Users\Masam\miniconda3\envs\tf\python.exe .\4__Predictor.py --predict
Write-Host "Predictor script completed. Waiting before next step..."
Start-Sleep -Seconds 60

Write-Host "Running Nightly broker script..."
C:\Users\Masam\miniconda3\envs\tf\python.exe .\7.2__NightlyBroker.py --RunStocks 1
Write-Host "This Weeks Buy Signals Saved to BuySignals.json"

# Stop logging
Stop-Transcript
