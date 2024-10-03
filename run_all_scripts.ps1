# Define the base path
$basePath = "C:\Users\Masam\Desktop\Stock-Market-LSTM"

# Define the log file path
$logFile = "$basePath\Data\logging\__run_all_scripts.log"
# Ensure the logging directory exists
$logDir = Split-Path $logFile -Parent
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force
}


# Start logging
Start-Transcript -Path $logFile -Append

try {
    # Change directory to the base path to ensure relative paths work
    Set-Location $basePath -ErrorAction Stop

    Write-Host "Starting the stock market data download and processing sequence..."

    # Specify the full path to the Python executable
    $pythonExe = "C:\Users\Masam\miniconda3\envs\tf\python.exe"

    # Define the scripts and their respective arguments
    $scripts = @(
        @{
            Name = "Ticker Downloader"
            File = "1__TickerDownloader.py"
            Args = @("--ImmediateDownload")
        },
        @{
            Name = "Bulk Price Downloader"
            File = "2__BulkPriceDownloader.py"
            Args = @("--RefreshMode", "--ClearOldData")
        },
        @{
            Name = "Indicators Script"
            File = "3__Indicators.py"
            Args = @()
        },
        @{
            Name = "Predictor Script"
            File = "4__Predictor.py"
            Args = @("--predict")
        },
        @{
            Name = "Nightly Broker Script"
            File = "5__NightlyBroker.py"
            Args = @("--RunStocks", "1")
        }
    )

    foreach ($script in $scripts) {
        # Add a timer for each script
        $timer = [System.Diagnostics.Stopwatch]::StartNew()

        Write-Host "Running $($script.Name)..."

        # Construct the argument list
        $argList = @($script.File) + $script.Args

        # Execute the Python script and capture the output
        $output = & $pythonExe $argList 2>&1

        if ($LASTEXITCODE -ne 0) {
            Write-Host "Error running $($script.Name). Exit code: $LASTEXITCODE" -ForegroundColor Red
            Write-Host "Output:`n$output" -ForegroundColor Red
        } else {
            Write-Host "$($script.Name) completed successfully." -ForegroundColor Green
            Write-Host "Time elapsed: $([math]::Round($timer.Elapsed.TotalSeconds,2)) seconds." -ForegroundColor Cyan
        }

        # Optional: Log a separator for readability
        Write-Host "----------------------------------------"

        # Pause briefly between scripts to ensure proper execution
        Start-Sleep -Seconds 5
    }

    Write-Host "This Week's Buy Signals Saved to trading_data.parquet" -ForegroundColor Yellow
} catch {
    Write-Host "An error occurred: $_" -ForegroundColor Red
} finally {
    # Stop logging
    Stop-Transcript
}
