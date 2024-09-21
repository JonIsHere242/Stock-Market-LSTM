# Define the base path
$basePath = "C:\apitesting\Stock-Market-LSTM"

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

    # Activate the Conda environment
    $env:Path = "C:\Users\Masam\miniconda3\envs\tf;C:\Users\Masam\miniconda3\envs\tf\Scripts;" + $env:Path

    $scripts = @(
        @{Name="Ticker Downloader"; File="1__TickerDownloader.py"; Args=@("--ImmediateDownload")},
        @{Name="Bulk Price Downloader"; File="2__BulkPriceDownloader.py"; Args=@("--RefreshMode")},  # Changed arguments
        @{Name="Indicators script"; File="3__Indicators.py"; Args=@()},
        @{Name="Predictor script"; File="4__Predictor.py"; Args=@("--predict")},
        @{Name="Nightly broker script"; File="5__NightlyBroker.py"; Args=@("--RunStocks", "1")}
    )

    foreach ($script in $scripts) {
        # Add a timer for each script
        $timer = [System.Diagnostics.Stopwatch]::StartNew()

        Write-Host "Running $($script.Name)..."
        $argList = @($script.File) + $script.Args
        $output = & python $argList 2>&1
        if ($LASTEXITCODE -ne 0) {
            Write-Host "Error running $($script.Name). Exit code: $LASTEXITCODE"
            Write-Host "Output: $output"
        } else {
            Write-Host "$($script.Name) completed successfully."
            Write-Host "Time elapsed: $([math]::Round($timer.Elapsed.TotalSeconds,2)) seconds."
        }
        Start-Sleep -Seconds 10
    }

    Write-Host "This Week's Buy Signals Saved to trading_data.parquet"
} catch {
    Write-Host "An error occurred: $_"
} finally {
    # Stop logging
    Stop-Transcript
}
