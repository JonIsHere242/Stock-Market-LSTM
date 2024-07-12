# Define the base path
$basePath = "C:\apitesting\Stock-Market-LSTM"

# Define the log file path
$logFile = "$basePath\run_all_scripts.log"

# Get the current day of the week (1 = Monday, 7 = Sunday)
$currentDay = (Get-Date).DayOfWeek.value__

# Check if it's a weekday (Monday to Friday)
if ($currentDay -ge 1 -and $currentDay -le 5) {
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
            @{Name="Bulk Price Downloader"; File="2__BulkPriceDownloader.py"; Args=@("--ClearOldData", "--RefreshMode")},
            @{Name="Indicators script"; File="3__Indicators.py"; Args=@()},
            @{Name="Predictor script"; File="4__Predictor.py"; Args=@("--predict", "--feature_cut", "30")},
            @{Name="Nightly broker script"; File="5__NightlyBroker.py"; Args=@("--RunStocks", "1")}
        )

        foreach ($script in $scripts) {
            Write-Host "Running $($script.Name)..."
            $argList = @($script.File) + $script.Args
            $output = & python $argList 2>&1
            if ($LASTEXITCODE -ne 0) {
                Write-Host "Error running $($script.Name). Exit code: $LASTEXITCODE"
                Write-Host "Output: $output"
            } else {
                Write-Host "$($script.Name) completed successfully."
            }
            Start-Sleep -Seconds 30
        }

        Write-Host "This days Buy Signals Saved to live_trading.db buy_signals table"
    } catch {
        Write-Host "An error occurred: $_"
    } finally {
        # Stop logging
        Stop-Transcript
    }
} else {
    Write-Host "Today is a weekend. The script has already been run before."
}


