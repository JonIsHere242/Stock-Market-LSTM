# Define the base path where your Python scripts are located
$basePath = "C:\Users\Masam\Desktop\Stock-Market-LSTM"

# Define the log file path
$logFile = "$basePath\Data\logging\__run_all_scripts.log"

# Define the path to the Python executable
$pythonExe = "C:\Users\Masam\miniconda3\envs\tf\python.exe"

# Define the scripts to run with their respective arguments
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

# -----------------------------
# Function Definitions
# -----------------------------

# Function to write log messages
function Write-Log {
    param(
        [string]$Message,
        [ValidateSet("White", "Gray", "Red", "Green", "Yellow", "Cyan")]
        [string]$Color = "White"
    )
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] $Message"
    Write-Host $logMessage -ForegroundColor $Color
    Add-Content -Path $logFile -Value $logMessage
}

# -----------------------------
# Initialization
# -----------------------------

# Ensure the logging directory exists
$logDir = Split-Path $logFile -Parent
if (-not (Test-Path $logDir)) {
    try {
        New-Item -ItemType Directory -Path $logDir -Force | Out-Null
        Write-Log "Created logging directory at $logDir" -Color Green
    }
    catch {
        Write-Host "Failed to create logging directory at $logDir. Exiting script." -ForegroundColor Red
        exit 1
    }
}

# Clear the host screen and display startup messages
Clear-Host
Write-Host "Starting the execution of Python scripts..." -ForegroundColor Cyan
Write-Host "Logs will be saved to $logFile" -ForegroundColor Yellow
Start-Sleep -Seconds 2

# Start logging
Write-Log "=========================================" -Color White
Write-Log "Script execution started. Base path: $basePath" -Color White
Write-Log "=========================================" -Color White

# -----------------------------
# Script Execution Loop
# -----------------------------

foreach ($script in $scripts) {
    Write-Log "Preparing to run '$($script.Name)'..." -Color Cyan

    # Wait for 5 seconds before executing the script
    Start-Sleep -Seconds 30

    # Start timer
    $timer = [System.Diagnostics.Stopwatch]::StartNew()

    try {
        # Construct the full path to the Python script
        $scriptPath = Join-Path -Path $basePath -ChildPath $script.File

        # Verify that the Python script exists
        if (-not (Test-Path $scriptPath)) {
            throw "Script file not found: $scriptPath"
        }

        # Verify that the Python executable exists
        if (-not (Test-Path $pythonExe)) {
            throw "Python executable not found at $pythonExe"
        }

        # Construct the argument list
        $argList = @($scriptPath) + $script.Args

        # Log the execution command
        $command = "$pythonExe " + ($argList -join ' ')
        Write-Log "Executing: $command" -Color Gray

        # Create a new process start info
        $pinfo = New-Object System.Diagnostics.ProcessStartInfo
        $pinfo.FileName = $pythonExe
        $pinfo.Arguments = ($argList -join ' ')
        $pinfo.UseShellExecute = $false
        $pinfo.RedirectStandardOutput = $true
        $pinfo.RedirectStandardError = $true
        $pinfo.CreateNoWindow = $true

        # Create and start the process
        $process = New-Object System.Diagnostics.Process
        $process.StartInfo = $pinfo
        
        # Set up event handlers for output and error streams
        $stdoutBuilder = New-Object System.Text.StringBuilder
        $stderrBuilder = New-Object System.Text.StringBuilder
        
        $process.OutputDataReceived = {
            param($sender, $e)
            if ($null -ne $e.Data) {
                $stdoutBuilder.AppendLine($e.Data)
                Write-Host $e.Data
                Add-Content -Path $logFile -Value $e.Data
            }
        }
        
        $process.ErrorDataReceived = {
            param($sender, $e)
            if ($null -ne $e.Data) {
                $stderrBuilder.AppendLine($e.Data)
                Write-Host $e.Data -ForegroundColor Red
                Add-Content -Path $logFile -Value "ERROR: $($e.Data)"
            }
        }

        # Start the process and begin reading output
        $process.Start() | Out-Null
        $process.BeginOutputReadLine()
        $process.BeginErrorReadLine()
        
        # Wait for the process to exit
        $process.WaitForExit()

        # Check the exit code
        if ($process.ExitCode -ne 0) {
            throw "Script '$($script.Name)' exited with code $($process.ExitCode)"
        }

        # Log successful execution
        Write-Log "'$($script.Name)' completed successfully." -Color Green
    }
    catch {
        # Log the error details
        Write-Log "Error running '$($script.Name)'. Error details:" -Color Red
        Write-Log $_.Exception.Message -Color Red
    }
    finally {
        # Stop the timer and log the elapsed time
        $timer.Stop()
        $elapsed = [math]::Round($timer.Elapsed.TotalSeconds, 2)
        Write-Log "Time elapsed for '$($script.Name)': $elapsed seconds." -Color Cyan
        Write-Log "-----------------------------------------" -Color White
    }
}

# -----------------------------
# Completion
# -----------------------------

Write-Log "All scripts have been executed." -Color Yellow
Write-Log "This Week's Buy Signals Saved to trading_data.parquet" -Color Yellow
Write-Log "Script execution completed." -Color Yellow
Write-Log "=========================================" -Color White

# Prompt to exit
Write-Host "All tasks completed. Press any key to exit..." -ForegroundColor Green
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")