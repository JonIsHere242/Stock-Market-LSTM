import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import csv

def load_data(file_path):
    cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Distance to Support (%)', 'Distance to Resistance (%)', 'UpProbability', 'UpPrediction']
    df = pd.read_parquet(file_path, columns=cols)
    df.set_index('Date', inplace=True)
    return df

def calculate_mean_percentage_change(df):
    changes = [((df['Close'].iloc[-i] - df['Close'].iloc[-i-1]) / df['Close'].iloc[-i-1]) * 100 for i in range(1, 8)]
    return np.mean(changes)

def check_buy_signal(df):
    if len(df) < 15:  # Ensure there is enough data to calculate moving averages and percentiles
        return False

    high_up_prob = np.percentile(df['UpProbability'].values, 90)
    up_prediction = df['UpPrediction'].iloc[-1] == 1
    dist_to_support = df['Distance to Support (%)'].iloc[-1] < 50
    dist_to_resistance = df['Distance to Resistance (%)'].iloc[-1] > 100
    recent_mean_percentage_change = calculate_mean_percentage_change(df)
    current_up_prob = df['UpProbability'].iloc[-1] > 0.53
    up_prob_ma = df['UpProbability'].rolling(window=14).mean().iloc[-1] > high_up_prob

    return (up_prob_ma and up_prediction and dist_to_support and dist_to_resistance and recent_mean_percentage_change > 1.5 and current_up_prob)

def main():
    data_directory = 'Data/RFpredictions'
    csv_file = 'BuySignals.csv'
    columns = ['Ticker', 'Date', 'isBought', 'sharesHeld', 'TimeSinceBought', 'HasHeldPreviously', 'WinLossPercentage']

    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(columns)

    buy_signals = pd.read_csv(csv_file) if os.path.exists(csv_file) else pd.DataFrame(columns=columns)

    files = [f for f in os.listdir(data_directory) if f.endswith('.parquet')]
    
    for file_name in tqdm(files, desc="Processing files"):
        file_path = os.path.join(data_directory, file_name)
        df = load_data(file_path)
        ticker = os.path.basename(file_path).replace('.parquet', '')

        if check_buy_signal(df):
            current_date = df.index[-1].date()
            new_signal = {
                'Ticker': ticker,
                'Date': str(current_date),
                'isBought': False,
                'sharesHeld': 0,
                'TimeSinceBought': 0,
                'HasHeldPreviously': False,
                'WinLossPercentage': 0.0
            }

            if ticker not in buy_signals['Ticker'].values:
                buy_signals = pd.concat([buy_signals, pd.DataFrame([new_signal])], ignore_index=True)

    buy_signals.to_csv(csv_file, index=False)

if __name__ == "__main__":
    main()
