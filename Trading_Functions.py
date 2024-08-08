import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

PARQUET_FILE = 'trading_data.parquet'

# Define the DataFrame columns
COLUMNS = [
    'Symbol', 'LastBuySignalDate', 'LastBuySignalPrice', 'IsCurrentlyBought', 
    'ConsecutiveLosses', 'LastTradedDate', 'UpProbability'
]

def initialize_parquet():
    """Create or reset the Parquet file with an empty DataFrame."""
    df = pd.DataFrame(columns=COLUMNS)
    df.to_parquet(PARQUET_FILE, index=False)

def read_trading_data():
    """Read all trading data from the Parquet file."""
    if not os.path.exists(PARQUET_FILE):
        initialize_parquet()
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_parquet(PARQUET_FILE)

def write_trading_data(df):
    """Write all trading data to the Parquet file."""
    df.to_parquet(PARQUET_FILE, index=False)

def update_buy_signal(symbol, date, price, up_probability):
    """Update or add a buy signal for a symbol."""
    df = read_trading_data()
    new_data = pd.DataFrame({
        'Symbol': [symbol],
        'LastBuySignalDate': [date],
        'LastBuySignalPrice': [price],
        'IsCurrentlyBought': [False],
        'ConsecutiveLosses': [0],
        'LastTradedDate': [None],
        'UpProbability': [up_probability]
    })
    # Remove existing entry for this symbol and concatenate the new data
    df = pd.concat([df[df['Symbol'] != symbol], new_data], ignore_index=True)
    write_trading_data(df)

def mark_position_as_bought(symbol):
    """Mark a symbol as currently bought."""
    df = read_trading_data()
    df.loc[df['Symbol'] == symbol, 'IsCurrentlyBought'] = True
    write_trading_data(df)

def update_trade_result(symbol, is_loss):
    """Update the consecutive losses and last traded date for a symbol."""
    df = read_trading_data()
    if symbol in df['Symbol'].values:
        if is_loss:
            df.loc[df['Symbol'] == symbol, 'ConsecutiveLosses'] += 1
        else:
            df.loc[df['Symbol'] == symbol, 'ConsecutiveLosses'] = 0
        df.loc[df['Symbol'] == symbol, 'LastTradedDate'] = datetime.now()
        df.loc[df['Symbol'] == symbol, 'IsCurrentlyBought'] = False
        write_trading_data(df)

def get_buy_signals(days=14):
    """Get buy signals from the last 14 days that haven't been bought."""
    df = read_trading_data()
    current_date = datetime.now()
    date_threshold = current_date - timedelta(days=days)
    
    signals = df[
        (df['IsCurrentlyBought'] == False) & 
        (df['LastBuySignalDate'] >= date_threshold)
    ].to_dict('records')
    
    return signals

def get_open_positions():
    """Get all currently bought positions."""
    df = read_trading_data()
    return df[df['IsCurrentlyBought'] == True]['Symbol'].tolist()

def calculate_position_size(total_value, cash, reserve_percent, max_positions, current_price):
    """Calculate the position size for a new trade."""
    workable_capital = total_value * (1 - reserve_percent)
    capital_per_position = workable_capital / max_positions
    
    if cash < capital_per_position or cash < (total_value * reserve_percent):
        return 0
    
    return int(capital_per_position / current_price) - 1

def should_sell(current_price, entry_price, entry_date, current_date, 
                stop_loss_percent=3.0, take_profit_percent=100.0, 
                max_hold_days=9, min_daily_return=0.25):
    """Determine if a position should be sold based on various criteria."""
    days_held = (current_date - entry_date).days
    profit_percent = (current_price - entry_price) / entry_price * 100
    
    stop_loss = current_price <= entry_price * (1 - stop_loss_percent / 100)
    take_profit = current_price >= entry_price * (1 + take_profit_percent / 100)
    max_hold = days_held > max_hold_days
    low_return = days_held > 3 and profit_percent / days_held < min_daily_return * days_held
    
    return stop_loss or take_profit or max_hold or low_return

def prioritize_buy_signals(buy_signals, correlation_data):
    """Prioritize buy signals based on group correlations and up probability."""
    merged_data = pd.merge(buy_signals, correlation_data, on='Symbol', how='left')
    group_allocations = merged_data.groupby('Cluster')['IsCurrentlyBought'].sum() / len(merged_data)
    merged_data['GroupAllocation'] = merged_data['Cluster'].map(group_allocations)
    return merged_data.sort_values(['GroupAllocation', 'UpProbability'], ascending=[True, False])

def get_cooldown_end(consecutive_losses):
    """Get the cooldown end date based on consecutive losses."""
    cooldown_days = {1: 7, 2: 28, 3: 90, 4: 282}
    days = cooldown_days.get(consecutive_losses, 282 if consecutive_losses >= 4 else 0)
    return datetime.now().date() + timedelta(days=days)

def is_in_cooldown(trading_data, symbol, current_date):
    """Check if a symbol is in cooldown period."""
    if symbol not in trading_data.index:
        return False
    
    consecutive_losses = trading_data.loc[symbol, 'ConsecutiveLosses']
    last_traded_date = trading_data.loc[symbol, 'LastTradedDate']
    
    if pd.isnull(last_traded_date):
        return False
    
    cooldown_end = get_cooldown_end(consecutive_losses)
    return current_date <= cooldown_end

def check_group_correlation(symbol, current_positions, correlation_data, max_positions_per_cluster=2):
    """Check if adding a new position would violate group correlation limits."""
    if symbol not in correlation_data.index:
        return True  # If we don't have correlation data, assume it's safe to trade

    symbol_cluster = correlation_data.loc[symbol, 'Cluster']
    current_clusters = [correlation_data.loc[pos, 'Cluster'] for pos in current_positions if pos in correlation_data.index]
    
    # Count how many positions are already in the same cluster
    cluster_count = sum(1 for cluster in current_clusters if cluster == symbol_cluster)
    
    return cluster_count < max_positions_per_cluster

def get_consecutive_losses(symbol):
    """Get the number of consecutive losses for a symbol."""
    df = read_trading_data()
    if symbol in df['Symbol'].values:
        return df.loc[df['Symbol'] == symbol, 'ConsecutiveLosses'].iloc[0]
    return 0

def get_days_since_last_trade(symbol):
    """Get the number of days since the last trade for a symbol."""
    df = read_trading_data()
    if symbol in df['Symbol'].values and pd.notnull(df.loc[df['Symbol'] == symbol, 'LastTradedDate'].iloc[0]):
        last_trade_date = df.loc[df['Symbol'] == symbol, 'LastTradedDate'].iloc[0]
        return (datetime.now() - last_trade_date).days
    return None

def colorize_output(value, label, good_threshold, bad_threshold, lower_is_better=False, reverse=False):
    def get_color_code(normalized_value):
        # Direct transition from red to green
        red = int(255 * (1 - normalized_value))
        green = int(255 * normalized_value)
        return f"\033[38;2;{red};{green};0m"

    # Adjust thresholds and normalization for reverse and lower_is_better scenarios
    if reverse:
        good_threshold, bad_threshold = bad_threshold, good_threshold

    if lower_is_better:
        if value <= good_threshold:
            color_code = "\033[92m"  # Bright Green
        elif value >= bad_threshold:
            color_code = "\033[91m"  # Bright Red
        else:
            range_span = bad_threshold - good_threshold
            normalized_value = (value - good_threshold) / range_span
            color_code = get_color_code(1 - normalized_value)
    else:
        if value >= good_threshold:
            color_code = "\033[92m"  # Bright Green
        elif value <= bad_threshold:
            color_code = "\033[91m"  # Bright Red
        else:
            range_span = good_threshold - bad_threshold
            normalized_value = (value - bad_threshold) / range_span
            color_code = get_color_code(normalized_value)

    return f"{label:<30}{color_code}{value:.2f}\033[0m"



# Initialize the Parquet file if it doesn't exist
if not os.path.exists(PARQUET_FILE):
    initialize_parquet()
