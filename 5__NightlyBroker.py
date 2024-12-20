#!/root/root/miniconda4/envs/tf/bin/python
import os
import time
import logging
import argparse
import random
import pandas as pd
import numpy as np
from backtrader.feeds import PandasData
import backtrader as bt
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
from tqdm import tqdm
from numba import jit
from functools import partial
import pyarrow.parquet as pq
from tqdm.asyncio import tqdm_asyncio
import multiprocessing
from functools import partial
from numba import njit
from functools import lru_cache
import traceback
from collections import Counter
from Trading_Functions import *
import sys
import pandas_market_calendars as mcal
from datetime import datetime, timedelta
import math


def setup_logging():
    """Set up logging configuration with detailed error handling and debugging."""
    try:
        log_file = 'Data/logging/5__NightlyBroker.log'
        log_dir = os.path.dirname(log_file)

        print(f"Log directory path: {log_dir}")
        print(f"Full log file path: {log_file}")
        print(f"Current working directory: {os.getcwd()}")

        print(f"Attempting to create log directory: {log_dir}")
        os.makedirs(log_dir, exist_ok=True)
        
        if os.path.exists(log_dir):
            print(f"Log directory created/exists: {log_dir}")
        else:
            print(f"Failed to create log directory: {log_dir}")

        print(f"Setting up logging to file: {log_file}")
        
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            filemode='a'  # Append mode
        )
        
        # Test log write
        logging.info("Logging setup completed successfully.")
        
        if os.path.exists(log_file):
            print(f"Log file created successfully: {log_file}")
            print(f"File size: {os.path.getsize(log_file)} bytes")
        else:
            print(f"Failed to create log file: {log_file}")

        print("Logging setup completed. Check the log file for a test message.")
    except Exception as e:
        print(f"Error setting up logging: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Python version: {sys.version}")
        print(f"OS: {os.name}")
        print("Directory contents:")
        for root, dirs, files in os.walk('.'):
            for name in files:
                print(os.path.join(root, name))






@njit
def fast_calculate_recent_mean_percentage_change(close_prices):
    diff = close_prices[1:] - close_prices[:-1]
    percentage_changes = diff / close_prices[:-1] * 100
    return np.mean(percentage_changes)




def arg_parser():
    parser = argparse.ArgumentParser(description="Backtesting Trading Strategy on Stock Data")
    parser.add_argument("--BrokerTest", type=float, default=0, help="Percentage of random files to backtest")
    parser.add_argument("--RunStocks", type=float, default=0, help="Percentage of random files to backtest")
    parser.add_argument("--force", action='store_true', help="Force the script to run even if data is not up to last trading date")
    return parser.parse_args()


class DailyPercentageRange(bt.Indicator):
    lines = ('pct_range',)

    plotinfo = {
        "plot": True,
        "subplot": True
    }

class PercentileIndicator(bt.Indicator):
    lines = ('high_up_prob', 'low_down_prob')
    params = (('period', 7),)  # Window size for percentile calculation

    def __init__(self):
        self.addminperiod(self.params.period)

    def next(self):
        up_probs = self.data0.UpProb.get(size=self.params.period)
        down_probs = self.data0.DownProb.get(size=self.params.period)
        self.lines.high_up_prob[0] = np.percentile(up_probs, 95)
        self.lines.low_down_prob[0] = np.percentile(down_probs, 95)

class ATRPercentage(bt.Indicator):
    lines = ('atr_percent',)
    params = (('period', 14),)

    def __init__(self):
        self.atr = bt.indicators.AverageTrueRange(self.data, period=self.params.period)

    def next(self):
        self.lines.atr_percent[0] = (self.atr[0] / self.data.close[0]) * 100

correlation_data = pd.read_parquet('Correlations.parquet').set_index('Ticker')
buysignal_data = pd.read_parquet('_Buy_Signals.parquet').set_index('Symbol')

class CustomPandasData(bt.feeds.PandasData):
    lines = ('dist_to_support', 'dist_to_resistance', 'UpProbability', 'UpPrediction')
    params = (
        ('datetime', 'Date'),
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
        ('openinterest', None),
        ('dist_to_support', 'Distance to Support (%)'),
        ('dist_to_resistance', 'Distance to Resistance (%)'),
        ('UpProbability', 'UpProbability'),
        ('UpPrediction', 'UpPrediction'),
    )










class MovingAverageCrossoverStrategy(bt.Strategy):
    params = (
        ('days_range', 5),
        ('fast_period', 14),
        ('slow_period', 60),
        ('max_positions', 4),
        ('reserve_percent', 0.1),
        ('stop_loss_percent', 3.0),
        ('take_profit_percent', 100.0),
        ('position_timeout', 9),
        ('trailing_stop_percent', 5.0),
        ('expected_profit_per_day_percentage', 0.25),
        ('rolling_period', 8),
        ('max_group_allocation', 0.45),
        ('correlation_data', None),
        ('parquet_file', '_Buy_Signals.parquet'),
    )

    def __init__(self):
        self.inds = {d: {} for d in self.datas}
        self.order_list = []
        self.entry_prices = {}
        self.position_dates = {}
        self.order_cooldown = {}
        self.buying_status = {}
        self.consecutive_losses = {}
        self.cooldown_end = {}
        self.open_positions = 0
        self.asset_groups = {}
        self.asset_correlations = {}
        self.group_allocations = {}
        self.total_groups = self.detect_total_groups()
        self.correlation_data = self.params.correlation_data
        self.trading_data = read_trading_data()

        self.total_bars = 252
        self.progress_bar = tqdm(
            total=self.total_bars, 
            desc="Strategy Progress", 
            unit="day", 
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]',
            ncols=100
        )
        self.day_count = 0

        for d in self.datas:
            self.inds[d] = {}
            self.inds[d]['up_prob'] = d.UpProbability
            self.inds[d]['dist_to_support'] = d.dist_to_support
            self.inds[d]['dist_to_resistance'] = d.dist_to_resistance
            self.inds[d]['UpProbMA'] = bt.indicators.SimpleMovingAverage(d.UpProbability, period=self.p.fast_period)

    def next(self):
        self.day_count += 1
        self.progress_bar.update(1)
        
        current_date = self.datetime.date()
        
        if any(len(data) == 0 for data in self.datas):
            return

        sell_data = [d for d in self.datas if self.getposition(d).size > 0]
        for d in sell_data:
            self.evaluate_sell_conditions(d, current_date)

        if self.open_positions < self.params.max_positions:
            buy_candidates = self.get_buy_candidates(current_date)
            if buy_candidates:
                self.process_buy_candidates(buy_candidates, current_date)

        if self.day_count == self.total_bars:
            buy_candidates = self.get_buy_candidates(current_date)
            if buy_candidates:
                self.process_buy_candidates(buy_candidates, current_date)









    def detect_total_groups(self):
        correlation_data = pd.read_parquet('Correlations.parquet')
        return correlation_data['Cluster'].nunique()

    def notify_order(self, order):
        if order.status in [order.Completed, order.Partial]:
            self.handle_order_execution(order)
        elif order.status in [order.Canceled, order.Margin, order.Rejected, order.Expired]:
            self.handle_order_failure(order)





    def handle_order_execution(self, order):
        if order.isbuy():
            self.handle_buy_execution(order)
        elif order.issell():
            self.handle_sell_execution(order)


    def update_group_data(self, order):
        data = order.data
        if hasattr(data, 'Cluster'):
            self.asset_groups[data._name] = data.Cluster[0]
            self.asset_correlations[data._name] = {
                'mean_intragroup_correlation': getattr(data, 'mean_intragroup_correlation', [0])[0],
                'diff_to_mean_group_corr': getattr(data, 'diff_to_mean_group_corr', [0])[0],
                **{f'correlation_{i}': getattr(data, f'correlation_{i}', [0])[0] for i in range(8)}
            }
        else:
            logging.warning(f"Cluster information not available for {data._name}")
        self.update_group_allocations()

    def update_group_allocations(self):
        total_value = self.broker.getvalue()
        self.group_allocations = {group: 0 for group in range(self.total_groups)}
        for data in self.datas:
            if self.getposition(data).size > 0:
                group = self.asset_groups.get(data._name)
                if group is not None:
                    position_value = self.getposition(data).size * data.close[0]
                    self.group_allocations[group] += position_value / total_value

    def handle_sell_execution(self, order):
        data = order.data
        self.open_positions -= 1
        self.buying_status[data._name] = False
        self.handle_position_closure(data, order)
        self.remove_asset_data(data)

        # Check if the entry price exists for this data
        if data in self.entry_prices:
            is_loss = order.executed.price < self.entry_prices[data]
        else:
            # If entry price is not found, assume it's not a loss
            # You may want to log this occurrence for further investigation
            logging.warning(f"Entry price not found for {data._name}. Assuming no loss.")
            is_loss = False

        update_trade_result(data._name, is_loss)

    def handle_buy_execution(self, order):
        data = order.data
        if data._name not in self.buying_status:
            self.open_positions += 1
        self.buying_status[data._name] = True
        self.entry_prices[data] = order.executed.price  # Ensure this line is present
        self.position_dates[data] = self.datetime.date()
        self.order_list.append(order)
        self.order_cooldown[data] = self.datetime.date() + timedelta(days=1)
        self.update_group_data(order)
        #mark_position_as_bought()

        position_size = order.executed.size  # Assuming you can get size from the executed order
        mark_position_as_bought(data._name, position_size)

        update_buy_signal(data._name, self.datetime.date(), order.executed.price, data.UpProbability[0])

    def handle_position_closure(self, data, order):
        if data in self.position_dates:
            days_held = (self.datetime.date() - self.position_dates[data]).days
            percentage = (order.executed.price - self.entry_prices[data]) / self.entry_prices[data] * 100
            self.update_consecutive_losses(data, percentage < 0)
            del self.entry_prices[data]
            del self.position_dates[data]
        if order in self.order_list:
            self.order_list.remove(order)

    def update_consecutive_losses(self, data, is_loss):
        if is_loss:
            self.consecutive_losses[data] = self.consecutive_losses.get(data, 0) + 1
            loss_count = self.consecutive_losses[data]
            cooldown_days = {1: 7, 2: 28, 3: 90, 4: 282}
            days = cooldown_days.get(loss_count, 282 if loss_count >= 4 else 0)
            self.cooldown_end[data] = self.datetime.date() + timedelta(days=days)
        else:
            self.consecutive_losses[data] = 0

    def remove_asset_data(self, data):
        if data._name in self.asset_groups:
            del self.asset_groups[data._name]
        if data._name in self.asset_correlations:
            del self.asset_correlations[data._name]

    def handle_order_failure(self, order):
        if order in self.order_list:
            self.order_list.remove(order)

    def get_buy_candidates(self, current_date):
        buy_candidates = []
        for d in self.datas:
            if not self.getposition(d).size and self.can_buy(d, current_date):
                size = self.calculate_position_size(d)
                if size > 0:
                    correlation = self.get_mean_correlation(d._name, [data._name for data in self.datas if self.getposition(data).size > 0])
                    buy_candidates.append((d, size, correlation))
        return buy_candidates

    def can_buy(self, data, current_date):
        return (not self.should_skip_buying(data, current_date) and
                self.can_buy_more_positions() and
                self.is_buy_signal(data))

    def process_buy_candidates(self, buy_candidates, current_date):
       if buy_candidates:
           buy_candidates = self.sort_buy_candidates(buy_candidates)
           self.save_best_buy_signals(buy_candidates)
           for d, size, _ in buy_candidates:
               if self.open_positions < self.params.max_positions:
                   self.execute_buy(d, current_date, size)
               else:
                   break

    def calculate_position_size(self, data):
        return calculate_position_size(
            self.broker.getvalue(),
            self.broker.getcash(),
            self.params.reserve_percent,
            self.params.max_positions,
            data.close[0]
        )

    def sort_buy_candidates(self, buy_candidates):
        current_positions = [d._name for d in self.datas if self.getposition(d).size > 0]
        try:
            candidates_with_correlation = []
            for candidate in buy_candidates:
                try:
                    if len(candidate) == 2:
                        d, size = candidate
                    elif len(candidate) == 3:
                        d, size, _ = candidate
                    else:
                        logging.error(f"Unexpected candidate format: {candidate}")
                        continue

                    correlation_column = f'correlation_{size}'
                    if correlation_column not in self.inds[d].lines:
                        logging.error(f"Column {correlation_column} not found for {d._name}")
                        continue

                    correlation = self.get_mean_correlation(d._name, current_positions)
                    candidates_with_correlation.append((d, size, correlation))

                except ValueError as e:
                    logging.error(f"Error unpacking candidate {candidate}: {e}")
                    continue
                
            return sorted(
                candidates_with_correlation,
                key=lambda x: (self.inds[x[0]]['up_prob'][0], -x[2]),
                reverse=True
            )
        except Exception as e:
            logging.error(f"Error sorting buy candidates: {e}")
            return sorted(buy_candidates, key=lambda x: self.inds[x[0]]['up_prob'][0], reverse=True)

    def get_mean_correlation(self, candidate_ticker, current_positions):
        correlation_data = pd.read_parquet('Correlations.parquet')
        if not current_positions:
            return 0
        candidate_data = correlation_data[correlation_data['Ticker'] == candidate_ticker]
        if candidate_data.empty:
            logging.warning(f"No correlation data found for ticker: {candidate_ticker}")
            return 0

        correlations = []
        for pos in current_positions:
            pos_data = correlation_data[correlation_data['Ticker'] == pos]
            if pos_data.empty:
                logging.warning(f"No correlation data found for position: {pos}")
                continue
            
            correlation_column = f'correlation_{pos_data.index[0]}'
            if correlation_column not in candidate_data.columns:
                logging.warning(f"Column {correlation_column} not found for {candidate_ticker}")
                continue
            
            correlations.append(candidate_data.iloc[0][correlation_column])

        return sum(correlations) / len(correlations) if correlations else 0

    def should_skip_buying(self, data, current_date):
        return (self.buying_status.get(data._name, False) or
                (data._name in self.cooldown_end and current_date <= self.cooldown_end[data._name]))

    def evaluate_sell_conditions(self, data, current_date):
        position = self.getposition(data)
        if not position.size:
            return

        current_price = data.close[0]
        entry_price = self.entry_prices[data]
        entry_date = self.position_dates[data]

        if should_sell(current_price, entry_price, entry_date, current_date,
                       self.params.stop_loss_percent, self.params.take_profit_percent,
                       self.params.position_timeout, self.params.expected_profit_per_day_percentage):
            self.close_position(data, "Sell condition met")

    def get_high_up_prob(self, data):
        return np.percentile(self.inds[data]['up_prob'], 90)

    def can_buy_more_positions(self):
        return self.open_positions < self.params.max_positions

    def is_buy_signal(self, data):
        if 'up_prob' not in self.inds[data]:
            return False

        high_up_prob = self.get_high_up_prob(data)
        up_prediction = self.inds[data].get('UpPrediction', [1])[0]

        favorable_support_resistance = (
            self.inds[data].get('dist_to_support', [100])[0] < 50 and
            self.inds[data].get('dist_to_resistance', [0])[0] > 100
        )

        close_prices = np.array(data.close.get(size=8))
        recent_mean_percentage_change = fast_calculate_recent_mean_percentage_change(close_prices)

        current_up_prob_good = self.inds[data]['up_prob'][0] > 0.55

        return (
            self.inds[data].get('UpProbMA', [0])[0] > high_up_prob and
            up_prediction and
            favorable_support_resistance and
            recent_mean_percentage_change > 1.5 and
            current_up_prob_good
        )

    def execute_buy(self, data, current_date, size):
        if self.check_group_allocation(data):
            self.buy(data=data, size=size, exectype=bt.Order.StopTrail, trailpercent=self.params.trailing_stop_percent / 100)
            self.order_cooldown[data] = current_date + timedelta(days=1)
            self.buying_status[data._name] = True
            self.open_positions += 1

    def check_group_allocation(self, data):
        group = self.asset_groups.get(data._name)
        if group is not None:
            current_allocation = self.group_allocations.get(group, 0)
            return current_allocation < self.params.max_group_allocation
        return True

    def close_position(self, data, reason):
        logging.info(f'Closing {data._name} due to {reason}')
        self.close(data=data)

    def save_best_buy_signals(self, buy_candidates):
       top_buy_candidates = buy_candidates[:3]
       for d, size, correlation in top_buy_candidates:
           self.save_buy_signal(d, self.datetime.date())


    def save_buy_signal(self, data, current_date):
        next_trading_date = get_next_trading_day(current_date)
        current_irl_date = datetime.now().date()
        days_diff = (current_irl_date - current_date).days
        if days_diff <= 5:
            symbol = data._name
            price = data.close[0]

            # Read existing data
            df = read_trading_data()

            # Update or add new signal
            new_signal = pd.DataFrame({
                'Symbol': [symbol],
                'LastBuySignalDate': [next_trading_date],  # Use next trading date
                'LastBuySignalPrice': [price],
                'IsCurrentlyBought': [False],
                'ConsecutiveLosses': [0],
                'LastTradedDate': [None],
                'UpProbability': [data.UpProbability[0]]
            })

            # Remove existing entry for this symbol (if any)
            df = df[df['Symbol'] != symbol]

            # Append new signal
            df = pd.concat([df, new_signal], ignore_index=True)

            # Write updated data back to Parquet file
            write_trading_data(df)

            print(f"Buy signal saved for {symbol} at price {price} on {next_trading_date}", flush=True)
            logging.info(f"Buy signal saved for {symbol} at price {price} on {next_trading_date}")





    def get_cluster_distribution(self, current_positions):
        cluster_distribution = {}
        for ticker in current_positions:
            cluster = self.asset_groups.get(ticker)
            if cluster is not None:
                cluster_distribution[cluster] = cluster_distribution.get(cluster, 0) + 1
        return cluster_distribution

    def log_portfolio_state(self):
        logging.info(f"Open Positions: {self.open_positions}")
        logging.info(f"Group Allocations: {self.group_allocations}")
        for data in self.datas:
            if self.getposition(data).size > 0:
                logging.info(f"Position: {data._name}, Size: {self.getposition(data).size}, Entry Price: {self.entry_prices.get(data, 'Unknown')}")

    def log_trade(self, order):
        if order.status == order.Completed:
            if order.isbuy():
                log_msg = (
                    f"BUY EXECUTED, Price: {order.executed.price:.2f}, "
                    f"Size: {order.executed.size:.0f}, "
                    f"Cost: {order.executed.value:.2f}, "
                    f"Comm: {order.executed.comm:.2f}"
                )
            else:  # Sell
                log_msg = (
                    f"SELL EXECUTED, Price: {order.executed.price:.2f}, "
                    f"Size: {order.executed.size:.0f}, "
                    f"Cost: {order.executed.value:.2f}, "
                    f"Comm: {order.executed.comm:.2f}"
                )
            logging.info(log_msg)

    def log_daily_info(self):
        logging.info(f'DateTime: {self.datas[0].datetime.date(0)}')
        logging.info(f'Portfolio Value: {self.broker.getvalue():.2f}')
        logging.info(f'Cash: {self.broker.getcash():.2f}')

    def stop(self):


        ##get the number of data feeds that have the day 2024-09-20 here expressed as a percentage 
        ##of the total number of data feeds

        DatafeedCounter = 0
        ##get the len of the number of data feeds
        lenDataFeeds = len(self.datas)
        for d in self.datas:
            if self.datetime.date() == datetime(2024, 9, 20).date():
                DatafeedCounter += 1
                
        
        ##percentage of files that have the most up to date data
        PercentageDataFeeds = (DatafeedCounter / lenDataFeeds) * 100
        print(f"Percentage of data feeds with the most up to date data: {PercentageDataFeeds:.2f}%")
        print(f"Strategy stopped. Last processed date: {self.datetime.date()}")
        print(f"Processed {self.day_count} days out of {self.total_bars} expected")

        if self.day_count < self.total_bars:
            print(f"Warning: Strategy stopped early. Processed {self.day_count}/{self.total_bars} days.")

        self.log_portfolio_state()
        logging.info("Strategy stopped. Final portfolio state logged.")
        logging.info(f"Processed {self.day_count} days out of {self.total_bars} expected")
        
        if PercentageDataFeeds == 100:
            self.progress_bar.close()






#=============================[ Control Logic ]=============================#
#=============================[ Control Logic ]=============================#
#=============================[ Control Logic ]=============================#

def select_random_files(directory, percent):
    all_files = [f for f in os.listdir(directory) if f.endswith('.parquet')]
    num_files = len(all_files)
    num_to_select = max(1, int(round(num_files * percent / 100)))
    selected_files = random.sample(all_files, num_to_select)
    return [os.path.join(directory, f) for f in selected_files]





def get_last_trading_date():
    # Initialize the NYSE calendar
    nyse = mcal.get_calendar('NYSE')
    
    # Get today's date in the local timezone
    today = datetime.now().date()
    
    # Retrieve the trading schedule for the past 10 days up to today
    schedule = nyse.schedule(start_date=today - timedelta(days=10), end_date=today)
    
    if schedule.empty:
        raise Exception("No trading days found in the past 10 days.")
    
    # Check if today is a trading day
    if today in schedule.index.date:
        # Get today's market open time in UTC (assuming 'market_open' is timezone-aware)
        today_market_open = schedule.loc[schedule.index.date == today, 'market_open'].iloc[0]
        
        # Ensure that 'today_market_open' is timezone-aware
        if today_market_open.tzinfo is None:
            today_market_open = today_market_open.replace(tzinfo=timezone.utc)
        
        # Get the current UTC time as a timezone-aware datetime
        now_utc = datetime.now(timezone.utc)
        
        # If the current time is before the market opens today, exclude today
        if now_utc < today_market_open:
            schedule = schedule[schedule.index.date < today]
    
    if not schedule.empty:
        # The last entry in the schedule is the last trading day
        last_trading_date = schedule.index[-1].date()
        return last_trading_date
    else:
        raise Exception("No trading days found in the past 10 days after excluding today.")



def load_data(file_path, last_trading_date):
    try:
        table = pq.read_table(file_path)
        df = table.to_pandas()
        
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        # Use the last trading date
        yesterday = last_trading_date
        start_date = yesterday - timedelta(days=400)
        
        df = df[(df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= yesterday)]
        
        if len(df) < 252:
            logging.info(f"Skipping {file_path} due to insufficient data: {len(df)} days")
            return None
        
        df = df.iloc[-252:]  # Keep up to 252 trading days (approx 1 year)
        
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 
                            'Distance to Support (%)', 'Distance to Resistance (%)', 
                            'UpProbability', 'UpPrediction']
        
        if all(col in df.columns for col in required_columns):
            for col in df.select_dtypes(include=['float64']).columns:
                df[col] = df[col].round(4).astype(np.float32)
            
            return (os.path.basename(file_path).replace('.parquet', ''), df)
        else:
            missing_columns = [col for col in required_columns if col not in df.columns]
            print(f"Skipping {file_path} due to missing columns: {missing_columns}")

    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        traceback.print_exc()

    return None











def parallel_load_data(file_paths, last_trading_date):
    with multiprocessing.Pool() as pool:
        results = list(tqdm(pool.starmap(load_data, [(fp, last_trading_date) for fp in file_paths]), total=len(file_paths), desc="Loading Files"))
    return [result for result in results if result is not None]


def find_common_start_date(dates, threshold=0.95):
    date_counter = Counter(dates)
    total_count = len(dates)
    for date, count in date_counter.most_common():
        if count / total_count >= threshold:
            return date
    return None





class FixedCommissionScheme(bt.CommInfoBase):
    params = (
        ('commission', 3.0),  # Fixed commission per trade (average of $1 to $3)
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_FIXED),  # Fixed commission type
    )
    def _getcommission(self, size, price, pseudoexec):
        return self.p.commission  # Return fixed commission regardless of trade size





def main():
    setup_logging()
    timer = time.time()
    cerebro = bt.Cerebro(maxcpus=None)
    cerebro = bt.Cerebro(cheat_on_open=False)

    comminfo = FixedCommissionScheme()
    cerebro.broker.addcommissioninfo(comminfo)
    InitialStartingCash = 5000
    cerebro.broker.set_cash(InitialStartingCash)
    args = arg_parser()

    if args.BrokerTest > 0:
        file_paths = select_random_files('Data/RFpredictions', args.BrokerTest)
    elif args.RunStocks > 0:
        file_paths = [os.path.join('Data/RFpredictions', f) for f in os.listdir('Data/RFpredictions') if f.endswith('.parquet')]
        file_paths.sort(key=lambda x: x.lower())

    last_trading_date = get_last_trading_date()
    print(f"Last trading date: {last_trading_date}")

    # Load data using the last trading date
    loaded_data = parallel_load_data(file_paths, last_trading_date)

    if not loaded_data:
        print("No data was successfully loaded. Exiting.")
        return

    if args.force:
        # Find the most common maximum date among the data feeds
        max_dates = [df['Date'].dt.date.max() for name, df in loaded_data]
        date_counts = Counter(max_dates)
        last_trading_date, count = date_counts.most_common(1)[0]
        print(f"Force mode: Using last trading date: {last_trading_date}")

    # Accept datasets that have data up to last_trading_date
    aligned_data = []
    for name, df in loaded_data:
        max_date = df['Date'].dt.date.max()
        if max_date == last_trading_date and len(df) >= 252:
            aligned_data.append((name, df))
        else:
            logging.info(f"Skipping {name} due to insufficient data or not up to last trading date.")

    if not aligned_data:
        print("No data remains after alignment. Exiting.")
        return

    print(f"Number of datasets after alignment: {len(aligned_data)}")
    print(f"Number of trading days: {len(aligned_data[0][1])}")

    for name, df in aligned_data:
        data = CustomPandasData(dataname=df)
        cerebro.adddata(data, name=name)

    if not loaded_data:
        print("No data was successfully loaded. Exiting.")
        return

    if len(cerebro.datas) == 0:
        print("WARNING: No data loaded into Cerebro. Exiting.")
        return

    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="TradeStats")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="DrawDown")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="SharpeRatio", riskfreerate=0.05)
    cerebro.addanalyzer(bt.analyzers.SQN, _name="SQN")

    cerebro.addstrategy(MovingAverageCrossoverStrategy, 
                        correlation_data=correlation_data,
                        parquet_file='_Buy_Signals.parquet')
    
    strategies = cerebro.run()
    first_strategy = strategies[0]
    print(f"Actually processed {first_strategy.day_count} days")
    

    # Extract Analyzers
    trade_stats = first_strategy.analyzers.TradeStats.get_analysis()
    drawdown = first_strategy.analyzers.DrawDown.get_analysis()
    sharpe_ratio = first_strategy.analyzers.SharpeRatio.get_analysis()
    sqn_value = first_strategy.analyzers.SQN.get_analysis().get('sqn', 0)

    # Determine SQN Description
    if sqn_value < 2.0:
        description = 'Poor, but tradeable'
    elif 2.0 <= sqn_value < 2.5:
        description = 'Average'
    elif 2.5 <= sqn_value < 3.0:
        description = 'Good'
    elif 3.0 <= sqn_value < 5.0:
        description = 'Excellent'
    elif 5.0 <= sqn_value < 7.0:
        description = 'Superb'
    elif sqn_value >= 7.0:
        description = 'Maybe the Holy Grail!'

    # Extract Trade Statistics
    won_total = trade_stats.get('won', {}).get('pnl', {}).get('total', 0)
    lost_total = trade_stats.get('lost', {}).get('pnl', {}).get('total', 0)
    total_closed = trade_stats.get('total', {}).get('closed', 0)
    won_avg = trade_stats.get('won', {}).get('pnl', {}).get('average', 0)
    lost_avg = trade_stats.get('lost', {}).get('pnl', {}).get('average', 0)
    won_max = trade_stats.get('won', {}).get('pnl', {}).get('max', 0)
    lost_max = trade_stats.get('lost', {}).get('pnl', {}).get('max', 0)
    net_total = trade_stats.get('pnl', {}).get('net', {}).get('total', 0)

    PercentGain = ((cerebro.broker.getvalue() / InitialStartingCash) * 100) - 100
    if won_total != 0 and lost_total != 0 and total_closed > 10:
        logging.info(f"====================== New Entry ===================")
        logging.info(f"Max Drawdown: {drawdown['max']['drawdown']:.2f}%")
        logging.info(f"Total Trades: {total_closed}")
        logging.info(f"Profitable Trades: {trade_stats['won']['total']}")
        logging.info(f"Sharpe Ratio: {sharpe_ratio['sharperatio']:.2f}")
        logging.info(f"SQN: {sqn_value:.2f}")
        logging.info(f"Profit Factor: {lost_total / won_total:.2f}")
        logging.info(f"Average Trade: {net_total / total_closed:.2f}")
        logging.info(f"Average Winning Trade: {won_avg:.2f}")
        logging.info(f"Average Losing Trade: {lost_avg:.2f}")
        logging.info(f"Largest Winning Trade: {won_max:.2f}")
        logging.info(f"Largest Losing Trade: {lost_max:.2f}")
        logging.info(f"Time taken: {time.time() - timer:.2f} seconds")
        logging.info(f"Initial Portfolio funding: ${InitialStartingCash:.2f}")
        logging.info(f"Final Portfolio Value: ${cerebro.broker.getvalue():.2f}")
        logging.info(f"Percentage Gain: {PercentGain:.2f}%")
        logging.info(f"Cash: ${cerebro.broker.getcash():.2f}")

    if total_closed > 0:
        percent_profitable = (trade_stats['won']['total'] / total_closed) * 100
    else:
        percent_profitable = 0

    final_portfolio_value = cerebro.broker.getvalue()
    percentage_gain = ((final_portfolio_value / InitialStartingCash) * 100) - 100
    PercentPerTrade = percentage_gain / total_closed if total_closed > 0 else 0

    if total_closed > 1:
        percentage_gain_without_largest_winner = (((final_portfolio_value - won_max) / InitialStartingCash) * 100) - 100
        PercentWithoutLargestWinningTradePerTrade = percentage_gain_without_largest_winner / total_closed

        if won_total > 0:
            won_max_percentage = (won_max / won_total) * 100
    else :
        percentage_gain_without_largest_winner = 0
        PercentWithoutLargestWinningTradePerTrade = 0
        won_max_percentage = 0
        sharpe_ratio['sharperatio'] = 0

    print(f"{'=' * 8} Trading Strategy Results {'=' * 8}")
    print(f"{'Initial Starting Cash:':<30}${cerebro.broker.startingcash:.2f}")
    print(colorize_output(final_portfolio_value, 'Final Portfolio Value:', InitialStartingCash * 2, InitialStartingCash * 1.25))
    print(colorize_output(drawdown['max']['len'], 'Max Drawdown Days:', 30, 90, lower_is_better=True))
    print(colorize_output(drawdown['max']['drawdown'], 'Max Drawdown percentage:', 10, 25, lower_is_better=True))
    print(colorize_output(total_closed, 'Total Trades:', 75, 110, lower_is_better=True))
    print(colorize_output(percent_profitable, 'Profitable Trades %:', 60, 40))
    print(colorize_output(won_avg, 'Average Profitable Trade:', 100, 50))
    print(colorize_output(lost_avg, 'Average Unprofitable Trade:', -20, -100, reverse=True))
    print(colorize_output(won_total, 'Total Profitable Trades:', 5000, 800))
    print(colorize_output(abs(lost_total), 'Total Unprofitable Trades:', good_threshold=won_total/2, bad_threshold=won_total*2, lower_is_better=True))    
    print(colorize_output(won_max_percentage, 'Largest Winning percent:', 5, 50))
    print(colorize_output(won_max, 'Largest Winning Trade:', 500, 100))
    print(colorize_output(abs(lost_max), 'Largest Losing Trade:', good_threshold=won_avg, bad_threshold=won_avg*3, lower_is_better=True))    
    print(colorize_output(sharpe_ratio['sharperatio'], 'Sharpe Ratio:', 2.0, 1.0))
    print(colorize_output(sqn_value, 'SQN:', 1.9, 5.0))
    print(f"{'SQN Description:':<30}{description}")
    print(colorize_output(PercentPerTrade, 'Gain Per Trade %:', 1.0, 0.5))
    print(colorize_output(PercentWithoutLargestWinningTradePerTrade, 'Gain without Largest:', 1.0, 0.75))
    print(colorize_output(percentage_gain, 'Percentage Gain:', 50, 10.0))

    # === New Code Starts Here ===

    # Extract individual trade PnLs from TradeAnalyzer
    individual_trades = trade_stats.get('trades', [])
    if not individual_trades:
        print("No individual trades found in TradeAnalyzer.")
    else:
        # Extract PnL for each trade
        trade_pnls = [trade.get('pnl', 0) for trade in individual_trades]

        # Identify the largest winning trade
        largest_trade_pnl = max(trade_pnls) if trade_pnls else 0

        # Exclude the largest winning trade (only one occurrence)
        trade_pnls_excl_largest = trade_pnls.copy()
        if largest_trade_pnl in trade_pnls_excl_largest:
            trade_pnls_excl_largest.remove(largest_trade_pnl)

        # Recalculate Sharpe Ratio without the largest trade
        mean_return = np.mean(trade_pnls_excl_largest) if trade_pnls_excl_largest else 0
        std_return = np.std(trade_pnls_excl_largest, ddof=1) if len(trade_pnls_excl_largest) > 1 else 0
        sharpe_ratio_excl = mean_return / std_return if std_return != 0 else 0

        # Recalculate SQN without the largest trade
        n_trades_excl = len(trade_pnls_excl_largest)
        sqn_excl = (mean_return / std_return) * math.sqrt(n_trades_excl) if (n_trades_excl > 0 and std_return != 0) else 0

        # Plotting the Trade PnLs
        plt.figure(figsize=(14, 6))

        # Histogram including the largest trade
        plt.subplot(1, 2, 1)
        plt.hist(trade_pnls, bins=30, alpha=0.7, color='blue', edgecolor='black')
        plt.title('Distribution of Trade PnLs (Including Largest Trade)')
        plt.xlabel('Trade PnL')
        plt.ylabel('Frequency')

        # Histogram excluding the largest trade
        plt.subplot(1, 2, 2)
        plt.hist(trade_pnls_excl_largest, bins=30, alpha=0.7, color='green', edgecolor='black')
        plt.title('Distribution of Trade PnLs (Excluding Largest Trade)')
        plt.xlabel('Trade PnL')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.show()

        # Print Adjusted Metrics
        print(f"{'=' * 8} Adjusted Trading Strategy Results {'=' * 8}")
        print(f"Sharpe Ratio (excluding largest trade): {sharpe_ratio_excl:.2f}")
        print(f"SQN (excluding largest trade): {sqn_excl:.2f}")
        print(f"Largest Winning Trade PnL: {largest_trade_pnl:.2f}")
        print(f"Total Trades: {len(trade_pnls)}")
        print(f"Trades Excluding Largest: {n_trades_excl}")

        # Log Adjusted Metrics
        logging.info(f"Adjusted Sharpe Ratio (excluding largest trade): {sharpe_ratio_excl:.2f}")
        logging.info(f"Adjusted SQN (excluding largest trade): {sqn_excl:.2f}")
        logging.info(f"Largest Winning Trade PnL: {largest_trade_pnl:.2f}")
        logging.info(f"Total Trades: {len(trade_pnls)}")
        logging.info(f"Trades Excluding Largest: {n_trades_excl}")

    # === New Code Ends Here ===

    # Continue with existing plotting if necessary
    if len(cerebro.datas) < 10:
        plt.style.use('dark_background')
        plt.rcParams['figure.facecolor'] = '#424242'
        plt.rcParams['axes.facecolor'] = '#424242'
        plt.rcParams['grid.color'] = 'None'
        cerebro.plot(style='candlestick',
                     iplot=False,
                     start=datetime.now().date() - pd.DateOffset(days=252),
                     end=datetime.now().date(),
                     width=20,
                     height=10,
                     dpi=100,
                     tight=True)

    print(f"Time taken: {time.time() - timer:.2f} seconds")

if __name__ == "__main__":
    main()