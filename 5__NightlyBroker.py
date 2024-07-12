#!/root/root/miniconda4/envs/tf/bin/python
import os
import time
import logging
import argparse
import random
import pandas as pd
import numpy as np
import backtrader as bt
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm
import csv
import traceback
from collections import Counter
from numba import njit
from functools import lru_cache
import multiprocessing
import pyarrow.parquet as pq
from live_trading_db import get_open_positions
import sqlite3
import scipy.stats as stats

def LoggingSetup():
    loggerfile = "__BrokerLog.log"
    logging.basicConfig(
        filename=loggerfile,
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filemode='a'  # Append mode
    )


def arg_parser():
    parser = argparse.ArgumentParser(description="Backtesting Trading Strategy on Stock Data")
    parser.add_argument("--BrokerTest", type=float, default=0, help="Percentage of random files to backtest")
    parser.add_argument("--RunStocks", type=float, default=0, help="Percentage of random files to backtest")

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

    def save_buy_signals(self):
        conn = sqlite3.connect('trading.db')
        cursor = conn.cursor()
        
        cursor.execute('''CREATE TABLE IF NOT EXISTS buy_signals
                          (symbol TEXT, date DATE, price REAL)''')

        for signal in self.buy_signals:
            cursor.execute("INSERT INTO buy_signals (symbol, date, price) VALUES (?, ?, ?)",
                           (signal['symbol'], signal['date'], signal['price']))

        conn.commit()
        conn.close()

correlation_data = pd.read_parquet('Correlations.parquet').set_index('Ticker')




























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
    __slots__ = ('inds', 'order_list', 'entry_prices', 'position_dates', 'order_cooldown',
                 'buying_status', 'consecutive_losses', 'cooldown_end', 'open_positions',
                 'asset_groups', 'asset_correlations', 'group_allocations', 'total_groups',
                 'correlation_data', 'workable_capital', 'capital_per_position')

    params = (
        ('days_range', 5),
        ('fast_period', 14),
        ('slow_period', 60),
        ('max_positions', 4),
        ('reserve_percent', 0.4),
        ('stop_loss_percent', 3.0),
        ('take_profit_percent', 100.0),
        ('position_timeout', 9),
        ('trailing_stop_percent', 5.0),
        ('rolling_period', 8),
        ('max_group_allocation', 0.5),
        ('correlation_data', None),
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
        self.open_positions = 0  # Start with 0 open positions
        self.asset_groups = {}
        self.asset_correlations = {}
        self.group_allocations = {}
        self.total_groups = self.detect_total_groups()
        self.correlation_data = self.params.correlation_data
        if self.params.correlation_data is None:
            self.correlation_data = pd.read_parquet('Correlations.parquet')
        else:
            self.correlation_data = self.params.correlation_data

            # Log information about the correlation data
            logging.info(f"Correlation data shape: {self.correlation_data.shape}")
            logging.info(f"Correlation data columns: {self.correlation_data.columns}")

            # Check if 'Ticker' column exists
            if 'Ticker' not in self.correlation_data.columns:
                logging.error("'Ticker' column not found in correlation data. Available columns are:")
                for col in self.correlation_data.columns:
                    logging.error(f"  - {col}")
                raise ValueError("'Ticker' column not found in correlation data")

            # Convert the correlation data to a more efficient format for lookups
            try:
                self.correlation_dict = {row['Ticker']: row.to_dict() for _, row in self.correlation_data.iterrows()}
            except KeyError as e:
                logging.error(f"Error creating correlation dictionary: {str(e)}")
                logging.error("First few rows of correlation data:")
                logging.error(self.correlation_data.head().to_string())
                raise
            
        # Get the length of the first data feed (they should all be the same length now)
        self.total_bars = 252
        print(f"Total bars in strategy: {self.total_bars}")
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
        # Update progress bar
        self.day_count += 1
        self.progress_bar.update(1)

        current_date = self.datetime.date()

        if any(len(data) == 0 for data in self.datas):
            return  # Skip this bar if any data feed has run out

        # Vectorized check for sell conditions
        sell_data = [d for d in self.datas if self.getposition(d).size > 0]
        for d in sell_data:
            self.evaluate_sell_conditions(d, current_date)

        # Only check buy conditions if we have room for new positions
        if self.open_positions < self.params.max_positions:
            buy_candidates = self.get_buy_candidates(current_date)
            if buy_candidates:
                self.process_buy_candidates(buy_candidates, current_date)

                if (datetime.now().date() - current_date).days <= 10:
                    self.save_buy_signals(buy_candidates)







    def save_buy_signals(self, buy_candidates):
        if not buy_candidates:
            return

        conn = sqlite3.connect('live_trading.db')
        cursor = conn.cursor()

        cursor.execute('''
        CREATE TABLE IF NOT EXISTS buy_signals (
            symbol TEXT,
            date DATE NOT NULL,
            price REAL,
            PRIMARY KEY (symbol, date)
        )
        ''')

        for candidate in buy_candidates[:5]:  # Save top 5 candidates
            if isinstance(candidate, tuple):
                if len(candidate) >= 2:
                    data = candidate[0]
                    # Ensure data is the correct type before accessing attributes
                    if hasattr(data, '_name') and hasattr(data, 'close'):
                        cursor.execute("INSERT OR REPLACE INTO buy_signals (symbol, date, price) VALUES (?, ?, ?)",
                                       (data._name, self.datetime.date(), data.close[0]))
                    else:
                        logging.warning(f"Unexpected data format in buy candidate: {candidate}")
                else:
                    logging.warning(f"Unexpected tuple length in buy candidate: {candidate}")
            else:
                logging.warning(f"Unexpected format for buy candidate: {candidate}")

        conn.commit()
        conn.close()







    def detect_total_groups(self):
        correlation_data = pd.read_parquet('Correlations.parquet')
        return correlation_data['Cluster'].nunique()

    def setup_indicator_plot(self, indicator):
        indicator.plotinfo.plot = True
        indicator.plotinfo.subplot = True
        indicator.plotinfo.plotlinelabels = True
        indicator.plotinfo.linecolor = 'blue'
        indicator.plotinfo.plotname = 'Up Probability MA'

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

    def handle_buy_execution(self, order):
        data = order.data
        if data._name not in self.buying_status:
            self.open_positions += 1
        self.buying_status[data._name] = True
        self.entry_prices[data] = order.executed.price
        self.position_dates[data] = self.datetime.date()
        self.order_list.append(order)
        self.order_cooldown[data] = self.datetime.date() + timedelta(days=1)
        self.update_group_data(order)






    def update_group_data(self, order):
        data = order.data
        ticker = data._name
        if ticker in self.correlation_dict:
            self.asset_groups[ticker] = self.correlation_dict[ticker].get('Cluster', 0)
            self.asset_correlations[ticker] = {
                'mean_intragroup_correlation': self.correlation_dict[ticker].get('mean_intragroup_correlation', 0),
                'diff_to_mean_group_corr': self.correlation_dict[ticker].get('diff_to_mean_group_corr', 0),
            }
        else:
            #logging.warning(f"Correlation data not found for {ticker}")
            self.asset_groups[ticker] = 0
            self.asset_correlations[ticker] = {
                'mean_intragroup_correlation': 0,
                'diff_to_mean_group_corr': 0,
            }
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

    def handle_position_closure(self, data, order):
        if data in self.position_dates:
            days_held = (self.datetime.date() - self.position_dates[data]).days
            percentage = (order.executed.price - self.entry_prices[data]) / self.entry_prices[data] * 100
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

    def update_open_positions_count(self):
        self.open_positions = sum(1 for d in self.datas if self.getposition(d).size > 0)

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
            for candidate in buy_candidates:
                if self.open_positions < self.params.max_positions:
                    if isinstance(candidate, tuple) and len(candidate) >= 2:
                        d, size = candidate[:2]
                        self.execute_buy(d, current_date, size)
                    else:
                        logging.warning(f"Unexpected buy candidate format: {candidate}")
                else:
                    break




    def calculate_position_size(self, data):
        total_value = self.broker.getvalue()
        cash_available = self.broker.getcash()
        workable_capital = total_value * (1 - self.params.reserve_percent)
        capital_for_position = workable_capital / self.params.max_positions

        size = int(capital_for_position / data.close[0]) - 1
        if (total_value * self.params.reserve_percent) > cash_available or size < 5:
            return 0
        return size if cash_available >= capital_for_position else 0





    def sort_buy_candidates(self, buy_candidates):
        current_positions = [d._name for d in self.datas if self.getposition(d).size > 0]
        candidates_with_correlation = []

        for candidate in buy_candidates:
            try:
                if len(candidate) == 2:
                    d, size = candidate
                elif len(candidate) == 3:
                    d, size, _ = candidate
                else:
                    logging.warning(f"Unexpected candidate format: {candidate}")
                    continue

                # Get the up probability directly from the data feed
                up_prob = d.UpProbability[0]

                correlation = self.get_mean_correlation(d._name, current_positions)
                candidates_with_correlation.append((d, size, correlation, up_prob))

            except Exception as e:
                logging.warning(f"Error processing candidate {candidate}: {e}")
                continue

        try:
            return sorted(
                candidates_with_correlation,
                key=lambda x: (x[3], -x[2]),  # x[3] is up_prob, x[2] is correlation
                reverse=True
            )
        except Exception as e:
            logging.error(f"Error sorting buy candidates: {e}")
            return sorted(buy_candidates, key=lambda x: x[0].UpProbability[0], reverse=True)


    def get_mean_correlation(self, candidate_ticker, current_positions):
        if not current_positions:
            return 0

        candidate_data = self.correlation_dict.get(candidate_ticker)
        if candidate_data is None:
            #logging.warning(f"No correlation data found for ticker: {candidate_ticker}")
            return 0.20  # Return a weak correlation instead of 0

        correlations = []
        for pos in current_positions:
            correlation_column = f'correlation_{pos}'
            if correlation_column in candidate_data:
                correlations.append(candidate_data[correlation_column])
            else:
                logging.warning(f"Column {correlation_column} not found for {candidate_ticker}")
                correlations.append(0.25)  # Append a weak correlation for missing data

        return sum(correlations) / len(correlations) if correlations else 0.20


    def should_skip_buying(self, data, current_date):
        return (self.buying_status.get(data._name, False) or
                (data._name in self.cooldown_end and current_date <= self.cooldown_end[data._name]))

    def is_position_open(self, data):
        return data in self.entry_prices

    def evaluate_sell_conditions(self, data, current_date):
        position = self.getposition(data)
        if not position.size:
            return

        current_price = data.close[0]
        entry_price = self.entry_prices[data]
        days_held = (current_date - self.position_dates[data]).days
        profit_percent = (current_price - entry_price) / entry_price * 100

        if (current_price <= entry_price * (1 - self.params.stop_loss_percent / 100) or
            current_price >= entry_price * (1 + self.params.take_profit_percent / 100) or
            days_held > self.params.position_timeout or
            (days_held > 3 and profit_percent / days_held < 0.25 * days_held)):
            self.close_position(data, "Sell condition met")

    def can_buy_more_positions(self):
        return self.open_positions < self.params.max_positions


    def is_buy_signal(self, data):
        # Define all parameters locally
        volatility_threshold = 1.0
        trend_threshold = 5.0
        up_prob_threshold = 0.70
        volume_percentile = 25.0
        support_threshold = 50
        resistance_threshold = 50

        return (
            self.check_volatility(data, volatility_threshold) and
            self.check_trend(data, trend_threshold) and
            self.check_up_probability(data, up_prob_threshold) and
            self.check_volume_spike(data, volume_percentile) and
            self.check_near_support(data, support_threshold) and 
            self.check_far_from_resistance(data, resistance_threshold)
        )

    def check_volatility(self, data, volatility_threshold):
        volatility = self.calculate_volatility(data)
        return volatility >= volatility_threshold

    def check_trend(self, data, trend_threshold):
        trend = self.calculate_trend(data)
        return trend*100 >= trend_threshold

    def check_up_probability(self, data, up_prob_threshold):
        if len(data.UpProbability) == 0:
            return False
        return data.UpProbability[0] >= up_prob_threshold

    def check_volume_spike(self, data, volume_percentile):
        return self.is_volume_spike(data, volume_percentile)

    def check_near_support(self, data, support_threshold):
        if len(data.dist_to_support) == 0:
            return False
        return data.dist_to_support[0] < support_threshold

    def check_far_from_resistance(self, data, resistance_threshold):
        if len(data.dist_to_resistance) == 0:
            return False
        return data.dist_to_resistance[0] > resistance_threshold
    

    def calculate_volatility(self, data):
        close_prices = np.array(data.close.get(size=21))
        if len(close_prices) < 21:
            return 0
        return (np.std(close_prices) / np.mean(close_prices)) * 100

    def calculate_trend(self, data):
        prices = np.array(data.close.get(size=21))
        if len(prices) < 21:
            return 0
        x = np.arange(len(prices))
        slope, _, _, _, _ = stats.linregress(x, prices)
        return slope

    def is_volume_spike(self, data, volume_percentile):
        recent_volume = data.volume[0]
        volume_history = np.array(data.volume.get(size=50))
        if len(volume_history) < 50 or recent_volume == 0:
            return False
        volume_percentile_value = np.percentile(volume_history, volume_percentile)
        return recent_volume > volume_percentile_value






    ###

            #def is_buy_signal(self, data):
            #    RecentPriceVolatility = self.RecentPriceVolatility(data)
            #    if RecentPriceVolatility < 5.0:
            #        return False
            #
            #
            #
            #    # Check if stock has been doing okay over the last month (approximately 21 trading days)
            #    monthly_close_prices = np.array(data.close.get(size=21))
            #    monthly_trend = self.calculate_trend(monthly_close_prices)
            #
            #    # Check if the last 3 days are doing okay
            #    recent_close_prices = np.array(data.close.get(size=3))
            #    recent_trend = self.calculate_trend(recent_close_prices)
            #
            #    # Get the 95th percentile of up_prob over the last 100 periods
            #    up_prob_95th_percentile = self.get_up_prob_percentile(data, percentile=85)
            #
            #    # Get the 5th percentile of up_prob over the last week (5 trading days)
            #    up_prob_5th_percentile_week = self.get_up_prob_percentile(data, percentile=5, lookback=5)
            #
            #    # Check conditions
            #    long_short_term_positive = monthly_trend > 0.05 and recent_trend > 0.01
            #    high_up_prob = self.inds[data]['up_prob'][0] >= up_prob_95th_percentile
            #    not_recently_very_low = self.inds[data]['up_prob'][0] > up_prob_5th_percentile_week
            #
            #    return (long_short_term_positive and 
            #            high_up_prob and 
            #            not_recently_very_low and 
            #            RecentPriceVolatility > 5.0)
            #
            #
            #
            #def calculate_trend(self, prices):
            #    """Calculate the trend of prices using linear regression."""
            #    if len(prices) < 2:
            #        return 0  # Return 0 (no trend) if there's not enough data
            #    x = np.arange(len(prices))
            #    slope, _ = np.polyfit(x, prices, 1)
            #    return slope
            #
            #def get_up_prob_percentile(self, data, percentile, lookback=100):
            #    """Calculate the specified percentile of up_prob over the last lookback periods."""
            #    up_prob_history = np.array(self.inds[data]['up_prob'].get(size=lookback))
            #
            #    if len(up_prob_history) == 0:
            #        return 0  # Return 0 to prevent false positives
            #    return np.percentile(up_prob_history, percentile)
            #
            #def RecentPriceVolatility(self, data):
            #    """Calculate the recent price volatility of the stock."""
            #    if len(data) < 21:
            #        return 0
            #    close_prices = np.array(data.close.get(size=21))
            #    return (np.std(close_prices) / np.mean(close_prices)) * 100
    



















    def execute_buy(self, data, current_date, size):
        if self.check_group_allocation(data):
            logging.info(f"Placing buy order for {data._name}, size: {size}")
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

    def stop(self):
        for data in self.datas:
            if data in self.entry_prices:
                self.close(data=data)

    def save_best_buy_signals(self, buy_candidates):
        self.save_buy_signals(buy_candidates[:5])

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
        self.progress_bar.close()
        print(f"Strategy stopped. Last processed date: {self.datetime.date()}")

        print(f"Processed {self.day_count} days out of {self.total_bars} expected")
        if self.day_count < self.total_bars:
            print(f"Warning: Strategy stopped early. Processed {self.day_count}/{self.total_bars} days.")
        super().stop()
        self.log_portfolio_state()
        logging.info("Strategy stopped. Final portfolio state logged.")
















##===================================[ Control logic ]===================================##
##===================================[ Control logic ]===================================##
##===================================[ Control logic ]===================================##
##===================================[ Control logic ]===================================##
##===================================[ Control logic ]===================================##

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

def select_random_files(directory, percent):
    all_files = [f for f in os.listdir(directory) if f.endswith('.parquet')]
    num_files = len(all_files)
    num_to_select = max(1, int(round(num_files * percent / 100)))
    selected_files = random.sample(all_files, num_to_select)
    return [os.path.join(directory, f) for f in selected_files]



def load_data(file_path):
    try:
        table = pq.read_table(file_path)
        df = table.to_pandas()
        
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        yesterday = datetime.now().date() - timedelta(days=1)
        start_date = yesterday - timedelta(days=400)
        
        df = df[(df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= yesterday)]
        
        if len(df) < 265:
            ##log a warning here
            #logging.info(f"Skipping {file_path} due to insufficient data: {len(df)} days")
            return None
        
        df = df.iloc[-265:]  # Keep up to 265 trading days
        
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
        global error_counter
        error_counter += 1
    return None


error_counter = 0





def parallel_load_data(file_paths):
    with multiprocessing.Pool() as pool:
        results = list(tqdm(pool.imap(load_data, file_paths), total=len(file_paths), desc="Loading Files"))

    logging.info(f"Successfully loaded {len(results) - error_counter} files")

    return [result for result in results if result is not None]

def find_common_start_date(dates, threshold=0.95):
    date_counter = Counter(dates)
    total_count = len(dates)
    for date, count in date_counter.most_common():
        if count / total_count >= threshold:
            return date
    return None

def main():
    LoggingSetup()
    timer = time.time()
    cerebro = bt.Cerebro(maxcpus=None)
    InitalStartingCash = 5000
    cerebro.broker.set_cash(InitalStartingCash)
    args = arg_parser()

    if args.BrokerTest > 0:
        file_paths = select_random_files('Data/RFpredictions', args.BrokerTest)
    elif args.RunStocks > 0:
        file_paths = [os.path.join('Data/RFpredictions', f) for f in os.listdir('Data/RFpredictions') if f.endswith('.parquet')]
        file_paths.sort(key=lambda x: x.lower())

    loaded_data = parallel_load_data(file_paths)

    if not loaded_data:
        print("No data was successfully loaded. Exiting.")
        return

    start_dates = [df['Date'].min().date() for _, df in loaded_data]
    common_start_date = find_common_start_date(start_dates)

    if not common_start_date:
        print("No common start date found. Exiting.")
        return

    print(f"Common start date: {common_start_date}")

    aligned_data = []
    for name, df in loaded_data:
        aligned_df = df[df['Date'].dt.date >= common_start_date]
        if len(aligned_df) >= 265:
            aligned_data.append((name, aligned_df.iloc[:265]))
        else:
            logging.info(f"Skipping {name} due to insufficient data: {len(aligned_df)} days")
    if not aligned_data:
        print("No data remains after alignment. Exiting.")
        return

    end_date = aligned_data[0][1]['Date'].max().date()
    print(f"Aligned data from {common_start_date} to {end_date}")
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

    # In your main function
    try:
        correlation_data = pd.read_parquet('Correlations.parquet')
        print(f"Loaded correlation data with shape: {correlation_data.shape}")
        print(f"Columns in correlation data: {correlation_data.columns}")
    except Exception as e:
        print(f"Error loading correlation data: {str(e)}")
        return

    cerebro.addstrategy(MovingAverageCrossoverStrategy, correlation_data=correlation_data)
    strategies = cerebro.run()
    first_strategy = strategies[0]
    print(f"Actually processed {first_strategy.day_count} days")

    ##enalble cheat on open
    trade_stats = first_strategy.analyzers.TradeStats.get_analysis()
    drawdown = first_strategy.analyzers.DrawDown.get_analysis()
    sharpe_ratio = first_strategy.analyzers.SharpeRatio.get_analysis()
    sqn_value = first_strategy.analyzers.SQN.get_analysis().get('sqn', 0)

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

    won_total = trade_stats.get('won', {}).get('pnl', {}).get('total', 0)
    lost_total = trade_stats.get('lost', {}).get('pnl', {}).get('total', 0)
    total_closed = trade_stats.get('total', {}).get('closed', 0)
    won_avg = trade_stats.get('won', {}).get('pnl', {}).get('average', 0)
    lost_avg = trade_stats.get('lost', {}).get('pnl', {}).get('average', 0)
    won_max = trade_stats.get('won', {}).get('pnl', {}).get('max', 0)
    lost_max = trade_stats.get('lost', {}).get('pnl', {}).get('max', 0)
    net_total = trade_stats.get('pnl', {}).get('net', {}).get('total', 0)

    PercentGain = ((cerebro.broker.getvalue() / InitalStartingCash) * 100) - 100
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
        logging.info(f"Inital Portfolio funding: ${InitalStartingCash:.2f}")
        logging.info(f"Final Portfolio Value: ${cerebro.broker.getvalue():.2f}")
        logging.info(f"Percentage Gain: {PercentGain:.2f}%")
        logging.info(f"Cash: ${cerebro.broker.getcash():.2f}")

    if total_closed > 0:
        percent_profitable = (trade_stats['won']['total'] / total_closed) * 100
    else:
        percent_profitable = 0

    final_portfolio_value = cerebro.broker.getvalue()
    percentage_gain = ((final_portfolio_value / InitalStartingCash) * 100) - 100
    PercentPerTrade = percentage_gain / total_closed if total_closed > 0 else 0

    if total_closed > 1:
        percentage_gain_without_largest_winner = (((final_portfolio_value - won_max) / InitalStartingCash) * 100) - 100
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
    print(colorize_output(final_portfolio_value, 'Final Portfolio Value:', InitalStartingCash * 2, InitalStartingCash * 1.25))
    print(colorize_output(drawdown['max']['len'], 'Max Drawdown Days:', 15, 90, lower_is_better=True))
    print(colorize_output(drawdown['max']['drawdown'], 'Max Drawdown percentage:', 5, 15, lower_is_better=True))
    print(colorize_output(total_closed, 'Total Trades:', 75, 110, lower_is_better=True))
    print(colorize_output(percent_profitable, 'Profitable Trades %:', 60, 40))
    print(colorize_output(won_avg, 'Average Profitable Trade:', 100, 50))
    print(colorize_output(lost_avg, 'Average Unprofitable Trade:', -20, -100, reverse=True))
    print(colorize_output(won_total, 'Total Profitable Trades:', 5000, 800))
    print(colorize_output(lost_total, 'Total Unprofitable Trades:', -20, -2500, reverse=True))
    print(colorize_output(won_max_percentage, 'Largest Winning percent:', 5, 50))
    print(colorize_output(won_max, 'Largest Winning Trade:', 500, 100))
    print(colorize_output(lost_max, 'Largest Losing Trade:', -50, -200, reverse=True))
    print(colorize_output(sharpe_ratio['sharperatio'], 'Sharpe Ratio:', 2.0, 1.0))
    print(colorize_output(sqn_value, 'SQN:', 1.9, 5.0))
    print(f"{'SQN Description:':<30}{description}")
    print(colorize_output(PercentPerTrade, 'Gain Per Trade %:', 1.0, 0.5))
    print(colorize_output(PercentWithoutLargestWinningTradePerTrade, 'Gain without Largest:', 1.0, 0.75))
    print(colorize_output(percentage_gain, 'Percentage Gain:', 50, 10.0))

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
