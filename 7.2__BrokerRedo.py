#!/root/root/miniconda4/envs/tf/bin/python
import os
import time
import logging
import argparse
import random
from joblib import dump, load
import pandas as pd
import numpy as np
from backtrader.feeds import PandasData
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import ta  # Technical Analysis library for financial indicators
import feature_engine  # Advanced feature engineering techniques
import backtrader as bt  # For backtesting trading strategies
import quantstats as qs  # Extends pyfolio for comprehensive analytics
from scipy.optimize import minimize  # For numerical optimization
import optuna  # For advanced hyperparameter optimization
import logging
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from datetime import datetime, timedelta


def LoggingSetup():
    loggerfile = "__BrokerLog.log"
    logging.basicConfig(
        filename="__BrokerLog.log",
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filemode='a'  # Append mode
    )

def arg_parser():
    parser = argparse.ArgumentParser(description="Backtesting Trading Strategy on Stock Data")
    parser.add_argument("--BrokerTest", type=float, default=0, help="Percentage of random files to backtest")
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



class CustomPandasData(bt.feeds.PandasData):
    lines = ('dist_to_support', 'dist_to_resistance', 'UpProb', 'DownProb', 'Weighted_UpProb')
    
    params = (
        ('datetime', None),
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
        ('openinterest', None),
        ('dist_to_support', 'Distance to Support (%)'),
        ('dist_to_resistance', 'Distance to Resistance (%)'),
        ('UpProb', 'UpProb_Shift_1'),
        ('DownProb', 'DownProb_Shift_1'),
        ('Weighted_UpProb', 'Weighted_UpProb'),
    )




class MovingAverageCrossoverStrategy(bt.Strategy):
    params = (
        ('fast_period', 14),
        ('slow_period', 60),
        ('max_positions', 4),
        ('reserve_percent', 0.4),
        ('stop_loss_percent', 5.0),
        ('take_profit_percent', 100.0),
        ('position_timeout', 14),
        ('trailing_stop_percent', 3.5),
        ('rolling_period', 8),
    )

    def __init__(self):
        self.inds = {}
        self.order_list = []
        self.entry_prices = {}
        self.position_dates = {}
        self.order_cooldown = {}
        self.buying_status = {}
        self.consecutive_losses = {}
        self.cooldown_end = {}
        self.open_positions = 0

        for d in self.datas:
            # Initialize indicators for each data feed
            self.inds[d] = {
                'up_prob': d.UpProb,
                'down_prob': d.DownProb,
                'Weighted_UpProb': d.Weighted_UpProb,
                'dist_to_support': d.dist_to_support,
                'dist_to_resistance': d.dist_to_resistance,
                'UpProbMA': bt.indicators.SimpleMovingAverage(d.UpProb, period=self.params.fast_period),
                'DownProbMA': bt.indicators.SimpleMovingAverage(d.DownProb, period=self.params.fast_period),
                'crossover': bt.indicators.CrossOver(d.UpProb, d.DownProb)
            }

            # Configure MA indicators for plotting
            self.inds[d]['UpProbMA'].plotinfo.plot = True
            self.inds[d]['UpProbMA'].plotinfo.subplot = True
            self.inds[d]['UpProbMA'].plotinfo.plotlinelabels = True
            self.inds[d]['UpProbMA'].plotinfo.linecolor = 'blue'
            self.inds[d]['UpProbMA'].plotinfo.plotname = 'Up Probability MA'

            self.inds[d]['DownProbMA'].plotinfo.plot = True
            self.inds[d]['DownProbMA'].plotinfo.subplot = True
            self.inds[d]['DownProbMA'].plotinfo.plotlinelabels = True
            self.inds[d]['DownProbMA'].plotinfo.linecolor = 'red'
            self.inds[d]['DownProbMA'].plotinfo.plotname = 'Down Probability MA'

    def next_trading_day(current_date):
        next_day = current_date + timedelta(days=1)
        # Simulate skipping weekends (assuming weekdays only trading)
        while next_day.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            next_day += timedelta(days=1)
        return next_day





    def notify_order(self, order):
        # Setup status name mapping
        status_names = {
            bt.Order.Completed: 'Completed',
            bt.Order.Canceled: 'Canceled',
            bt.Order.Submitted: 'Submitted',
            bt.Order.Accepted: 'Accepted',
            bt.Order.Partial: 'Partially Executed',
            bt.Order.Margin: 'Margin Call Failed',
            bt.Order.Rejected: 'Rejected by Broker',
            bt.Order.Expired: 'Expired'
        }

        order_status = status_names.get(order.status, "Unknown")
        logging.info(f"Order {order_status} for {order.data._name}")

        if order.status in [bt.Order.Completed, bt.Order.Partial]:
            if order.isbuy():
                # Check if current date is after the cooldown period
                ##if self.datetime.date() >= self.order_cooldown.get(order.data, datetime.date.min):


                ##ensure that there is a 1 day cooldown period between orders
                if self.datetime.date() >= self.order_cooldown.get(order.data, datetime.date):
                    self.buying_status[order.data._name] = False
                    self.open_positions += 1
                    self.entry_prices[order.data] = order.executed.price
                    self.position_dates[order.data] = self.datetime.date()
                    logging.info(f"Bought: {order.data._name} at {order.executed.price}")
                    self.order_list.append(order)
                    self.order_cooldown[order.data] = self.datetime.date() + timedelta(days=1)
            elif order.issell():
                self.open_positions -= 1
                if order.data in self.position_dates:
                    days_held = (self.datetime.date() - self.position_dates[order.data]).days
                    percentage = (order.executed.price - self.entry_prices[order.data]) / self.entry_prices[order.data] * 100
                    trade_result = (order.executed.price - self.entry_prices[order.data]) / self.entry_prices[order.data]
                    if percentage < 0:
                        # Loss
                        self.consecutive_losses[order.data] = self.consecutive_losses.get(order.data, 0) + 1
                        loss_count = self.consecutive_losses[order.data]
                        cooldown_days = {1: 14, 2: 28, 3: 60, 4: 282}
                        days = cooldown_days.get(loss_count, 282 if loss_count >= 4 else 0)
                        self.cooldown_end[order.data] = self.datetime.date() + timedelta(days=days)
                    else:
                        # Profit
                        self.consecutive_losses[order.data] = 0
                    logging.info(f"Sold: {order.data._name} at {order.executed.price} for a {percentage:.2f}% gain/loss after {days_held} days.")
                    del self.entry_prices[order.data]
                    del self.position_dates[order.data]




                if order in self.order_list:
                    self.order_list.remove(order)
        elif order.status in [bt.Order.Canceled, bt.Order.Margin, bt.Order.Rejected, bt.Order.Expired, bt.Order.Submitted, bt.Order.Accepted]:
            if order in self.order_list:
                self.order_list.remove(order)







    def calculate_position_size(self, data):
        total_value = self.broker.getvalue()
        reserved_cash = total_value * self.params.reserve_percent  # 40% reserved
        investable_capital = total_value - reserved_cash  # 60% of total value
        chunk_size = total_value * 0.15  # Each chunk is 15% of total value

        size = int(chunk_size / data.close[0])
        return size if self.broker.getcash() >= chunk_size else 0



    def CloseWrapper(self, data, message):
        self.close(data=data)
        logging.info(f"{message} for {data._name} at {data.close[0]} on {self.datetime.date()}")




    def next(self):
            for d in self.datas:
                current_date = self.datetime.date()
                if d in self.entry_prices:  # Check if there is an entry price stored
                    if d.close[0] <= self.entry_prices[d] * (1 - self.params.stop_loss_percent / 100):
                        message = "Stop loss"
                        self.CloseWrapper(d, message)  # Execute stop loss
                        continue

                    # Check for take profit
                    if d.close[0] >= self.entry_prices[d] * (1 + self.params.take_profit_percent / 100):
                        message = "Take profit"
                        self.CloseWrapper(d, message)
                        continue
                    NewStopLossTesting = False
                    
                    if NewStopLossTesting:
                        days_held = (current_date - self.position_dates[d]).days
                        initial_stop_loss_percent = 1.0  # Tighter stop loss for initial days
                        regular_stop_loss_percent = 5.0  # Regular stop loss after initial period
                        initial_period = 3  # Number of days for the initial stop loss
            
                        # Determine the current stop loss percentage based on days held
                        current_stop_loss_percent = initial_stop_loss_percent if days_held < initial_period else regular_stop_loss_percent
            
                        # Execute stop loss based on current stop loss percentage
                        if d.close[0] <= self.entry_prices[d] * (1 - current_stop_loss_percent / 100):
                            self.CloseWrapper(d, "Stop loss due to stop loss percentage")






                # Position timeout check
                if d in self.position_dates:
                    days_held = (self.datetime.date() - self.position_dates[d]).days
                    if days_held > self.params.position_timeout:
                        message = "Position timeout"
                        self.CloseWrapper(d, message)
                        continue

                if d._name in self.cooldown_end and self.datetime.date() <= self.cooldown_end[d._name]:
                    continue  # Skip trading this stock if still in cooldown period


                if self.buying_status.get(d._name, False):  # Check if there's an ongoing buying process
                    continue  # Skip if already trying to buy this stock

                if self.open_positions > self.params.max_positions:
                    continue  # Limit number of open positions
                else:
                    current_position = self.getposition(d)
                    if not current_position.size > 0:
                        ##get the cross ovver indicator made in the self.inds dictionary
                        pass





                        HighUpProb = np.percentile(self.inds[d]['up_prob'], 95)
                        LowDownProb = np.percentile(self.inds[d]['down_prob'], 50)



                        BuySignal = (
                            (self.inds[d]['UpProbMA'][0] > self.inds[d]['DownProbMA'][0]) and
                            ##(self.inds[d]['crossover'][0] == 1) and
                            (self.inds[d]['up_prob'][0] > HighUpProb*1.1) and 
                            (self.inds[d]['down_prob'][0] < LowDownProb*1.1)

                        )

                        if BuySignal:  # UpProb MA crosses above DownProb MA
                        

                            size = self.calculate_position_size(d)
                            if size > 0:
                                #print(f"DistToSupport: {DistToSupport[0]}")
                                #print(f"DistToResistance: {DistToResistance[0]}")
                                ##print(f"DistRatio: {DistRatio}")
                                self.buy(data=d, size=size, exectype=bt.Order.StopTrail, trailpercent=self.params.trailing_stop_percent / 100)
                                self.order_cooldown[d] = current_date + timedelta(days=1)
                                self.buying_status[d._name] = True
                                ##logging.info(f"Bought: {d._name} at {d.close[0]} on {self.datetime.date()}")










    def stop(self):
        ##close all orders at the end of the backtest
        for d in self.datas:
            if d in self.entry_prices:  # Check if there is an entry price stored
                self.close(data=d)  # Execute stop loss
                logging.info(f"End Loss triggered for {d._name} at {d.close[0]} on {self.datetime.date()}")
                        
        
        logging.info(f"Final Portfolio Value: ${self.broker.getvalue():.2f}")
        logging.info(f"Final Cash: ${self.broker.getcash():.2f}")






















##############################################################################################################
##############################################################################################################
##############################################################################################################
##############################################################################################################

def select_random_files(directory, percent):
    all_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    num_files = len(all_files)
    num_to_select = max(1, int(round(num_files * percent / 100)))
    selected_files = random.sample(all_files, num_to_select)
    return [os.path.join(directory, f) for f in selected_files]


def load_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    if len(df) < 500:
        return None
    
    df = df[-282:]
    return df

def load_and_add_data(cerebro, file_path, FailedFileCounter):
    df = load_data(file_path)
    if df is not None:
        data = CustomPandasData(dataname=df)
        cerebro.adddata(data, name=os.path.basename(file_path))
    else:
        FailedFileCounter += 1
        
def main():
    LoggingSetup()
    timer = time.time()
    cerebro = bt.Cerebro()
    cerebro.broker.set_cash(5000)
    args = arg_parser()
    FailedFileCounter = 0

    # Load data into cerebro
    for file_path in select_random_files('Data/RFpredictions', args.BrokerTest):
        load_and_add_data(cerebro, file_path, FailedFileCounter)
        if FailedFileCounter > 0:
            logging.info(f"Failed to load {FailedFileCounter} files.")

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="TradeStats")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="DrawDown")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="SharpeRatio", riskfreerate=0.0)

    # Add strategy
    cerebro.addstrategy(MovingAverageCrossoverStrategy)

    # Run backtest
    strategies = cerebro.run()
    first_strategy = strategies[0]

    # Get analyzer results
    trade_stats = first_strategy.analyzers.TradeStats.get_analysis()
    drawdown = first_strategy.analyzers.DrawDown.get_analysis()
    sharpe_ratio = first_strategy.analyzers.SharpeRatio.get_analysis()

    # Print the results
    print(f"Max Drawdown: {drawdown.max.drawdown:.2f}%")
    print(f"Total Trades: {trade_stats.total.closed}")
    print(f"Profitable Trades: {trade_stats.won.total}")
    print(f"Sharpe Ratio: {sharpe_ratio['sharperatio']:.2f}")

    # Calculate percentage of profitable trades
    if trade_stats.total.closed > 0:
        percent_profitable = (trade_stats.won.total / trade_stats.total.closed) * 100
    else:
        percent_profitable = 0
    print(f"Percentage of Profitable Trades: {percent_profitable:.2f}%")

    print(f"Time taken: {time.time() - timer:.2f} seconds")
    print("Final Portfolio Value: $%.2f" % cerebro.broker.getvalue())
    print("Cash: $%.2f" % cerebro.broker.getcash())

    if args.BrokerTest < 1.0:
        plt.style.use('dark_background')
        plt.rcParams['figure.facecolor'] = '#424242'
        plt.rcParams['axes.facecolor'] = '#424242'
        plt.rcParams['grid.color'] = 'None'

        cerebro.plot(style='candlestick',
                     iplot=False,
                     start=datetime.now().date() - pd.DateOffset(days=282),
                     end=datetime.now().date(),
                     width=20,
                     height=10,
                     dpi=100,
                     tight=True)



if __name__ == "__main__":
    main()
