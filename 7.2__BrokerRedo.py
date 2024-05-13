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
from tabulate import tabulate


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
    parser.add_argument("--RunTickers", type=float, default=0, help="Percentage of random files to backtest")

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
        while next_day.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            next_day += timedelta(days=1)
        return next_day




    def notify_order(self, order):
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

        if order.status in [bt.Order.Completed, bt.Order.Partial]:
            if order.isbuy():
                ##ensure that there is a 1 day cooldown period between orders
                if self.datetime.date() >= self.order_cooldown.get(order.data, datetime.date):
                    self.buying_status[order.data._name] = False
                    self.open_positions += 1
                    self.entry_prices[order.data] = order.executed.price
                    self.position_dates[order.data] = self.datetime.date()
                    #logging.info(f"Bought: {order.data._name} at {order.executed.price}")
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
                        cooldown_days = {1: 7, 2: 28, 3: 90, 4: 282}
                        days = cooldown_days.get(loss_count, 282 if loss_count >= 4 else 0)
                        self.cooldown_end[order.data] = self.datetime.date() + timedelta(days=days)
                    else:
                        # Profit
                        self.consecutive_losses[order.data] = 0
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
        #logging.info(f"{message} for {data._name} at {data.close[0]} on {self.datetime.date()}")






    def next(self):
            for d in self.datas:                

                if self.buying_status.get(d._name, False):  # Check if there's an ongoing buying process
                    continue  # Skip if already trying to buy this stock

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
                    
                    NewStopLossTesting = True
                    
                    if NewStopLossTesting:
                        days_held = (self.datetime.date() - self.position_dates[d]).days
                        InitalTime = 3
                        

                        PercantageList = []

                        for i in range(1, 90):
                            percentageChange = ((d.close[-i] - d.close[-i-1]) / d.close[-i-1]) * 100
                            if percentageChange < 0:
                                PercantageList.append(percentageChange)

                       
                        InitalPercentile = 80.0
                        FinalPercentile = 50.0
                        ##in the inital period set the stop loss to the maximum allowed inital stop loss or the 90th percentile of the negative percentage changes
                        if days_held < InitalTime:
                            current_stop_loss_percent = abs(np.percentile(PercantageList, InitalPercentile))
                            #print(f"current_stop_loss_percent: {current_stop_loss_percent}")
                        else:
                            current_stop_loss_percent = abs(np.percentile(PercantageList, FinalPercentile))
                            #print(f"current_stop_loss_percent after inital time: {current_stop_loss_percent}")
                            ##if the current price is less than the stop loss percentage then close the position





                        if d.close[0] <= self.entry_prices[d] * (1 - current_stop_loss_percent / 100):
                            currentLoss = ((d.close[0] - self.entry_prices[d]) / self.entry_prices[d]) * 100
                            #print(f"Stop loss due to stop loss percentage: {currentLoss} exceeding {current_stop_loss_percent}")
                            self.CloseWrapper(d, "Stop loss due to stop loss percentage")









                # Position timeout check
                if d in self.position_dates:
                    days_held = (self.datetime.date() - self.position_dates[d]).days
                    if days_held > self.params.position_timeout:
                        self.CloseWrapper(d, "Position timeout")
                        continue

                if d._name in self.cooldown_end and self.datetime.date() <= self.cooldown_end[d._name]:
                    continue  # Skip trading this stock if still in cooldown period


                if self.buying_status.get(d._name, False):  # Check if there's an ongoing buying process
                    continue  # Skip if already trying to buy this stock

                if self.open_positions >= self.params.max_positions:
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
        for d in self.datas:
            if d in self.entry_prices:  # Check if there is an entry price stored
                self.close(data=d)  # Execute stop loss





















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
    
    last_date = df.index.max()
    one_year_ago = last_date - timedelta(days=365)

    df = df[df.index >= one_year_ago]
    return df




def load_and_add_data(cerebro, file_path, FailedFileCounter):
    df = load_data(file_path)
    if df is not None:
        data = CustomPandasData(dataname=df)
        cerebro.adddata(data, name=os.path.basename(file_path))
    else:
        FailedFileCounter += 1
        



def colorize_output(value, label, good_threshold, bad_threshold, reverse=False):
    """
    Colors terminal output based on performance.
    value: the numeric value to evaluate
    label: the label for the output
    good_threshold: the value threshold for good performance (green)
    bad_threshold: the value threshold for bad performance (red)
    reverse: if True, smaller values are better
    """
    if reverse:
        if value <= good_threshold:
            color_code = "\033[92m"  # Bright Green
        elif value >= bad_threshold:
            color_code = "\033[91m"  # Bright Red
        else:
            # Calculate interpolation
            range_span = bad_threshold - good_threshold
            normalized_value = (value - good_threshold) / range_span
            red = int(255 * normalized_value)
            green = int(255 * (1 - normalized_value))
            blue = 0
            color_code = f"\033[38;2;{red};{green};{blue}m"
    else:
        if value >= good_threshold:
            color_code = "\033[92m"  # Bright Green
        elif value <= bad_threshold:
            color_code = "\033[91m"  # Bright Red
        else:
            # Calculate interpolation
            range_span = good_threshold - bad_threshold
            normalized_value = (value - bad_threshold) / range_span
            red = int(255 * (1 - normalized_value))
            green = int(255 * normalized_value)
            blue = 0
            color_code = f"\033[38;2;{red};{green};{blue}m"

    # Formatted output with reset at the end
    return f"{label:<30}{color_code}{value:.2f}\033[0m"





def load_ticker_data(cerebro, directory, tickers, FailedFileCounter):
    # Ensure tickers are lower case for comparison
    ticker_set = set([ticker.lower() for ticker in tickers])
    for file_name in os.listdir(directory):
        if file_name.endswith('.csv'):
            ticker_name = file_name.split('.')[0].lower()
            if ticker_name in ticker_set:
                file_path = os.path.join(directory, file_name)
                load_and_add_data(cerebro, file_path, FailedFileCounter)

def main():
    LoggingSetup()
    timer = time.time()
    cerebro = bt.Cerebro()
    InitalStartingCash = 5000
    cerebro.broker.set_cash(InitalStartingCash)
    args = arg_parser()
    FailedFileCounter = 0

    # Check if we are running a percentage test
    if args.BrokerTest > 0:
        for file_path in select_random_files('Data/RFpredictions', args.BrokerTest):
            load_and_add_data(cerebro, file_path, FailedFileCounter)
            if FailedFileCounter > 0:
                logging.info(f"Failed to load {FailedFileCounter} files.")
    # Check if RunTickers is specified
    elif args.RunTickers > 0:
        # Load the CSV with ticker information
        df_tickers = pd.read_csv('TradingModelStats.csv')
        tickers = df_tickers['Ticker'].tolist()
        
        # Load data for specified tickers
        load_ticker_data(cerebro, 'Data/RFpredictions', tickers, FailedFileCounter)
        if FailedFileCounter > 0:
            logging.info(f"Failed to load {FailedFileCounter} files.")




    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="TradeStats")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="DrawDown")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="SharpeRatio", riskfreerate=0.05)
    cerebro.addanalyzer(bt.analyzers.SQN, _name="SQN")

    ##add the trades 
    cerebro.addstrategy(MovingAverageCrossoverStrategy)

    # Run backtest
    strategies = cerebro.run()
    first_strategy = strategies[0]

    # Get analyzer results
    trade_stats = first_strategy.analyzers.TradeStats.get_analysis()
    drawdown = first_strategy.analyzers.DrawDown.get_analysis()
    sharpe_ratio = first_strategy.analyzers.SharpeRatio.get_analysis()
    sqn_value = first_strategy.analyzers.SQN.get_analysis()['sqn']  # Assuming 'sqn' is the correct key
    

    
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





    PercentGain = ((cerebro.broker.getvalue() / InitalStartingCash) * 100) -100
    if trade_stats.won.pnl.total != 0 and trade_stats.lost.pnl.total != 0 and trade_stats.total.closed > 10:
        logging.info(f"====================== New Entry ===================")
        logging.info(f"Max Drawdown: {drawdown.max.drawdown:.2f}%")
        logging.info(f"Total Trades: {trade_stats.total.closed}")
        logging.info(f"Profitable Trades: {trade_stats.won.total}")
        logging.info(f"Sharpe Ratio: {sharpe_ratio['sharperatio']:.2f}")
        logging.info(f"SQN: {sqn_value:.2f}")


        logging.info(f"Profit Factor: {trade_stats.won.pnl.total / trade_stats.lost.pnl.total:.2f}")


        logging.info(f"Average Trade: {trade_stats.pnl.net.total / trade_stats.total.closed:.2f}")
        logging.info(f"Average Winning Trade: {trade_stats.won.pnl.average:.2f}")
        logging.info(f"Average Losing Trade: {trade_stats.lost.pnl.average:.2f}")
        logging.info(f"Largest Winning Trade: {trade_stats.won.pnl.max:.2f}")
        logging.info(f"Largest Losing Trade: {trade_stats.lost.pnl.max:.2f}")
        #logging.info(f"Percentage of Profitable Trades: {percent_profitable:.2f}%")
        logging.info(f"Time taken: {time.time() - timer:.2f} seconds")
        logging.info(f"Inital Portfolio funding: ${InitalStartingCash:.2f}")
        logging.info(f"Final Portfolio Value: ${cerebro.broker.getvalue():.2f}")
        logging.info(f"Percentage Gain: {PercentGain:.2f}%")
        logging.info(f"Cash: ${cerebro.broker.getcash():.2f}")



    # Calculate percentage of profitable trades
    if trade_stats.total.closed > 0:
        percent_profitable = (trade_stats.won.total / trade_stats.total.closed) * 100
    else:
        percent_profitable = 0




    #print(f"====================== ADVANCED ===================")
    #print(f"{'Initial Starting Cash:':<30}${cerebro.broker.startingcash:.2f}")
    #print(f"{'Final Portfolio Value:':<30}${cerebro.broker.getvalue():.2f}")
    #print(f"{'Max Drawdown Days:':<30}{drawdown.max.len:.2f}")
    #print(f"{'Max Drawdown percentage:':<30}{drawdown.max.drawdown:.2f}%")
    #print(f"{'Total Trades:':<30}{trade_stats.total.closed}")
    #print(f"{'Percentage Profitable Trades:':<30}{percent_profitable:.2f}%")
    #print(f"{'Average Profitable Trade:':<30}${trade_stats.won.pnl.average:.2f}")
    #print(f"{'Average Unprofitable Trade:':<30}${trade_stats.lost.pnl.average:.2f}")
    #print(f"{'Largest Winning Trade:':<30}${trade_stats.won.pnl.max:.2f}")
    #print(f"{'Largest Losing Trade:':<30}${trade_stats.lost.pnl.max:.2f}")
    #print(f"{'Sharpe Ratio:':<30}{sharpe_ratio['sharperatio']:.2f}")
    #print(f"{'SQN:':<30}{sqn_value:.2f}")
    #print(f"{'SQN Description:':<30}{description}")
    #print(f"{'Profit Factor:':<30}{trade_stats.won.pnl.total / trade_stats.lost.pnl.total:.2f}")
    #print(f"{'Percentage Gain:':<30}{PercentGain:.2f}%")


    print(f"====================== ADVANCED ===================")
    print(f"{'Initial Starting Cash:':<30}${cerebro.broker.startingcash:.2f}")
    print(colorize_output(cerebro.broker.getvalue(), 'Final Portfolio Value:', 10000, 5400))
    #print(colorize_output((trade_stats.won.pnl.total -trade_stats.lost.pnl.total)), 'Total Profit:', 5000, 1.0)

    print(colorize_output(drawdown.max.len, 'Max Drawdown Days:', 15, 90, reverse=True))
    print(colorize_output(drawdown.max.drawdown, 'Max Drawdown percentage:', 5, 15, reverse=True))
    print(f"{'Total Trades:':<30}{trade_stats.total.closed}")
    print(colorize_output(percent_profitable, 'Percentage Profitable Trades:', 60, 40))
    print(colorize_output(trade_stats.won.pnl.average, 'Average Profitable Trade:', 100, 50))
    print(colorize_output(trade_stats.lost.pnl.average, 'Average Unprofitable Trade:', -20, -trade_stats.won.pnl.average, reverse=True))
    print(colorize_output(trade_stats.won.pnl.max, 'Largest Winning Trade:', 400, 100))
    print(colorize_output(trade_stats.lost.pnl.max, 'Largest Losing Trade:', -50, -200, reverse=True))
    print(colorize_output(sharpe_ratio['sharperatio'], 'Sharpe Ratio:', 2.0, 0.5))
    print(colorize_output(sqn_value, 'SQN:', 7, 0))
    print(f"{'SQN Description:':<30}{description}")
    print(colorize_output(PercentGain, 'Percentage Gain:', 100, 10))




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







# At the end of your main function, call this plotting function
if __name__ == "__main__":
    main()
    
