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
import logging
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import datetime, timedelta
from tqdm import tqdm

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
    lines = ('dist_to_support', 'dist_to_resistance', 'UpProbability')
    
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
        ('UpProbability', 'UpProbability'),
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
                'up_prob': d.UpProbability,
                'dist_to_support': d.dist_to_support,
                'dist_to_resistance': d.dist_to_resistance,
                'UpProbMA': bt.indicators.SimpleMovingAverage(d.UpProbability, period=self.params.fast_period),
            }

            self.inds[d]['UpProbMA'].plotinfo.plot = True
            self.inds[d]['UpProbMA'].plotinfo.subplot = True
            self.inds[d]['UpProbMA'].plotinfo.plotlinelabels = True
            self.inds[d]['UpProbMA'].plotinfo.linecolor = 'blue'
            self.inds[d]['UpProbMA'].plotinfo.plotname = 'Up Probability MA'

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
                if self.datetime.date() >= self.order_cooldown.get(order.data, datetime.date):
                    self.buying_status[order.data._name] = False
                    self.open_positions += 1
                    self.entry_prices[order.data] = order.executed.price
                    self.position_dates[order.data] = self.datetime.date()
                    self.order_list.append(order)
                    self.order_cooldown[order.data] = self.datetime.date() + timedelta(days=1)
            elif order.issell():
                self.open_positions -= 1
                if order.data in self.position_dates:
                    days_held = (self.datetime.date() - self.position_dates[order.data]).days
                    percentage = (order.executed.price - self.entry_prices[order.data]) / self.entry_prices[order.data] * 100
                    trade_result = (order.executed.price - self.entry_prices[order.data]) / self.entry_prices[order.data]
                    if percentage < 0:
                        self.consecutive_losses[order.data] = self.consecutive_losses.get(order.data, 0) + 1
                        loss_count = self.consecutive_losses[order.data]
                        cooldown_days = {1: 7, 2: 28, 3: 90, 4: 282}
                        days = cooldown_days.get(loss_count, 282 if loss_count >= 4 else 0)
                        self.cooldown_end[order.data] = self.datetime.date() + timedelta(days=days)
                    else:
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
        reserved_cash = total_value * self.params.reserve_percent
        investable_capital = total_value - reserved_cash
        chunk_size = total_value * 0.15

        size = int(chunk_size / data.close[0])
        return size if self.broker.getcash() >= chunk_size else 0

    def CloseWrapper(self, data, message):
        self.close(data=data)

    def next(self):
        for d in self.datas:                
            if self.buying_status.get(d._name, False):
                continue

            current_date = self.datetime.date()
            if d in self.entry_prices:
                if d.close[0] <= self.entry_prices[d] * (1 - self.params.stop_loss_percent / 100):
                    self.CloseWrapper(d, "Stop loss")
                    continue

                if d.close[0] >= self.entry_prices[d] * (1 + self.params.take_profit_percent / 100):
                    self.CloseWrapper(d, "Take profit")
                    continue
                
                NewStopLossTesting = True

                if NewStopLossTesting:
                    days_held = (self.datetime.date() - self.position_dates[d]).days
                    InitalTime = 3

                    PercantageList = []

                    for i in range(1, 80):
                        percentageChange = ((d.close[-i] - d.close[-i-1]) / d.close[-i-1]) * 100
                        if percentageChange < 0:
                            PercantageList.append(percentageChange)

                    InitalPercentile = 80.0
                    FinalPercentile = 50.0
                    if days_held < InitalTime:
                        current_stop_loss_percent = abs(np.percentile(PercantageList, InitalPercentile)) if PercantageList else self.params.stop_loss_percent
                    else:
                        current_stop_loss_percent = abs(np.percentile(PercantageList, FinalPercentile)) if PercantageList else self.params.stop_loss_percent

                    if d.close[0] <= self.entry_prices[d] * (1 - current_stop_loss_percent / 100):
                        currentLoss = ((d.close[0] - self.entry_prices[d]) / self.entry_prices[d]) * 100
                        self.CloseWrapper(d, "Stop loss due to stop loss percentage")




            if d in self.position_dates:
                days_held = (self.datetime.date() - self.position_dates[d]).days
                if days_held > self.params.position_timeout:
                    self.CloseWrapper(d, "Position timeout")
                    continue

            if d._name in self.cooldown_end and self.datetime.date() <= self.cooldown_end[d._name]:
                continue

            if self.buying_status.get(d._name, False):
                continue

            if self.open_positions > self.params.max_positions:
                continue

            else:


                current_position = self.getposition(d)
                if not current_position.size > 0:
                    #HighUpProb = np.percentile(self.inds[d]['up_prob'], 95)
                    LowDownProb = np.percentile(self.inds[d]['up_prob'], 99)
                    #randomBuy = random.randint(1, 5) == 1
                    #print(f"LowDownProb: {LowDownProb}")
                    #print(f"UpProbMA: {self.inds[d]['UpProbMA'][0]}")
                    #print(f"RandomBuy: {randomBuy}")
                    BuySignal = (
                        (self.inds[d]['UpProbMA'][0] > LowDownProb)
                        #randomBuy == True
                    )


                    ###make a random buy signal that is 1 in 1000
                    #BuySignal = random.randint(1, 1000) == 1



                    if BuySignal:
                        size = self.calculate_position_size(d)
                        if size > 0:
                            self.buy(data=d, size=size, exectype=bt.Order.StopTrail, trailpercent=self.params.trailing_stop_percent / 100)
                            self.order_cooldown[d] = current_date + timedelta(days=1)
                            self.buying_status[d._name] = True






    def stop(self):
        for d in self.datas:
            if d in self.entry_prices:
                self.close(data=d)





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

        ##ensure that the data has at least 100 rows
        if len(df) < 100:
            FailedFileCounter += 1
            return None


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
        file_paths = select_random_files('Data/RFpredictions', args.BrokerTest)
        for file_path in tqdm(file_paths, desc="Loading Random Files"):
            load_and_add_data(cerebro, file_path, FailedFileCounter)
            if FailedFileCounter > 0:
                logging.info(f"Failed to load {FailedFileCounter} files.")

    # Check if RunTickers is specified
    elif args.RunTickers > 0:
        df_tickers = pd.read_csv('TradingModelStats.csv')
        tickers = df_tickers['Ticker'].tolist()
        file_names = os.listdir('Data/RFpredictions')
        for file_name in tqdm([f for f in file_names if f.endswith('.csv') and f.split('.')[0].lower() in tickers], desc="Loading Ticker Files"):
            file_path = os.path.join('Data/RFpredictions', file_name)
            load_and_add_data(cerebro, file_path, FailedFileCounter)
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


    print(f"{'Initial Starting Cash:':<30}${cerebro.broker.startingcash:.2f}")
    print(colorize_output(cerebro.broker.getvalue(), 'Final Portfolio Value:', 10000, 5400))
    print(colorize_output(drawdown.max.len, 'Max Drawdown Days:', 15, 90, reverse=True))
    print(colorize_output(drawdown.max.drawdown, 'Max Drawdown percentage:', 5, 15, reverse=True))
    print(colorize_output(trade_stats.total.closed, 'Total Trades:', 100, 250, reverse=True))

    #print(f"{'Total Trades:':<30}{trade_stats.total.closed}")
    print(colorize_output(percent_profitable, 'Percentage Profitable Trades:', 60, 40))
    print(colorize_output(trade_stats.won.pnl.average, 'Average Profitable Trade:', 100, 50))
    print(colorize_output(trade_stats.lost.pnl.average, 'Average Unprofitable Trade:', -20, -trade_stats.won.pnl.average, reverse=True))
    print(colorize_output(trade_stats.won.pnl.max, 'Largest Winning Trade:', 500, 100))
    print(colorize_output(trade_stats.lost.pnl.max, 'Largest Losing Trade:', -50, -200, reverse=True))
    print(colorize_output(sharpe_ratio['sharperatio'], 'Sharpe Ratio:', 3.0, 0.5))
    print(colorize_output(sqn_value, 'SQN:', 5.0, 2.0))
    print(f"{'SQN Description:':<30}{description}")
    print(colorize_output(PercentGain, 'Percentage Gain:', 100, 10))

    if args.BrokerTest < 1.0:
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







# At the end of your main function, call this plotting function
if __name__ == "__main__":
    main()
    
