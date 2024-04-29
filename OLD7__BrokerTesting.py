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


def LoggingSetup():
    loggerfile = "__BrokerLog.log"
    logging.basicConfig(
        filename="__BrokerLog.log",
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filemode='a'  # Append mode
    )

 




def arg_parser():
    parser = argparse.ArgumentParser(description="Backtesting Trading Strategy on Stock Data")
    parser.add_argument("--BrokerTest", type=float, default=0, help="Percentage of random files to backtest")
    return parser.parse_args()
  


class CustomPandasData(bt.feeds.PandasData):
    lines = ('UpProb', 'DownProb')

    params = (
        ('datetime', None),  # use None for index, or column name for datetime field
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
        ('UpProb', "UpProb"),
        ('DownProb', "DownProb"),
        ('openinterest', None),  # Pandas data may not have this column by default
    )


class MovingAverageCrossoverStrategy(bt.Strategy):
    params = (
        ("Abs_StopLoss", 5.0),
        ("Abs_TakeProfit", 100.0),
        ("Max_DaysToHold", 5),
        ("max_investment_per_trade", 15),
        ("cooldown_period", 10),  # base cooldown period in days
        ("max_positions", 4),
        ("percent_per_trade", 15),
        ("reserve_percent", 40),
        ("base_cooldown", 10),  # base cooldown period in days
        ("significant_loss_threshold", 20),  # threshold for significant loss in percent

    )

    def __init__(self):
        self.inds = {}
        self.open_positions = 0
        self.start_cash = self.broker.get_cash()
        for data in self.datas:
            self.inds[data._name] = dict(
                order=None,
                entry_price=None,
                position_size=0,
                is_long_position=False,
                bars_since_last_loss=0,
                atr=bt.indicators.AverageTrueRange(data, period=14),
                trades_data=[],
                current_date=datetime.now().date(),
                total_profit=0,
                trade_occurred=False,
                position_holding_days=0,
                max_position_holding_days = 10,
                last_trade_loss=False,
                days_since_last_trade=0,
                consecutive_losses=0,  # tracks consecutive losses
                last_percent_loss=0,  # to track the percent loss of the last trade

                filename=os.path.basename(data._name).split('.')[0] if data._name else "Unknown"
            )



    def can_buy(self):
        """Check if a new position can be opened"""
        if self.open_positions < self.params.max_positions:
            return True
        else:
            return False

    def calculate_position_size(self, data):
        account_value = self.broker.get_value()  # Total value of the account including current cash and holdings
        reserve = account_value * 0.4  # 40% reserve
        available_for_trading = account_value - reserve  # 60% of account value available for trading
        chunk_size = available_for_trading / 4  # Divide the available amount by 4

        # Check if the cash on hand is sufficient to consider for a chunk investment
        if self.broker.get_cash() > chunk_size:
            size = int(chunk_size / data.close[0])  # Number of shares you can buy with one chunk
            ##print(f"Account Value: {account_value}, Reserve: {reserve}, Available for Trading: {available_for_trading}, \
            ##       Chunk Size: {chunk_size}, Tradable Size: {size}")
            return size
        else:
            print(f"Insufficient cash to invest. Cash available: {self.broker.get_cash()}, Required: {chunk_size}")
            return 0


        ##==========================[Main Strategy Logic]===============================##
        ##==========================[Main Strategy Logic]===============================##
        ##==========================[Main Strategy Logic]===============================##




    def next(self):
        for data in self.datas:
            d = self.inds[data._name]
            d['days_since_last_trade'] += 1  

            if d['is_long_position']:
                d['position_holding_days'] += 1



            if self.can_buy(data, d):
                self.check_signals(data)
            



    def check_signals(self, data):
        d = self.inds[data._name]



        # Debugging outputs for buy and sell logic
        if not d['is_long_position'] and self.open_positions < self.params.max_positions and not d['trade_occurred']:
            if self.calculate_buy_signals(data, d) and self.calculate_hold_signals(data, d):
                size = self.calculate_position_size(data)
                ##make sure the price is above 1 and below 400
                if size > 0 and (data.close[0] > 1) and (data.close[0] < 400):
                    try :
                        ##logging.info(f"Successfully bought {size} shares of {data._name} at {data.close[0]}")
                        d['order'] = self.buy(data=data, size=size)

                        d['is_long_position'] = True
                        self.open_positions += 1
                        d['position_size'] = size
                        d['entry_price'] = data.close[0]  # Set entry price here
                        ##print(f"Buying {size} shares of {data._name} at price {d['entry_price']}.")
                    except Exception as e:
                        logging.error(f"Error buying {size} shares of {data._name} at {data.close[0]}. Error: {e}")






        if d['is_long_position']:

            if d['entry_price'] is not None:  # Ensure entry price is not None
                # Sell if the price is above 400 to weed out the shitters
                if data.close[0] < 1:
                    self.log_trade_stats(data, d, self.open_positions, 'PriceBelow1Sell')
                    self.ClosingTrade(data, d)
                    return

                #sell if the price is above 400
                if data.close[0] > 400:
                    self.log_trade_stats(data, d, self.open_positions, 'PriceAbove400Sell')
                    self.ClosingTrade(data, d)
                    return

                # Stop loss logic
                if ((data.close[0] - d['entry_price']) / d['entry_price']) * 100 < -self.params.Abs_StopLoss and d['position_size'] > 0:
                    self.log_trade_stats(data, d, self.open_positions, 'StoplossSell')
                    self.ClosingTrade(data, d)
                    return

                ##relative take profit based on atr 
                if (data.close[0] - d['entry_price']) > 3 * d['atr'][0] and d['position_size'] > 0:
                    self.log_trade_stats(data, d, self.open_positions, 'ATRTakeProfitSell')
                    self.ClosingTrade(data, d)
                    return

                #absolute take profit based on take profit percentage in the params 
                if ((data.close[0] - d['entry_price']) / d['entry_price']) * 100 > self.params.Abs_TakeProfit and d['position_size'] > 0:
                    self.log_trade_stats(data, d, self.open_positions, 'ABSTakeProfitSell')
                    self.ClosingTrade(data, d)
                    return

            if d['position_holding_days'] > 1:
                if d['position_holding_days'] > self.params.Max_DaysToHold or not self.calculate_hold_signals(data, d) and d['position_size'] > 0:
                    self.log_trade_stats(data, d, self.open_positions, 'TimeOutSell')
                    self.ClosingTrade(data, d)
                    return





    def ClosingTrade(self, data, d):
        d['order'] = self.close(data=data)
        d['is_long_position'] = False
        self.open_positions -= 1
        d['position_holding_days'] = 0
        d['position_size'] = 0




    def log_trade_stats(self, data, d, openpositions, action_flag):
        trade_stats = {
            "AccountValue": self.broker.get_value(),
            'Timestamp': data.datetime.date(0),
            'Filename': d['filename'],
            'OpenPositions': openpositions,
            'Action': action_flag,
            'ClosePrice': data.close[0],
            'EntryPrice': d['entry_price'] if d['entry_price'] is not None else 'N/A',
            'PercentageChange': round(((data.close[0] - d['entry_price']) / d['entry_price']) * 100, 2) if d['entry_price'] else 'N/A',


            'ATR': round(d['atr'][0], 2) if 'atr' in d else 'N/A',
            'HoldingDays': d['position_holding_days'],
            'PositionSize': d['position_size']
        }
        log_message = f"Trade Stats with Cooldown: {trade_stats}"
        logging.info(log_message)







    def calculate_buy_signals(self, data, d):
        if len(data) < 70:
            return False
        HighUpProb = np.percentile(data.UpProb.get(size=70), 85)  # Top 10% high confidence for upward trend
        LowDownProb = np.percentile(data.DownProb.get(size=70), 15)  # Bottom 10% low confidence for downward trend


        PercentageAtr = d['atr'][0] / data.close[0] * 100
        BuySignal = (
                data.UpProb[0] > HighUpProb and
                data.DownProb[0] < LowDownProb and
                PercentageAtr > 7.5
        )
        return BuySignal





    def calculate_hold_signals(self, data, d):
        if len(data) < 70:
            return False
        
        HighUpProb = np.percentile(data.UpProb.get(size=70), 60)
        HoldSignal = (data.UpProb[0] > HighUpProb)

        return HoldSignal











    def can_buy(self, data, d):
        """Enhanced check if a new position can be opened considering additional loss-based cooldown"""
        base_cooldown = self.params.base_cooldown * (2 ** d['consecutive_losses'])
        significant_loss_cooldown = d['last_percent_loss'] + 1  # Extra days for each percent over threshold
        cooldown = max(base_cooldown, significant_loss_cooldown)
        if (self.open_positions < self.params.max_positions and
                d['days_since_last_trade'] >= cooldown and
                not d['last_trade_loss']):
            return True
        return False



    def notify_order(self, order):
        d = self.inds[order.data._name]
        if order.status in [order.Completed]:
            if order.isbuy():
                d['position_size'] += order.executed.size
            elif order.issell():
                d['position_size'] -= order.executed.size
                # Calculate percent loss
                percent_loss = ((order.executed.price - d['entry_price']) / d['entry_price']) * 100
                if percent_loss < 0:
                    d['last_trade_loss'] = True
                    d['consecutive_losses'] += 1
                    d['last_percent_loss'] = abs(percent_loss)  # store the absolute value of percent loss
                    if abs(percent_loss) > self.params.significant_loss_threshold:
                        d['last_percent_loss'] = abs(percent_loss) - self.params.significant_loss_threshold
                    else:
                        d['last_percent_loss'] = 0
                else:
                    d['last_trade_loss'] = False
                    d['consecutive_losses'] = 0
                    d['last_percent_loss'] = 0
                if d['position_size'] < 0:
                    logging.error(f"Negative position size for {order.data._name}: {d['position_size']}")
                    d['position_size'] = 0
            d['order'] = None




    def stop(self):
        for data in self.datas:
            d = self.inds[data._name]
            if d['is_long_position']:
                self.ClosingTrade(data, d)  # Corrected call
                self.log_trade_stats(data, d, self.open_positions, 'EndOfBacktestSell')
































##utils ===================================================================================
##utils ===================================================================================
##utils ===================================================================================

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
    

    df = df.iloc[-282:]
    ##remove the last 10 rows
    



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
    ##open("Data/RFpredictions/__BackTesting.log", 'w').close()

    timer = time.time()
    cerebro = bt.Cerebro()
    cerebro.broker.set_cash(5000)
    args = arg_parser()
    FailedFileCounter = 0
    for file_path in select_random_files('Data/RFpredictions', args.BrokerTest):
        load_and_add_data(cerebro, file_path, FailedFileCounter)
    logging.info(f"Failed to load {FailedFileCounter} files.")
    cerebro.addstrategy(MovingAverageCrossoverStrategy)
    cerebro.run()
    print(f"Time taken: {time.time() - timer:.2f} seconds")
    print("Final Portfolio Value: $%.2f" % cerebro.broker.getvalue())
    print("Cash: $%.2f" % cerebro.broker.getcash())

    if args.BrokerTest < 0.5: 
        plt.style.use('dark_background')
        plt.rcParams['figure.facecolor'] = '#424242'
        plt.rcParams['axes.facecolor'] = '#424242'
        plt.rcParams['grid.color'] = 'None'
        
        cerebro.plot(style='candlestick', iplot=False, start=datetime.now().date() - pd.DateOffset(days=282))


if __name__ == "__main__":
    main()
