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


    ##('commission', 0.006),   # USD per share
    ##('clearing_fee', 0.00017),  # CAD per share
    ##('sec_fee', 0.00011),       # CAD per share


class InteractiveBrokersCommissionScheme(bt.CommInfoBase):
    params = (
        ('commission', 0.00),   # USD per share
        ('clearing_fee', 0.0000),  # CAD per share
        ('sec_fee', 0.00000),       # CAD per share
        ('stocklike', True),
        ('commtype', bt.CommInfoBase.COMM_FIXED)  # Use COMM_FIXED for fixed commissions
    )

    def _getcommission(self, size, price, pseudoexec):
        """
        Calculate the commission for a trade in terms of size and price.
        """
        return abs(size) * (self.p.commission + self.p.clearing_fee + self.p.sec_fee)

config = {
    "initialCash": 5000,
    "commission": 0.0,
    "plotShow": True,
}

def LoggingSetup():
    loggerfile = "Data/RFpredictions/__BackTesting.log"
    # Configure logging to append to the log file, with a specific format and date
    logging.basicConfig(filename=loggerfile, level=logging.INFO, 
                        format='%(asctime)s %(levelname)s: %(message)s', 
                        datefmt='%Y-%m-%d %H:%M:%S',
                        filemode='a')  # 'a' for append mode

def arg_parser():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Backtesting Trading Strategy on Stock Data")
    parser.add_argument("--file", type=str, default=None, help="Path to the CSV file to backtest on")
    parser.add_argument("--randomfile", action='store_true', help="Select a random file from the directory for backtesting")
    parser.add_argument("--RandomFilePercent", type=int, default=100, help="Percentage of random files to backtest")
    return parser.parse_args()
  
def select_random_file(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    if not files:
        logging.error("No CSV files found in the directory")
        return None
    return os.path.join(directory, random.choice(files))

def select_random_files(directory, percent):
    all_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    num_files = len(all_files)
    num_to_select = max(1, (num_files * percent) // 100)
    selected_files = random.sample(all_files, num_to_select)
    return [os.path.join(directory, f) for f in selected_files]

class CustomPandasData(bt.feeds.PandasData):
    lines = ('prediction', 'UpProb', 'DownProb', "UnsureProb", 'MagnitudePrediction', 'Open', 'High', 'Low', 'Close', 'Volume')

    params = (
        ('datetime', -1),
        ('Open', 'Open'),
        ('High', 'High'),
        ('Low', 'Low'),
        ('Close', 'Close'),
        ('Volume', 'Volume'),
        ('prediction', "Prediction"),
        ('UpProb', "UpProb"),
        ('DownProb', "DownProb"),
        ('UnsureProb', "UnsureProb"),
        ('MagnitudePrediction', "MagnitudePrediction"),
        ("cooldown_period", 60),
    )
    datafields = bt.feeds.PandasData.datafields + ['prediction', 'UpProb', 'DownProb', "UnsureProb", 'MagnitudePrediction', 'Open', 'High', 'Low', 'Close', 'Volume']

def load_data(file_path):
    try:
        df = pd.read_csv(file_path, parse_dates=['Date'])
        if not all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'Prediction', "MagnitudePrediction"]):
            logging.error(f"Missing required columns in the data: {file_path}")
            return None
        return df
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
    except Exception as e:
        logging.error(f"An error occurred while reading {file_path}: {e}")
        return None


class MovingAverageCrossoverStrategy(bt.Strategy):

    params = (
        ('filename', None),  # Add 'filename' parameter
        ("Abs_StopLoss", 5.0),
        ("Abs_TakeProfit", 30.0),
        ("Max_DaysToHold", 7.0),
        ("max_investment_per_trade", 20),
        ("cooldown_period", 60),
    )

    def __init__(self):
        self.filename = self.params.filename
        self.ticker = os.path.basename(self.params.filename).split('.')[0]
        self.current_date = datetime.now().date()
        self.order = None
        self.entry_price = None
        self.candle_counter = 0
        self.is_long_position = False
        self.position_size = 0  # Added to track the size of the position
        self.start_cash = self.broker.get_cash()
        self.total_profit = 0
        self.bars_since_last_loss = 0
        self.mean_slope_history = []
        self.atr = bt.indicators.AverageTrueRange(period=14)
        self.very_low_up_prob_indices = []  # To store the indices of very low UpProb values
        self.trade_occurred = False
        self.trades_data = []
        self.short_term_ma = bt.indicators.SimpleMovingAverage(self.data.close, period=5)
        self.long_term_ma = bt.indicators.SimpleMovingAverage(self.data.close, period=60)
        self.base_max_investment_per_trade = self.params.max_investment_per_trade  # Store the base max investment value
        self.atr_stop_loss = None  # To store the ATR-based stop loss level
        self.closed_trades = []
        self.position_holding_days = 0
        self.MaxDaysToHold = 7


    def next(self):
        self.bars_since_last_loss += 1
        BuySignal = False
        HoldSignal = False

        if self.bars_since_last_loss < self.params.cooldown_period:
            return

        if self.order:
            return
        
        if len(self.data.UpProb) < 91:
            return
        
        if len(self.data.MagnitudePrediction) < 91:
            print("Not enough data")
            return

        if self.data.close[0] < 1.5:
            if self.is_long_position:
                self.sell_trade()
            return
        

        if self.data.close[0] > 250.0:
            if self.is_long_position:
                self.sell_trade()
            return
        
        
        

        ####Method 1 working with Profit Factor: 1.43
        ##AtrMagnatude = self.data.MagnitudePrediction[0] - self.atr[0]
        ##HighMagnatude = np.percentile(self.data.MagnitudePrediction.get(size=len(self.data.MagnitudePrediction)), 50) - self.atr[0]
        ##HighUpProb = np.percentile(self.data.UpProb.get(size=len(self.data.MagnitudePrediction)), 1.0)
        ##BuySignal = self.data.UpProb[0] < HighUpProb and AtrMagnatude < 0 ##and HighMagnatude
        ##HoldSignal = self.data.UpProb[0] > HighUpProb*1.3


        ####method 2 PF 1.37 lower trade volume
        ##PercentageAtr = self.atr[0] / self.data.close[0] * 100
        ##ExpectedReturn = PercentageAtr * self.base_max_investment_per_trade
        ##AtrMagnatude = self.data.MagnitudePrediction[0] - self.atr[0]
        ##HighUpProb = np.percentile(self.data.UpProb.get(size=len(self.data.MagnitudePrediction)), 1.0)
        ##BuySignal = self.data.UpProb[0] < HighUpProb and AtrMagnatude < 0 and ExpectedReturn > 100 and PercentageAtr > 5.0
        ##HoldSignal = self.data.UpProb[0] > HighUpProb*1.2

        ##method 3 PF 2.0 lower trade volume 174 trades
        ##PercentageAtr = self.atr[0] / self.data.close[0] * 100
        ##AtrMagnatude = self.data.MagnitudePrediction[0] - self.atr[0]
        ##ExpectedReturn = PercentageAtr * self.base_max_investment_per_trade
        ##HighUpProb = np.percentile(self.data.UpProb.get(size=len(self.data.MagnitudePrediction)), 5.0)
        ##BuySignal = self.data.UpProb[0] < HighUpProb and AtrMagnatude < 0 and ExpectedReturn > 100 and PercentageAtr > 10.3 
        ##HoldSignal = self.data.UpProb[0] > HighUpProb*1.2

        ##method 4 PF 2.32 lower trade volume 61 trades 50%APR
        ##PercentageAtr = self.atr[0] / self.data.close[0] * 100
        ##AtrMagnatude = self.data.MagnitudePrediction[0] - self.atr[0]
        ##ExpectedReturn = PercentageAtr * self.base_max_investment_per_trade
        ##HighUpProb = np.percentile(self.data.UpProb.get(size=len(self.data.MagnitudePrediction)), 5.0)
        ##BuySignal = self.data.UpProb[0] < HighUpProb and AtrMagnatude < 0 and ExpectedReturn > 300 and PercentageAtr > 10.3
        ##HoldSignal = self.data.UpProb[0] > HighUpProb*1.3

        ##method 5 PF 3.07 lower trade volume 55 trades 50%APR
        ##PercentageAtr = self.atr[0] / self.data.close[0] * 100
        ##AtrMagnatude = self.data.MagnitudePrediction[0] - self.atr[0]
        ##ExpectedReturn = PercentageAtr * self.base_max_investment_per_trade
        ##HighUpProb = np.percentile(self.data.UpProb.get(size=len(self.data.MagnitudePrediction)), 8.0)
        ##BuySignal = self.data.UpProb[0] < HighUpProb and AtrMagnatude < 0 and ExpectedReturn > 300 and PercentageAtr > 10.3
        ##HoldSignal = self.data.UpProb[0] > HighUpProb*1.15



        ##===========================[ Strategy Implementation ]========================##
        ##===========================[ Strategy Implementation ]========================##
        ##===========================[ Strategy Implementation ]========================##


        
        PercentageAtr = self.atr[0] / self.data.close[0] * 100
        AtrMagnatude = self.data.MagnitudePrediction[0] - self.atr[0]
        ExpectedReturn = PercentageAtr * self.base_max_investment_per_trade
        HighUpProb = np.percentile(self.data.UpProb.get(size=len(self.data.MagnitudePrediction)), 8.0)
        BuySignal = self.data.UpProb[0] < HighUpProb and AtrMagnatude < 0 and ExpectedReturn > 100 and PercentageAtr > 10.3
        HoldSignal = self.data.UpProb[0] > HighUpProb*1.15


        ##==============================[ Strategy Buy/sell ]============================##
        ##==============================[ Strategy Buy/sell ]============================##
        ##==============================[ Strategy Buy/sell ]============================##

        MatchingDates = self.data.datetime.date(0) == self.current_date

        if BuySignal or HoldSignal and MatchingDates:
            trade_info = {
                'Ticker': self.ticker,  # Assuming you have a way to get the ticker
                'BuySignal': BuySignal,
                'HoldSignal': HoldSignal,
                'SignalTime': self.data.datetime.datetime(0),
            }
            self.trades_data.append(trade_info)


        if self.is_long_position:
            self.position_holding_days += 1
            CurrentProfitPercentage = (self.data.close[0] - self.entry_price) / self.entry_price * 100    

            if CurrentProfitPercentage > self.params.Abs_TakeProfit:      
                print(CurrentProfitPercentage)
                self.sell_trade()
                return

            if CurrentProfitPercentage < -self.params.Abs_StopLoss:
                self.sell_trade()
                return

            for i in range(1, 9):
                if self.position_holding_days == i and CurrentProfitPercentage > i*0.5:
                    self.sell_trade()            
            



        if self.position_holding_days >= self.MaxDaysToHold:
            print(self.position_holding_days)
            self.sell_trade()


        if BuySignal and not self.is_long_position:
            self.dynamic_investment_allocation()
            self.buy_trade()
            logging.info(f"Filename: {self.filename}, Buy Signal Detected - Price: {self.data.close[0]:.2f}, ATR: {self.atr[0]:.2f}, UpProb: {self.data.UpProb[0]:.2f}, DownProb: {self.data.DownProb[0]:.2f}, UnsureProb: {self.data.UnsureProb[0]:.2f}, MagnitudePrediction: {self.data.MagnitudePrediction[0]:.2f}")
            self.trailing_stop = self.data.close[0] * (1 - self.params.Abs_StopLoss / 100)  # Initialize trailing stop
            return
        
        
        if self.is_long_position:
            if not HoldSignal:
                if self.data.close[0] > self.entry_price * (1 + self.params.Abs_TakeProfit / 100):
                    self.sell_trade()
                    return
        
        
        if self.is_long_position:
            # Update trailing stop
            self.trailing_stop = max(self.trailing_stop, self.data.close[0] * (1 - self.params.Abs_StopLoss / 100))

            # Check if we should sell
            if self.position_holding_days >= self.params.Max_DaysToHold or self.data.close[0] < self.trailing_stop:
                self.sell_trade()
                return

            if self.data.close[0] < self.entry_price * (1 - self.params.Abs_TakeProfit / 100):
                self.sell_trade()
                return
 



    def calculate_atr_fraction(self, current_profit):
        # Decrease the ATR fraction as profit increases
        if current_profit > 1:
            return 0.5  # Tighter stop-loss for higher profit
        return 1.0  # Looser stop-loss initially






    def dynamic_investment_allocation(self):
        UpProbMean = np.mean(self.data.UpProb.get(size=len(self.data.UpProb)))
        UpProbStd = np.std(self.data.UpProb.get(size=len(self.data.UpProb)))
        UpProbZ = (self.data.UpProb[0] - UpProbMean) / UpProbStd

        if len(self.data) < 30:  # Use the longer MA period
            return  # Skip the allocation adjustment

        # Condition for Negative Momentum and High Volatility
        negative_momentum = self.short_term_ma[0] < self.long_term_ma[0]
        high_volatility = self.atr[0] > np.mean(self.atr.get(size=30))  # High ATR compared to recent average

        # Adjust Scaling Factor based on Market Conditions
        scaling_factor = 1 + 0.2 * max(0, UpProbZ)
        if negative_momentum and high_volatility:
            scaling_factor *= 0.5  # Reduce the scaling factor by 50% in negative momentum
        self.params.max_investment_per_trade = self.base_max_investment_per_trade * scaling_factor

       


    def buy_trade(self):
        max_investment = (self.params.max_investment_per_trade / 100) * self.broker.get_cash()
        self.position_size = max_investment / self.data.close[0]
        self.position_size = int(self.position_size)
        self.buy(size=self.position_size, coc=False)
        
        self.is_long_position = True
        self.position_holding_days = 1
        atr_value = self.atr[0]
        self.atr_stop_loss = self.data.close[0] - 0.25 * atr_value






    def sell_trade(self):
        self.order = self.sell(size=self.position_size, coc=False)
        self.is_long_position = False
        self.position_holding_days = 0 
        self.position_size = 0  # Reset position size

    def notify_order(self, order):
        if order.status in [order.Completed]:
            if order.isbuy():
                self.entry_price = order.executed.price
                # Log details at the time of order execution
                logging.info(f"Filename: {self.filename}, Buy Order Executed - Size: {order.executed.size}, Price: {order.executed.price:.2f}, Value: {order.executed.value:.2f}")
            elif order.issell():
                self.entry_price = None
                # Log details at the time of order execution
                logging.info(f"Filename: {self.filename}, Sell Order Executed - Size: {order.executed.size}, Price: {order.executed.price:.2f}, Value: {order.executed.value:.2f}")
            self.candle_counter = 0
            self.order = None

    def has_trade_occurred(self):
        return self.trade_occurred



    def notify_trade(self, trade):
        if trade.justopened:
            self.entry_date = self.datas[0].datetime.date(0)  # Log entry date
            self.trade_occurred = True

        if trade.isclosed:
            exit_date = self.datas[0].datetime.date(0)  # Log exit date
            duration = (exit_date - self.entry_date).days
            self.log_trade(trade, self.entry_date, exit_date, duration)
            self.entry_date = None  # Reset entry date

            if trade.pnl < 0:
                self.bars_since_last_loss = 0


    def log_trade(self, trade, entry_date, exit_date, duration):
        if trade.isclosed:
            position_size = self.position_size
            mean_close = np.mean(self.datas[0].close.get(size=len(self.datas[0].close)))
            std_dev_close = np.std(self.datas[0].close.get(size=len(self.datas[0].close)))
            logging.info(f"Filename: {self.filename}, Trade Closed - EntryDate: {entry_date}, ExitDate: {exit_date}, Duration: {duration} days, PnL: {trade.pnl:.2f}, Mean Close Price: {mean_close:.2f}, Std Dev Close Price: {std_dev_close:.2f}")
            self.total_profit += trade.pnl

    def stop(self):
        
        if self.is_long_position:
            self.sell_trade()

        if self.data.datetime.date(0) != self.current_date:
            debug = False
            if debug:
                logging.error(f"Last date in the data does not match the current date: {self.current_date}")
            
        trades_df = pd.DataFrame(self.trades_data)
        trades_df.to_csv("__trades_data.csv", index=False)







def get_ticker(file_path):
    return os.path.basename(file_path).split('.')[0]








def data_integrity_check(df):
    len_df = len(df)
    dupe_counter = 0
    AccecpatbleDataDupe = 0.05
    if len_df != len(df['Close'].unique()):
        dupe_counter += 1
        if len_df / dupe_counter > AccecpatbleDataDupe:
            logging.error("Data contains only duplicate rows beyond acceptable limit")
            logging.error(f"Data contains {dupe_counter} duplicate rows")
            dataloss = len_df / dupe_counter > AccecpatbleDataDupe
            print(f"Data has {dataloss} duplicate rows")
            return False
    return True



def collect_stats(strategy, filename):
    starting_cash = strategy.broker.startingcash
    ending_cash = strategy.broker.getvalue()
    trades = strategy.closed_trades
    num_trades = len(trades)
    num_profitable_trades = sum(1 for trade in trades if trade.pnl > 0)
    num_unprofitable_trades = num_trades - num_profitable_trades
    perc_profitable_trades = (num_profitable_trades / num_trades) * 100 if num_trades > 0 else 0
    mean_pl = sum(trade.pnl for trade in trades) / num_trades if num_trades > 0 else 0
    max_gain = max(trade.pnl for trade in trades) if trades else 0
    max_loss = min(trade.pnl for trade in trades) if trades else 0

    logging.info(f"Filename: {filename}, Date: {datetime.now()}, StartingCash: {starting_cash}, EndingCash: {ending_cash}, NumberOfTrades: {num_trades}, NumberOfProfitableTrades: {num_profitable_trades}, NumberOfUnprofitableTrades: {num_unprofitable_trades}, PercentageOfProfitableTrades: {perc_profitable_trades}, MeanPL: {mean_pl}, MaxGain: {max_gain}, MaxLoss: {max_loss}")
    return {
        'starting_cash': starting_cash,
        'ending_cash': ending_cash,
        'num_trades': num_trades,
        'num_profitable_trades': num_profitable_trades,
        'num_unprofitable_trades': num_unprofitable_trades,
        'perc_profitable_trades': perc_profitable_trades,
        'mean_pl': mean_pl,
        'max_gain': max_gain,
        'max_loss': max_loss
    }


def delete_small_files(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    for file in files:
        df = pd.read_csv(os.path.join(directory, file))
        if df.shape[0] < 282*3:
            os.remove(os.path.join(directory, file))



def main():
    LoggingSetup()
    args = arg_parser()
    open('Data/RFpredictions/__BackTesting.log', 'w').close()

    trade_occurred = False
    SkippedFiles = 0
    
    if args.RandomFilePercent > 0 and not args.randomfile:
        file_paths = select_random_files('Data/RFpredictions', args.RandomFilePercent)
        for file_path in tqdm(file_paths, desc="Running Simulation"):
            
            df = load_data(file_path)
            if data_integrity_check == False:
                continue
            if df.shape[0] < 90:  # Minimum number of rows required for the strategy
                print(f"Insufficient data for file: {file_path}")
                continue
            
            df = df.tail(282 * 1)  # Modify this line if needed
            if df is not None:
                cerebro = bt.Cerebro()
                cerebro.broker.set_coc(True)
                data = CustomPandasData(dataname=df, datetime=0)
                cerebro.adddata(data)
                cerebro.broker.setcash(config['initialCash'])
                cerebro.addstrategy(MovingAverageCrossoverStrategy, filename=file_path)
                cerebro.run()

            else:
                logging.error(f"Data loading failed for file: {file_path}")



    else:
        while not trade_occurred:
            file_path = args.file if args.file else select_random_file('Data/RFpredictions')
            if file_path:
                df = load_data(file_path)
                if len(df) < 90:
                    SkippedFiles += 1
                    print(f"Skipping file: {file_path} Skipped Files: {SkippedFiles}")
                    continue
                df = df.tail(282)  # Modify this line if needed
                cerebro = bt.Cerebro()
                cerebro.broker.set_coc(True)
                data = CustomPandasData(dataname=df, datetime=0)
                cerebro.adddata(data)
                cerebro.broker.setcash(config['initialCash'])
                strategy = MovingAverageCrossoverStrategy
                cerebro.addstrategy(strategy, filename=file_path)
                results = cerebro.run()
                strategy_instance = results[0]
                if strategy_instance.has_trade_occurred():
                    trade_occurred = True
                    if config["plotShow"]:
                        plot_trades(cerebro, file_path)
                    else:
                        print(f"No trades occurred for file: {file_path}")
            else:
                print("No valid file path provided.")
                break

def plot_trades(cerebro, file_path):
    plt.style.use('dark_background')
    plt.rcParams['figure.facecolor'] = '#424242'
    plt.rcParams['axes.facecolor'] = '#424242'
    plt.rcParams['grid.color'] = 'None'
    cerebro.plot(style='candlestick')
    plt.get_current_fig_manager().set_window_title(file_path)  # Set window title

if __name__ == "__main__":
    main()
