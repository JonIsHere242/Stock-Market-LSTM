#!/root/root/miniconda4/envs/tf/bin/python
import backtrader as bt
from backtrader_ib_insync import IBStore
import threading
import cmd
import logging
import ib_insync as ibi
import pandas as pd
from datetime import datetime
import time 
import sqlite3
import datetime
import pytz
import numpy as np
from datetime import datetime, timedelta

from BuySignalParquet import *
from Trading_Functions import *


# Set up logging
logging.basicConfig(filename='__BrokerLive.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

POSITIONS_FILE = 'positions.csv'

def fetch_and_save_positions(ib):
    positions = ib.positions()
    positions_list = []
    for position in positions:
        if position.position != 0:
            contract = position.contract
            positions_list.append({
                'symbol': contract.symbol,
                'exchange': contract.exchange,
                'secType': contract.secType,
                'position': position.position,
                'averageCost': position.avgCost
            })
            logging.info(f'Position found: {contract.symbol}, {position.position}, {position.avgCost}')
    df = pd.DataFrame(positions_list)
    df.to_csv(POSITIONS_FILE, index=False)
    return df

def load_positions_from_csv():
    try:
        df = pd.read_csv(POSITIONS_FILE)
        return df.to_dict(orient='records')
    except FileNotFoundError:
        logging.warning(f"{POSITIONS_FILE} not found, returning empty positions list")
        return []
    except Exception as e:
        logging.error(f"Error loading {POSITIONS_FILE}: {e}")
        return []

def save_positions_to_csv(positions_data):
    df = pd.DataFrame.from_dict(positions_data, orient='index')
    df.to_csv(POSITIONS_FILE, index=False)

def check_market_status():
    # Define market hours in Eastern Time
    market_open = datetime.time(9, 30)
    market_close = datetime.time(16, 0)
    
    # Get current time in Eastern Time
    eastern_tz = pytz.timezone('US/Eastern')
    now = datetime.datetime.now(eastern_tz)
    
    # Check if it's a weekday
    if now.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        next_open = now.replace(hour=market_open.hour, minute=market_open.minute, second=0, microsecond=0)
        while next_open.weekday() >= 5:
            next_open += datetime.timedelta(days=1)
        return False, next_open
    
    # Convert current time to a time object for comparison
    current_time = now.time()
    
    # Check if market is open
    if market_open <= current_time < market_close:
        next_close = now.replace(hour=market_close.hour, minute=market_close.minute, second=0, microsecond=0)
        return True, next_close
    else:
        if current_time < market_open:
            next_open = now.replace(hour=market_open.hour, minute=market_open.minute, second=0, microsecond=0)
        else:
            next_open = (now + datetime.timedelta(days=1)).replace(hour=market_open.hour, minute=market_open.minute, second=0, microsecond=0)
            while next_open.weekday() >= 5:
                next_open += datetime.timedelta(days=1)
        return False, next_open




def read_buy_signals(csv_file):
    try:
        df = pd.read_csv(csv_file)
        buy_signals = df[df['isBought'] == False]['Ticker'].tolist()
        return buy_signals
    except Exception as e:
        logging.error(f"Error reading {csv_file}: {e}")
        return []

def get_open_positions(ib):
    positions = ib.positions()
    positions_list = []
    for position in positions:
        if position.position != 0:
            contract = position.contract
            positions_list.append((contract, position))
            logging.info(f'Position found: {contract.symbol}, {position.position}')
    return positions_list




def calculate_sharpe_ratio(trades, risk_free_rate=0.02):
    if not trades:
        return 0

    returns = [(trade['exit_price'] - trade['entry_price']) / trade['entry_price'] for trade in trades]
    excess_returns = [r - (risk_free_rate / 252) for r in returns]  # Assuming daily returns

    if len(excess_returns) < 2:
        return 0

    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)  # Annualized

def calculate_win_loss_ratio(trades):
    if not trades:
        return 0

    wins = sum(1 for trade in trades if trade['profit_loss'] > 0)
    losses = sum(1 for trade in trades if trade['profit_loss'] < 0)

    if losses == 0:
        return float('inf') if wins > 0 else 0

    return wins / losses

def calculate_avg_profit_per_trade(trades):
    if not trades:
        return 0

    total_profit = sum(trade['profit_loss'] for trade in trades)
    return total_profit / len(trades)

































class MyStrategy(bt.Strategy):
    params = (
        ('max_positions', 4),
        ('position_size', 0),  # Will be set dynamically
        ('stop_loss_percent', 10),
        ('take_profit_percent', 20),
        ('position_timeout', 9),
        ('expected_profit_per_day_percentage', 0.25),
    )

    def __init__(self):
        self.order_dict = {}
        self.market_open = True
        self.data_ready = {d: False for d in self.datas}
        self.barCounter = 0
        self.trading_data = read_trading_data()

    def notify_data(self, data, status):
        logging.info(f'Data Status => {data._getstatusname(status)}')
        print('Data Status =>', data._getstatusname(status))
        if status == data.LIVE:
            self.data_ready[data] = True

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.process_completed_order(order)
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            self.order_dict.pop(order.data._name, None)

    def next(self):
        self.barCounter += 1
        current_date = bt.num2date(self.datas[0].datetime[0]).date()
        
        is_market_open, _ = self.check_market_status()
        if not is_market_open:
            return

        for data in self.datas:
            if not self.data_ready[data]:
                continue

            symbol = data._name
            position = self.broker.getposition(data)
            
            if position.size != 0:
                self.evaluate_sell_conditions(data, current_date)
            else:
                self.process_buy_signal(data, current_date)
            
            print(f'{symbol}: Current Price: {data.close[0]}')

    def check_market_status(self):
        eastern_tz = pytz.timezone('US/Eastern')
        now = datetime.now(eastern_tz)
        market_open = datetime.time(9, 30)
        market_close = datetime.time(16, 0)
        
        if now.weekday() >= 5:
            return False, None
        
        current_time = now.time()
        if market_open <= current_time < market_close:
            return True, None
        return False, None

    def process_completed_order(self, order):
        symbol = order.data._name
        if order.isbuy():
            self.handle_buy_order(order, symbol)
        elif order.issell():
            self.handle_sell_order(order, symbol)

    def handle_buy_order(self, order, symbol):
        entry_date = self.data.datetime.date(0)
        entry_price = order.executed.price
        mark_position_as_bought(symbol)
        update_buy_signal(symbol, entry_date, entry_price)
        logging.info(f'Buy order executed for {symbol} at {entry_price}')

    def handle_sell_order(self, order, symbol):
        exit_date = self.data.datetime.date(0)
        exit_price = order.executed.price
        entry_data = self.trading_data[self.trading_data['Symbol'] == symbol].iloc[0]
        entry_price = entry_data['LastBuySignalPrice']
        size = order.executed.size
        profit_loss = (exit_price - entry_price) * size

        update_trade_result(symbol, profit_loss < 0)
        logging.info(f'Sell order executed for {symbol} at {exit_price}. Profit/Loss: {profit_loss}')

    def process_buy_signal(self, data, current_date):
        symbol = data._name
        buy_signals = get_buy_signals()
        symbol_data = next((s for s in buy_signals if s['Symbol'] == symbol), None)
        
        if symbol_data and not symbol_data['IsCurrentlyBought'] and symbol not in self.order_dict:
            cash = self.broker.getcash()
            if cash >= self.params.position_size:
                size = int(self.params.position_size / data.close[0])
                order = self.buy(data=data, size=size)
                self.order_dict[symbol] = order
                logging.info(f'Buy order placed for {symbol}')

    def evaluate_sell_conditions(self, data, current_date):
        symbol = data._name
        if symbol in self.order_dict:
            return

        position = self.broker.getposition(data)
        if position.size == 0:
            return

        symbol_data = self.trading_data[self.trading_data['Symbol'] == symbol].iloc[0]
        entry_price = symbol_data['LastBuySignalPrice']
        current_price = data.close[0]

        if self.check_stop_loss(current_price, entry_price):
            self.close_position(data, "Stop loss")
        elif self.check_take_profit(current_price, entry_price):
            self.close_position(data, "Take profit")
        elif self.check_position_timeout(current_date, symbol):
            self.close_position(data, "Position timeout")
        elif self.check_expected_profit(current_date, symbol, current_price, entry_price):
            self.close_position(data, "Expected Profit Per Day not met")

    def check_stop_loss(self, current_price, entry_price):
        return current_price <= entry_price * (1 - self.params.stop_loss_percent / 100)

    def check_take_profit(self, current_price, entry_price):
        return current_price >= entry_price * (1 + self.params.take_profit_percent / 100)

    def check_position_timeout(self, current_date, symbol):
        symbol_data = self.trading_data[self.trading_data['Symbol'] == symbol].iloc[0]
        entry_date = symbol_data['LastBuySignalDate']
        days_held = (current_date - entry_date).days
        return days_held > self.params.position_timeout

    def check_expected_profit(self, current_date, symbol, current_price, entry_price):
        symbol_data = self.trading_data[self.trading_data['Symbol'] == symbol].iloc[0]
        entry_date = symbol_data['LastBuySignalDate']
        days_held = (current_date - entry_date).days
        if days_held > 3:
            total_profit_percentage = ((current_price - entry_price) / entry_price) * 100
            daily_profit_percentage = total_profit_percentage / days_held
            expected_profit_per_day = self.params.expected_profit_per_day_percentage * days_held
            return daily_profit_percentage < expected_profit_per_day
        return False

    def close_position(self, data, reason):
        if data._name not in self.order_dict:
            logging.info(f'Closing {data._name} due to {reason}')
            print(f'Closing {data._name} due to {reason}')
            order = self.close(data=data)
            self.order_dict[data._name] = order

    def stop(self):
        total_portfolio_value = self.broker.getvalue()
        logging.info(f'Final Portfolio Value: {total_portfolio_value}')
        print(f'Final Portfolio Value: {total_portfolio_value}')





















































class TradingCmd(cmd.Cmd):
    intro = 'Welcome to the trading console. Type help or ? to list commands.\n'
    prompt = '(trading) '
    cerebro = None
    store = None
    ib = None

    def do_end(self, arg):
        'Stop the strategy and disconnect from IB.'
        print('Ending the trading session...')
        logging.info('Ending the trading session...')

        try:
            if self.cerebro:
                self.cerebro.runstop()  # Stop the backtrader run loop
                logging.info('Cerebro run loop stopped.')

            if self.ib and self.ib.isConnected():
                self.ib.disconnect()  # Disconnect from IB
                logging.info('Disconnected from Interactive Brokers.')
            elif self.store and hasattr(self.store, 'broker') and self.store.broker:
                if self.store.broker.ib.isConnected():
                    self.store.broker.ib.disconnect()  # Disconnect from IB via store broker
                    logging.info('Disconnected from Interactive Brokers via store broker.')

        except Exception as e:
            logging.error(f'Error ending the trading session: {e}')
            print(f'Error ending the trading session: {e}')
            return False

        print('Trading session ended successfully.')
        logging.info('Trading session ended successfully.')
        return True

    def do_positions(self, arg):
        'List current open positions.'
        if self.ib:
            positions = get_open_positions(self.ib)
            print('Current Positions:', positions)
            logging.info(f'Current Positions: {positions}')
        else:
            print('IB is not connected.')
            logging.info('IB is not connected.')

def is_valid_exchange(exchange):
    return exchange in ['NASDAQ', 'NYSE']








def start():
    cerebro = bt.Cerebro()

    try:
        print('Connecting to Interactive Brokers TWS...')
        logging.info('Connecting to Interactive Brokers TWS...')
        ib = ibi.IB()
        ib.connect('127.0.0.1', 7497, clientId=1)
        store = IBStore(port=7497)
        time.sleep(3)  # Allow time for connection to establish
    except Exception as e:
        print('Failed to connect to Interactive Brokers TWS')
        logging.error(f'Failed to connect to Interactive Brokers TWS: {e}')
        return

    # Load positions from Parquet file
    open_positions = get_open_positions()
    
    # Load buy signals from Parquet file
    buy_signals = get_buy_signals()
    
    all_symbols = set(open_positions + [signal['Symbol'] for signal in buy_signals])

    print(f'Total symbols to trade: {len(all_symbols)}')
    logging.info(f'Total symbols to trade: {len(all_symbols)}')

    # Add data feeds for all symbols
    for symbol in all_symbols:
        try:
            contract = ibi.Stock(symbol, 'SMART', 'USD')
            if contract.exchange in ['NASDAQ', 'NYSE']:
                data = store.getdata(dataname=symbol, sectype=contract.secType, exchange=contract.exchange)
                data._name = symbol
                cerebro.resampledata(data, timeframe=bt.TimeFrame.Seconds, compression=15)
                print(f'Added data feed for {symbol}')
                logging.info(f'Added data feed for {symbol}')
            else:
                print(f'Skipping data for {symbol} - Invalid Exchange: {contract.exchange}')
                logging.info(f'Skipping data for {symbol} - Invalid Exchange: {contract.exchange}')
        except Exception as e:
            print(f'Error adding data for {symbol}: {e}')
            logging.error(f'Error adding data for {symbol}: {e}')

    # Create the broker from the store
    broker = store.getbroker()
    cerebro.setbroker(broker)

    # Add the strategy
    cerebro.addstrategy(MyStrategy)

    # Run the backtrader engine
    try:
        cerebro.run()
    except Exception as e:
        print(f'Error during Cerebro run: {e}')
        logging.error(f'Error during Cerebro run: {e}')
    finally:
        # Ensure we disconnect from IB even if an error occurs
        if ib.isConnected():
            ib.disconnect()
            print('Disconnected from Interactive Brokers TWS')
            logging.info('Disconnected from Interactive Brokers TWS')

if __name__ == '__main__':
    start()