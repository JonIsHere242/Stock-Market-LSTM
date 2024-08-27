#!/root/root/miniconda4/envs/tf/bin/python
import backtrader as bt
from backtrader_ib_insync import IBStore
import threading
import cmd
import logging
from logging.handlers import RotatingFileHandler

import ib_insync as ibi
import pandas as pd
import sqlite3
import numpy as np



####Time related headaches
from datetime import datetime, time
import pytz





from Trading_Functions import *
import traceback


def setup_logging(log_to_console=True):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create a rotating file handler
    file_handler = RotatingFileHandler('__BrokerLive.log', maxBytes=1024 * 1024, backupCount=5)
    file_handler.setLevel(logging.INFO)

    # Create a formatter and add it to the handler
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    if log_to_console:
        # Create a console handler only if log_to_console is True
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

# You can now control logging to the console by setting this parameter when you call setup_logging
setup_logging(log_to_console=False)  # Set to False to disable logging to the terminal





def get_open_positions(ib):
    try:
        positions = ib.positions()
        positions_list = []
        for position in positions:
            if position.position != 0:
                contract = position.contract
                positions_list.append(contract.symbol)
                logging.info(f'Position found: {contract.symbol}, {position.position}')
        
        if not positions_list:
            logging.info('No open positions found')
        
        return positions_list
    except Exception as e:
        logging.error(f'Error fetching positions: {e}')
        return []











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

        # Add a timer for heartbeat
        self.add_timer(
            when=bt.Timer.SESSION_START,
            offset=timedelta(minutes=0),  # No offset at the session start
            repeat=timedelta(minutes=1),  # Repeat every minute
            weekdays=[0, 1, 2, 3, 4],  # Monday to Friday
        )


    def check_market_status(self):
        eastern_tz = pytz.timezone('US/Eastern')
        now = datetime.now(eastern_tz)
        market_open = time(9, 30)
        market_close = time(16, 0)
        
        if now.weekday() >= 5:
            return False, None
        
        current_time = now.time()
        if market_open <= current_time < market_close:
            return True, None
        return False, None




    def notify_timer(self, timer, when, *args, **kwargs):
        ##change this to be a simple heartbeat check 
        print(f'Heartbeat: {self.safe_get_date()}')




    def safe_get_date(self):
        try:
            # Using Backtrader's num2date to convert plot datetime numbers to datetime objects
            return bt.num2date(self.datas[0].datetime[0]).date()
        except AttributeError:
            # If datetime is an integer, convert it using fromtimestamp
            return datetime.fromtimestamp(self.datas[0].datetime[0], tz=datetime.timezone.utc).date()
        except Exception as e:
            logging.error(f"Error getting date: {e}")
            # Fallback to current UTC date
            return datetime.now(datetime.timezone.utc).date()



    def notify_data(self, data, status, *args, **kwargs):
        if status == data.LIVE:
            self.data_ready[data] = True
            logging.info(f'Data Status: {data._name} is now LIVE')
        else:
            self.data_ready[data] = False
            logging.info(f'Data Status: {data._name} is {data._getstatusname(status)}')

    def notify_order(self, order):
        if order.status in [order.Completed]:
            self.process_completed_order(order)
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            self.order_dict.pop(order.data._name, None)




    def next(self):
            self.barCounter += 1
            current_date = self.safe_get_date()
            logging.info(f"Next method called. Bar: {self.barCounter}, Date: {current_date}")

            if not self.market_open:
                logging.info("Market is closed. Skipping this bar.")
                return

            for data in self.datas:
                if data not in self.data_ready or not self.data_ready[data]:
                    continue  # Skip this data feed if it's not ready
                
                symbol = data._name
                position = self.broker.getposition(data)

                if position.size != 0:
                    logging.info(f'{symbol}: Current Position Size: {position.size}')
                    self.evaluate_sell_conditions(data, current_date)
                else:
                    logging.info(f'{symbol}: No position. Evaluating buy signals.')
                    self.process_buy_signal(data, current_date)
                logging.info(f'{symbol}: Current Price: {data.close[0]}')






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
        logging.info(f"Checking expected profit for {symbol}: Days held: {days_held}, Current price: {current_price}, Entry price: {entry_price}")
        if days_held > 3:
            total_profit_percentage = ((current_price - entry_price) / entry_price) * 100
            daily_profit_percentage = total_profit_percentage / days_held
            expected_profit_per_day = self.params.expected_profit_per_day_percentage * days_held
            logging.info(f"{symbol}: Total profit %: {total_profit_percentage:.4f}%, Daily profit %: {daily_profit_percentage:.4f}%, Expected daily %: {self.params.expected_profit_per_day_percentage:.4f}%")
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

        ##wait until the connection is established
        while not ib.isConnected():
            pass


        open_positions = get_open_positions(ib)
        buy_signals = get_buy_signals()
        all_symbols = list(set(open_positions + [signal['Symbol'] for signal in buy_signals]))

        all_symbols.sort(key=lambda x: x in open_positions, reverse=True)

        print(f'Total symbols to trade: {len(all_symbols)}')
        logging.info(f'Total symbols to trade: {len(all_symbols)}')

        for symbol in all_symbols:
            try:
                contract = ibi.Stock(symbol, 'SMART', 'USD')
                data = store.getdata(
                    dataname=symbol,
                    sectype=contract.secType,
                    exchange='SMART',
                    currency='USD',
                    rtbar=True,
                    what='TRADES',
                    useRTH=True,
                    qcheck=1.0,
                    backfill_start=False,
                    reconnect=True,
                    timeframe=bt.TimeFrame.Seconds,
                    compression=5,
                    live=True,
                )

                data._name = symbol
                cerebro.adddata(data)
                cerebro.resampledata(data, timeframe=bt.TimeFrame.Minutes, compression=1)  # Resample to 1-minute bars
                
                print(f'Added live data feed for {symbol}')
            except Exception as e:
                print(f'Error adding data for {symbol}: {e}')




        broker = store.getbroker()
        cerebro.setbroker(broker)

        logging.info('Starting Cerebro run')
        try:

            cerebro.addstrategy(MyStrategy)

            cerebro.run()
        except Exception as e:
            logging.error(f"Error during Cerebro run: {e}")
            logging.error(f"Traceback: {traceback.format_exc()}")
        logging.info('Cerebro run completed')

    except Exception as e:
        print(f'Error during execution: {e}')
        logging.error(f'Error during execution: {e}')
        logging.error(f"Traceback: {traceback.format_exc()}")
    finally:
        if 'ib' in locals() and ib.isConnected():
            ib.disconnect()
            print('Disconnected from Interactive Brokers TWS')
            logging.info('Disconnected from Interactive Brokers TWS')





if __name__ == '__main__':
    start()


