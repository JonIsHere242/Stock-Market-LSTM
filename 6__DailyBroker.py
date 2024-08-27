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
        ('debug', True),
        ('assume_live', True),  # New parameter to control this behavior

    )

    def __init__(self):
        self.order_dict = {}
        self.market_open = True
        self.data_ready = {d: False for d in self.datas}
        self.barCounter = 0
        self.trading_data = read_trading_data(is_live=True)  # Read live trading data
        self.position_closed = {d._name: False for d in self.datas}  # Track if a position has been closed

        # Add a timer for heartbeat
        self.add_timer(
            when=bt.Timer.SESSION_START,
            offset=timedelta(minutes=0),  # No offset at the session start
            repeat=timedelta(minutes=1),  # Repeat every minute
            weekdays=[0, 1, 2, 3, 4],  # Monday to Friday
        )





    def check_market_status(self):
        return is_market_open(), None

    def notify_timer(self, timer, when, *args, **kwargs):
        print(f'Heartbeat: {self.safe_get_date()}')

    def safe_get_date(self):
        try:
            return bt.num2date(self.datas[0].datetime[0]).date()
        except AttributeError:
            return datetime.fromtimestamp(self.datas[0].datetime[0], tz=datetime.timezone.utc).date()
        except Exception as e:
            logging.error(f"Error getting date: {e}")
            return datetime.now(datetime.timezone.utc).date()


    def debug(self, msg):
        if self.p.debug:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[DEBUG] {current_time}: {msg}")
            logging.debug(msg)


    def notify_data(self, data, status, *args, **kwargs):
        super().notify_data(data, status, *args, **kwargs)
        print(f"Data status change for {data._name}: {data._getstatusname(status)}")
        if status == data.LIVE:
            logging.info(f"{data._name} is now live.")
        elif status == data.DISCONNECTED:
            logging.warning(f"{data._name} disconnected.")
        elif status == data.DELAYED:
            logging.info(f"{data._name} is in delayed mode.")



    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.handle_buy_order(order, order.data._name)
            elif order.issell():
                self.handle_sell_order(order, order.data._name)
                self.position_closed[order.data._name] = True  # Mark the position as closed

            self.order_dict.pop(order.data._name, None)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            logging.warning(f'Order {order.data._name} failed with status: {order.getstatusname()}')
            self.order_dict.pop(order.data._name, None)




    def next(self):
        self.barCounter += 1
        current_date = self.safe_get_date()
        self.trading_data = read_trading_data(is_live=True)  # Read live trading data

        self.market_open, _ = self.check_market_status()

        if not self.market_open:
            logging.info("Market is closed. Disconnecting from Cerebro.")
            self.stop()
            return

        print(f"Bar {self.barCounter}: Open: {self.data.open[0]}, High: {self.data.high[0]}, Low: {self.data.low[0]}, Close: {self.data.close[0]}, Volume: {self.data.volume[0]}")

        # Print currently owned stocks and their purchase dates
        current_positions = self.trading_data[self.trading_data['IsCurrentlyBought'] == True]
        if not current_positions.empty:
            print("\nCurrently owned stocks:")
            for _, position in current_positions.iterrows():
                symbol = position['Symbol']
                purchase_date = position['LastBuySignalDate']
                print(f"Symbol: {symbol}, Purchase Date: {purchase_date}")

        print(f"Number of data feeds: {len(self.datas)}")

        for data in self.datas:
            symbol = data._name
            position = self.getposition(data)
            logging.info(f"{symbol} - Position size: {position.size}, Closed flag: {self.position_closed[symbol]}")

            if position.size > 0 and not self.position_closed[symbol]:
                logging.info(f"{symbol} - Current position size: {position.size} at average price: {position.price}")
                self.evaluate_sell_conditions(data, current_date)
            elif position.size == 0 and self.position_closed[symbol]:
                # Reset the flag if the position is actually closed
                self.position_closed[symbol] = False
                logging.info(f"{symbol} - Position closed and flag reset")




            # testing sell conditions initally
            #else:
            #    logging.info(f'{symbol}: No position. Evaluating buy signals.')
            #    self.process_buy_signal(data, current_date)
            #
            #logging.info(f'{symbol}: Current Price: {data.close[0]}')













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
        logging.info(f'Buy order executed for {symbol} at {entry_price}')




    def handle_sell_order(self, order, symbol):
        exit_date = self.data.datetime.date(0)
        exit_price = order.executed.price
        entry_data = self.trading_data[self.trading_data['Symbol'] == symbol].iloc[0]
        entry_price = entry_data['LastBuySignalPrice']
        size = order.executed.size
        profit_loss = (exit_price - entry_price) * size


        self.position_closed[symbol] = True
        logging.info(f'Position closed for {symbol}')


        update_trade_result(symbol, profit_loss < 0, exit_price, exit_date, is_live=True)
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
        if symbol in self.order_dict or self.position_closed[symbol]:
            return


        symbol_data = self.trading_data[self.trading_data['Symbol'] == symbol].iloc[0]
        entry_price = symbol_data['LastBuySignalPrice']
        entry_date = pd.to_datetime(symbol_data['LastBuySignalDate']).date()  # Ensure this is a datetime.date object
        current_price = data.close[0]
        current_date = pd.to_datetime(current_date).date()  # Ensure consistency in date type

        print(f"Current Price: {current_price}, Entry Price: {entry_price}, Entry Date: {entry_date}, Current Date: {current_date}")

        if should_sell(current_price, entry_price, entry_date, current_date, 
                       self.params.stop_loss_percent, self.params.take_profit_percent, 
                       self.params.position_timeout, self.params.expected_profit_per_day_percentage):
            self.close_position(data, "Sell conditions met")
            return








    def close_position(self, data, reason):
        if data._name not in self.order_dict and not self.position_closed[data._name]:
            logging.info(f'Closing {data._name} due to {reason}')
            print(f'Closing {data._name} due to {reason}')

            order = self.close(data=data)
            self.order_dict[data._name] = order
            logging.info(f"Close order placed for {data._name}")





    def stop(self):
        total_portfolio_value = self.broker.getvalue()
        logging.info(f'Final Portfolio Value: {total_portfolio_value}')
        print(f'Final Portfolio Value: {total_portfolio_value}')


















































###==============================================[TRADING COMMANDS FOR LATER]==================================================###
###==============================================[TRADING COMMANDS FOR LATER]==================================================###
###==============================================[TRADING COMMANDS FOR LATER]==================================================###
###==============================================[TRADING COMMANDS FOR LATER]==================================================###

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















def start():
    logging.info('')
    logging.info('============================[ Starting new trading session ]==============')
    logging.info('')

    cerebro = bt.Cerebro()

    try:
        print('Connecting to Interactive Brokers TWS...')
        logging.info('Connecting to Interactive Brokers TWS...')
        ib = ibi.IB()
        ib.connect('127.0.0.1', 7497, clientId=1)
        store = IBStore(port=7497)

        while not ib.isConnected():
            print('Waiting for IB to connect...')
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
                    contract=contract,  # Ensure contract is set correctly
                    sectype=contract.secType,
                    exchange=contract.exchange,
                    currency=contract.currency,
                    rtbar=True,
                    what='TRADES',
                    useRTH=True,
                    qcheck=1.0,
                    backfill_start=False,
                    backfill=False,
                    reconnect=True,
                    timeframe=bt.TimeFrame.Seconds,
                    compression=5,
                    live=True,
                )
                resampled_data = cerebro.resampledata(data, timeframe=bt.TimeFrame.Seconds, compression=30)
                resampled_data._name = symbol  # Give the resampled data the symbol name





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


