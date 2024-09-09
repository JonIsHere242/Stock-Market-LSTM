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
        ('reserve_percent', 0.4),
        ('stop_loss_percent', 10),
        ('take_profit_percent', 50.0),
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




    def initialize_order_dict(self):
        active_orders = self.fetch_active_orders()
        for order in active_orders:
            self.order_dict[order.orderId] = order
        logging.info(f"Initialized with {len(self.order_dict)} active orders.")

    def fetch_active_orders(self):
        """Fetch active orders from IB."""
        return self.ib.reqAllOpenOrders()

    def update_order_dict(self):
        """Update the order dictionary with currently active orders."""
        active_orders = self.fetch_active_orders()
        for order in active_orders:
            self.order_dict[order.orderId] = order
        
        # Remove completed or canceled orders
        for order_id in list(self.order_dict.keys()):
            if self.order_dict[order_id].status not in ['Submitted', 'PreSubmitted']:
                del self.order_dict[order_id]








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
        # Handle submitted or accepted orders
        if order.status in [order.Submitted, order.Accepted]:
            self.order_dict[order.ref] = order  # Store the order in the dictionary if submitted or accepted
            return

        # Handle completed orders (both buy and sell)
        if order.status in [order.Completed]:
            if order.isbuy():
                self.handle_buy_order(order, order.data._name)
            elif order.issell():
                self.handle_sell_order(order, order.data._name)
                self.position_closed[order.data._name] = True  # Mark the position as closed

            # Remove the completed order from the order dictionary
            if order.ref in self.order_dict:
                del self.order_dict[order.ref]

        # Handle canceled, margin, or rejected orders
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            logging.warning(f'Order {order.data._name} failed with status: {order.getstatusname()}')

            # Remove the failed order from the order dictionary
            if order.ref in self.order_dict:
                del self.order_dict[order.ref]


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
        
            elif position.size == 0 and not self.position_closed[symbol]:
                # When there's no position, evaluate buy signals
                logging.info(f'{symbol}: No position flag detected fresh check.')
                logging.info(f'{symbol}: No position. Evaluating buy signals for data.')
                print(f'{symbol}: No position. Evaluating buy signals for data.')
                
                self.process_buy_signal(data, current_date)

            elif position.size < 0:
                # Error case where we have shorted a stock
                logging.error('\n\n\n')
                logging.error(f'-------------------------------------------------------------------------------------')
                logging.error(f'{symbol}: ERROR - We have shorted this stock! Current position size: {position.size}')
                logging.error(f'-------------------------------------------------------------------------------------')
                logging.error('\n\n\n')

            
            



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









    
    ###================[Update system to see orders on restart]=================#

    def process_completed_order(self, order):
        symbol = order.data._name
        if order.isbuy():
            self.handle_buy_order(order, symbol)
        elif order.issell():
            self.handle_sell_order(order, symbol)

    def handle_buy_order(self, order):
        symbol = order.data._name
        entry_price = order.executed.price
        entry_date = self.data.datetime.date(0)
        position_size = order.executed.size

        # Update live trades Parquet file
        update_filled_order(symbol, entry_price, entry_date, position_size, is_live=True)
        logging.info(f'Buy order executed for {symbol} at {entry_price}, size: {position_size}')

    def handle_sell_order(self, order):
        symbol = order.data._name
        exit_price = order.executed.price
        exit_date = self.data.datetime.date(0)
    
        # Mark the position as not currently bought in the live trades Parquet file
        update_trade_result(symbol, is_loss=False, exit_price=exit_price, exit_date=exit_date, is_live=True)
        logging.info(f'Sell order executed for {symbol} at {exit_price}')





    def calculate_position_size(self, data):
        total_value = self.broker.getvalue()
        cash_available = self.broker.getcash()
        workable_capital = total_value * (1 - self.params.reserve_percent)
        capital_for_position = workable_capital / self.params.max_positions
        size = int(capital_for_position / data.close[0])
        if cash_available < (total_value * self.params.reserve_percent) or cash_available < capital_for_position:
            return 0
        return size






    #================[PLACE ORDER WITH TWS ]=================#
    #================[PLACE ORDER WITH TWS ]=================#
    #================[PLACE ORDER WITH TWS ]=================#
    #================[PLACE ORDER WITH TWS ]=================#
    #================[PLACE ORDER WITH TWS ]=================#
    #================[PLACE ORDER WITH TWS ]=================#
    #================[PLACE ORDER WITH TWS ]=================#

    #def process_buy_signal(self, data):
    #    symbol = data._name
    #    if symbol not in self.live_trades[self.live_trades['IsCurrentlyBought']]['Symbol'].values:
    #        size = self.calculate_position_size(data)
    #        if size > 0:
    #            order = self.buy(data=data, size=size)
    #            self.order_dict[symbol] = order
    #            logging.info(f'Market buy order placed for {symbol}, position size: {size}')


    def process_buy_signal(self, data):
        symbol = data._name
        if symbol not in self.live_trades[self.live_trades['IsCurrentlyBought']]['Symbol'].values:
            size = self.calculate_position_size(data)
            if size > 0:
                # Create the market buy order (transmit=False so that the children orders can be attached)
                parent_order = self.buy(
                    data=data,
                    size=size,
                    **{'orderType': 'MKT', 'transmit': False, 'goodTillCancel': True}
                )

                # Store the parent order in the order dictionary
                self.order_dict[parent_order.ref] = {
                    'data': data,
                    'size': size,
                    'parent_order': parent_order,
                    'tp_order': None,
                    'ts_order': None
                }

                # Once the parent order is filled, attach the child orders
                def submit_child_orders(order):
                    entry_price = order.executed.price

                    # Calculate the take-profit price (100% above entry)
                    take_profit_price = entry_price * 2.0

                    # Create the take-profit order (transmit=False to allow chaining)
                    take_profit_order = self.sell(
                        data=data,
                        size=size,
                        exectype=bt.Order.Limit,
                        price=take_profit_price,
                        parent=parent_order,
                        transmit=False,
                        **{'goodTillCancel': True}
                    )

                    # Create the trailing stop order (transmit=True to send the full bracket order)
                    trailing_stop_order = self.sell(
                        data=data,
                        size=size,
                        **{
                            'orderType': 'TRAIL',
                            'trailingPercent': 3,  # 3% trailing stop
                            'parent': parent_order,
                            'transmit': True,
                            'goodTillCancel': True
                        }
                    )

                    # Update the dictionary with child orders
                    self.order_dict[parent_order.ref]['tp_order'] = take_profit_order
                    self.order_dict[parent_order.ref]['ts_order'] = trailing_stop_order

                    logging.info(f"Submitted take-profit at {take_profit_price} and 3% trailing stop for {symbol}")

                # Use Backtrader's order notification system to trigger child order submission after parent order is filled
                self.notify_order = lambda order: (
                    submit_child_orders(order)
                    if order.status == bt.Order.Completed and order.ref == parent_order.ref
                    else None
                )

                logging.info(f"Market buy order placed for {symbol}, position size: {size}")


    #================[CLOSE POSITION WITH TWS ]=================#
    #================[CLOSE POSITION WITH TWS ]=================#
    #================[CLOSE POSITION WITH TWS ]=================#
    #================[CLOSE POSITION WITH TWS ]=================#

    #def close_position(self, data, reason):
    #    if data._name not in self.order_dict and not self.position_closed[data._name]:
    #        logging.info(f'Closing {data._name} due to {reason}')
    #        print(f'Closing {data._name} due to {reason}')
    #
    #        order = self.close(data=data)
    #        self.order_dict[data._name] = order
    #        logging.info(f"Close order placed for {data._name}")
    #

    def close_position(self, data):
        """ Close the position and cancel related take-profit and trailing stop orders. """
        for order_ref, order_info in list(self.order_dict.items()):
            if order_info['data'] == data:
                # Cancel the take-profit and trailing stop orders
                if order_info['tp_order']:
                    self.cancel(order_info['tp_order'])
                if order_info['ts_order']:
                    self.cancel(order_info['ts_order'])

                # Close the parent order position
                self.close(data=data)

                # Remove from the order dictionary
                del self.order_dict[order_ref]

                logging.info(f"Closed position for {data._name} and removed related take-profit and trailing stop orders.")
                break



    def stop(self):
        total_portfolio_value = self.broker.getvalue()
        logging.info(f'Final Portfolio Value: {total_portfolio_value}')
        print(f'Final Portfolio Value: {total_portfolio_value}')


















































###==============================================[INITALIZATION]==================================================###
###==============================================[INITALIZATION]==================================================###
###==============================================[INITALIZATION]==================================================###
###==============================================[INITALIZATION]==================================================###

def start():
    logging.info('============================[ Starting new trading session ]==============')

    cerebro = bt.Cerebro()

    try:
        # Connect to Interactive Brokers TWS
        logging.info('Connecting to Interactive Brokers TWS...')
        ib = ibi.IB()
        ib.connect('127.0.0.1', 7497, clientId=1)

        # Initialize the store for data and broker interaction
        store = IBStore(port=7497)

        # Wait for connection
        while not ib.isConnected():
            logging.info('Waiting for IB to connect...')
            time.sleep(3)

        # Fetch currently open positions from IB
        open_positions = get_open_positions(ib)

        # Fetch buy signals from backtesting and live trading data
        buy_signals_backtesting = get_buy_signals(is_live=False)
        buy_signals_live = get_buy_signals(is_live=True)
        buy_signals = buy_signals_backtesting + buy_signals_live

        # Create a unified set of symbols to process, ensuring no duplicates
        all_symbols = set(open_positions + [signal.get('Symbol') for signal in buy_signals if 'Symbol' in signal])

        logging.info(f'Buy signals: {buy_signals}')
        logging.info(f'Total symbols to trade: {len(all_symbols)}')

        # Add data feeds for each symbol, sorting to prioritize open positions
        for symbol in sorted(all_symbols, key=lambda x: x in open_positions, reverse=True):
            try:
                # Configure and add the data feed for each symbol
                contract = ibi.Stock(symbol, 'SMART', 'USD')
                data = store.getdata(
                    dataname=symbol,
                    contract=contract,
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
                resampled_data._name = symbol

                logging.info(f'Added live data feed for {symbol}')
            except Exception as e:
                logging.error(f'Error adding data for {symbol}: {e}')

        # Set the broker to the one retrieved from IBStore
        broker = store.getbroker()
        cerebro.setbroker(broker)

      

        # Start the Cerebro run loop with the strategy
        cerebro.addstrategy(MyStrategy)
        cerebro.run()
    except Exception as e:
        logging.error(f'Error during execution: {e}')
        logging.error(f"Traceback: {traceback.format_exc()}")
    finally:
        # Ensure disconnection from IB at the end of the session
        if 'ib' in locals() and ib.isConnected():
            ib.disconnect()
            logging.info('Disconnected from Interactive Brokers TWS')





if __name__ == '__main__':
    start()



