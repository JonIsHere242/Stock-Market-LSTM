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
setup_logging(log_to_console=True)  # Set to False to disable logging to the terminal





















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
        ('stop_loss_percent', 5),
        ('take_profit_percent', 100),
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
        self.live_trades = pd.DataFrame()  # Initialize live_trades as an empty DataFrame
        self.position_data = self.load_position_data()

        # Add a timer for heartbeat
        self.add_timer(
            when=bt.Timer.SESSION_START,
            offset=timedelta(minutes=0),  # No offset at the session start
            repeat=timedelta(minutes=1),  # Repeat every minute
            weekdays=[0, 1, 2, 3, 4],  # Monday to Friday
        )


    def load_position_data(self):
        df = read_trading_data(is_live=True)
        return {row['Symbol']: row.to_dict() for _, row in df.iterrows()}



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
            # Store the order in the dictionary if submitted or accepted
            self.order_dict[order.ref] = {'main_order': order}
            logging.info(f"Order for {order.data._name} {order.getstatusname()}")
            return

        # Handle completed orders (both buy and sell)
        if order.status in [order.Completed]:
            if order.isbuy():
                self.handle_buy_order(order)
            elif order.issell():
                self.handle_sell_order(order)
                self.position_closed[order.data._name] = True  # Mark the position as closed

            logging.info(f"Order for {order.data._name} {order.getstatusname()}")

        # Handle canceled, margin, or rejected orders
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            logging.warning(f'Order {order.data._name} failed with status: {order.getstatusname()}')

        # Remove the completed or failed order from the order dictionary
        if order.ref in self.order_dict:
            del self.order_dict[order.ref]
            logging.info(f"Removed order for {order.data._name} from order dictionary")





    def next(self):
        self.barCounter += 1
        current_date = self.safe_get_date()
        self.trading_data = read_trading_data(is_live=True)  # Read live trading data
        self.live_trades = self.trading_data  # Update live_trades with the latest data

        logging.info(f"Bar {self.barCounter}: Current date: {current_date}")
        logging.info(f"Live trades data shape: {self.live_trades.shape}")
        logging.info(f"Live trades columns: {self.live_trades.columns}")

        self.market_open, _ = self.check_market_status()

        if not self.market_open:
            logging.info("Market is closed. Disconnecting from Cerebro.")
            self.stop()
            return

        print(f"Bar {self.barCounter}: Open: {self.data.open[0]}, High: {self.data.high[0]}, Low: {self.data.low[0]}, Close: {self.data.close[0]}, Volume: {self.data.volume[0]}")

        # Print currently owned stocks and their purchase dates
        current_positions = self.live_trades[self.live_trades['IsCurrentlyBought'] == True] if 'IsCurrentlyBought' in self.live_trades.columns else pd.DataFrame()
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
                self.position_closed[symbol] = False
                logging.info(f"{symbol} - Position closed and flag reset")
            elif position.size == 0 and not self.position_closed[symbol]:
                logging.info(f'{symbol}: No position flag detected fresh check.')
                logging.info(f'{symbol}: No position. Evaluating buy signals for data.')
                print(f'{symbol}: No position. Evaluating buy signals for data.')
                self.process_buy_signal(data)
            elif position.size < 0:
                logging.error(f'{symbol}: ERROR - We have shorted this stock! Current position size: {position.size}')

            
            



    def evaluate_sell_conditions(self, data, current_date):
        symbol = data._name
        if symbol in self.order_dict or self.position_closed[symbol]:
            return
    
        symbol_data = self.trading_data[self.trading_data['Symbol'] == symbol].iloc[0]
        entry_price = symbol_data['LastBuySignalPrice']
        entry_date = pd.to_datetime(symbol_data['LastBuySignalDate']).date()  # Ensure this is a datetime.date object
        current_price = data.close[0]
        current_date = pd.to_datetime(current_date).date()  # Ensure consistency in date type
    
        logging.info(f"Current Price: {current_price}, Entry Price: {entry_price}, Entry Date: {entry_date}, Current Date: {current_date}")
    
        if should_sell(current_price, entry_price, entry_date, current_date, 
                      self.params.stop_loss_percent, self.params.take_profit_percent, 
                      self.params.position_timeout, self.params.expected_profit_per_day_percentage, verbose=True):
            logging.info(f"Sell conditions met for {symbol}. Initiating close_position.")
            self.close_position(data, "Sell conditions met")
            return
        else:
            logging.info(f"Sell conditions not met for {symbol}.")










    
    ###================[Update system to see orders on restart]=================#



    def handle_buy_order(self, order):
        symbol = order.data._name
        entry_price = order.executed.price
        entry_date = self.data.datetime.date(0)
        position_size = order.executed.size

        logging.info(f"Buy order for {symbol} completed. Size: {position_size}, Price: {entry_price}")

        # Update the Parquet file
        update_trade_data(symbol, 'buy', entry_price, entry_date, position_size, is_live=True)

        if not hasattr(self, 'active_positions'):
            self.active_positions = set()

        if symbol not in self.active_positions:
            self.active_positions.add(symbol)
            logging.info(f"Added {symbol} to active positions after buy order completion")






    def handle_sell_order(self, order):
        symbol = order.data._name
        exit_price = order.executed.price
        exit_date = self.data.datetime.date(0)
        position_size = order.executed.size

        logging.info(f"Sell order for {symbol} completed. Size: {position_size}, Price: {exit_price}")

        # Update the Parquet file
        update_trade_data(symbol, 'sell', exit_price, exit_date, 0, is_live=True)

        if hasattr(self, 'active_positions') and symbol in self.active_positions:
            self.active_positions.remove(symbol)
            logging.info(f"Removed {symbol} from active positions after sell order completion")







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

    def process_buy_signal(self, data):
        symbol = data._name
        logging.info(f"Processing buy signal for {symbol}")

        # Check if an order is already pending for this symbol
        if any(order['data']._name == symbol for order in self.order_dict.values()):
            logging.info(f"Order already pending for {symbol}, skipping buy signal")
            return

        # Check if the symbol is already in active positions
        if hasattr(self, 'active_positions') and symbol in self.active_positions:
            logging.info(f"{symbol} is already in active positions, skipping buy signal")
            return

        if 'IsCurrentlyBought' not in self.live_trades.columns:
            logging.warning("'IsCurrentlyBought' column not found in live_trades")
            return

        currently_bought = self.live_trades[self.live_trades['IsCurrentlyBought'] == True]
        logging.info(f"Currently bought symbols: {currently_bought['Symbol'].tolist() if 'Symbol' in currently_bought.columns else 'Symbol column not found'}")

        if symbol not in currently_bought['Symbol'].values:
            size = self.calculate_position_size(data)
            if size > 0:
                # Generate a unique OCO group ID
                oco_id = f"OCO_{symbol}_{int(time.time())}"

                # Create the parent market buy order
                parent_order = self.buy(
                    data=data,
                    size=size,
                    exectype=bt.Order.Market,
                    transmit=False,
                    **{'goodTillCancel': True}
                )

                # Estimate entry price (since it's a market order, use current price)
                entry_price = data.close[0]
                take_profit_price = entry_price * 2.0  # 100% profit target

                # Create the take-profit limit sell order
                take_profit_order = self.sell(
                    data=data,
                    size=size,
                    exectype=bt.Order.Limit,
                    price=take_profit_price,
                    parent=parent_order,
                    transmit=False,
                    oco=oco_id,
                    **{'goodTillCancel': True}
                )

                # Create the trailing stop sell order
                trailing_stop_order = self.sell(
                    data=data,
                    size=size,
                    exectype=bt.Order.StopTrail,
                    trailpercent=3.0,
                    parent=parent_order,
                    transmit=True,  # Transmit=True on the last order to send all orders
                    oco=oco_id,
                    **{'goodTillCancel': True}
                )

                # Store the orders
                self.order_dict[parent_order.ref] = {
                    'main_order': parent_order,
                    'take_profit_order': take_profit_order,
                    'trailing_stop_order': trailing_stop_order,
                    'data': data
                }

                logging.info(f"Bracket order placed for {symbol}, position size: {size}")
            else:
                logging.info(f"Not enough capital to open position for {symbol}")
        else:
            logging.info(f"{symbol} is already bought, skipping buy signal")


    ##=====================[CLOSE POSITION]====================##
    ##=====================[CLOSE POSITION]====================##
    ##=====================[CLOSE POSITION]====================##
    ##=====================[CLOSE POSITION]====================##
    ##=====================[CLOSE POSITION]====================##

    def close_position(self, data, reason):
        """Close the position and cancel related take-profit and trailing stop orders."""
        symbol = data._name
        logging.info(f"Attempting to close position for {symbol} due to: {reason}")

        try:
            # Cancel any pending orders related to this position
            for order_ref, order_info in list(self.order_dict.items()):
                if order_info['data'] == data:
                    if 'take_profit_order' in order_info and order_info['take_profit_order']:
                        self.cancel(order_info['take_profit_order'])
                        logging.info(f"Canceled take-profit order for {symbol}")
                    if 'trailing_stop_order' in order_info and order_info['trailing_stop_order']:
                        self.cancel(order_info['trailing_stop_order'])
                        logging.info(f"Canceled trailing stop order for {symbol}")

            # Close the position
            order = self.close(data=data)
            self.order_dict[order.ref] = order
            logging.info(f"Sell order submitted for {symbol}, Order ID: {order.ref}")
        except Exception as e:
            logging.error(f"Failed to close position for {symbol}: {e}")




    def stop(self):
        total_portfolio_value = self.broker.getvalue()
        logging.info(f'Final Portfolio Value: {total_portfolio_value}')
        print(f'Final Portfolio Value: {total_portfolio_value}')
        super().stop()


















































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



