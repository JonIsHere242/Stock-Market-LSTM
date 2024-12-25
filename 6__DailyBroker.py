#!/root/root/miniconda4/envs/tf/bin/python
import backtrader as bt
from backtrader_ib_insync import IBStore
import logging
from logging.handlers import RotatingFileHandler

import ib_insync as ibi
import pandas as pd
import numpy as np
from datetime import datetime
import time
import uuid

from Trading_Functions import *
import traceback
import sys
import time
import socket



DEBUG_MODE = True 



def dprint(message):
    """Debug print function that can be toggled on/off."""
    if DEBUG_MODE:
        print(f"[DEBUG] {message}")



def setup_logging(log_to_console=True):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = RotatingFileHandler('__BrokerLive.log', maxBytes=1024 * 1024, backupCount=5)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if log_to_console:
        # Create a console handler only if log_to_console is True
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

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




def check_tws_connection(host='127.0.0.1', port=7497, timeout=5):
    try:
        # Try to establish a socket connection to TWS
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False



def wait_for_tws(max_attempts=5, wait_time=10):
    for attempt in range(max_attempts):
        if check_tws_connection():
            logging.info("Successfully connected to TWS")
            return True
        else:
            remaining_attempts = max_attempts - attempt - 1
            message = (
                "\n=== TWS Connection Error ===\n"
                "Interactive Brokers Trader Workstation (TWS) is not running or not accessible.\n"
                f"Please ensure TWS is:\n"
                "1. Launched and running\n"
                "2. Properly logged in\n"
                "3. API connections are enabled\n"
                "4. Port 7497 is open and configured\n"
                f"\nRetrying in {wait_time} seconds... ({remaining_attempts} attempts remaining)\n"
            )
            print(message)
            logging.warning(message)
            time.sleep(wait_time)
    
    final_message = (
        "\n=== Connection Failed ===\n"
        "Could not connect to TWS after multiple attempts.\n"
        "Please start TWS and run this program again.\n"
        "Exiting...\n"
    )
    print(final_message)
    logging.error(final_message)
    return False






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
        self.start_time = datetime.now()
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



    def safe_get_date(self):
        """Safely get current date from data feed or system"""
        try:
            return bt.num2date(self.datas[0].datetime[0]).date()
        except AttributeError:
            return datetime.now().date()
        except Exception as e:
            logging.error(f"Error getting date: {e}")
            return datetime.now().date()


    def notify_timer(self, timer, when, *args, **kwargs):
        elapsed_mins = (datetime.now() - self.start_time).total_seconds() / 60
        if elapsed_mins >= 5:  # 5 minute runtime
            logging.info('Strategy runtime of 5 minutes reached. Starting shutdown sequence...')

            # Stop cerebro
            self.env.runstop()
            # Disconnect from IB
            try:
                store = self.broker.store
                if hasattr(store, 'disconnect'):
                    store.disconnect()
                    logging.info('Disconnected from Interactive Brokers')
            except Exception as e:
                logging.error(f"Error disconnecting from IB: {e}")

            # Log final message
            logging.info('Shutdown sequence complete. Exiting in 5 seconds...')

            # Wait 5 seconds then exit
            time.sleep(5)
            sys.exit(0)

        else:
            current_time = datetime.now().strftime("%H:%M:%S")
            print(f'Heartbeat: {current_time} - Running for {elapsed_mins:.1f} minutes')





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
        """Updated order notification handler with proper dictionary structure"""
        # Handle submitted or accepted orders
        if order.status in [order.Submitted, order.Accepted]:
            # Store the order in the dictionary with proper structure
            self.order_dict[order.ref] = {
                'order': order,
                'data_name': order.data._name,
                'status': order.getstatusname()
            }
            logging.info(f"Order for {order.data._name} {order.getstatusname()}")
            return

        # Handle completed orders
        if order.status in [order.Completed]:
            if order.isbuy():
                self.handle_buy_order(order)
            elif order.issell():
                self.handle_sell_order(order)
                self.position_closed[order.data._name] = True
            
            logging.info(f"Order for {order.data._name} {order.getstatusname()}")

        # Handle failed orders
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            logging.warning(f'Order {order.data._name} failed with status: {order.getstatusname()}')

        # Remove completed or failed orders from dictionary
        if order.ref in self.order_dict:
            del self.order_dict[order.ref]




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

            


            
    ##============[EVALUATE SELL CONDITIONS]==================##
    ##============[EVALUATE SELL CONDITIONS]==================##
    ##============[EVALUATE SELL CONDITIONS]==================##
    
    def evaluate_sell_conditions(self, data, current_date):
        """Evaluate whether to sell a position with improved data handling"""
        symbol = data._name
        if symbol in self.order_dict or self.position_closed[symbol]:
            return

        try:
            # Read the latest trading data
            self.trading_data = read_trading_data(is_live=True)

            # Get symbol data with better error handling
            symbol_data = self.trading_data[self.trading_data['Symbol'] == symbol]
            if symbol_data.empty:
                logging.warning(f"No trading data found for {symbol}, attempting to recreate from current position")

                # Recreate trading data from current position
                position = self.getposition(data)
                current_data = {
                    'Symbol': symbol,
                    'LastBuySignalDate': pd.Timestamp(current_date - timedelta(days=1)),  # Assume bought yesterday if unknown
                    'LastBuySignalPrice': position.price,
                    'IsCurrentlyBought': True,
                    'ConsecutiveLosses': 0,
                    'LastTradedDate': pd.Timestamp(current_date),
                    'UpProbability': 0.0,
                    'PositionSize': position.size
                }

                # Update the trading data
                self.trading_data = pd.concat([self.trading_data, pd.DataFrame([current_data])], ignore_index=True)
                write_trading_data(self.trading_data, is_live=True)
                symbol_data = self.trading_data[self.trading_data['Symbol'] == symbol].iloc[0]
            else:
                symbol_data = symbol_data.iloc[0]

            # Get all the necessary values for sell evaluation
            entry_price = float(symbol_data['LastBuySignalPrice'])
            entry_date = pd.to_datetime(symbol_data['LastBuySignalDate']).date()
            current_price = data.close[0]
            current_date = pd.to_datetime(current_date).date()

            logging.info(f"{symbol} Position Analysis:")
            logging.info(f"Current Price: {current_price}, Entry Price: {entry_price}")
            logging.info(f"Entry Date: {entry_date}, Current Date: {current_date}")
            logging.info(f"Days Held: {(current_date - entry_date).days}")

            if should_sell(current_price, entry_price, entry_date, current_date, 
                          self.params.stop_loss_percent, self.params.take_profit_percent, 
                          self.params.position_timeout, self.params.expected_profit_per_day_percentage, verbose=True):
                logging.info(f"Sell conditions met for {symbol}. Initiating close_position.")
                self.close_position(data, "Sell conditions met")
                return
            else:
                logging.info(f"Sell conditions not met for {symbol}.")

        except Exception as e:
            logging.error(f"Error evaluating sell conditions for {symbol}: {e}")
            logging.error(traceback.format_exc())



    ###================[Update system to see orders on restart]=================#
    def handle_buy_order(self, order):
        """Handle completed buy orders with proper trade data updating"""
        symbol = order.data._name
        entry_price = order.executed.price
        entry_date = self.data.datetime.date(0)
        position_size = order.executed.size

        logging.info(f"Buy order for {symbol} completed. Size: {position_size}, Price: {entry_price}")

        try:
            # First read existing data
            df = read_trading_data(is_live=True)

            # Create new trade data
            new_data = {
                'Symbol': symbol,
                'LastBuySignalDate': pd.Timestamp(entry_date),
                'LastBuySignalPrice': entry_price,
                'IsCurrentlyBought': True,
                'ConsecutiveLosses': 0,
                'LastTradedDate': pd.Timestamp(entry_date),
                'UpProbability': 0.0,  # You might want to pass this as a parameter
                'PositionSize': position_size
            }

            # Update or append the data
            if symbol in df['Symbol'].values:
                for col, value in new_data.items():
                    df.loc[df['Symbol'] == symbol, col] = value
            else:
                df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)

            # Write back to parquet
            write_trading_data(df, is_live=True)
            logging.info(f"Successfully updated trade data for buy order: {symbol}")

            # Update active positions set
            if not hasattr(self, 'active_positions'):
                self.active_positions = set()
            if symbol not in self.active_positions:
                self.active_positions.add(symbol)

        except Exception as e:
            logging.error(f"Error updating trade data for {symbol}: {e}")
            logging.error(traceback.format_exc())









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
    def process_buy_signalNOTAJUSTEDTOCONTRACT(self, data):
        """Process buy signals with proper price rounding and order handling"""
        symbol = data._name
        logging.info(f"Processing buy signal for {symbol}")

        # Check for existing orders/positions using the correct dictionary structure
        if symbol in [order_info.get('data_name') for order_info in self.order_dict.values()]:
            logging.info(f"Order already pending for {symbol}, skipping buy signal")
            return

        if hasattr(self, 'active_positions') and symbol in self.active_positions:
            logging.info(f"{symbol} is already in active positions, skipping buy signal")
            return

        currently_bought = self.live_trades[self.live_trades['IsCurrentlyBought'] == True]

        if symbol not in currently_bought['Symbol'].values:
            size = self.calculate_position_size(data)
            if size > 0:
                try:
                    # Get current price and round it properly
                    current_price = data.close[0]
                    # Round to 2 decimal places for most stocks, 4 for penny stocks
                    tick_size = 0.01 if current_price >= 1.0 else 0.0001
                    limit_price = round(current_price * 1.001 / tick_size) * tick_size

                    # Create main market buy order with rounded price
                    main_order = self.buy(
                        data=data,
                        size=size,
                        exectype=bt.Order.Limit,
                        price=limit_price,
                        transmit=True  # Changed to True to ensure order is sent
                    )

                    # Only create trailing stop if main order is valid
                    if main_order and main_order.ref:
                        # Create trailing stop order
                        trail_stop = self.sell(
                            data=data,
                            size=size,
                            exectype=bt.Order.StopTrail,
                            trailpercent=0.03,  # 3% trailing stop
                            parent=main_order,
                            transmit=True
                        )

                        # Store orders in dictionary only if both are created successfully
                        if trail_stop and trail_stop.ref:
                            self.order_dict[main_order.ref] = {
                                'order': main_order,
                                'trail_stop': trail_stop,
                                'data_name': symbol,
                                'status': 'SUBMITTED'
                            }
                            logging.info(f"Orders placed successfully for {symbol} at limit price {limit_price}")
                        else:
                            logging.error(f"Failed to create trailing stop order for {symbol}")
                    else:
                        logging.error(f"Failed to create main order for {symbol}")

                except Exception as e:
                    logging.error(f"Error placing orders for {symbol}: {str(e)}")
                    # Cancel any partial orders if there was an error
                    if 'main_order' in locals() and main_order:
                        self.cancel(main_order)
                    if 'trail_stop' in locals() and trail_stop:
                        self.cancel(trail_stop)




    ##=====================[GET CONTRACT LIMITATIONS FOR DYNAMIC TRADE BASED ON STOCK]====================##
    ##=====================[GET CONTRACT LIMITATIONS FOR DYNAMIC TRADE BASED ON STOCK]====================##
    ##=====================[GET CONTRACT LIMITATIONS FOR DYNAMIC TRADE BASED ON STOCK]====================##
    ##=====================[GET CONTRACT LIMITATIONS FOR DYNAMIC TRADE BASED ON STOCK]====================##
    ##=====================[GET CONTRACT LIMITATIONS FOR DYNAMIC TRADE BASED ON STOCK]====================##
    ##=====================[GET CONTRACT LIMITATIONS FOR DYNAMIC TRADE BASED ON STOCK]====================##


    def get_contract_limitations(self, data):
        """Get contract details and market rules for a symbol"""
        dprint("Entering get_contract_limitations method.")
        dprint(f"data object: {data}")
        try:
            dprint("Inside try block of get_contract_limitations.")

            # Get contract details
            symbol = data._name
            dprint(f"Extracted symbol from data._name: {symbol}")

            contract = self.broker.store.makecontract(data)  # Get IB contract object
            dprint(f"IB contract created: {contract}")

            details_list = self.broker.store.reqContractDetails(contract)
            dprint(f"Contract details list fetched: {details_list}")

            details = details_list[0]  # Get first contract detail
            dprint(f"Details extracted (first item): {details}")

            # Get order types allowed
            dprint("Determining order types from details.orderTypes.")
            order_types = {
                'allowMarketOrders': details.orderTypes.find('MKT') != -1,
                'allowLimitOrders': details.orderTypes.find('LMT') != -1,
                'allowStopOrders': details.orderTypes.find('STP') != -1,
                'allowTrailingStop': details.orderTypes.find('TRAIL') != -1
            }
            dprint(f"order_types derived: {order_types}")

            # Get lot size requirements
            dprint("Extracting min_tick from details.")
            min_tick = details.minTick
            dprint(f"min_tick = {min_tick}")

            dprint("Checking if details has 'minSize' attribute for min_size.")
            min_size = details.minSize if hasattr(details, 'minSize') else 1
            dprint(f"min_size = {min_size}")

            # Get market rule for precise price increments
            dprint("Splitting marketRuleIds to retrieve the first market rule ID.")
            market_rule_id = details.marketRuleIds.split(',')[0]  # Get first market rule
            dprint(f"market_rule_id = {market_rule_id}")

            result_dict = {
                'order_types': order_types,
                'min_tick': min_tick,
                'min_size': min_size,
                'market_rule_id': market_rule_id
            }
            dprint(f"Returning result from get_contract_limitations: {result_dict}")
            return result_dict

        except Exception as e:
            dprint(f"Caught exception in get_contract_limitations: {str(e)}")
            logging.error(f"Error getting contract limitations for {data._name}: {str(e)}")
            return None

    def adjust_order_size(self, size, min_size):
        """Adjust order size to meet minimum lot requirements"""
        dprint("Entering adjust_order_size method.")
        dprint(f"Received parameters -> size: {size}, min_size: {min_size}")
        if min_size > 1:
            dprint("min_size > 1, performing integer division to find largest multiple.")
            # Round down to nearest lot size
            adjusted_size = (size // min_size) * min_size
            dprint(f"adjusted_size after integer division: {adjusted_size}")

            if adjusted_size < min_size:
                dprint(f"adjusted_size < min_size ({min_size}), returning 0.")
                return 0  # Can't meet minimum lot requirement

            dprint(f"Returning adjusted_size: {adjusted_size}")
            return adjusted_size

        dprint("min_size <= 1, returning original size.")
        return size

    def create_optimized_order(self, data, size, contract_limits):
        """Create the most appropriate order type based on contract limitations"""
        dprint("Entering create_optimized_order method.")
        dprint(f"Parameters -> data: {data}, size: {size}, contract_limits: {contract_limits}")

        current_price = data.close[0]
        dprint(f"current_price = {current_price}")

        # Adjust size for minimum lot requirements
        adjusted_size = self.adjust_order_size(size, contract_limits['min_size'])
        dprint(f"adjusted_size from adjust_order_size: {adjusted_size}")

        if adjusted_size == 0:
            dprint(f"adjusted_size == 0, logging warning and returning (None, None).")
            logging.warning(f"Order size {size} too small for minimum lot size {contract_limits['min_size']}")
            return None, None

        try:
            dprint("Attempting to create main_order and (possibly) trail_stop.")
            main_order = None
            trail_stop = None
            order_types = contract_limits['order_types']
            dprint(f"Parsed order_types from contract_limits: {order_types}")

            if order_types['allowMarketOrders'] and order_types['allowTrailingStop']:
                dprint("allowMarketOrders and allowTrailingStop both True. Using Market order + trailing stop.")
                # Preferred method: Market order with trailing stop
                main_order = self.buy(
                    data=data,
                    size=adjusted_size,
                    exectype=bt.Order.Market,
                    transmit=True
                )
                dprint(f"Market buy order created: {main_order}")

            elif order_types['allowLimitOrders']:
                dprint("Falling back to limit order since market+trailing not possible.")
                # Fallback: Limit order at slight premium
                tick_size = contract_limits['min_tick']
                dprint(f"tick_size from contract_limits: {tick_size}")
                limit_price_raw = current_price * 1.001
                dprint(f"Raw limit price (1.001 * current_price): {limit_price_raw}")

                # Round limit_price to the nearest multiple of tick_size
                limit_price = round(limit_price_raw / tick_size) * tick_size
                dprint(f"limit_price after rounding: {limit_price}")

                main_order = self.buy(
                    data=data,
                    size=adjusted_size,
                    exectype=bt.Order.Limit,
                    price=limit_price,
                    transmit=True
                )
                dprint(f"Limit buy order created: {main_order}")
            else:
                dprint("No suitable order type found, logging error and returning (None, None).")
                logging.error(f"No suitable order type found for {data._name}")
                return None, None

            # Create trailing stop if main order is valid
            if main_order and main_order.ref and order_types['allowTrailingStop']:
                dprint("Main order valid and trailing stop allowed. Creating trailing stop.")
                trail_stop = self.sell(
                    data=data,
                    size=adjusted_size,
                    exectype=bt.Order.StopTrail,
                    trailpercent=0.03,
                    parent=main_order,
                    transmit=True
                )
                dprint(f"Trailing stop order created: {trail_stop}")

            dprint(f"Returning main_order and trail_stop: {main_order}, {trail_stop}")
            return main_order, trail_stop

        except Exception as e:
            dprint(f"Caught exception in create_optimized_order: {str(e)}")
            logging.error(f"Error creating optimized order: {str(e)}")
            return None, None

    def process_buy_signal(self, data):
        """Enhanced process_buy_signal with contract limitation checks"""
        dprint("Entering process_buy_signal method.")
        symbol = data._name
        dprint(f"Symbol extracted: {symbol}")
        logging.info(f"Processing buy signal for {symbol}")

        # Check for existing orders/positions
        current_order_symbols = [order_info.get('data_name') for order_info in self.order_dict.values()]
        dprint(f"Symbols with pending orders: {current_order_symbols}")

        if symbol in current_order_symbols:
            dprint(f"{symbol} in current_order_symbols, skipping buy signal.")
            logging.info(f"Order already pending for {symbol}, skipping")
            return

        dprint("Checking if symbol is in active_positions attribute (if exists).")
        if hasattr(self, 'active_positions') and symbol in self.active_positions:
            dprint(f"{symbol} found in self.active_positions, skipping.")
            logging.info(f"{symbol} already in active positions, skipping")
            return

        currently_bought = self.live_trades[self.live_trades['IsCurrentlyBought'] == True]
        dprint(f"Currently bought DataFrame subset:\n{currently_bought}")

        if symbol in currently_bought['Symbol'].values:
            dprint(f"{symbol} found in currently_bought['Symbol'], returning.")
            return

        # Get contract limitations
        dprint("Requesting contract limitations from get_contract_limitations.")
        contract_limits = self.get_contract_limitations(data)
        dprint(f"contract_limits received: {contract_limits}")

        if not contract_limits:
            dprint("contract_limits is None, logging error and returning.")
            logging.error(f"Unable to get contract limitations for {symbol}")
            return

        # Calculate position size
        dprint(f"Calculating position size via self.calculate_position_size(data).")
        size = self.calculate_position_size(data)
        dprint(f"Position size calculated: {size}")

        if size <= 0:
            dprint(f"Size <= 0, returning without placing orders.")
            return

        try:
            dprint("Attempting to create optimized orders.")
            main_order, trail_stop = self.create_optimized_order(data, size, contract_limits)
            dprint(f"main_order: {main_order}, trail_stop: {trail_stop}")

            if main_order and main_order.ref:
                dprint("main_order valid, storing orders in order_dict.")
                self.order_dict[main_order.ref] = {
                    'order': main_order,
                    'trail_stop': trail_stop,
                    'data_name': symbol,
                    'status': 'SUBMITTED'
                }
                dprint(f"order_dict updated for order ref: {main_order.ref}")
                logging.info(f"Orders placed successfully for {symbol}")
            else:
                dprint("main_order invalid or missing ref, logging error.")
                logging.error(f"Failed to create orders for {symbol}")

        except Exception as e:
            dprint(f"Caught exception in process_buy_signal: {str(e)}")
            logging.error(f"Error in process_buy_signal for {symbol}: {str(e)}")

            # Cancel any partial orders if there was an error
            if 'main_order' in locals() and main_order:
                dprint("Canceling main_order due to exception.")
                self.cancel(main_order)
            if 'trail_stop' in locals() and trail_stop:
                dprint("Canceling trail_stop due to exception.")
                self.cancel(trail_stop)



















    ##=====================[CLOSE POSITION]====================##
    ##=====================[CLOSE POSITION]====================##
    ##=====================[CLOSE POSITION]====================##

    def close_position(self, data, reason):
        """Close position and cancel associated trailing stop."""
        symbol = data._name
        logging.info(f"Attempting to close position for {symbol} due to: {reason}")

        # Find and cancel trailing stop first
        for order_ref, order_info in list(self.order_dict.items()):
            if order_info['data_name'] == symbol:
                if 'trail_stop' in order_info and order_info['trail_stop']:
                    self.cancel(order_info['trail_stop'])
                    logging.info(f"Cancelled trailing stop for {symbol}")
                del self.order_dict[order_ref]

        # Close the position
        order = self.close(data=data)
        logging.info(f"Closing order submitted for {symbol}")


    def stop(self):
        total_portfolio_value = self.broker.getvalue()
        logging.info(f'Final Portfolio Value: {total_portfolio_value}')
        print(f'Final Portfolio Value: {total_portfolio_value}')
        super().stop()













##=============================[Final Data Sync ]============================##
##=============================[Final Data Sync ]============================##
##=============================[Final Data Sync ]============================##
##=============================[Final Data Sync ]============================##
##=============================[Final Data Sync ]============================##



def finalize_positions_sync(ib, trading_data_path='live_trading_data.parquet'):
    """
    Fetch IB's official open positions, then update local data to reflect 
    any mismatches so the next session starts with the correct state.
    """
    try:
        # 1. Get real open positions from IB
        ib_positions = ib.positions()
        real_positions = {}  # { 'AAPL': 100, 'TSLA': 50, ... }
        for pos in ib_positions:
            if pos.position != 0:
                symbol = pos.contract.symbol
                size = pos.position
                real_positions[symbol] = size
        
        # 2. Read your local trades DataFrame
        df = pd.read_parquet(trading_data_path)

        # Just in case the columns differ, double-check you have 'Symbol' & 'IsCurrentlyBought' columns
        if 'Symbol' not in df.columns or 'IsCurrentlyBought' not in df.columns:
            logging.warning("DataFrame missing required columns (Symbol, IsCurrentlyBought).")
        
        # Convert local positions to a dictionary for easy comparison
        local_positions = (
            df[df['IsCurrentlyBought'] == True]
            .set_index('Symbol')['PositionSize']
            .to_dict()
        )  # e.g. { 'AAPL': 100, 'TSLA': 50 }

        # 3. Reconcile differences
        # ----------------------------------------------------
        # Case A: Symbol in IB but not in local -> Mark it as bought
        # Case B: Symbol in local but not in IB -> Mark it as closed
        # Case C: Position size mismatch -> Correct local data
        # ----------------------------------------------------

        # A: Symbol in IB but not in local or mismatch in size
        for symbol, real_size in real_positions.items():
            if symbol not in local_positions:
                logging.info(f"{symbol} is open in IB but not marked locally; updating local data.")
                # Append or update a row in df to reflect this new open position
                new_data = {
                    'Symbol': symbol,
                    'IsCurrentlyBought': True,
                    'PositionSize': real_size,
                    'LastBuySignalDate': pd.Timestamp.now(),  # or unknown
                    'LastBuySignalPrice': 0,  # or unknown
                }
                # If the symbol doesn't exist at all, append row
                if symbol not in df['Symbol'].values:
                    df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
                else:
                    # If row for symbol exists but was previously closed
                    for col, val in new_data.items():
                        df.loc[df['Symbol'] == symbol, col] = val
            else:
                local_size = local_positions[symbol]
                if local_size != real_size:
                    logging.info(
                        f"Mismatch for {symbol}: local size={local_size}, IB size={real_size}. Correcting local data."
                    )
                    df.loc[df['Symbol'] == symbol, 'PositionSize'] = real_size
        
        # B: Symbol in local but not in IB -> Mark as closed
        for symbol, local_size in local_positions.items():
            if symbol not in real_positions:
                logging.info(f"{symbol} is locally open but not in IB; marking as closed in local data.")
                df.loc[df['Symbol'] == symbol, 'IsCurrentlyBought'] = False
                df.loc[df['Symbol'] == symbol, 'PositionSize'] = 0
                df.loc[df['Symbol'] == symbol, 'LastTradedDate'] = pd.Timestamp.now()

        # 4. Save the updated DataFrame
        df.to_parquet(trading_data_path, index=False)
        logging.info("Final data sync completed successfully. Local data now matches IB reality.")
    
    except Exception as e:
        logging.error(f"Error in finalize_positions_sync: {e}")
        logging.error(traceback.format_exc())




























###==============================================[INITALIZATION]==================================================###
###==============================================[INITALIZATION]==================================================###
###==============================================[INITALIZATION]==================================================###
###==============================================[INITALIZATION]==================================================###

def start():
    logging.info('============================[ Starting new trading session ]==============')
    cerebro = bt.Cerebro()

    if not wait_for_tws():
        return

    ib = None
    try:
        # Connect to Interactive Brokers TWS
        logging.info('Initializing IB connection...')
        ib = ibi.IB()
        try:
            ib.connect('127.0.0.1', 7497, clientId=1)
        except ConnectionRefusedError:
            logging.error("TWS refused connection. Please check if TWS is running and configured properly.")
            return
        except Exception as conn_err:
            logging.error(f"Unexpected error during IB connection: {conn_err}")
            return

        try:
            store = IBStore(port=7497)
        except Exception as store_err:
            logging.error(f"Failed to initialize IBStore: {store_err}")
            return

        connection_attempts = 0
        while not ib.isConnected() and connection_attempts < 5:
            logging.info('Waiting for IB connection to stabilize...')
            time.sleep(3)
            connection_attempts += 1
            
        if not ib.isConnected():
            logging.error("Failed to establish stable connection with TWS")
            return

        # Data Collection Phase
        try:
            open_positions = get_open_positions(ib)
            buy_signals_backtesting = get_buy_signals(is_live=False)
            buy_signals_live = get_buy_signals(is_live=True)
            buy_signals = buy_signals_backtesting + buy_signals_live
            
            all_symbols = set(open_positions + [signal.get('Symbol') for signal in buy_signals if 'Symbol' in signal])
            
            if not all_symbols:
                logging.warning("No symbols found to trade")
                return
                
            logging.info(f'Buy signals: {buy_signals}')
            logging.info(f'Total symbols to trade: {len(all_symbols)}')
            
        except Exception as data_err:
            logging.error(f"Error collecting trading data: {data_err}")
            return

        # Data Feed Setup Phase
        failed_symbols = []
        for symbol in sorted(all_symbols, key=lambda x: x in open_positions, reverse=True):
            try:
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
                resampled_data = cerebro.resampledata(
                    data, 
                    timeframe=bt.TimeFrame.Seconds, 
                    compression=30
                )
                resampled_data._name = symbol
                logging.info(f'Successfully added live data feed for {symbol}')
                
            except Exception as feed_err:
                logging.error(f'Failed to add data feed for {symbol}: {feed_err}')
                failed_symbols.append(symbol)
                continue

        if len(failed_symbols) == len(all_symbols):
            logging.error("Failed to add data feeds for all symbols")
            return

        # Strategy Execution Phase
        try:
            broker = store.getbroker()
            cerebro.setbroker(broker)
            cerebro.addstrategy(MyStrategy)
            cerebro.run()
            
        except Exception as strat_err:
            logging.error(f"Strategy execution failed: {strat_err}")
            logging.error(traceback.format_exc())
            
    except Exception as e:
        logging.error(f"Unexpected error in main execution loop: {e}")
        logging.error(traceback.format_exc())
        
    finally:
        try:
            finalize_positions_sync(ib, 'live_trading_data.parquet')
        except Exception as sync_err:
            logging.error(f"Error during final data sync: {sync_err}")

        try:
            if ib and ib.isConnected():
                ib.disconnect()
                logging.info('Successfully disconnected from Interactive Brokers TWS')
        except Exception as disconnect_err:
            logging.error(f"Error during TWS disconnection: {disconnect_err}")

if __name__ == '__main__':
    setup_logging(log_to_console=True)
    start()

