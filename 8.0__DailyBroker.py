import backtrader as bt
from backtrader_ib_insync import IBStore
import threading
import cmd
import logging
import ib_insync as ibi
import pandas as pd
from datetime import datetime
import time 
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

class MyStrategy(bt.Strategy):
    params = (
        ('stop_loss_percent', 10),
        ('take_profit_percent', 20),
        ('position_timeout', 30),
    )

    def __init__(self):
        self.data_ready = {d: False for d in self.datas}
        self.barCounter = 0

        # Load saved positions
        self.positions_data = {pos['symbol']: pos for pos in load_positions_from_csv()}

        # Maintain a list of active orders
        self.active_orders = []

    def notify_data(self, data, status):
        logging.info(f'Data Status => {data._getstatusname(status)}')
        print('Data Status =>', data._getstatusname(status))
        if status == data.LIVE:
            self.data_ready[data] = True

    def notify_order(self, order):
        symbol = order.data._name
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            if order in self.active_orders:
                self.active_orders.remove(order)
        if order.status == order.Completed:
            if order.isbuy():
                current_shares = self.positions_data[symbol].get('position', 0)
                new_shares = order.executed.size
                self.positions_data[symbol]['averageCost'] = (
                    (self.positions_data[symbol].get('averageCost', 0) * current_shares) +
                    (order.executed.price * new_shares)
                ) / (current_shares + new_shares) if (current_shares + new_shares) != 0 else order.executed.price
                self.positions_data[symbol]['position'] = current_shares + new_shares
                self.positions_data[symbol]['last_time_bought'] = datetime.now().strftime('%Y-%m-%d')
                logging.info(f'Bought {symbol} at {order.executed.price}')
                print(f'Bought {symbol} at {order.executed.price}')
                save_positions_to_csv(self.positions_data)
            elif order.issell():
                self.positions_data[symbol]['position'] -= order.executed.size
                self.positions_data[symbol]['last_time_sold'] = datetime.now().strftime('%Y-%m-%d')
                logging.info(f'Sold {symbol} at {order.executed.price}')
                print(f'Sold {symbol} at {order.executed.price}')
                save_positions_to_csv(self.positions_data)

    def next(self):
        self.barCounter += 1
        current_date = bt.num2date(self.datas[0].datetime[0])
        
        # Check for commands
        self.check_commands()

        for data in self.datas:
            if not self.data_ready[data]:
                continue

            symbol = data._name
            position = self.broker.getposition(data)
            
            if position.size != 0:
                self.evaluate_sell_conditions(data, current_date)
            
            print(f'{symbol}: Current Price: {data.close[0]}')

    def evaluate_sell_conditions(self, data, current_date):
        symbol = data._name
        if symbol in self.active_orders:
            return  # Skip if there's already an active order for this symbol

        position = self.broker.getposition(data)
        if position.size == 0:
            return  # No position to close

        entry_price = self.positions_data[symbol]['averageCost']
        current_price = data.close[0]

        # Stop loss condition
        if current_price <= entry_price * (1 - self.params.stop_loss_percent / 100):
            self.CloseWrapper(data, "Stop loss")
            return

        # Take profit condition
        if current_price >= entry_price * (1 + self.params.take_profit_percent / 100):
            self.CloseWrapper(data, "Take profit")
            return

        # Position timeout condition
        days_held = (current_date - datetime.strptime(self.positions_data[symbol]['last_time_bought'], '%Y-%m-%d')).days
        if days_held > self.params.position_timeout:
            self.CloseWrapper(data, "Position timeout")
            return

        if days_held > 3:
            total_profit_percentage = ((current_price - entry_price) / entry_price) * 100
            daily_profit_percentage = total_profit_percentage / days_held
            expected_profit_per_day = self.params.expected_profit_per_day_percentage * days_held

            if daily_profit_percentage < expected_profit_per_day:
                self.CloseWrapper(data, "Expected Profit Per Day not met")
                return

    def CloseWrapper(self, data, reason):
        if data._name not in self.active_orders:
            logging.info(f'Closing {data._name} due to {reason}')
            print(f'Closing {data._name} due to {reason}')
            order = self.close(data=data)
            self.active_orders[data._name] = order

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






# Function to check if the stock is listed on NASDAQ or NYSE
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
        ##time.sleep for 3 seconds to finish the connection
        time.sleep(3)
        fetch_and_save_positions(ib)
    except Exception as e:
        print('Failed to connect to Interactive Brokers TWS')
        logging.error('Failed to connect to Interactive Brokers TWS')
        print(e)
        logging.error(e)
        return





    # Fetch currently held positions using ib_insync and add data to Cerebro
    positionsList = get_open_positions(ib)
    print(f'currently held positions: {len(positionsList)}')
    print(f"Stocks Held: {positionsList}")
    logging.info(f'currently held positions: {len(positionsList)}')
    logging.info(f"Stocks Held: {positionsList}")

    for contract, position in positionsList:
        if is_valid_exchange(contract.exchange):
            try:
                data = store.getdata(dataname=contract.symbol, sectype=contract.secType, exchange=contract.exchange)
                data._name = contract.symbol  # Optional: Name the data feed
                cerebro.resampledata(data, timeframe=bt.TimeFrame.Seconds, compression=15)  # Resample to 15-second bars
            except Exception as e:
                print(f'Error adding data for {contract.symbol}')
                logging.error(f'Error adding data for {contract.symbol}')
                logging.error(e)
        else:
            print(f'Skipping data for {contract.symbol} - Invalid Exchange: {contract.exchange}')
            logging.info(f'Skipping data for {contract.symbol} - Invalid Exchange: {contract.exchange}')

    # Read buy signals from CSV
    buy_signals = read_buy_signals('BuySignals.csv')
    print(f'Buy Signals: {buy_signals}')
    logging.info(f'Buy Signals: {buy_signals}')

    for ticker in buy_signals:
        try:
            contract = ibi.Stock(ticker, 'SMART', 'USD')
            if is_valid_exchange(contract.exchange):
                data = store.getdata(dataname=ticker, sectype=contract.secType, exchange=contract.exchange)
                data._name = ticker  # Optional: Name the data feed
                cerebro.resampledata(data, timeframe=bt.TimeFrame.Seconds, compression=15)  # Resample to 15-second bars
            else:
                print(f'Skipping data for {ticker} - Invalid Exchange: {contract.exchange}')
                logging.info(f'Skipping data for {ticker} - Invalid Exchange: {contract.exchange}')
        except Exception as e:
            print(f'Error adding data for {ticker}')
            logging.error(f'Error adding data for {ticker}')
            logging.error(e)

    # Create the broker from the store
    broker = store.getbroker()
    cerebro.setbroker(broker)

    # Add the strategy
    cerebro.addstrategy(MyStrategy)

    # Pass cerebro, store, and ib to the command handler
    TradingCmd.cerebro = cerebro
    TradingCmd.store = store
    TradingCmd.ib = ib

    # Start the command handler in a separate thread
    cmd_thread = threading.Thread(target=TradingCmd().cmdloop)
    cmd_thread.start()

    # Run the backtrader engine
    cerebro.run()

if __name__ == '__main__':
    start()
