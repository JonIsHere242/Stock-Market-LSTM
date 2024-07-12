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
import datetime
import pytz
import numpy as np
from datetime import datetime, timedelta
from live_trading_db import (
    get_db_connection, update_portfolio_data, update_performance_metrics,
    update_system_state, get_recent_trades, calculate_performance_metrics, get_open_positions, add_trade_history, get_strategy_data, update_strategy_data, add_open_position
)

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

def get_recent_trades(days=30):
    conn = get_db_connection()
    cur = conn.cursor()
    
    date_threshold = datetime.now().date() - timedelta(days=days)
    
    cur.execute('''
    SELECT * FROM trade_history
    WHERE exit_date >= ?
    ''', (date_threshold,))
    
    trades = [dict(row) for row in cur.fetchall()]
    conn.close()
    
    return trades

def calculate_performance_metrics():
    recent_trades = get_recent_trades()
    
    sharpe_ratio = calculate_sharpe_ratio(recent_trades)
    win_loss_ratio = calculate_win_loss_ratio(recent_trades)
    avg_profit_per_trade = calculate_avg_profit_per_trade(recent_trades)
    
    return sharpe_ratio, win_loss_ratio, avg_profit_per_trade
























class MyStrategy(bt.Strategy):
    
    params = (
        ('max_positions', 4),
        ('position_size', 0),  # Will be set dynamically
        ('stop_loss_percent', 10),
        ('take_profit_percent', 20),
        ('position_timeout', 9),
        ('expected_profit_per_day_percentage', 0.25),
        ('order_cooldown', 45),  # 45 seconds to wait before submitting another order for the same symbol
    )

    def __init__(self):
        self.open_positions = {pos['symbol']: dict(pos) for pos in get_open_positions()}
        self.buy_signals = self.load_buy_signals()
        self.order_dict = {}
        self.market_open = True
        self.data_ready = {d: False for d in self.datas}
        self.barCounter = 0
        self.pending_orders = {}
        self.order_cooldown_end = {}

        # Load strategy data for each symbol
        self.strategy_data = {}
        for data in self.datas:
            symbol = data._name
            strategy_data = get_strategy_data(symbol)
            if strategy_data:
                self.strategy_data[symbol] = dict(strategy_data)
            else:
                self.strategy_data[symbol] = {
                    'consecutive_losses': 0,
                    'last_trade_date': None
                }

    def load_buy_signals(self):
        signals = read_buy_signals('BuySignals.csv')
        return {ticker.replace('.parquet', ''): date for ticker, date in signals}

    def notify_data(self, data, status):
        logging.info(f'Data Status => {data._getstatusname(status)}')
        print('Data Status =>', data._getstatusname(status))
        if status == data.LIVE:
            self.data_ready[data] = True

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return  # Order pending, do nothing

        if order.status in [order.Completed]:
            if order.isbuy():
                self.handle_buy_order(order)
            elif order.issell():
                self.handle_sell_order(order)
            
            # Remove the pending order
            self.pending_orders.pop(order.data._name, None)
            
            # Set cooldown
            self.order_cooldown_end[order.data._name] = self.datas[0].datetime.datetime(0) + timedelta(seconds=self.params.order_cooldown)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f'Order Canceled/Margin/Rejected: {order.status}')
            # Remove the pending order
            self.pending_orders.pop(order.data._name, None)

    def next(self):
        self.barCounter += 1
        current_date = bt.num2date(self.datas[0].datetime[0])
        
        is_market_open, _ = check_market_status()
        if not self.handle_market_status(is_market_open):
            return

        self.check_commands()

        for data in self.datas:
            if not self.data_ready[data]:
                continue

            symbol = data._name
            position = self.broker.getposition(data)
            
            if position.size != 0:
                self.evaluate_sell_conditions(data, current_date)
            else:
                self.process_buy_signal(data)
            
            print(f'{symbol}: Current Price: {data.close[0]}')

    def handle_market_status(self, is_market_open):
        if not is_market_open and self.market_open:
            self.market_open = False
            self.stop()
            return False
        self.market_open = is_market_open
        return is_market_open

    def handle_order_notification(self, order):
        if order.status in [order.Completed]:
            self.process_completed_order(order)
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            self.order_dict.pop(order.data._name, None)

    def process_completed_order(self, order):
        symbol = order.data._name
        if order.isbuy():
            self.handle_buy_order(order, symbol)
        elif order.issell():
            self.handle_sell_order(order, symbol)

    def handle_buy_order(self, order, symbol):
        entry_date = self.data.datetime.date(0)
        entry_price = order.executed.price
        size = order.executed.size
        add_open_position(symbol, entry_date, entry_price, size)

        # Reset consecutive losses for this symbol
        self.strategy_data[symbol]['consecutive_losses'] = 0
        self.strategy_data[symbol]['last_trade_date'] = entry_date
        update_strategy_data(symbol, 0, entry_date)

    def handle_sell_order(self, order, symbol):
        if symbol in self.open_positions:
            exit_date = self.data.datetime.date(0)
            entry_date = self.open_positions[symbol]['entry_date']
            entry_price = self.open_positions[symbol]['entry_price']
            size = self.open_positions[symbol]['position_size']
            exit_price = order.executed.price
            profit_loss = (exit_price - entry_price) * size

            add_trade_history(symbol, entry_date, entry_price, exit_date, exit_price, size, profit_loss, "Position Closed")

            # Update consecutive losses if necessary
            if profit_loss < 0:
                self.strategy_data[symbol]['consecutive_losses'] += 1
            else:
                self.strategy_data[symbol]['consecutive_losses'] = 0

            self.strategy_data[symbol]['last_trade_date'] = exit_date
            update_strategy_data(symbol, self.strategy_data[symbol]['consecutive_losses'], exit_date)

            del self.open_positions[symbol]

    def process_buy_signal(self, data):
        symbol = data._name
        if symbol in self.buy_signals and symbol not in self.order_dict:
            cash = self.broker.getcash()
            if cash >= self.params.position_size:
                size = int(self.params.position_size / data.close[0])
                order = self.buy(data=data, size=size)
                self.order_dict[symbol] = order

    def evaluate_sell_conditions(self, data, current_date):
        symbol = data._name

        # Check if there's a pending order or if we're in cooldown
        if symbol in self.pending_orders or (symbol in self.order_cooldown_end and current_date < self.order_cooldown_end[symbol]):
            return

        position = self.broker.getposition(data)
        if position.size == 0:
            return

        entry_price = self.open_positions[symbol]['entry_price']
        current_price = data.close[0]

        if self.check_stop_loss(current_price, entry_price):
            self.close_position(data, "Stop loss")
        elif self.check_take_profit(current_price, entry_price):
            self.close_position(data, "Take profit")
        elif self.check_position_timeout(current_date, symbol):
            self.close_position(data, "Position timeout")
        elif self.check_expected_profit(current_date, symbol, current_price, entry_price):
            self.close_position(data, "Expected Profit Per Day not met")

    def check_position_timeout(self, current_date, symbol):
        entry_date = self.open_positions[symbol]['entry_date']
        days_held = (current_date - entry_date).days
        return days_held > self.params.position_timeout

    def check_expected_profit(self, current_date, symbol, current_price, entry_price):
        entry_date = self.open_positions[symbol]['entry_date']
        days_held = (current_date - entry_date).days
        if days_held > 3:
            total_profit_percentage = ((current_price - entry_price) / entry_price) * 100
            daily_profit_percentage = total_profit_percentage / days_held
            expected_profit_per_day = self.params.expected_profit_per_day_percentage * days_held
            return daily_profit_percentage < expected_profit_per_day
        return False

    def check_stop_loss(self, current_price, entry_price):
        return current_price <= entry_price * (1 - self.params.stop_loss_percent / 100)

    def check_take_profit(self, current_price, entry_price):
        return current_price >= entry_price * (1 + self.params.take_profit_percent / 100)

    def close_position(self, data, reason):
        if data._name not in self.pending_orders:
            logging.info(f'Closing {data._name} due to {reason}')
            print(f'Closing {data._name} due to {reason}')
            order = self.close(data=data)
            self.pending_orders[data._name] = order

    def stop(self):
        cash_balance = self.broker.getcash()
        total_portfolio_value = self.broker.getvalue()
        update_portfolio_data(cash_balance, total_portfolio_value)

        # Calculate and update performance metrics
        sharpe_ratio, win_loss_ratio, avg_profit_per_trade = calculate_performance_metrics()
        update_performance_metrics(sharpe_ratio, win_loss_ratio, avg_profit_per_trade)

        # Update system state
        update_system_state("Stopped", "Daily trading session completed")






























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
        time.sleep(3)  # Allow time for connection to establish
        fetch_and_save_positions(ib)
    except Exception as e:
        print('Failed to connect to Interactive Brokers TWS')
        logging.error(f'Failed to connect to Interactive Brokers TWS: {e}')
        return

    # Load positions from database
    db_positions = get_open_positions()
    db_position_symbols = set(pos['symbol'] for pos in db_positions)

    # Fetch currently held positions using ib_insync
    ib_positions = get_open_positions(ib)
    ib_position_symbols = set(contract.symbol for contract, _ in ib_positions)

    # Combine positions from database and IB
    all_position_symbols = db_position_symbols.union(ib_position_symbols)

    print(f'Total positions: {len(all_position_symbols)}')
    logging.info(f'Total positions: {len(all_position_symbols)}')

    # Add data feeds for all positions
    for symbol in all_position_symbols:
        try:
            contract = next((c for c, _ in ib_positions if c.symbol == symbol), None)
            if not contract:
                contract = ibi.Stock(symbol, 'SMART', 'USD')
            
            if is_valid_exchange(contract.exchange):
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

    # Read buy signals from CSV
    buy_signals = read_buy_signals('BuySignals.csv')
    print(f'Buy Signals: {buy_signals}')
    logging.info(f'Buy Signals: {buy_signals}')

    # Add data feeds for buy signals
    for ticker in buy_signals:
        if ticker not in all_position_symbols:
            try:
                contract = ibi.Stock(ticker, 'SMART', 'USD')
                if is_valid_exchange(contract.exchange):
                    data = store.getdata(dataname=ticker, sectype=contract.secType, exchange=contract.exchange)
                    data._name = ticker
                    cerebro.resampledata(data, timeframe=bt.TimeFrame.Seconds, compression=15)
                    print(f'Added data feed for buy signal {ticker}')
                    logging.info(f'Added data feed for buy signal {ticker}')
                else:
                    print(f'Skipping buy signal for {ticker} - Invalid Exchange: {contract.exchange}')
                    logging.info(f'Skipping buy signal for {ticker} - Invalid Exchange: {contract.exchange}')
            except Exception as e:
                print(f'Error adding data for buy signal {ticker}: {e}')
                logging.error(f'Error adding data for buy signal {ticker}: {e}')

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
