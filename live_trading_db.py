import sqlite3
from datetime import datetime, date

DB_NAME = 'live_trading.db'


def get_db_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def create_tables():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute('''
    CREATE TABLE IF NOT EXISTS open_positions (
        symbol TEXT PRIMARY KEY,
        entry_date DATE NOT NULL,
        entry_price REAL NOT NULL,
        position_size INTEGER NOT NULL,
        current_stop_loss REAL,
        current_take_profit REAL
    )
    ''')

    cur.execute('''
    CREATE TABLE IF NOT EXISTS trade_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        symbol TEXT NOT NULL,
        entry_date DATE NOT NULL,
        entry_price REAL NOT NULL,
        exit_date DATE NOT NULL,
        exit_price REAL NOT NULL,
        position_size INTEGER NOT NULL,
        profit_loss REAL NOT NULL,
        exit_reason TEXT
    )
    ''')

    cur.execute('''
    CREATE TABLE IF NOT EXISTS strategy_data (
        symbol TEXT PRIMARY KEY,
        consecutive_losses INTEGER NOT NULL DEFAULT 0,
        last_trade_date DATE
    )
    ''')

    cur.execute('''
    CREATE TABLE IF NOT EXISTS portfolio_data (
        date DATE PRIMARY KEY,
        cash_balance REAL NOT NULL,
        total_portfolio_value REAL NOT NULL
    )
    ''')

    cur.execute('''
    CREATE TABLE IF NOT EXISTS risk_management (
        date DATE PRIMARY KEY,
        max_drawdown REAL NOT NULL,
        daily_pl REAL NOT NULL,
        weekly_pl REAL NOT NULL,
        monthly_pl REAL NOT NULL
    )
    ''')

    cur.execute('''
    CREATE TABLE IF NOT EXISTS system_state (
        last_run_date DATE PRIMARY KEY,
        status TEXT NOT NULL,
        message TEXT
    )
    ''')

    cur.execute('''
    CREATE TABLE IF NOT EXISTS performance_metrics (
        date DATE PRIMARY KEY,
        sharpe_ratio REAL,
        win_loss_ratio REAL,
        avg_profit_per_trade REAL
    )
    ''')

    cur.execute('''
    CREATE TABLE IF NOT EXISTS buy_signals (
        symbol TEXT,
        date DATE NOT NULL,
        price REAL,
        PRIMARY KEY (symbol, date)
    )
    ''')

    conn.commit()
    conn.close()

# Open Positions functions
def add_open_position(symbol, entry_date, entry_price, position_size, stop_loss=None, take_profit=None):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('''
    INSERT OR REPLACE INTO open_positions 
    (symbol, entry_date, entry_price, position_size, current_stop_loss, current_take_profit)
    VALUES (?, ?, ?, ?, ?, ?)
    ''', (symbol, entry_date, entry_price, position_size, stop_loss, take_profit))
    conn.commit()
    conn.close()

def get_open_positions():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT * FROM open_positions')
    positions = cur.fetchall()
    conn.close()
    return positions

def update_position_exit_levels(symbol, stop_loss, take_profit):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('''
    UPDATE open_positions 
    SET current_stop_loss = ?, current_take_profit = ?
    WHERE symbol = ?
    ''', (stop_loss, take_profit, symbol))
    conn.commit()
    conn.close()

# Trade History functions
def add_trade_history(symbol, entry_date, entry_price, exit_date, exit_price, position_size, profit_loss, exit_reason):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('''
    INSERT INTO trade_history 
    (symbol, entry_date, entry_price, exit_date, exit_price, position_size, profit_loss, exit_reason)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (symbol, entry_date, entry_price, exit_date, exit_price, position_size, profit_loss, exit_reason))
    conn.commit()
    conn.close()

# Strategy Data functions
def update_strategy_data(symbol, consecutive_losses, last_trade_date):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('''
    INSERT OR REPLACE INTO strategy_data 
    (symbol, consecutive_losses, last_trade_date)
    VALUES (?, ?, ?)
    ''', (symbol, consecutive_losses, last_trade_date))
    conn.commit()
    conn.close()

def get_strategy_data(symbol):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT * FROM strategy_data WHERE symbol = ?', (symbol,))
    data = cur.fetchone()
    conn.close()
    return data

def update_performance_metrics(sharpe_ratio, win_loss_ratio, avg_profit_per_trade):
    conn = get_db_connection()
    cur = conn.cursor()
    today = datetime.now().date()
    
    cur.execute('''
    INSERT OR REPLACE INTO performance_metrics 
    (date, sharpe_ratio, win_loss_ratio, avg_profit_per_trade)
    VALUES (?, ?, ?, ?)
    ''', (today, sharpe_ratio, win_loss_ratio, avg_profit_per_trade))
    
    conn.commit()
    conn.close()

def get_latest_performance_metrics():
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute('''
    SELECT * FROM performance_metrics
    ORDER BY date DESC
    LIMIT 1
    ''')
    
    metrics = cur.fetchone()
    conn.close()
    
    return metrics

# Portfolio Data functions
def update_portfolio_data(cash_balance, total_portfolio_value):
    conn = get_db_connection()
    cur = conn.cursor()
    today = date.today()
    cur.execute('''
    INSERT OR REPLACE INTO portfolio_data 
    (date, cash_balance, total_portfolio_value)
    VALUES (?, ?, ?)
    ''', (today, cash_balance, total_portfolio_value))
    conn.commit()
    conn.close()

def get_latest_portfolio_data():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT * FROM portfolio_data ORDER BY date DESC LIMIT 1')
    data = cur.fetchone()
    conn.close()
    return data

# Risk Management functions
def update_risk_management(max_drawdown, daily_pl, weekly_pl, monthly_pl):
    conn = get_db_connection()
    cur = conn.cursor()
    today = date.today()
    cur.execute('''
    INSERT OR REPLACE INTO risk_management 
    (date, max_drawdown, daily_pl, weekly_pl, monthly_pl)
    VALUES (?, ?, ?, ?, ?)
    ''', (today, max_drawdown, daily_pl, weekly_pl, monthly_pl))
    conn.commit()
    conn.close()

# System State functions
def update_system_state(status, message):
    conn = get_db_connection()
    cur = conn.cursor()
    today = date.today()
    cur.execute('''
    INSERT OR REPLACE INTO system_state 
    (last_run_date, status, message)
    VALUES (?, ?, ?)
    ''', (today, status, message))
    conn.commit()
    conn.close()

def get_last_system_state():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT * FROM system_state ORDER BY last_run_date DESC LIMIT 1')
    state = cur.fetchone()
    conn.close()
    return state

def update_buy_signal(symbol, buy_signal):
    conn = get_db_connection()
    cur = conn.cursor()
    today = date.today()
    cur.execute('''
    INSERT OR REPLACE INTO buy_signals 
    (symbol, buy_signal, date_updated)
    VALUES (?, ?, ?)
    ''', (symbol, buy_signal, today))
    conn.commit()
    conn.close()

def get_buy_signals():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT * FROM buy_signals')
    signals = cur.fetchall()
    conn.close()
    return signals

# Initialize the database
if __name__ == "__main__":
    create_tables()
    print(f"Database '{DB_NAME}' created successfully with all required tables.")
