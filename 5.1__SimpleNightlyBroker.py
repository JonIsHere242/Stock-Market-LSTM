import os
import random
import pandas as pd
import numpy as np
import backtrader as bt
import logging
from datetime import datetime, timedelta
from tqdm import tqdm




# Set up logging
logging.basicConfig(filename='__SimplifiedNightlyBroker.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')




class SimplifiedMAStrategy(bt.Strategy):
    
    params = (
        ('fast_period', 50),
        ('slow_period', 200),
        ('max_positions', 4),
        ('max_total_exposure', 0.60),
        ('max_single_exposure', 0.15),
        ('trailing_stop_pct', 0.05),  # 5% trailing stop
        ('position_timeout', 9),  # 5 days position timeout
        ('min_daily_profit_pct', 0.001),  # 0.1% minimum daily profit
    )

    def __init__(self):
        self.fast_ma = {}
        self.slow_ma = {}
        for d in self.datas:
            try:
                self.fast_ma[d] = bt.indicators.SMA(d, period=self.p.fast_period)
                self.slow_ma[d] = bt.indicators.SMA(d, period=self.p.slow_period)
            except Exception as e:
                logging.warning(f"Error initializing SMA for {d._name}: {str(e)}")
        self.order = {}
        self.correlation_data = self.load_correlation_data()
        self.my_positions = {}
        self.position_entry_dates = {}
        self.position_entry_prices = {}

    def load_correlation_data(self):
        corr_df = pd.read_parquet('Correlations.parquet')
        return {row['Ticker']: row.to_dict() for _, row in corr_df.iterrows()}

    def next(self):
        for d in self.datas:
            if d not in self.fast_ma or d not in self.slow_ma:
                continue
            
            if d._name in self.my_positions:
                self.check_and_close_position(d)
            
            if not self.getposition(d).size and len(self.my_positions) < self.p.max_positions:
                if self.fast_ma[d][0] > self.slow_ma[d][0] and self.fast_ma[d][-1] <= self.slow_ma[d][-1]:
                    self.buy_signal(d)

    def check_and_close_position(self, data):
        days_held = len(self) - self.position_entry_dates[data._name]
        entry_price = self.position_entry_prices[data._name]
        current_price = data.close[0]
        total_profit_pct = (current_price - entry_price) / entry_price
        daily_profit_pct = total_profit_pct / days_held if days_held > 0 else 0

        # Check for position timeout
        if days_held >= self.p.position_timeout:
            self.close_position(data, "timeout")
        # Check for minimum daily profit
        elif daily_profit_pct < self.p.min_daily_profit_pct:
            self.close_position(data, "insufficient daily profit")
        # Check for MA crossover (sell signal)
        elif self.fast_ma[data][0] < self.slow_ma[data][0] and self.fast_ma[data][-1] >= self.slow_ma[data][-1]:
            self.close_position(data, "MA crossover")

    def close_position(self, data, reason):
        self.close(data=data)
        del self.my_positions[data._name]
        del self.position_entry_dates[data._name]
        del self.position_entry_prices[data._name]
        logging.info(f"Closing position for {data._name} due to {reason}")

    def buy_signal(self, data):
        portfolio_value = self.broker.getvalue()
        max_position_value = portfolio_value * self.p.max_single_exposure
        current_exposure = sum(self.broker.get_value([d]) for d in self.datas if d._name in self.my_positions)

        if (current_exposure + max_position_value) / portfolio_value <= self.p.max_total_exposure:
            size = int(max_position_value / data.close[0])
            if size > 0:
                # Use StopTrail order type with trailpercent
                self.buy(data=data, size=size, exectype=bt.Order.StopTrail, trailpercent=self.p.trailing_stop_pct)
                self.my_positions[data._name] = size
                self.position_entry_dates[data._name] = len(self)
                self.position_entry_prices[data._name] = data.close[0]
                logging.info(f"Buy signal for {data._name} at {data.close[0]}, size: {size}, trail: {self.p.trailing_stop_pct:%}")

    def stop(self):
        self.save_buy_signals()

    def save_buy_signals(self):
        buy_signals = []
        for d in self.datas:
            if d not in self.fast_ma or d not in self.slow_ma:
                continue
            if self.fast_ma[d][0] > self.slow_ma[d][0] and not self.getposition(d).size:
                correlation = self.get_mean_correlation(d._name)
                buy_signals.append({
                    'Symbol': d._name,
                    'Date': d.datetime.date(0),
                    'Close': d.close[0],
                    'Correlation': correlation
                })
        
        df = pd.DataFrame(buy_signals)
        if not df.empty:
            df = df.sort_values('Correlation', ascending=True).head(10)  # Top 10 least correlated signals
            df.to_csv('BuySignals.csv', index=False)
            logging.info(f"Saved {len(df)} buy signals to BuySignals.csv")
        else:
            logging.info("No buy signals to save.")

    def get_mean_correlation(self, symbol):
        if symbol in self.correlation_data:
            correlations = [v for k, v in self.correlation_data[symbol].items() if k.startswith('correlation_') and v != 1]
            return np.mean(correlations) if correlations else 0
        return 0
















def run_backtest(data_dir, start_date, end_date):
    cerebro = bt.Cerebro()
    cerebro.broker.set_cash(5000)  # Set initial capital

    logging.info(f"===========================[ Backtest: {start_date} - {end_date} ]===========================")
    logging.info(f"===========================[ Backtest: {start_date} - {end_date} ]===========================")
    logging.info(f"===========================[ Backtest: {start_date} - {end_date} ]===========================")


    # Load data
    for file in tqdm(os.listdir(data_dir)):
        if file.endswith('.parquet'):
            df = pd.read_parquet(os.path.join(data_dir, file))
            df['datetime'] = pd.to_datetime(df['Date'])
            df = df[(df['datetime'].dt.date >= start_date) & (df['datetime'].dt.date <= end_date)]
            
            # Check if there's enough data for SMA calculation
            if len(df) > max(SimplifiedMAStrategy.params.fast_period, SimplifiedMAStrategy.params.slow_period):
                df.set_index('datetime', inplace=True)
                data = bt.feeds.PandasData(dataname=df, name=file.replace('.parquet', ''))
                cerebro.adddata(data)
            else:
                logging.warning(f"Skipping {file} due to insufficient data points")

    cerebro.addstrategy(SimplifiedMAStrategy)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=15)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe_ratio')
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')

    # Run the backtest
    results = cerebro.run()

    # Print results
    strat = results[0]
    sharpe_ratio = strat.analyzers.sharpe_ratio.get_analysis()['sharperatio']
    max_drawdown = strat.analyzers.drawdown.get_analysis()['max']['drawdown']
    total_return = strat.analyzers.returns.get_analysis()['rtot']

    logging.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    logging.info(f"Max Drawdown: {max_drawdown:.2%}")
    logging.info(f"Total Return: {total_return:.2%}")
    logging.info(f"Final Portfolio Value: ${cerebro.broker.getvalue():.2f}")

if __name__ == '__main__':
    data_dir = 'Data/RFpredictions'
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365)  # 1 year of data
    run_backtest(data_dir, start_date, end_date)