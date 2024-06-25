#!/root/root/miniconda4/envs/tf/bin/python
import os
import time
import logging
import argparse
import random
import pandas as pd
import numpy as np
from backtrader.feeds import PandasData
import backtrader as bt
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from tqdm import tqdm
import json
import csv



def LoggingSetup():
    loggerfile = "__BrokerLog.log"
    logging.basicConfig(
        filename=loggerfile,
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filemode='a'  # Append mode
    )

def arg_parser():
    parser = argparse.ArgumentParser(description="Backtesting Trading Strategy on Stock Data")
    parser.add_argument("--BrokerTest", type=float, default=0, help="Percentage of random files to backtest")
    parser.add_argument("--RunStocks", type=float, default=0, help="Percentage of random files to backtest")

    return parser.parse_args()

class DailyPercentageRange(bt.Indicator):
    lines = ('pct_range',)

    plotinfo = {
        "plot": True,
        "subplot": True
    }

class PercentileIndicator(bt.Indicator):
    lines = ('high_up_prob', 'low_down_prob')
    params = (('period', 7),)  # Window size for percentile calculation

    def __init__(self):
        self.addminperiod(self.params.period)

    def next(self):
        up_probs = self.data0.UpProb.get(size=self.params.period)
        down_probs = self.data0.DownProb.get(size=self.params.period)

        self.lines.high_up_prob[0] = np.percentile(up_probs, 95)
        self.lines.low_down_prob[0] = np.percentile(down_probs, 95)



class ATRPercentage(bt.Indicator):
    lines = ('atr_percent',)
    params = (('period', 14),)

    def __init__(self):
        self.atr = bt.indicators.AverageTrueRange(self.data, period=self.params.period)

    def next(self):
        self.lines.atr_percent[0] = (self.atr[0] / self.data.close[0]) * 100




class CustomPandasData(bt.feeds.PandasData):
    lines = (
        'dist_to_support', 
        'dist_to_resistance', 
        'UpProbability', 
        'UpPrediction',
        'Cluster',
        'mean_intragroup_correlation',
        'diff_to_mean_group_corr',
        'correlation_0', 
        'correlation_1', 
        'correlation_2', 
        'correlation_3', 
        'correlation_4',
        'correlation_5', 
    )
    
    params = (
        ('datetime', None),
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
        ('openinterest', None),
        ('dist_to_support', 'Distance to Support (%)'),
        ('dist_to_resistance', 'Distance to Resistance (%)'),
        ('UpProbability', 'UpProbability'),
        ('UpPrediction', 'UpPrediction'),
        ('Cluster', 'Cluster'),
        ('mean_intragroup_correlation', 'mean_intragroup_correlation'),
        ('diff_to_mean_group_corr', 'diff_to_mean_group_corr'),
        ('correlation_0', 'correlation_0'),
        ('correlation_1', 'correlation_1'),
        ('correlation_2', 'correlation_2'),
        ('correlation_3', 'correlation_3'),
        ('correlation_4', 'correlation_4'),
        ('correlation_5', 'correlation_5'),
    )





class MovingAverageCrossoverStrategy(bt.Strategy):
    params = (
        ('days_range', 5), # Number of days to consider for trading
        ('fast_period', 14),
        ('slow_period', 60),
        ('max_positions', 4),
        ('reserve_percent', 0.4),
        ('stop_loss_percent', 3.0),
        ('take_profit_percent', 100.0),
        ('position_timeout', 9),
        ('trailing_stop_percent', 5.0),
        ('rolling_period', 8),
    )

    def __init__(self):
        self.inds = {}
        self.order_list = []
        self.entry_prices = {}
        self.position_dates = {}
        self.order_cooldown = {}
        self.buying_status = {}
        self.consecutive_losses = {}
        self.cooldown_end = {}
        self.open_positions = 0
        self.asset_groups = {}  # Track group membership of each asset
        self.asset_correlations = {}  # Track correlations of each asset

        for d in self.datas:
            self.inds[d] = {
                'up_prob': d.UpProbability,
                'dist_to_support': d.dist_to_support,
                'dist_to_resistance': d.dist_to_resistance,
                'UpProbMA': bt.indicators.SimpleMovingAverage(d.UpProbability, period=self.params.fast_period),
            }

            self.inds[d]['UpProbMA'].plotinfo.plot = True
            self.inds[d]['UpProbMA'].plotinfo.subplot = True
            self.inds[d]['UpProbMA'].plotinfo.plotlinelabels = True
            self.inds[d]['UpProbMA'].plotinfo.linecolor = 'blue'
            self.inds[d]['UpProbMA'].plotinfo.plotname = 'Up Probability MA'

    def notify_order(self, order):
        status_names = {
            bt.Order.Completed: 'Completed',
            bt.Order.Canceled: 'Canceled',
            bt.Order.Submitted: 'Submitted',
            bt.Order.Accepted: 'Accepted',
            bt.Order.Partial: 'Partially Executed',
            bt.Order.Margin: 'Margin Call Failed',
            bt.Order.Rejected: 'Rejected by Broker',
            bt.Order.Expired: 'Expired'
        }
        order_status = status_names.get(order.status, "Unknown")

        if order.status in [bt.Order.Completed, bt.Order.Partial]:
            if order.isbuy():

                if order.data._name not in self.buying_status:
                    self.open_positions += 1
                self.buying_status[order.data._name] = True
                self.entry_prices[order.data] = order.executed.price
                self.position_dates[order.data] = self.datetime.date()
                self.order_list.append(order)
                self.order_cooldown[order.data] = self.datetime.date() + timedelta(days=1)

                # Track group and correlations
                self.asset_groups[order.data._name] = order.data.Cluster[0]
                self.asset_correlations[order.data._name] = {
                    'mean_intragroup_correlation': order.data.mean_intragroup_correlation[0],
                    'correlation_0': order.data.correlation_0[0],
                    'correlation_1': order.data.correlation_1[0],
                    'correlation_2': order.data.correlation_2[0],
                    'correlation_3': order.data.correlation_3[0],
                    'correlation_4': order.data.correlation_4[0],
                    'correlation_5': order.data.correlation_5[0],
                }
            elif order.issell():
                #print(f"SELL ORDER {order_status}: {order.executed.price}")
                self.open_positions -= 1
                self.buying_status[order.data._name] = False
                if order.data in self.position_dates:
                    days_held = (self.datetime.date() - self.position_dates[order.data]).days
                    percentage = (order.executed.price - self.entry_prices[order.data]) / self.entry_prices[order.data] * 100
                    if percentage < 0:
                        self.consecutive_losses[order.data] = self.consecutive_losses.get(order.data, 0) + 1
                        loss_count = self.consecutive_losses[order.data]
                        cooldown_days = {1: 7, 2: 28, 3: 90, 4: 282}
                        days = cooldown_days.get(loss_count, 282 if loss_count >= 4 else 0)
                        self.cooldown_end[order.data] = self.datetime.date() + timedelta(days=days)
                    else:
                        self.consecutive_losses[order.data] = 0
                    del self.entry_prices[order.data]
                    del self.position_dates[order.data]

                if order in self.order_list:
                    self.order_list.remove(order)

                # Remove group and correlation data
                if order.data._name in self.asset_groups:
                    del self.asset_groups[order.data._name]
                if order.data._name in self.asset_correlations:
                    del self.asset_correlations[order.data._name]

        elif order.status in [bt.Order.Canceled, bt.Order.Margin, bt.Order.Rejected, bt.Order.Expired]:
            if order in self.order_list:
                self.order_list.remove(order)






    def calculate_position_size(self, data):
        total_value = self.broker.getvalue()
        cash_available = self.broker.getcash()
        workable_capital = total_value * 0.6
        capital_for_position = workable_capital / 4  # Each position is 15% of the total account value

        # Calculate the size of the position
        size = int(capital_for_position / data.close[0])
        size = size - 1

        # Check if we can maintain 40% cash and still afford the position
        if (total_value * 0.4) > cash_available or size < 5:
            return 0

        return size if cash_available >= capital_for_position else 0






    def get_mean_correlation(self, candidate_ticker, current_positions):
        correlation_data = pd.read_parquet('Correlations.parquet')
        if not current_positions:
            return 0  # No current positions, so correlation is 0

        # Filter the correlation data for the current candidate and the current positions
        candidate_data = correlation_data[correlation_data['Ticker'] == candidate_ticker]
        correlations = []
        for pos in current_positions:
            pos_data = correlation_data[correlation_data['Ticker'] == pos]
            if not pos_data.empty:
                correlations.append(candidate_data.iloc[0][f'correlation_{pos_data.index[0]}'])

        # Calculate the mean correlation
        if correlations:
            return sum(correlations) / len(correlations)
        else:
            return 0  # Default to 0 if no correlations found




    def next(self):
        current_date = self.datetime.date()
        numoforders = sum([1 for d in self.datas if self.getposition(d).size > 0])
        buy_candidates = []



        for d in self.datas:
            # Selling Logic
            if self.is_position_open(d):
                self.evaluate_sell_conditions(d, current_date)
                # Update numoforders after potential sell
                numoforders = sum([1 for d in self.datas if self.getposition(d).size > 0])

            # Collecting Buy Candidates
            if not self.should_skip_buying(d, current_date) and self.can_buy_more_positions() and self.is_buy_signal(d):
                size = self.calculate_position_size(d)
                if size > 0:
                    buy_candidates.append((d, size))
        # Process Buy Candidates
        if buy_candidates:
            buy_candidates = self.sort_buy_candidates(buy_candidates)
            self.save_best_buy_signals(buy_candidates)

            for d, size in buy_candidates:
                if numoforders < self.params.max_positions:
                    self.execute_buy(d, current_date, size)
                    numoforders += 1
                else:
                    break
        if numoforders > 4:
            print(f"Number of open positions: {numoforders}")




###=====================[ DISCRIMATIONNNNN ]====================###
###=====================[ DISCRIMATIONNNNN ]====================###

    def sort_buy_candidates(self, buy_candidates):
        current_positions = [d._name for d in self.datas if self.getposition(d).size > 0]
        try:
            # Add mean correlation to each candidate
            candidates_with_correlation = []
            for d, size in buy_candidates:
                ticker = d._name
                mean_corr = self.get_mean_correlation(ticker, current_positions)
                candidates_with_correlation.append((d, size, mean_corr))

            # Sort by up_prob first and then by mean correlation (lowest first)
            candidates_with_correlation = sorted(candidates_with_correlation, 
                                                 key=lambda x: (self.inds[x[0]]['up_prob'][0], -x[2]), 
                                                 reverse=True)

            # Remove the correlation value for the final buy candidates list
            buy_candidates = [(d, size) for d, size, _ in candidates_with_correlation]
        except Exception as e:
            logging.error(f"Error sorting buy candidates: {e}")
            ##backup sort by up probability
            buy_candidates = sorted(buy_candidates, key=lambda x: self.inds[x[0]]['up_prob'][0], reverse=True)
        return buy_candidates

    def sort_buy_candidatesUNTESTED(self, buy_candidates):
        current_positions = [d._name for d in self.datas if self.getposition(d).size > 0]
        try:
            # Add mean correlation, cluster information, and a random factor to each candidate
            candidates_with_details = []
            for d, size in buy_candidates:
                ticker = d._name
                mean_corr = self.get_mean_correlation(ticker, current_positions)
                cluster = self.cluster_data.loc[self.cluster_data['Ticker'] == ticker, 'correlation'].values[0]
                random_factor = random.random()  # Random factor for perturbation
                candidates_with_details.append((d, size, mean_corr, cluster, random_factor))

            # Get the current distribution of clusters in the portfolio
            current_cluster_distribution = self.get_cluster_distribution(current_positions)

            # Sort by up_prob, mean correlation (lowest first), and random factor
            candidates_with_details = sorted(candidates_with_details, 
                                             key=lambda x: (self.inds[x[0]]['up_prob'][0], -x[2], x[4]), 
                                             reverse=True)

            # Ensure no more than 3 out of 4 positions are in the same group
            final_buy_candidates = []
            cluster_count = {i: current_cluster_distribution.get(i, 0) for i in range(8)}  # Initialize cluster counts

            for candidate in candidates_with_details:
                d, size, mean_corr, cluster, random_factor = candidate

                if cluster_count[cluster] < 3 or len(final_buy_candidates) < 3:
                    final_buy_candidates.append((d, size))
                    cluster_count[cluster] += 1

                if len(final_buy_candidates) >= 4:
                    break

            # If there are not enough candidates, add the rest based on the primary sorting criteria
            if len(final_buy_candidates) < 4:
                for candidate in candidates_with_details:
                    if candidate[:2] not in final_buy_candidates:
                        final_buy_candidates.append((candidate[0], candidate[1]))
                        if len(final_buy_candidates) >= 4:
                            break

            buy_candidates = final_buy_candidates[:4]  # Ensure only the top 4 candidates are selected
        except Exception as e:
            logging.error(f"Error sorting buy candidates: {e}")
            # Backup sort by up probability
            buy_candidates = sorted(buy_candidates, key=lambda x: self.inds[x[0]]['up_prob'][0], reverse=True)[:4]
        return buy_candidates

    def save_best_buy_signals(self, buy_candidates):
        top_buy_candidates = buy_candidates[:5]  # Adjust the number of top candidates as needed
        for d, size in top_buy_candidates:
            self.save_buy_signal(d, self.datetime.date())

    def save_buy_signal(self, d, current_date):
        CurrentIRLDate = datetime.now().date()
        days_diff = (CurrentIRLDate - current_date).days
    
        # Only save signals within the last 5 IRL days
        if days_diff <= 5:
            csv_file = 'BuySignals.csv'
            columns = ['Ticker', 'Date', 'isBought', 'sharesHeld', 'TimeSinceBought', 'HasHeldPreviously', 'WinLossPercentage']
            
            if not os.path.exists(csv_file):
                with open(csv_file, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(columns)
    
            # Read existing buy signals into a DataFrame
            try:
                buy_signals = pd.read_csv(csv_file)
            except pd.errors.EmptyDataError:
                buy_signals = pd.DataFrame(columns=columns)
    
            new_signal = {
                'Ticker': d._name,
                'Date': str(current_date),
                'isBought': False,
                'sharesHeld': 0,
                'TimeSinceBought': 0,
                'HasHeldPreviously': False,
                'WinLossPercentage': 0.0
            }
    
            if d._name not in buy_signals['Ticker'].values:
                buy_signals = pd.concat([buy_signals, pd.DataFrame([new_signal])], ignore_index=True)
                buy_signals.to_csv(csv_file, index=False)





    def get_cluster_distribution(self, current_positions):
        cluster_distribution = {}
        for ticker in current_positions:
            cluster = self.cluster_data.loc[self.cluster_data['Ticker'] == ticker, 'Cluster'].values[0]
            if cluster in cluster_distribution:
                cluster_distribution[cluster] += 1
            else:
                cluster_distribution[cluster] = 1
        return cluster_distribution




    def should_skip_buying(self, d, current_date):
        if self.buying_status.get(d._name, False):
            return True
        if d._name in self.cooldown_end and current_date <= self.cooldown_end[d._name]:
            return True
        return False

    def is_position_open(self, d):
        return d in self.entry_prices

    def evaluate_sell_conditions(self, d, current_date):
        # Stop loss condition
        if d.close[0] <= self.entry_prices[d] * (1 - self.params.stop_loss_percent / 100):
            self.CloseWrapper(d, "Stop loss")
            return

        # Take profit condition
        if d.close[0] >= self.entry_prices[d] * (1 + self.params.take_profit_percent / 100):
            self.CloseWrapper(d, "Take profit")
            return

        # Position timeout condition
        days_held = (current_date - self.position_dates[d]).days
        if days_held > self.params.position_timeout:
            self.CloseWrapper(d, "Position timeout")
            return
        
        if days_held > 3:
            ExpectedProfitPerDayPercentage = 0.25
            total_profit_percentage = ((d.close[0] - self.entry_prices[d]) / self.entry_prices[d]) * 100
            DailyProfitPercentage = total_profit_percentage / days_held
            ExpectedProfitPerDay = ExpectedProfitPerDayPercentage * days_held

            if DailyProfitPercentage < ExpectedProfitPerDay:
                self.CloseWrapper(d, "Expected Profit Per Day not met")
                return


    def can_buy_more_positions(self):
        return self.open_positions < self.params.max_positions

    def is_buy_signal(self, d):
        HighUpProb = np.percentile(self.inds[d]['up_prob'], 90)
        UpPrediction = self.inds[d]['UpPrediction'] = 1

        FavorableSupportAndResistance = (
            self.inds[d]['dist_to_support'][0] < 50 and
            self.inds[d]['dist_to_resistance'][0] > 100
        )

        RecentMeanPercentageChange = np.mean([
            ((d.close[-i] - d.close[-i-1]) / d.close[-i-1]) * 100
            for i in range(1, 8)
        ])

        CurrentUpProbGood = self.inds[d]['up_prob'][0] > 0.55

        BuySignal = (
            self.inds[d]['UpProbMA'][0] > HighUpProb and
            UpPrediction and
            FavorableSupportAndResistance and
            RecentMeanPercentageChange > 1.5 and
            CurrentUpProbGood
        )
        return BuySignal


    def execute_buy(self, d, current_date, size):
        self.buy(data=d, size=size, exectype=bt.Order.StopTrail, trailpercent=self.params.trailing_stop_percent / 100)
        self.order_cooldown[d] = current_date + timedelta(days=1)
        self.buying_status[d._name] = True
        self.open_positions += 1


    def CloseWrapper(self, data, message):
        self.close(data=data)


    def stop(self):
        for d in self.datas:
            if d in self.entry_prices:
                self.close(data=d)


#=============================[ Control Logic ]=============================#
#=============================[ Control Logic ]=============================#
#=============================[ Control Logic ]=============================#

def select_random_files(directory, percent):
    all_files = [f for f in os.listdir(directory) if f.endswith('.parquet')]
    num_files = len(all_files)
    num_to_select = max(1, int(round(num_files * percent / 100)))
    selected_files = random.sample(all_files, num_to_select)
    return [os.path.join(directory, f) for f in selected_files]

#=============[ ADD DATA TO CEREBRO ]==============#
#=============[ ADD DATA TO CEREBRO ]==============#
#=============[ ADD DATA TO CEREBRO ]==============#

def load_data(file_path):
    Cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Distance to Support (%)', 'Distance to Resistance (%)', 'UpProbability', 'UpPrediction']
    
    df = pd.read_parquet(file_path, columns=Cols)
    
    df.set_index('Date', inplace=True)
    if len(df) < 500:
        return None
    
    last_date = df.index.max()
    one_year_ago = last_date - timedelta(days=365)
    
    df = df[df.index >= one_year_ago]
    return df

def load_correlation_data(file_path):
    # Read the correlation parquet file
    df_corr = pd.read_parquet(file_path)
    return df_corr

def load_and_add_data(cerebro, file_path, correlation_file_path, FailedFileCounter):
    df = load_data(file_path)
    if df is not None:
        if len(df) < 100:
            FailedFileCounter += 1
            return None

        # Load correlation data
        df_corr = load_correlation_data(correlation_file_path)

        # Merge or add the correlation data if necessary
        ticker = os.path.basename(file_path).replace('.parquet', '')
        correlation_info = df_corr[df_corr['Ticker'] == ticker]

        if not correlation_info.empty:
            correlation_info = correlation_info.iloc[0]  # Assuming only one row per ticker
            for col in correlation_info.index:
                if col not in df.columns:  # Only add new columns
                    df[col] = correlation_info[col]

            data = CustomPandasData(dataname=df)
            cerebro.adddata(data, name=os.path.basename(file_path))
        else:
            logging.warning(f"Correlation data not found for {ticker}")
    else:
        FailedFileCounter += 1



def colorize_output(value, label, good_threshold, bad_threshold, lower_is_better=False, reverse=False):
    def get_color_code(normalized_value):
        # Direct transition from red to green
        red = int(255 * (1 - normalized_value))
        green = int(255 * normalized_value)
        return f"\033[38;2;{red};{green};0m"

    # Adjust thresholds and normalization for reverse and lower_is_better scenarios
    if reverse:
        good_threshold, bad_threshold = bad_threshold, good_threshold

    if lower_is_better:
        if value <= good_threshold:
            color_code = "\033[92m"  # Bright Green
        elif value >= bad_threshold:
            color_code = "\033[91m"  # Bright Red
        else:
            range_span = bad_threshold - good_threshold
            normalized_value = (value - good_threshold) / range_span
            color_code = get_color_code(1 - normalized_value)
    else:
        if value >= good_threshold:
            color_code = "\033[92m"  # Bright Green
        elif value <= bad_threshold:
            color_code = "\033[91m"  # Bright Red
        else:
            range_span = good_threshold - bad_threshold
            normalized_value = (value - bad_threshold) / range_span
            color_code = get_color_code(normalized_value)

    return f"{label:<30}{color_code}{value:.2f}\033[0m"





def main():
    Path_to_Correlations = 'Correlations.parquet'
    LoggingSetup()
    timer = time.time()
    cerebro = bt.Cerebro()
    InitalStartingCash = 5000
    cerebro.broker.set_cash(InitalStartingCash)
    args = arg_parser()
    FailedFileCounter = 0

    if args.BrokerTest > 0:
        file_paths = select_random_files('Data/RFpredictions', args.BrokerTest)
        for file_path in tqdm(file_paths, desc="Loading Random Files"):
            load_and_add_data(cerebro, file_path, Path_to_Correlations, FailedFileCounter)
            if FailedFileCounter > 0:
                logging.info(f"Failed to load {FailedFileCounter} files.")
    elif args.RunStocks > 0:
        file_names = [f for f in os.listdir('Data/RFpredictions') if f.endswith('.parquet')]
        file_names.sort(key=lambda x: x.lower())
        for file_name in tqdm(file_names, desc="Loading Ticker Files"):
            file_path = os.path.join('Data/RFpredictions', file_name)
            load_and_add_data(cerebro, file_path, Path_to_Correlations, FailedFileCounter)
            if FailedFileCounter > 0:
                logging.info(f"Failed to load {FailedFileCounter} files.")




    if len(cerebro.datas) == 0:
        print("WARNING No data loaded into Cerebro. Exiting.")
        return

    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="TradeStats")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="DrawDown")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="SharpeRatio", riskfreerate=0.05)
    cerebro.addanalyzer(bt.analyzers.SQN, _name="SQN")

    # Call the strategy
    cerebro.addstrategy(MovingAverageCrossoverStrategy)
    strategies = cerebro.run()
    first_strategy = strategies[0]

    ##enalble cheat on open
    trade_stats = first_strategy.analyzers.TradeStats.get_analysis()
    drawdown = first_strategy.analyzers.DrawDown.get_analysis()
    sharpe_ratio = first_strategy.analyzers.SharpeRatio.get_analysis()
    sqn_value = first_strategy.analyzers.SQN.get_analysis().get('sqn', 0)

    if sqn_value < 2.0:
        description = 'Poor, but tradeable'
    elif 2.0 <= sqn_value < 2.5:
        description = 'Average'
    elif 2.5 <= sqn_value < 3.0:
        description = 'Good'
    elif 3.0 <= sqn_value < 5.0:
        description = 'Excellent'
    elif 5.0 <= sqn_value < 7.0:
        description = 'Superb'
    elif sqn_value >= 7.0:
        description = 'Maybe the Holy Grail!'

    won_total = trade_stats.get('won', {}).get('pnl', {}).get('total', 0)
    lost_total = trade_stats.get('lost', {}).get('pnl', {}).get('total', 0)
    total_closed = trade_stats.get('total', {}).get('closed', 0)
    won_avg = trade_stats.get('won', {}).get('pnl', {}).get('average', 0)
    lost_avg = trade_stats.get('lost', {}).get('pnl', {}).get('average', 0)
    won_max = trade_stats.get('won', {}).get('pnl', {}).get('max', 0)
    lost_max = trade_stats.get('lost', {}).get('pnl', {}).get('max', 0)
    net_total = trade_stats.get('pnl', {}).get('net', {}).get('total', 0)

    PercentGain = ((cerebro.broker.getvalue() / InitalStartingCash) * 100) - 100
    if won_total != 0 and lost_total != 0 and total_closed > 10:
        logging.info(f"====================== New Entry ===================")
        logging.info(f"Max Drawdown: {drawdown['max']['drawdown']:.2f}%")
        logging.info(f"Total Trades: {total_closed}")
        logging.info(f"Profitable Trades: {trade_stats['won']['total']}")
        logging.info(f"Sharpe Ratio: {sharpe_ratio['sharperatio']:.2f}")
        logging.info(f"SQN: {sqn_value:.2f}")
        logging.info(f"Profit Factor: {lost_total / won_total:.2f}")
        logging.info(f"Average Trade: {net_total / total_closed:.2f}")
        logging.info(f"Average Winning Trade: {won_avg:.2f}")
        logging.info(f"Average Losing Trade: {lost_avg:.2f}")
        logging.info(f"Largest Winning Trade: {won_max:.2f}")
        logging.info(f"Largest Losing Trade: {lost_max:.2f}")
        logging.info(f"Time taken: {time.time() - timer:.2f} seconds")
        logging.info(f"Inital Portfolio funding: ${InitalStartingCash:.2f}")
        logging.info(f"Final Portfolio Value: ${cerebro.broker.getvalue():.2f}")
        logging.info(f"Percentage Gain: {PercentGain:.2f}%")
        logging.info(f"Cash: ${cerebro.broker.getcash():.2f}")

    if total_closed > 0:
        percent_profitable = (trade_stats['won']['total'] / total_closed) * 100
    else:
        percent_profitable = 0

    final_portfolio_value = cerebro.broker.getvalue()
    percentage_gain = ((final_portfolio_value / InitalStartingCash) * 100) - 100
    PercentPerTrade = percentage_gain / total_closed if total_closed > 0 else 0

    if total_closed > 1:
        percentage_gain_without_largest_winner = (((final_portfolio_value - won_max) / InitalStartingCash) * 100) - 100
        PercentWithoutLargestWinningTradePerTrade = percentage_gain_without_largest_winner / total_closed

        if won_total > 0:
            won_max_percentage = (won_max / won_total) * 100
    else :
        percentage_gain_without_largest_winner = 0
        PercentWithoutLargestWinningTradePerTrade = 0
        won_max_percentage = 0
        sharpe_ratio['sharperatio'] = 0

    print(f"{'=' * 8} Trading Strategy Results {'=' * 8}")
    print(f"{'Initial Starting Cash:':<30}${cerebro.broker.startingcash:.2f}")
    print(colorize_output(final_portfolio_value, 'Final Portfolio Value:', InitalStartingCash * 2, InitalStartingCash * 1.25))
    print(colorize_output(drawdown['max']['len'], 'Max Drawdown Days:', 15, 90, lower_is_better=True))
    print(colorize_output(drawdown['max']['drawdown'], 'Max Drawdown percentage:', 5, 15, lower_is_better=True))
    print(colorize_output(total_closed, 'Total Trades:', 75, 110, lower_is_better=True))
    print(colorize_output(percent_profitable, 'Profitable Trades %:', 60, 40))
    print(colorize_output(won_avg, 'Average Profitable Trade:', 100, 50))
    print(colorize_output(lost_avg, 'Average Unprofitable Trade:', -20, -100, reverse=True))
    print(colorize_output(won_total, 'Total Profitable Trades:', 5000, 800))
    print(colorize_output(lost_total, 'Total Unprofitable Trades:', -20, -2500, reverse=True))
    print(colorize_output(won_max_percentage, 'Largest Winning percent:', 5, 50))
    print(colorize_output(won_max, 'Largest Winning Trade:', 500, 100))
    print(colorize_output(lost_max, 'Largest Losing Trade:', -50, -200, reverse=True))
    print(colorize_output(sharpe_ratio['sharperatio'], 'Sharpe Ratio:', 2.0, 1.0))
    print(colorize_output(sqn_value, 'SQN:', 1.9, 5.0))
    print(f"{'SQN Description:':<30}{description}")
    print(colorize_output(PercentPerTrade, 'Gain Per Trade %:', 1.0, 0.5))
    print(colorize_output(PercentWithoutLargestWinningTradePerTrade, 'Gain without Largest:', 1.0, 0.75))
    print(colorize_output(percentage_gain, 'Percentage Gain:', 50, 10.0))

    if len(cerebro.datas) < 10:
        plt.style.use('dark_background')
        plt.rcParams['figure.facecolor'] = '#424242'
        plt.rcParams['axes.facecolor'] = '#424242'
        plt.rcParams['grid.color'] = 'None'
        cerebro.plot(style='candlestick',
                     iplot=False,
                     start=datetime.now().date() - pd.DateOffset(days=252),
                     end=datetime.now().date(),
                     width=20,
                     height=10,
                     dpi=100,
                     tight=True)


if __name__ == "__main__":
    main()