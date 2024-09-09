import os
import pandas as pd
import numpy as np
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from gplearn.fitness import make_fitness
from sklearn.metrics import mean_squared_error, mean_absolute_error, mutual_info_score
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from scipy.stats import entropy, spearmanr, zscore, kendalltau
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy.special import kl_div
import random
import datetime
import time
from numba import njit, prange
import joblib  # Used to save/load preprocessed datasets


#import the mutual_info_score

# Updated CONFIG with Dataset Caching Options
CONFIG = {
    'input_directory': 'Data/PriceData',
    'data_sample_percentage': 3.0,  # Increased from 3 to use more data
    'output_directory': 'Results',
    'dataset_cache_path': 'cached_dataset.pkl',
    'use_cached_dataset': True,
    'population_size': 10000,  # Reduced from 10000 to allow more generations
    'generations': 10,  # Increased from 10 to allow for more evolution
    'tournament_size': 5,  # Increased from 3 to apply more selection pressure
    'stopping_criteria': -1e9,  # Set to a very low value to prevent early stopping
    'const_range': (-1, 1),  # Expanded range to allow for more diverse constants
    'init_depth': (2, 5),  # Increased max depth to allow for more complex initial programs
    'init_method': 'half and half',
    'function_set': ('add', 'sub', 'mul', 'div', 'abs'),  # Added 'abs'
    'parsimony_coefficient': 0.05,  # Reduced to allow for more complex programs
    'p_crossover': 0.7,
    'p_subtree_mutation': 0.1,
    'p_hoist_mutation': 0.05,
    'p_point_mutation': 0.1,
    'p_point_replace': 0.05,
    'max_samples': 0.2,  # Increased from 0.05 to use more of the data in each generation
    'verbose': 1,
    'n_jobs': -1,
    'low_memory': False,
    'random_state': 3301,
    'max_depth': 10  # Increased to allow for more complex programs
}


def safe_entropy(arr):
    hist, _ = np.histogram(arr, bins=5)
    if np.sum(hist) > 0:
        return entropy(hist)
    else:
        return np.nan  # Return NaN for empty or invalid slices

def dynamic_round(x):
    abs_x = np.abs(x)
    
    if abs_x < 1:  # Less than $1
        return np.round(x, 3)  # Round to nearest 0.1 cent
    elif abs_x < 10:  # $1 to $9.99
        return np.round(x, 2)  # Round to nearest cent
    elif abs_x < 100:  # $10 to $99.99
        return np.round(x, 1)  # Round to nearest 10 cents
    else:  # $100 and above
        return np.round(x, 0)  # Round to nearest dollar

# Vectorize the function for efficiency
dynamic_round_vec = np.vectorize(dynamic_round)


def calculate_time_weighted_target(df, daily_weight=0.90, weekly_weight=0.09, yearly_weight=0.01):
    # Calculate daily, weekly, and yearly future returns
    df['Future_Returns_Daily'] = df['Close'].shift(-1) / df['Close'] - 1  # 1 day ahead
    df['Future_Returns_Weekly'] = df['Close'].shift(-5) / df['Close'] - 1  # 5 days ahead (weekly)
    df['Future_Returns_Yearly'] = df['Close'].shift(-252) / df['Close'] - 1  # 252 trading days ahead (yearly)

    # Apply exponential decay weights to daily, weekly, and yearly returns
    weight_sum = daily_weight + weekly_weight + yearly_weight
    normalized_daily_weight = daily_weight / weight_sum
    normalized_weekly_weight = weekly_weight / weight_sum
    normalized_yearly_weight = yearly_weight / weight_sum

    # Calculate the time-weighted future returns
    df['Weighted_Future_Returns'] = (
        normalized_daily_weight * df['Future_Returns_Daily'] +
        normalized_weekly_weight * df['Future_Returns_Weekly'] +
        normalized_yearly_weight * df['Future_Returns_Yearly']
    )

    # Shift the target to align with the current row for machine learning purposes
    df['Target'] = df['Weighted_Future_Returns'].shift(-1)

    # Drop intermediate columns
    df = df.drop(columns=['Future_Returns_Daily', 'Future_Returns_Weekly', 'Future_Returns_Yearly', 'Weighted_Future_Returns'])

    return df

def simple_entropy_filter(df, window_size=64, min_periods=32, lower_quantile=0.1, upper_quantile=0.95):
    # Calculate returns and fill any NaN values
    df['Return'] = df['Close'].pct_change().fillna(0)

    # Define a simple entropy function based on histogram
    def calculate_entropy(arr):
        hist, _ = np.histogram(arr, bins=5)
        return entropy(hist) if np.sum(hist) > 0 else np.nan

    # Calculate rolling entropy with a minimum of 32 periods
    df['Return_Entropy'] = df['Return'].rolling(window=window_size, min_periods=min_periods).apply(calculate_entropy, raw=True)

    # Scale entropy to be between 0 and 1
    scaler = MinMaxScaler()
    df[['Return_Entropy']] = scaler.fit_transform(df[['Return_Entropy']].fillna(0))

    # Calculate dynamic thresholds based on quantiles
    lower_threshold = df['Return_Entropy'].quantile(lower_quantile)
    upper_threshold = df['Return_Entropy'].quantile(upper_quantile)

    # Filter out the bottom 10% least interesting and the top 5% most random data
    df_filtered = df[(df['Return_Entropy'] > lower_threshold) & (df['Return_Entropy'] < upper_threshold)]

    # Drop intermediate columns
    df_filtered = df_filtered.drop(columns=['Return', 'Return_Entropy'])
    
    return df_filtered





def load_and_prepare_data(file_paths, window_size=14, lower_quantile=0.1, upper_quantile=0.9):
    original_total_rows = 0
    filtered_total_rows = 0
    all_data = []

    for file_path in file_paths:
        df = pd.read_parquet(file_path)
        
        if 'Date' not in df.columns and df.index.name == 'Date':
            df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date'])
        
        if len(df) < 100:
            continue

        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df = df.assign(
            Open=dynamic_round_vec(df['Open']),
            High=dynamic_round_vec(df['High']),
            Low=dynamic_round_vec(df['Low']),
            Close=dynamic_round_vec(df['Close'])
        )

        original_total_rows += len(df)
        
        # Apply dynamic entropy filtering before lag features and other transformations
        df = simple_entropy_filter(df)
        filtered_total_rows += len(df)

        for col in ['High', 'Low', 'Volume']:
            lag_times = [1, 2, 3] if col == 'Volume' else [2, 5]
            for lag in lag_times:
                df[f'{col}_Lag{lag}'] = df[col].shift(lag)

        df['High_Low'] = df['High'] - df['Low']
        df['High_Close'] = df['High'] - df['Close']
        df['Low_Close'] = df['Low'] - df['Close']

        df = df.dropna()
        df = calculate_time_weighted_target(df)
        df = df.dropna().replace([np.inf, -np.inf], np.nan).dropna()
        all_data.append(df)

    combined_data = pd.concat(all_data, ignore_index=True).drop_duplicates()
    combined_data = combined_data.dropna()

    # Explicitly list the feature columns
    feature_columns = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'High_Lag2', 'High_Lag5', 'Low_Lag2', 'Low_Lag5',
        'Volume_Lag1', 'Volume_Lag2', 'Volume_Lag3',
        'High_Low', 'High_Close', 'Low_Close'
    ]
    X = combined_data[feature_columns]
    y = combined_data['Target']
    
    percentage_removed = ((original_total_rows - filtered_total_rows) / original_total_rows) * 100 if original_total_rows > 0 else 0
    
    print(f"Original data shape: {original_total_rows} rows")
    print(f"Filtered data shape: {filtered_total_rows} rows")
    print(f"Percentage of data removed: {percentage_removed:.2f}%")

    return X, y















def safe_operation(operation, default_value):
    try:
        result = operation()
        return default_value if np.isnan(result) or np.isinf(result) else result
    except:
        return default_value

def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    direction_accuracy = np.mean((np.sign(y_true) == np.sign(y_pred)))
    
    print(f"{model_name} Performance:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"Correlation: {correlation:.6f}")
    print(f"Direction Accuracy: {direction_accuracy:.6f}")
    print()

@njit(parallel=True)
def fast_mse(y, y_pred):
    return np.mean((y - y_pred)**2)

@njit(parallel=True)
def fast_mae(y, y_pred):
    return np.mean(np.abs(y - y_pred))

@njit(parallel=True)
def calculate_directional_accuracy(y, y_pred):
    return np.mean((np.sign(y) == np.sign(y_pred)))

@njit(parallel=True)
def calculate_profit_factor(y, y_pred):
    profits = np.where((y > 0) & (y_pred > 0), y, 0)
    losses = np.where((y < 0) & (y_pred > 0), -y, 0)
    return np.sum(profits) / (np.sum(losses) + 1e-10)

@njit(parallel=True)
def calculate_sharpe_ratio(y, y_pred):
    returns = y * np.sign(y_pred)
    return np.mean(returns) / (np.std(returns) + 1e-10)

@njit(parallel=True)
def calculate_information_ratio(y, y_pred):
    active_returns = y - y_pred
    return np.mean(active_returns) / (np.std(active_returns) + 1e-10)

@njit
def preprocess_data(y, y_pred):
    mask = np.isfinite(y) & np.isfinite(y_pred)
    return y[mask], y_pred[mask]

@njit
def calculate_all_metrics(y, y_pred):
    mse = fast_mse(y, y_pred)
    mae = fast_mae(y, y_pred)
    da = calculate_directional_accuracy(y, y_pred)
    pf = calculate_profit_factor(y, y_pred)
    sr = calculate_sharpe_ratio(y, y_pred)
    ir = calculate_information_ratio(y, y_pred)
    return mse, mae, da, pf, sr, ir







#####===============================[Custom Fitness Function]==================================#####
#####===============================[Custom Fitness Function]==================================#####
#####===============================[Custom Fitness Function]==================================#####
#####===============================[Custom Fitness Function]==================================#####
#####===============================[Custom Fitness Function]==================================#####


#
#def y_true_discrete(y, bins=10):
#    """
#    Discretize continuous `y` values into discrete bins for use in mutual information calculations.
#    
#    Args:
#    y (array-like): Continuous target values.
#    bins (int): The number of bins to discretize the data into.
#    
#    Returns:
#    array: Discretized version of `y` into bins.
#    """
#    return np.digitize(y, np.linspace(np.min(y), np.max(y), bins))
#
#def _custom_fitness(y, y_pred, w):
#    """
#    Custom fitness function for evaluating model performance using Information Coefficient (IC), 
#    Mutual Information (MI), and Directional Accuracy (DA).
#    
#    Args:
#    y (array-like): True target values.
#    y_pred (array-like): Predicted values from the model.
#    w: Weighting parameter (not used directly in this implementation but required for compatibility).
#    
#    Returns:
#    float: Fitness score, with lower values indicating better performance.
#    """
#    
#    # If either `y` or `y_pred` has no data, return a very large error (1e10).
#    if len(y) == 0 or len(y_pred) == 0:
#        return 1e10
#    
#    # If all predictions are the same, return a large error (1e9) to penalize lack of variability in the predictions.
#    if np.all(y_pred == y_pred[0]):
#        return 1e9
#    
#    try:
#        # Spearman's rank correlation is calculated between the true and predicted values.
#        # This correlation is also called Information Coefficient (IC).
#        ic = abs(spearmanr(y, y_pred)[0])  # Use absolute value since direction doesn't matter.
#        
#        # Mutual Information (MI) measures the amount of shared information between two variables.
#        # It is useful for assessing the dependency between the true and predicted values.
#        # The values are first discretized into bins before applying `mutual_info_score`.
#        mi = mutual_info_score(y_true_discrete(y), y_true_discrete(y_pred))
#
#        # Calculate the entropy of the true values `y`. This is used for normalizing the mutual information (MI).
#        # This tells us how much uncertainty is present in the target values `y` by themselves.
#        y_entropy = mutual_info_score(y_true_discrete(y), y_true_discrete(y))
#
#        # Normalize MI by dividing by the entropy of `y`. This gives a relative measure of dependency.
#        # If the entropy is zero (no variation in `y`), we avoid division by zero by setting normalized_mi to 0.
#        normalized_mi = mi / y_entropy if y_entropy != 0 else 0
#        
#        # Directional accuracy (DA) calculates how often the predicted direction (positive/negative)
#        # matches the direction of the true values.
#        da = np.mean((np.sign(y) == np.sign(y_pred)))
#        
#        # The combined fitness score is a weighted sum of the absolute IC, normalized MI, and DA.
#        # IC and normalized MI are given more weight than directional accuracy.
#        combined_score = (
#            0.45 * ic +                # 45% weight on absolute Information Coefficient
#            0.45 * normalized_mi +     # 45% weight on Normalized Mutual Information
#            0.10 * da                  # 10% weight on Directional Accuracy
#        )
#        
#        # Return the reciprocal of the combined score to ensure that lower fitness values
#        # (i.e., higher combined scores) indicate better performance.
#        return 1 / combined_score
#    except Exception as e:
#        # If any error occurs during fitness calculation, return a very large error value (1e32) to signify failure.
#        print(f"Error in fitness calculation: {e}")
#        return 1e32
#
## Register the custom fitness function for use in genetic programming models.
#custom_fitness = make_fitness(function=_custom_fitness, greater_is_better=False)


@njit
def y_true_discrete(y, bins=10):
    """
    Discretize continuous `y` values into discrete bins for use in mutual information calculations.
    """
    min_y, max_y = np.min(y), np.max(y)
    bin_edges = np.linspace(min_y, max_y, bins)
    return np.digitize(y, bin_edges)

@njit
def concordance(y, y_pred):
    """
    Calculate concordance as the percentage of times the true values and predicted values move in the same direction.
    """
    y_change = np.diff(y)
    y_pred_change = np.diff(y_pred)
    return np.mean(np.sign(y_change) == np.sign(y_pred_change))

def _custom_fitness(y, y_pred, w):
    """
    Optimized custom fitness function with concordance.
    """
    # Check for empty or identical predictions more efficiently
    if len(y) == 0 or len(y_pred) == 0:
        return 1e10

    if np.all(y_pred == y_pred[0]):
        return 1e9

    try:
        # Spearman's rank correlation coefficient (IC)
        ic = abs(spearmanr(y, y_pred)[0])

        # Discretize `y` and `y_pred` once and reuse
        y_discrete = y_true_discrete(y)
        y_pred_discrete = y_true_discrete(y_pred)

        # Mutual Information (MI)
        mi = mutual_info_score(y_discrete, y_pred_discrete)

        # Calculate the entropy of `y` once
        y_entropy = mutual_info_score(y_discrete, y_discrete)

        # Normalize MI
        normalized_mi = mi / y_entropy if y_entropy != 0 else 0

        # Directional Accuracy (DA)

        # Concordance (CON)
        con = abs(concordance(y, y_pred))

        # Combine scores with weights
        combined_score = (0.1 * ic) + (0.1 * normalized_mi) + (0.8 * con)

        # Return inverse of combined score (lower is better)
        return 1 / combined_score
    except Exception as e:
        return 1e32

# Register the optimized custom fitness function
custom_fitness = make_fitness(function=_custom_fitness, greater_is_better=False)
































def translate_program_to_plain_text(program, feature_names):
    """Translate the symbolic program into plain text using column names."""
    program_str = str(program)
    
    # Loop over each feature and replace the symbolic names (X1, X2, ...) with the actual column names
    for i, feature_name in enumerate(feature_names):
        program_str = program_str.replace(f'X{i}', feature_name)
    
    return program_str

def store_best_program_with_plain_text(program, feature_names, test_metrics):
    """Store the best program with translated plain text formula."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Translate program into plain text formula
    translated_program = translate_program_to_plain_text(program, feature_names)
    
    # Save the best program along with its metrics
    with open(os.path.join(CONFIG['output_directory'], 'best_programs_plain_text.txt'), 'a') as f:
        f.write(f"\n\n--- New Best Program ({timestamp}) ---\n")
        f.write(f"Program: {translated_program}\n")
        if test_metrics:
            f.write("Test Set Metrics:\n")
            for metric, value in test_metrics.items():
                f.write(f"  {metric}: {value:.6f}\n")
        else:
            f.write("Unable to calculate test metrics (no valid data)\n")







def calculate_metrics(y_true, y_pred, num_bins=10):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    
    if len(y_true) == 0:
        return None
    
    mse = safe_operation(lambda: mean_squared_error(y_true, y_pred), np.inf)
    mae = safe_operation(lambda: mean_absolute_error(y_true, y_pred), np.inf)
    correlation = safe_operation(lambda: np.corrcoef(y_true, y_pred)[0, 1], 0)
    ic = safe_operation(lambda: spearmanr(y_true, y_pred)[0], 0)
    
    y_true_discrete = pd.qcut(y_true, q=num_bins, labels=False, duplicates='drop')
    y_pred_discrete = pd.qcut(y_pred, q=num_bins, labels=False, duplicates='drop')
    
    mi = safe_operation(lambda: mutual_info_score(y_true_discrete, y_pred_discrete), 0)
    
    y_true_norm = y_true - np.min(y_true) + 1e-10
    y_pred_norm = y_pred - np.min(y_pred) + 1e-10
    y_true_norm /= np.sum(y_true_norm)
    y_pred_norm /= np.sum(y_pred_norm)
    kl_divergence = safe_operation(lambda: np.sum(kl_div(y_true_norm, y_pred_norm)), np.inf)
    
    return {
        'MSE': mse,
        'MAE': mae,
        'Correlation': correlation,
        'Information Coefficient': ic,
        'Mutual Information': mi,
        'KL Divergence': kl_divergence
    }

def calculate_baseline_mse(y):
    """Calculate baseline MSEs for different types of random guesses."""
    naive_predictions = np.roll(y, 1)
    naive_mse = np.mean((y[1:] - naive_predictions[1:]) ** 2)

    mean_y = np.mean(y)
    std_y = np.std(y)
    random_predictions = np.random.normal(mean_y, std_y, size=len(y))
    random_mse = np.mean((y - random_predictions) ** 2)

    return naive_mse, random_mse







def calculate_baseline_ic_mi(y_true):
    """Calculate IC and MI baselines using naive and random methods."""
    # Naive baseline: Predict the mean of y_true for every point
    naive_pred = np.mean(y_true) * np.ones_like(y_true)
    
    # Random baseline: Generate random predictions based on y_true's distribution
    random_pred = np.random.normal(np.mean(y_true), np.std(y_true), size=len(y_true))
    
    # Calculate Information Coefficient (IC) for naive and random baselines
    # Handle constant array case for naive IC
    naive_ic = spearmanr(y_true, naive_pred)[0] if np.std(naive_pred) > 0 else np.nan
    random_ic = spearmanr(y_true, random_pred)[0]
    
    # For Mutual Information, handle constant array for naive prediction
    try:
        naive_mi = mutual_info_score(np.digitize(y_true, bins=10), np.digitize(naive_pred, bins=10))
    except ValueError:
        naive_mi = np.nan  # Set to NaN when digitizing fails due to constant values
    
    random_mi = mutual_info_score(np.digitize(y_true, bins=10), np.digitize(random_pred, bins=10))
    
    return naive_pred, random_pred, {
        'naive_ic': naive_ic,
        'naive_mi': naive_mi,
        'random_ic': random_ic,
        'random_mi': random_mi
    }

def display_edge_over_random_ic_mi(y_train, y_test, y_pred_train, y_pred_test, train_metrics, test_metrics):
    # Calculate baseline IC and MI for training and testing sets, and get predictions
    naive_pred_train, random_pred_train, train_baselines = calculate_baseline_ic_mi(y_train)
    naive_pred_test, random_pred_test, test_baselines = calculate_baseline_ic_mi(y_test)
    
    # Calculate the edge in IC and MI over naive and random baselines
    edge_naive_ic_train = train_metrics['Information Coefficient'] - train_baselines['naive_ic'] if not np.isnan(train_baselines['naive_ic']) else np.nan
    edge_naive_mi_train = train_metrics['Mutual Information'] - train_baselines['naive_mi'] if not np.isnan(train_baselines['naive_mi']) else np.nan
    edge_naive_ic_test = test_metrics['Information Coefficient'] - test_baselines['naive_ic'] if not np.isnan(test_baselines['naive_ic']) else np.nan
    edge_naive_mi_test = test_metrics['Mutual Information'] - test_baselines['naive_mi'] if not np.isnan(test_baselines['naive_mi']) else np.nan
    
    # Percentage improvement/unimprovement over naive method
    if not np.isnan(train_baselines['naive_ic']):
        percent_ic_train = (train_metrics['Information Coefficient'] - train_baselines['naive_ic']) / abs(train_baselines['naive_ic']) * 100
    else:
        percent_ic_train = np.nan

    if not np.isnan(train_baselines['naive_mi']):
        percent_mi_train = (train_metrics['Mutual Information'] - train_baselines['naive_mi']) / abs(train_baselines['naive_mi']) * 100
    else:
        percent_mi_train = np.nan
    
    if not np.isnan(test_baselines['naive_ic']):
        percent_ic_test = (test_metrics['Information Coefficient'] - test_baselines['naive_ic']) / abs(test_baselines['naive_ic']) * 100
    else:
        percent_ic_test = np.nan

    if not np.isnan(test_baselines['naive_mi']):
        percent_mi_test = (test_metrics['Mutual Information'] - test_baselines['naive_mi']) / abs(test_baselines['naive_mi']) * 100
    else:
        percent_mi_test = np.nan

    # Prepare data for the table (Train set)
    train_table = pd.DataFrame({
        'True Values (Train)': y_train,
        'Model Predictions (Train)': y_pred_train,
        'Naive Predictions (Train)': naive_pred_train
    })
    
    # Prepare data for the table (Test set)
    test_table = pd.DataFrame({
        'True Values (Test)': y_test,
        'Model Predictions (Test)': y_pred_test,
        'Naive Predictions (Test)': naive_pred_test
    })
    
    # Display the tables
    print("\nTrain Set Predictions Comparison:\n")
    print(train_table.head())  # Displaying the first few rows of the table
    print("\nTest Set Predictions Comparison:\n")
    print(test_table.head())  # Displaying the first few rows of the table
    
    # Display percentage improvement/unimprovement
    print("\nPercentage Improvement Over Naive Baseline (Train):")
    if not np.isnan(percent_ic_train):
        print(f"Information Coefficient (IC): {percent_ic_train:.2f}%")
    else:
        print("Information Coefficient (IC): NaN (Naive prediction was constant)")

    if not np.isnan(percent_mi_train):
        print(f"Mutual Information (MI): {percent_mi_train:.2f}%")
    else:
        print("Mutual Information (MI): NaN (Naive prediction was constant)")

    print("\nPercentage Improvement Over Naive Baseline (Test):")
    if not np.isnan(percent_ic_test):
        print(f"Information Coefficient (IC): {percent_ic_test:.2f}%")
    else:
        print("Information Coefficient (IC): NaN (Naive prediction was constant)")

    if not np.isnan(percent_mi_test):
        print(f"Mutual Information (MI): {percent_mi_test:.2f}%")
    else:
        print("Mutual Information (MI): NaN (Naive prediction was constant)")

    # Optionally, display full metrics as before
    print("\nTraining Set Edge Over Naive Baseline:")
    print(f"Edge Over Naive IC (Train): {edge_naive_ic_train:.6f}")
    print(f"Edge Over Naive MI (Train): {edge_naive_mi_train:.6f}")
    
    print("\nTest Set Edge Over Naive Baseline:")
    print(f"Edge Over Naive IC (Test): {edge_naive_ic_test:.6f}")
    print(f"Edge Over Naive MI (Test): {edge_naive_mi_test:.6f}")










def main():
    timer = time.time()

    if CONFIG['use_cached_dataset'] and os.path.exists(CONFIG['dataset_cache_path']):
        print(f"Loading cached dataset from {CONFIG['dataset_cache_path']}...")
        X, y = joblib.load(CONFIG['dataset_cache_path'])
    else:
        all_file_paths = [os.path.join(CONFIG['input_directory'], f) for f in os.listdir(CONFIG['input_directory']) if f.endswith('.parquet')]
        num_files_to_use = max(1, int(len(all_file_paths) * CONFIG['data_sample_percentage'] / 100))
        selected_file_paths = random.sample(all_file_paths, num_files_to_use)
        print(f"Using {num_files_to_use} out of {len(all_file_paths)} files.")
        
        X, y = load_and_prepare_data(selected_file_paths)
        
        joblib.dump((X, y), CONFIG['dataset_cache_path'])
        print(f"Cached dataset saved to {CONFIG['dataset_cache_path']}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    est_gp = SymbolicRegressor(
        population_size=CONFIG['population_size'],
        generations=CONFIG['generations'],
        tournament_size=CONFIG['tournament_size'],
        stopping_criteria=CONFIG['stopping_criteria'],
        const_range=CONFIG['const_range'],
        init_depth=CONFIG['init_depth'],
        init_method=CONFIG['init_method'],
        function_set=CONFIG['function_set'],
        metric=custom_fitness,
        parsimony_coefficient=CONFIG['parsimony_coefficient'],
        p_crossover=CONFIG['p_crossover'],
        p_subtree_mutation=CONFIG['p_subtree_mutation'],
        p_hoist_mutation=CONFIG['p_hoist_mutation'],
        p_point_mutation=CONFIG['p_point_mutation'],
        p_point_replace=CONFIG['p_point_replace'],
        max_samples=CONFIG['max_samples'],
        verbose=CONFIG['verbose'],
        n_jobs=CONFIG['n_jobs'],
        low_memory=True,
        random_state=CONFIG['random_state']
    )

    ##print off all the column names that are going into the model
    print(X.columns)
    


    est_gp.fit(X_train, y_train)
    
    y_pred_train = est_gp.predict(X_train)
    y_pred_test = est_gp.predict(X_test)
    
    print("Best program:", est_gp._program)

    train_metrics = calculate_metrics(y_train, y_pred_train)
    test_metrics = calculate_metrics(y_test, y_pred_test)
    
    if train_metrics:
        print("Genetic Programming (Train) Performance:")
        for metric, value in train_metrics.items():
            print(f"{metric}: {value:.6f}")
        print()
    
    if test_metrics:
        print("Genetic Programming (Test) Performance:")
        for metric, value in test_metrics.items():
            print(f"{metric}: {value:.6f}")
        print()

    os.makedirs(CONFIG['output_directory'], exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    


    
    store_best_program_with_plain_text(est_gp._program, X.columns, test_metrics)





    print("Final Time: ", round(time.time() - timer, 2))

if __name__ == "__main__":
    main()
