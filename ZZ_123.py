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
import joblib 
import antropy as ant 
from tqdm import tqdm




import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import entropy, spearmanr
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt





# Updated CONFIG with Dataset Caching Options
CONFIG = {
    'input_directory': 'Data/PriceData',
    'data_sample_percentage': 1,  # Increased from 3 to use more data
    'output_directory': 'Results',
    'dataset_cache_path': 'cached_dataset.pkl',
    'use_cached_dataset': True,
    'population_size': 50000,  # Reduced from 10000 to allow more generations
    'generations': 15,  # Increased from 10 to allow for more evolution
    'tournament_size': 9,  # Increased from 3 to apply more selection pressure
    'stopping_criteria': -1e9,  # Set to a very low value to prevent early stopping
    'const_range': (-1, 1),  # Expanded range to allow for more diverse constants
    'init_depth': (2, 5),  # Increased max depth to allow for more complex initial programs
    'init_method': 'half and half',
    'function_set': ('add', 'sub', 'mul', 'div', 'abs'),  # Added 'abs'
    'parsimony_coefficient': 0.03,  # Reduced to allow for more complex programs
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








def calculate_dynamic_weighted_target(df, horizons=[1, 5, 10, 20, 60]):
    # Calculate future returns for each horizon
    for h in horizons:
        df[f'Future_Returns_{h}'] = df['Close'].shift(-h) / df['Close'] - 1

    # Compute correlations with past returns to determine weights
    weights = []
    for h in horizons:
        corr = df['Close'].pct_change().corr(df[f'Future_Returns_{h}'])
        weights.append(abs(corr) if not np.isnan(corr) else 0)

    # Normalize weights, handle total_weight == 0
    total_weight = sum(weights)
    if total_weight == 0:
        # Assign equal weights if total_weight is zero
        weights = [1.0 / len(weights)] * len(weights)
    else:
        weights = [w / total_weight for w in weights]

    # Calculate weighted target
    df['Target'] = sum(
        w * df[f'Future_Returns_{h}'] for w, h in zip(weights, horizons)
    )

    # Drop intermediate columns
    df = df.drop(columns=[f'Future_Returns_{h}' for h in horizons])
    return df









##===================================[Noise detection]===================================##
##===================================[Noise detection]===================================##
##===================================[Noise detection]===================================##
##===================================[Noise detection]===================================##
##===================================[Noise detection]===================================##
##===================================[Noise detection]===================================##
##===================================[Noise detection]===================================##
##===================================[Noise detection]===================================##
##===================================[Noise detection]===================================##




def calculate_spectral_slope(y, fs=1.0):
    """
    Calculate the slope of the power spectral density (PSD) on a log-log scale.
    """
    freqs, psd = welch(y, fs=fs, nperseg=min(256, len(y)))
    # Avoid log(0) by adding a small constant and exclude zero frequencies
    freqs = freqs[1:]
    psd = psd[1:]
    psd += 1e-10  # Prevent log(0)
    log_freqs = np.log(freqs)
    log_psd = np.log(psd)
    # Check if there are enough points to perform polyfit
    if len(log_freqs) < 2 or len(log_psd) < 2:
        return np.nan
    slope, _ = np.polyfit(log_freqs, log_psd, 1)
    return slope

def calculate_hurst_exponent(ts):
    """
    Calculate the Hurst Exponent of a time series.
    """
    lags = range(2, min(100, len(ts)//2))  # Ensure lags do not exceed half the series length
    tau = []
    for lag in lags:
        if lag >= len(ts):
            continue
        diff = ts[lag:] - ts[:-lag]
        std_diff = np.std(diff)
        if std_diff > 0:
            tau.append(np.sqrt(std_diff))
        else:
            tau.append(np.nan)
    tau = np.array(tau)
    valid = ~np.isnan(tau) & (tau > 0)
    if np.sum(valid) < 2:
        return np.nan
    poly = np.polyfit(np.log(np.array(lags)[valid]), np.log(tau[valid]), 1)
    return poly[0] * 2.0

def calculate_spectral_entropy(y, fs=1.0, nperseg=256):
    """
    Calculate the spectral entropy of a time series.
    """
    freqs, psd = welch(y, fs=fs, nperseg=min(nperseg, len(y)))
    psd_norm = psd / np.sum(psd)
    psd_norm += 1e-10  # Prevent log(0)
    return entropy(psd_norm)

def calculate_autocorrelation(y, lag=1):
    """
    Calculate the autocorrelation of the time series at a given lag.
    """
    if len(y) <= lag:
        return np.nan
    return spearmanr(y[:-lag], y[lag:]).correlation if len(y[:-lag]) > 0 else np.nan




def simple_randomness_filter(df, window_size=20, data_removal_target=0.10):
    """
    Enhanced filter to identify and handle different types of noise in the data.
    
    Parameters:
    - df: pandas DataFrame with at least a 'Close' column.
    - window_size: Rolling window size for calculations.
    - data_removal_target: Fraction of data to remove based on noise metrics.
    
    Returns:
    - df_filtered: Filtered DataFrame.
    - noise_metrics_df: DataFrame containing noise metrics for each window.
    """
    df = df.copy()
    
    # Calculate returns
    df['Return'] = df['Close'].pct_change()
    
    # Handle any NaN or infinite values in returns
    df['Return'].replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=['Return'], inplace=True)
    
    # Calculate rolling metrics
    metrics = {
        'Spectral_Slope': [],
        'Spectral_Entropy': [],
        'Hurst_Exponent': [],
        'Autocorrelation': []
    }
    
    # Initialize lists with NaN for the initial window
    for metric in metrics:
        metrics[metric] = [np.nan] * (window_size - 1)
    
    # Iterate over the rolling window
    for i in range(window_size, len(df) + 1):
        window = df['Return'].iloc[i - window_size:i].values
        if np.all(np.isfinite(window)):
            # Normalize window
            scaler = MinMaxScaler()
            window_norm = scaler.fit_transform(window.reshape(-1, 1)).flatten()
            
            # Calculate metrics
            slope = calculate_spectral_slope(window_norm)
            se = calculate_spectral_entropy(window_norm)
            hurst = calculate_hurst_exponent(window_norm)
            autocorr = calculate_autocorrelation(window_norm)
            
            metrics['Spectral_Slope'].append(slope)
            metrics['Spectral_Entropy'].append(se)
            metrics['Hurst_Exponent'].append(hurst)
            metrics['Autocorrelation'].append(autocorr)
        else:
            metrics['Spectral_Slope'].append(np.nan)
            metrics['Spectral_Entropy'].append(np.nan)
            metrics['Hurst_Exponent'].append(np.nan)
            metrics['Autocorrelation'].append(np.nan)
    
    # Append metrics to DataFrame
    for metric, values in metrics.items():
        df[metric] = values
    
    # Drop initial rows with NaN metrics
    df_filtered = df.dropna(subset=['Spectral_Slope', 'Spectral_Entropy', 'Hurst_Exponent', 'Autocorrelation']).copy()
    
    # Define thresholds based on desired noise characteristics
    # These thresholds can be tuned based on domain knowledge or exploratory analysis
    # Example thresholds:
    # - Spectral Slope: closer to -1 (pink noise) or -2 (brown noise) indicates more structured noise
    # - Spectral Entropy: lower values indicate more predictable patterns
    # - Hurst Exponent: far from 0.5 indicates trending or mean-reverting behavior
    # - Autocorrelation: higher absolute values indicate stronger dependencies
    
    # For demonstration, let's set some arbitrary thresholds
    # These should be adjusted based on your specific data and requirements
    slope_threshold_low = -2.0  # More structured (brown) noise
    slope_threshold_high = -0.5  # Less structured noise
    entropy_threshold = 0.5      # Lower entropy indicates more predictability
    hurst_threshold_low = 0.4     # Mean-reverting
    hurst_threshold_high = 0.6    # Trending
    autocorr_threshold = 0.2      # Minimum autocorrelation
    
    # Create a composite noise score
    # You can weight these metrics based on their importance
    df_filtered['Noise_Score'] = (
        ((df_filtered['Spectral_Slope'] >= slope_threshold_low) & 
         (df_filtered['Spectral_Slope'] <= slope_threshold_high)) * 1.0 +
        (df_filtered['Spectral_Entropy'] <= entropy_threshold) * 1.0 +
        ((df_filtered['Hurst_Exponent'] <= hurst_threshold_low) | 
         (df_filtered['Hurst_Exponent'] >= hurst_threshold_high)) * 1.0 +
        (np.abs(df_filtered['Autocorrelation']) >= autocorr_threshold) * 1.0
    )
    
    # Define rows to remove based on the Noise_Score
    # For example, remove rows where Noise_Score >= 3 (indicating high structured noise)
    noise_filter = df_filtered['Noise_Score'] >= 3
    num_to_remove = int(len(df_filtered) * data_removal_target)
    rows_to_remove = df_filtered[noise_filter].nlargest(num_to_remove, 'Noise_Score').index
    
    # Filter out the noisy rows
    df_final_filtered = df_filtered.drop(index=rows_to_remove).copy()
    
    # Drop intermediate columns
    columns_to_drop = ['Return', 'Spectral_Slope', 'Spectral_Entropy', 'Hurst_Exponent', 'Autocorrelation', 'Noise_Score']
    df_final_filtered.drop(columns=columns_to_drop, inplace=True)
    
    # Optionally, return the noise metrics for further analysis
    noise_metrics_df = df_filtered[['Spectral_Slope', 'Spectral_Entropy', 'Hurst_Exponent', 'Autocorrelation', 'Noise_Score']].copy()
    
    return df_final_filtered, noise_metrics_df

















##====================[Working noise filter]====================##
##====================[Working noise filter]====================##
##====================[Working noise filter]====================##
##====================[Working noise filter]====================##
##====================[Working noise filter]====================##
##====================[Working noise filter]====================##
##====================[Working noise filter]====================##
##====================[Working noise filter]====================##










# Noise metric calculation functions
def calculate_spectral_slope(y, fs=1.0):
    freqs, psd = welch(y, fs=fs, nperseg=min(256, len(y)))
    freqs = freqs[1:]
    psd = psd[1:] + 1e-10  # Prevent log(0)
    log_freqs = np.log(freqs)
    log_psd = np.log(psd)
    if len(log_freqs) < 2 or len(log_psd) < 2:
        return np.nan
    slope, _ = np.polyfit(log_freqs, log_psd, 1)
    return slope

def calculate_spectral_entropy(y, fs=1.0, nperseg=256):
    freqs, psd = welch(y, fs=fs, nperseg=min(nperseg, len(y)))
    psd_sum = np.sum(psd)
    if psd_sum == 0:
        return np.nan  # Cannot compute entropy if PSD sum is zero
    psd_norm = psd / psd_sum + 1e-10
    return entropy(psd_norm)


def calculate_hurst_exponent(ts):
    lags = range(2, min(100, len(ts)//2))
    tau = [np.sqrt(np.std(ts[lag:] - ts[:-lag])) for lag in lags if np.std(ts[lag:] - ts[:-lag]) > 0]
    tau = np.array(tau)
    if len(tau) < 2:
        return np.nan
    return np.polyfit(np.log(lags[:len(tau)]), np.log(tau), 1)[0] * 2.0

def calculate_autocorrelation(y, lag=1):
    if len(y) <= lag or np.std(y) == 0 or np.std(y[lag:]) == 0:
        return np.nan
    return spearmanr(y[:-lag], y[lag:]).correlation


# Enhanced randomness filter function
def enhanced_randomness_filter(df, window_size=20):
    df = df.copy()
    df['Return'] = df['Close'].pct_change().replace([np.inf, -np.inf], np.nan).dropna()

    metrics = {'Spectral_Slope': [], 'Spectral_Entropy': [], 'Hurst_Exponent': [], 'Autocorrelation': []}
    
    # Pad the metrics with NaN at the start
    for key in metrics:
        metrics[key] = [np.nan] * (window_size - 1)

    # Iterate over the rolling window to calculate noise metrics
    for i in range(window_size, len(df) + 1):
        window = df['Return'].iloc[i - window_size:i].values
        if np.all(np.isfinite(window)):
            window_norm = MinMaxScaler().fit_transform(window.reshape(-1, 1)).flatten()
            metrics['Spectral_Slope'].append(calculate_spectral_slope(window_norm))
            metrics['Spectral_Entropy'].append(calculate_spectral_entropy(window_norm))
            metrics['Hurst_Exponent'].append(calculate_hurst_exponent(window_norm))
            metrics['Autocorrelation'].append(calculate_autocorrelation(window_norm))
        else:
            for key in metrics:
                metrics[key].append(np.nan)

    # Add the metrics back to the DataFrame
    for key, values in metrics.items():
        df[key] = values

    # Filter the DataFrame based on the calculated metrics
    df_filtered = df.dropna(subset=metrics.keys()).copy()

    return df_filtered





def load_and_prepare_data(file_paths, window_size=14):
    original_total_rows = 0
    filtered_total_rows = 0
    all_data = []

    # Initialize tqdm progress bar
    for file_path in tqdm(file_paths, desc="Processing Files", unit="file"):
        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue  # Skip this file and continue with the next

        # Reset index if 'Date' is the index and not a column
        if 'Date' not in df.columns and df.index.name == 'Date':
            df = df.reset_index()

        # Ensure 'Date' column is in datetime format
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])  # Drop rows where 'Date' couldn't be parsed

        # Skip files with insufficient data
        if len(df) < 100:
            continue

        # Select relevant columns and apply dynamic rounding
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        df = df.assign(
            Open=dynamic_round_vec(df['Open']),
            High=dynamic_round_vec(df['High']),
            Low=dynamic_round_vec(df['Low']),
            Close=dynamic_round_vec(df['Close'])
        )

        original_total_rows += len(df)

        # Apply enhanced randomness filter to calculate noise metrics
        df_filtered = enhanced_randomness_filter(df, window_size=20)

        # Proceed if df_filtered is not empty
        if df_filtered.empty:
            continue

        # --- Add Custom Factor Features ---

        # 1. Momentum Features (over custom horizons)
        for horizon in [1, 5, 10]:
            df_filtered[f'Momentum_{horizon}'] = df_filtered['Close'] - df_filtered['Close'].shift(horizon)

        # 2. Volatility Features
        df_filtered['True_Range'] = df_filtered['High'] - df_filtered['Low']
        df_filtered['High_Low_Spread'] = df_filtered['High'] - df_filtered['Low']
        
        # 3. Directional Movement Features
        df_filtered['Directional_Bias'] = df_filtered['Close'] - (df_filtered['High'] + df_filtered['Low']) / 2
        
        # 4. Volume-Based Metrics
        df_filtered['Volume_Acceleration'] = df_filtered['Volume'].pct_change().shift(1)
        df_filtered['Volume_Weighted_Price'] = (df_filtered['Close'] * df_filtered['Volume']) / df_filtered['Volume'].sum()

        # Create lagged features
        for col in ['High', 'Low', 'Volume']:
            lag_times = [1, 2, 3] if col == 'Volume' else [2, 5]
            for lag in lag_times:
                df_filtered[f'{col}_Lag{lag}'] = df_filtered[col].shift(lag)

        # Create additional features
        df_filtered['High_Low'] = df_filtered['High'] - df_filtered['Low']
        df_filtered['High_Close'] = df_filtered['High'] - df_filtered['Close']
        df_filtered['Low_Close'] = df_filtered['Low'] - df_filtered['Close']

        # Drop rows with NaN values resulting from lagging
        df_filtered = df_filtered.dropna()

        # Calculate dynamic weighted target
        df_filtered = calculate_dynamic_weighted_target(df_filtered)

        # Replace infinite values and drop any remaining NaNs
        df_filtered = df_filtered.replace([np.inf, -np.inf], np.nan).dropna()

        filtered_total_rows += len(df_filtered)

        # Append processed DataFrame to the list
        all_data.append(df_filtered)

    if not all_data:
        raise ValueError("No data available after processing. Check your filters and input data.")

    # Concatenate all processed DataFrames
    combined_data = pd.concat(all_data, ignore_index=True).drop_duplicates()
    combined_data = combined_data.dropna()

    # Collect the noise metrics
    noise_metrics = combined_data[['Spectral_Slope', 'Spectral_Entropy', 'Hurst_Exponent', 'Autocorrelation']]

    # Calculate the 10th and 90th percentiles for Spectral_Entropy
    lower_quantile = noise_metrics['Spectral_Entropy'].quantile(0.10)
    upper_quantile = noise_metrics['Spectral_Entropy'].quantile(0.90)

    # Filter out the top 10% and bottom 10% based on Spectral_Entropy
    filtered_data = combined_data[
        (combined_data['Spectral_Entropy'] >= lower_quantile) &
        (combined_data['Spectral_Entropy'] <= upper_quantile)
    ]

    # Now proceed with creating features and target variable
    feature_columns = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'High_Lag2', 'High_Lag5', 'Low_Lag2', 'Low_Lag5',
        'Volume_Lag1', 'Volume_Lag2', 'Volume_Lag3',
        'High_Low', 'High_Close', 'Low_Close',
        # Include custom features
        'Momentum_1', 'Momentum_5', 'Momentum_10',
        'True_Range', 'High_Low_Spread', 'Directional_Bias',
        'Volume_Acceleration', 'Volume_Weighted_Price'
    ]

    # Verify that all feature columns exist in the filtered data
    missing_features = set(feature_columns) - set(filtered_data.columns)
    if missing_features:
        raise KeyError(f"The following feature columns are missing from the data: {missing_features}")

    X = filtered_data[feature_columns]
    y = filtered_data['Target']

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










def _custom_fitness3333(y, y_pred, w):
    """
    Custom fitness function focusing on IC and MI, optimized for speed.
    """
    # Check for empty or identical predictions
    if len(y) == 0 or len(y_pred) == 0:
        return 1e10

    if np.all(y_pred == y_pred[0]):
        return 1e9

    try:
        # Spearman's rank correlation coefficient (IC)
        ic = abs(spearmanr(y, y_pred)[0])

        # Discretize y and y_pred using fewer bins to speed up MI calculation
        y_discrete = y_true_discrete(y, bins=5)
        y_pred_discrete = y_true_discrete(y_pred, bins=5)

        # Mutual Information (MI)
        mi = mutual_info_score(y_discrete, y_pred_discrete)

        # Normalize MI
        y_entropy = mutual_info_score(y_discrete, y_discrete)
        normalized_mi = mi / y_entropy if y_entropy != 0 else 0

        # Combine IC and MI with adjusted weights
        ic_weight = 0.6
        mi_weight = 0.4
        combined_score = (ic_weight * ic) + (mi_weight * normalized_mi)

        # Return inverse of combined score (lower is better)
        return 1 / combined_score if combined_score != 0 else 1e10

    except Exception as e:
        return 1e32

# Register the optimized custom fitness function
#custom_fitness = make_fitness(function=_custom_fitness, greater_is_better=False)




def _custom_fitness(y, y_pred, w):
    """
    Custom fitness function focusing on Adjusted IC and MI, optimized for speed.
    """
    # Check for empty or identical predictions
    if len(y) == 0 or len(y_pred) == 0:
        return 1e10

    if np.all(y_pred == y_pred[0]):
        return 1e9

    try:
        # Spearman's rank correlation coefficient (IC)
        ic = abs(spearmanr(y, y_pred)[0])

        # Discretize y and y_pred using fewer bins to speed up MI calculation
        y_discrete = y_true_discrete(y, bins=5)
        y_pred_discrete = y_true_discrete(y_pred, bins=5)

        # Mutual Information (MI)
        mi = mutual_info_score(y_discrete, y_pred_discrete)

        # Compute the entropy of y_discrete as a measure of problem difficulty
        counts = np.bincount(y_discrete)
        y_entropy = entropy(counts, base=2)

        # Avoid division by zero
        epsilon = 1e-10

        # Adjust IC based on problem difficulty (entropy)
        adjusted_ic = ic / (y_entropy + epsilon)

        # Normalize MI
        normalized_mi = mi / (y_entropy + epsilon)

        # Combine adjusted IC and normalized MI with weights
        ic_weight = 0.6
        mi_weight = 0.4
        combined_score = (ic_weight * adjusted_ic) + (mi_weight * normalized_mi)

        # Return inverse of combined score (lower is better)
        return 1 / combined_score if combined_score != 0 else 1e10

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
