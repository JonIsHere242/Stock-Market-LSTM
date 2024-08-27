import os
import pandas as pd
import numpy as np
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
from gplearn.fitness import make_fitness
from sklearn.metrics import mean_squared_error, mean_absolute_error, mutual_info_score
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from scipy.stats import entropy, spearmanr, zscore
from sklearn.preprocessing import MinMaxScaler
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
    'data_sample_percentage': 2.5,  # Increased from 3 to use more data
    'output_directory': 'Results',
    'dataset_cache_path': 'cached_dataset.pkl',
    'use_cached_dataset': True,
    'population_size': 10000,  # Reduced from 10000 to allow more generations
    'generations': 10,  # Increased from 10 to allow for more evolution
    'tournament_size': 5,  # Increased from 3 to apply more selection pressure
    'stopping_criteria': -1e9,  # Set to a very low value to prevent early stopping
    'const_range': (-1, 1),
    'init_depth': (2, 5),  # Increased max depth to allow for more complex initial programs
    'init_method': 'half and half',
    'function_set': ('add', 'sub', 'mul', 'div', 'log', 'sqrt', 'sin', 'cos', 'abs'),  # Added 'abs'
    'parsimony_coefficient': 0.05,  # Reduced to allow for more complex programs
    'p_crossover': 0.7,
    'p_subtree_mutation': 0.1,
    'p_hoist_mutation': 0.05,
    'p_point_mutation': 0.1,
    'p_point_replace': 0.05,
    'max_samples': 0.2,  # Increased from 0.05 to use more of the data in each generation
    'verbose': 1,
    'n_jobs': -1,
    'low_memory': True,
    'random_state': 3301,
    'max_depth': 10  # Increased to allow for more complex programs
}

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




def calculate_target_col(df):
    # Calculate simple future returns for various days ahead
    df['Future_Returns_1'] = df['Close'].shift(-1) / df['Close'] - 1
    df['Future_Returns_2'] = df['Close'].shift(-2) / df['Close'] - 1
    df['Future_Returns_3'] = df['Close'].shift(-3) / df['Close'] - 1

    # Calculate logarithmic returns with direction
    df['Log_Returns_1'] = np.sign(df['Future_Returns_1']) * np.log(df['Close'].shift(-1) / df['Close']).abs()
    df['Log_Returns_2'] = np.sign(df['Future_Returns_2']) * np.log(df['Close'].shift(-2) / df['Close']).abs()
    df['Log_Returns_3'] = np.sign(df['Future_Returns_3']) * np.log(df['Close'].shift(-3) / df['Close']).abs()

    # Weighted combination of log returns
    df['Weighted_Future_Returns'] = (
        0.75 * df['Log_Returns_1'] +
        0.20 * df['Log_Returns_2'] +
        0.05 * df['Log_Returns_3']
    )

    return df


def load_and_prepare_data(file_paths):
    original_total_rows = 0
    filtered_total_rows = 0
    all_data = []

    for file_path in file_paths:
        df = pd.read_parquet(file_path)
        
        # Ensure 'Date' is in the DataFrame
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
        df['Return'] = df['Close'].pct_change().fillna(0)
        df['Volume_Change'] = df['Volume'].pct_change().fillna(0)
        
        window_size = 14
        df['Return_Entropy'] = df['Return'].rolling(window=window_size).apply(
            lambda x: entropy(np.histogram(x, bins=5)[0]), raw=True).fillna(0)
        df['Volume_Entropy'] = df['Volume_Change'].rolling(window=window_size).apply(
            lambda x: entropy(np.histogram(x, bins=5)[0]), raw=True).fillna(0)
        
        scaler = MinMaxScaler()
        df[['Return_Entropy', 'Volume_Entropy']] = scaler.fit_transform(df[['Return_Entropy', 'Volume_Entropy']])
        
        entropy_threshold_low = 0.05
        entropy_threshold_high = 0.90
        df = df[
            (df['Return_Entropy'] > entropy_threshold_low) & 
            (df['Return_Entropy'] < entropy_threshold_high) & 
            (df['Volume_Entropy'] > entropy_threshold_low) & 
            (df['Volume_Entropy'] < entropy_threshold_high)
        ]
        
        filtered_total_rows += len(df)
        
        for col in ['High', 'Low']:
            df[f'{col}_Lag1'] = df[col].shift(2)
            df[f'{col}_Lag2'] = df[col].shift(5)
        
        for col in ['Volume']:
            df[f'{col}_Lag1'] = df[col].shift(1)
            df[f'{col}_Lag2'] = df[col].shift(2)
            df[f'{col}_Lag3'] = df[col].shift(3)

        df = df.dropna()

        df = calculate_target_col(df)
        
        df = df.dropna().replace([np.inf, -np.inf], np.nan).dropna()
        all_data.append(df)

    combined_data = pd.concat(all_data, ignore_index=True).drop_duplicates()
    feature_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    combined_data = combined_data.dropna()

    X = combined_data[[col for col in combined_data.columns if col in feature_columns or '_Lag' in col]]
    y = combined_data['Weighted_Future_Returns']
    
    percentage_removed = ((original_total_rows - filtered_total_rows) / original_total_rows) * 100 if original_total_rows > 0 else 0
    
    print(f"Original data shape: {original_total_rows} rows")
    print(f"Filtered data shape: {filtered_total_rows} rows")
    print(f"Percentage of data removed: {percentage_removed:.2f}%")
    print(f"Final data loaded with {len(X)} samples and {len(X.columns)} features.")
    
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


def y_true_discrete(y):
    return pd.qcut(y, q=10, labels=False, duplicates='drop')


def _custom_fitness(y, y_pred, w):
    y, y_pred = preprocess_data(y, y_pred)
    
    if len(y) == 0 or len(y_pred) == 0:
        return 1e10  # Return a large number if no valid predictions are made
    
    if np.all(y_pred == y_pred[0]):  # Check for constant predictions
        return 1e9  # Penalize constant predictions heavily
    
    try:
        # Calculate Spearman's rank correlation as Information Coefficient
        ic = spearmanr(y, y_pred)[0]
        # Calculate Mutual Information
        mi = mutual_info_score(y_true_discrete(y), y_true_discrete(y_pred))  # Discretization needed for MI
        
        # Gather all other metrics
        mse = fast_mse(y, y_pred)
        mae = fast_mae(y, y_pred)
        da = calculate_directional_accuracy(y, y_pred)
        pf = calculate_profit_factor(y, y_pred)
        sr = calculate_sharpe_ratio(y, y_pred)
        ir = calculate_information_ratio(y, y_pred)
        
        # Construct the combined score with modified weights
        combined_score = (
            0.25 * abs(ic) +  # Higher weight for Information Coefficient
            0.25 * mi +       # Higher weight for Mutual Information
            0.10 / (1 + mse) +  # Reduced emphasis on MSE
            0.10 / (1 + mae) +  # Reduced emphasis on MAE
            0.10 * da +        # Emphasis on directional accuracy
            0.10 * pf +        # Profit factor in the scoring
            0.05 * sr +        # Sharpe ratio for risk-adjusted returns
            0.05 * ir          # Information ratio to measure relative returns
        )
        
        return 1 / combined_score  # Invert the score to make lower better for optimization
    except Exception as e:
        print(f"Error in fitness calculation: {e}")
        return 1e32  # Return a large number in case of any error

# Register the custom fitness function
custom_fitness = make_fitness(function=_custom_fitness, greater_is_better=False)







def translate_program(program, feature_names):
    program_str = str(program)
    for i, name in enumerate(feature_names):
        if '_Lag' in name:
            base, lag = name.split('_Lag')
            program_str = program_str.replace(f'X{i}', f'{base}(t-{lag})')
        elif name.startswith('PriceToVolume'):
            metric = name.replace('PriceToVolume', '')
            program_str = program_str.replace(f'X{i}', f'{metric}/Volume')
        else:
            program_str = program_str.replace(f'X{i}', name)
    return program_str

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

def display_edge_over_random(y_train, y_test, train_metrics, test_metrics):
    naive_mse_train, random_mse_train = calculate_baseline_mse(y_train)
    naive_mse_test, random_mse_test = calculate_baseline_mse(y_test)
    
    edge_naive_train = naive_mse_train - train_metrics['MSE']
    edge_random_train = random_mse_train - train_metrics['MSE']
    edge_naive_test = naive_mse_test - test_metrics['MSE']
    edge_random_test = random_mse_test - test_metrics['MSE']
    
    print(f"Edge Over Naive Prediction (Train): {edge_naive_train:.6f}")
    print(f"Baseline Naive MSE (Train): {naive_mse_train:.6f}")
    print(f"Edge Over Random Prediction (Train): {edge_random_train:.6f}")
    print(f"Baseline Random MSE (Train): {random_mse_train:.6f}")
    
    print(f"Edge Over Naive Prediction (Test): {edge_naive_test:.6f}")
    print(f"Baseline Naive MSE (Test): {naive_mse_test:.6f}")
    print(f"Edge Over Random Prediction (Test): {edge_random_test:.6f}")
    print(f"Baseline Random MSE (Test): {random_mse_test:.6f}")

def baseline_model_predictions(y_true):
    mean_y = np.mean(y_true)
    std_y = np.std(y_true)
    return np.random.normal(mean_y, std_y, size=len(y_true))

def calculate_mi_ic_benchmarks(y_true, y_pred, num_bins=10):
    # Discretize the predictions and true values for mutual information calculation
    y_true_discrete = pd.qcut(y_true, q=num_bins, labels=False, duplicates='drop')
    y_pred_discrete = pd.qcut(y_pred, q=num_bins, labels=False, duplicates='drop')

    shuffled_y_pred = np.random.permutation(y_pred_discrete)
    random_mi = mutual_info_score(y_true_discrete, shuffled_y_pred)
    random_ic = spearmanr(y_true_discrete, shuffled_y_pred)[0]
    
    print(f"Random MI: {random_mi:.6f}, Random IC: {random_ic:.6f}")

    baseline_y_pred = baseline_model_predictions(y_true)
    baseline_y_pred_discrete = pd.qcut(baseline_y_pred, q=num_bins, labels=False, duplicates='drop')
    baseline_mi = mutual_info_score(y_true_discrete, baseline_y_pred_discrete)
    baseline_ic = spearmanr(y_true_discrete, baseline_y_pred_discrete)[0]
    
    print(f"Baseline Model MI: {baseline_mi:.6f}, Baseline Model IC: {baseline_ic:.6f}")




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
    




    feature_names = X.columns.tolist()
    
    with open(os.path.join(CONFIG['output_directory'], 'best_programs.txt'), 'a') as f:
        f.write(f"\n\n--- New Best Program ({timestamp}) ---\n")
        f.write(f"Program: {translate_program(est_gp._program, feature_names)}\n")
        if test_metrics:
            f.write("Test Set Metrics:\n")
            for metric, value in test_metrics.items():
                f.write(f"  {metric}: {value:.6f}\n")
        else:
            f.write("Unable to calculate test metrics (no valid data)\n")



    if train_metrics and test_metrics:
        display_edge_over_random(y_train, y_test, train_metrics, test_metrics)
        ##also dispaly the ic 
        calculate_mi_ic_benchmarks(y_train, y_pred_train)


    print("Final Time: ", round(time.time() - timer, 2))

if __name__ == "__main__":
    main()
