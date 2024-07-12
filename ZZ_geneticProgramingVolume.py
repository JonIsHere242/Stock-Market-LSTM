import os
import pandas as pd
import numpy as np
from gplearn.genetic import SymbolicRegressor
from gplearn.fitness import make_fitness
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import datetime
from scipy.stats import spearmanr

FILECONFIG = {
    'input_directory': 'Data/PriceData',
    'output_directory': 'Results',
}

CONFIG = {
    'population_size': 15000,
    'generations': 3,
    'tournament_size': 10,
    'stopping_criteria': 0.0001,
    'const_range': None,
    'init_depth': (1, 6),
    'init_method': 'half and half',
    'function_set': ('add', 'sub', 'mul', 'div', 'log', 'abs', 'neg', 'sqrt', 'sin', 'cos'),
    'parsimony_coefficient': 0.01,
    'p_crossover': 0.7,
    'p_subtree_mutation': 0.1,
    'p_hoist_mutation': 0.05,
    'p_point_mutation': 0.1,
    'p_point_replace': 0.05,
    'max_samples': 0.9,
    'verbose': 1,
    'n_jobs': -1,
    'random_state': 3301
}


def calculate_obv(df):
    obv = (np.sign(df['Close'].diff()) * df['Volume']).cumsum()
    return obv


def calculate_cmf(df, period=20):
    mfv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']) * df['Volume']
    cmf = mfv.rolling(window=period).sum() / df['Volume'].rolling(window=period).sum()
    return cmf

def calculate_vwap(df):
    return (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()


def calculate_adl(df):
    mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    mfv = mfm * df['Volume']
    return mfv.cumsum()



def calculate_mfi(df, period=14):
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    raw_money_flow = typical_price * df['Volume']
    positive_flow = (raw_money_flow.where(typical_price > typical_price.shift(1), 0)).rolling(window=period).sum()
    negative_flow = (raw_money_flow.where(typical_price < typical_price.shift(1), 0)).rolling(window=period).sum()
    money_ratio = positive_flow / negative_flow
    return 100 - (100 / (1 + money_ratio))




def load_and_prepare_data(file_paths):
    all_data = []
    for file_path in file_paths:
        df = pd.read_parquet(file_path)
        if 'Date' not in df.columns and df.index.name == 'Date':
            df = df.reset_index()
        df['Date'] = pd.to_datetime(df['Date'])
        
        if len(df) < 100:
            continue
        
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        df['Future_Volume'] = (
            0.90 * df['Volume'].shift(-1) +
            0.09 * df['Volume'].shift(-2) +
            0.01 * df['Volume'].shift(-3)
        )
        
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[f'{col}_Lag1'] = df[col].shift(5)
            df[f'{col}_Lag2'] = df[col].shift(14)
        
        df['PriceToVolumeClose'] = df['Close'] / df['Volume']
        df['PriceToVolumeHigh'] = df['High'] / df['Volume']
        df['PriceToVolumeLow'] = df['Low'] / df['Volume']
        
        df = df.dropna().replace([np.inf, -np.inf], np.nan).dropna().round(7)


   


        for col in df.columns:
            if df[col].dtype == 'float64':
                df[col] = df[col].astype('float32')
        
        all_data.append(df)
    
    combined_data = pd.concat(all_data, ignore_index=True)
    
    feature_columns = [col for col in combined_data.columns if col not in ['Date', 'Future_Volume']]
    


    X = combined_data[feature_columns]
    y = combined_data['Future_Volume']
    
    return X, y



def preprocess_data(y, y_pred):
    mask = np.isfinite(y) & np.isfinite(y_pred)
    return y[mask], y_pred[mask]





def _custom_fitness(y, y_pred, w):
    y, y_pred = preprocess_data(y, y_pred)

    if len(y) == 0 or len(y_pred) == 0:
        return 1e10

    pred_variance = np.var(y_pred)
    if pred_variance < 1e-4:
        return 1e10

    if np.mean(np.abs(y_pred)) < 1e-4:
        return 1e10

    # We'll use the weight array directly as the current volume
    current_volume = w

    relative_accuracy = safe_operation(lambda: np.mean(np.abs(y - y_pred) / np.where(y == 0, np.nan, y)), 1e8)
    mse = safe_operation(lambda: mean_squared_error(y, y_pred), 1e8)
    correlation = safe_operation(lambda: np.corrcoef(y, y_pred)[0, 1], 0)
    volume_trend_accuracy = safe_operation(lambda: np.mean((y > y.mean()) == (y_pred > y_pred.mean())), 1e8)
    large_pred_penalty = safe_operation(lambda: np.mean(np.maximum(0, np.abs(y_pred) - 10 * np.max(y))), 1e8)
    sign_balance = safe_operation(lambda: np.abs(np.mean(np.sign(y_pred - y))), 1e8)
    extreme_mask = (y < np.percentile(y, 10)) | (y > np.percentile(y, 90))
    extreme_performance = safe_operation(lambda: np.mean(np.abs((y[extreme_mask] - y_pred[extreme_mask]) / np.where(y[extreme_mask] == 0, np.nan, y[extreme_mask]))), 1e8)
    non_linearity = safe_operation(lambda: 1 - np.abs(np.corrcoef(y, y_pred**2)[0, 1] - correlation), 1e8)
    direction_acc = safe_operation(lambda: directional_accuracy(y, y_pred, current_volume), 0)

    combined_score = (
        0.20 * relative_accuracy +
        0.15 * (mse / np.var(y)) +
        0.15 * (1 - abs(correlation)) +
        0.15 * (1 - volume_trend_accuracy) +
        0.10 * large_pred_penalty +
        0.05 * sign_balance +
        0.05 * extreme_performance +
        0.05 * (1 - non_linearity) +
        0.10 * (1 - direction_acc)  # New component for directional accuracy
    )

    complexity_penalty = 0.1 / (len(str(y_pred)) + 1)
    
    final_score = combined_score + complexity_penalty

    return np.clip(final_score, 0.001, 1e10)

def directional_accuracy(y_true, y_pred, current_volume):
    true_direction = np.sign(y_true - current_volume)
    pred_direction = np.sign(y_pred - current_volume)
    return np.mean(true_direction == pred_direction)

custom_fitness = make_fitness(function=_custom_fitness, greater_is_better=False)




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


def safe_operation(operation, default_value):
    try:
        result = operation()
        return default_value if np.isnan(result) or np.isinf(result) else result
    except Exception:
        return default_value

def calculate_metrics(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    
    if len(y_true) == 0:
        return None
    
    mse = safe_operation(lambda: mean_squared_error(y_true, y_pred), np.inf)
    mae = safe_operation(lambda: mean_absolute_error(y_true, y_pred), np.inf)
    correlation = safe_operation(lambda: np.corrcoef(y_true, y_pred)[0, 1], 0)
    relative_accuracy = safe_operation(lambda: np.mean(np.abs(y_true - y_pred) / np.where(y_true == 0, np.nan, y_true)), np.inf)
    volume_trend_accuracy = safe_operation(lambda: np.mean((y_true > y_true.mean()) == (y_pred > y_pred.mean())), 0)
    
    # New metrics
    direction_accuracy = safe_operation(lambda: np.mean(np.sign(y_true[1:] - y_true[:-1]) == np.sign(y_pred[1:] - y_pred[:-1])), 0)
    rank_correlation = safe_operation(lambda: spearmanr(y_true, y_pred)[0], 0)
    mape = safe_operation(lambda: np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, np.nan, y_true))) * 100, np.inf)
    ic = safe_operation(lambda: np.corrcoef(y_pred, y_true)[0, 1], 0)
    
    return {
        'MSE': mse,
        'MAE': mae,
        'Correlation': correlation,
        'Relative Accuracy': relative_accuracy,
        'Volume Trend Accuracy': volume_trend_accuracy,
        'Direction Accuracy': direction_accuracy,
        'Rank Correlation': rank_correlation,
        'MAPE': mape,
        'Information Coefficient': ic
    }

def benchmark_average_volume(X, y):
    return np.full(len(y), np.mean(X['Volume']))

def benchmark_moving_average_volume(X, y, window=5):
    return X['Volume'].rolling(window=window).mean().fillna(method='bfill').values

def main():
    file_paths = [os.path.join(FILECONFIG['input_directory'], f) for f in os.listdir(FILECONFIG['input_directory']) if f.endswith('.parquet')]
    ##only use 500 files
    file_paths = file_paths[:100]


    X, y = load_and_prepare_data(file_paths)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    est_gp = SymbolicRegressor(**CONFIG, metric=custom_fitness)
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
    
    y_pred_avg = benchmark_average_volume(X_test, y_test)
    y_pred_ma = benchmark_moving_average_volume(X_test, y_test)
    
    avg_metrics = calculate_metrics(y_test, y_pred_avg)
    ma_metrics = calculate_metrics(y_test, y_pred_ma)
    
    print("Average Volume Benchmark Performance:")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.6f}")
    print()
    
    print("Moving Average Volume Benchmark Performance:")
    for metric, value in ma_metrics.items():
        print(f"{metric}: {value:.6f}")
    print()
    
    os.makedirs(FILECONFIG['output_directory'], exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    feature_names = X.columns.tolist()
    
    with open(os.path.join(FILECONFIG['output_directory'], 'best_programs.txt'), 'a') as f:
        f.write(f"\n\n--- New Best Program ({timestamp}) ---\n")
        f.write(f"Program: {translate_program(est_gp._program, feature_names)}\n")
        if test_metrics:
            f.write("Test Set Metrics:\n")
            for metric, value in test_metrics.items():
                f.write(f"{metric}: {value:.6f}\n")
        else:
            f.write("Unable to calculate test metrics (no valid data)\n")
        
        f.write("\nBenchmark Comparisons:\n")
        f.write("Average Volume Benchmark:\n")
        for metric, value in avg_metrics.items():
            f.write(f"{metric}: {value:.6f}\n")
        f.write("\nMoving Average Volume Benchmark:\n")
        for metric, value in ma_metrics.items():
            f.write(f"{metric}: {value:.6f}\n")

if __name__ == "__main__":
    main()
