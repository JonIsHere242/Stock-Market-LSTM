# symbolic_regressor.py
import numpy as np
from scipy.stats import spearmanr
from gplearn.genetic import SymbolicRegressor
from gplearn.fitness import make_fitness
import pandas as pd
from datetime import datetime
from Z1_GenFuncs import DataPrep
import os
import argparse
from sklearn.metrics import make_scorer


##===============================[ Fitness Functions ]==================================##
##===============================[ Fitness Functions ]==================================##
##===============================[ Fitness Functions ]==================================##
##===============================[ Fitness Functions ]==================================##
##===============================[ Fitness Functions ]==================================##



def calculate_trading_metrics_from_predictions(y_true, y_pred, threshold=0):
    """
    Calculate trading metrics based on predictions and actual values
    
    Parameters:
    y_true: actual returns
    y_pred: predicted values (signals)
    threshold: threshold for considering a trade (default: 0)
    
    Returns:
    dict: trading metrics
    """
    # Generate trade signals based on predictions
    trade_signals = np.sign(y_pred - threshold)
    
    # Calculate trade outcomes (multiply signals by actual returns)
    trades = trade_signals * y_true
    
    # Filter out zero trades (no position)
    trades = trades[trade_signals != 0]
    
    if len(trades) == 0:
        return {
            'expected_value': 0.0,
            'profit_factor': 0.0
        }
    
    winning_trades = trades[trades > 0]
    losing_trades = trades[trades < 0]
    
    # Calculate win rate and average profits/losses
    win_rate = len(winning_trades) / len(trades)
    avg_win = np.mean(winning_trades) if len(winning_trades) > 0 else 0
    avg_loss = np.mean(losing_trades) if len(losing_trades) < 0 else 0
    
    # Expected value
    expected_value = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    
    # Profit factor
    profit_factor = abs(np.sum(winning_trades) / np.sum(losing_trades)) if (len(losing_trades) > 0 and np.sum(losing_trades) != 0) else 0
    
    return {
        'expected_value': expected_value,
        'profit_factor': profit_factor
    }



def combined_fitness(y_true, y_pred, sample_weight=None):
    """
    Combined fitness function with better numerical stability
    """
    try:
        # Initial cleanup and clipping of predictions
        y_pred = np.clip(y_pred, -1e6, 1e6)  # Clip extreme values
        y_pred = np.nan_to_num(y_pred, nan=0, posinf=0, neginf=0)
        
        # Check for valid inputs
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        
        # Standardize predictions to help prevent overflow
        if len(y_pred) > 0 and np.std(y_pred) != 0:
            y_pred = (y_pred - np.mean(y_pred)) / np.std(y_pred)
        
        # Early exit conditions with penalties
        if len(y_true) == 0:
            return 1.0
            
        # Penalty for constant solutions (using safe comparison)
        unique_vals = np.unique(np.round(y_pred, 6))
        if len(unique_vals) < 2:
            return 1.0
            
        # Safe ratio calculations
        total_len = len(y_pred)
        if total_len == 0:
            return 1.0
            
        # Check for too many zeros
        zero_mask = np.abs(y_pred) < 1e-6
        zero_ratio = np.sum(zero_mask) / total_len
        if zero_ratio > 0.2:
            return 1.0
            
        # Variance check
        if np.std(y_pred) < 1e-6:
            return 5.0
        
        # Calculate trading metrics with safety checks
        try:
            # Normalize predictions to prevent overflow in trading calculations
            y_pred_norm = y_pred / np.max(np.abs(y_pred)) if np.max(np.abs(y_pred)) > 0 else y_pred
            metrics = calculate_trading_metrics_from_predictions(y_true, y_pred_norm)
            
            # Normalize and clip metrics
            ev = np.clip(metrics['expected_value'], -1e3, 1e3)
            pf = np.clip(metrics['profit_factor'], 0, 1e3)
            
            # Safe normalization
            std_true = np.std(y_true)
            if std_true == 0:
                return 1.0
            
            normalized_ev = ev / std_true
            normalized_pf = np.log1p(pf)  # Using log1p for better numerical stability
            
            # Calculate IC with safety
            try:
                ic = spearmanr(y_true, y_pred_norm)[0]
                ic_score = -abs(ic) if not np.isnan(ic) else 0
            except:
                ic_score = 0
            
            # Combine scores with safety checks
            weighted_score = np.clip(
                0.25 * normalized_ev +
                0.5 * normalized_pf +
                0.25 * ic_score,
                -1e3, 1e3
            )
            
            return weighted_score if np.isfinite(weighted_score) else 5.0
            
        except Exception as e:
            return 5.0
            
    except Exception as e:
        return 5.0

# Create the fitness function
combined_fitness_function = make_fitness(
    function=combined_fitness,
    greater_is_better=False
)



















def translate_program(program_str, feature_cols):
    translated = program_str
    for i, col in enumerate(feature_cols):
        translated = translated.replace(f'X{i}', col)
    return translated

def evaluate_metric(name, value):
    """Get ranking and interpretation for each metric"""
    rankings = {
        'EV': [
            (2.0, "UNICORN: Exceptional expected value"),
            (1.5, "Elite: Production-ready returns"),
            (1.0, "Strong: Near production quality"),
            (0.75, "Good: Shows promise"),
            (0.5, "Mediocre: Needs work"),
            (0.25, "Weak: Major revision needed"),
            (-float('inf'), "Poor: Back to drawing board")
        ],
        'PF': [
            (1.5, "UNICORN: Superior profit consistency"),
            (1.3, "Elite: Highly profitable"),
            (1.1, "Strong: Consistently profitable"),
            (1.0, "Good: Break-even with potential"),
            (0.9, "Mediocre: Slight losses"),
            (0.8, "Weak: Significant losses"),
            (-float('inf'), "Poor: Catastrophic losses")
        ],
        'IC': [
            (0.25, "UNICORN: Crystal-clear prediction"),
            (0.2, "Elite: Exceptional signal"),
            (0.15, "Strong: Clear signal"),
            (0.1, "Good: Useful signal"),
            (0.07, "Mediocre: Weak signal"),
            (0.05, "Weak: Barely above noise"),
            (-float('inf'), "Poor: Random noise")
        ]
    }
    
    for threshold, interpretation in rankings[name]:
        if value >= threshold:
            return interpretation
    return rankings[name][-1][1]


def log_solution(formula, feature_cols, model, X_train, X_test, y_train, y_test, log_file="ZZZ_GeneticSolutions.txt"):
    """Enhanced logging with metric interpretations"""
    human_readable = translate_program(formula, feature_cols)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Safe prediction with overflow handling
    train_pred = np.clip(model.predict(X_train), -1e6, 1e6)
    test_pred = np.clip(model.predict(X_test), -1e6, 1e6)
    
    # Calculate metrics with safety checks
    train_metrics = calculate_trading_metrics_from_predictions(y_train, train_pred)
    test_metrics = calculate_trading_metrics_from_predictions(y_test, test_pred)
    
    # Calculate IC with absolute value
    train_ic = abs(spearmanr(y_train, train_pred)[0])
    test_ic = abs(spearmanr(y_test, test_pred)[0])
    
    log_entry = (
        f"Timestamp: {timestamp}\n"
        f"Formula: {human_readable}\n"
        f"Train Metrics:\n"
        f"  - Expected Value: {train_metrics['expected_value']:.4f}\n"
        f"    {evaluate_metric('EV', train_metrics['expected_value'])}\n"
        f"  - Profit Factor: {train_metrics['profit_factor']:.4f}\n"
        f"    {evaluate_metric('PF', train_metrics['profit_factor'])}\n"
        f"  - IC: {train_ic:.4f}\n"
        f"    {evaluate_metric('IC', train_ic)}\n"
        f"Test Metrics:\n"
        f"  - Expected Value: {test_metrics['expected_value']:.4f}\n"
        f"    {evaluate_metric('EV', test_metrics['expected_value'])}\n"
        f"  - Profit Factor: {test_metrics['profit_factor']:.4f}\n"
        f"    {evaluate_metric('PF', test_metrics['profit_factor'])}\n"
        f"  - IC: {test_ic:.4f}\n"
        f"    {evaluate_metric('IC', test_ic)}\n"
        f"Raw Formula: {formula}\n"
        f"{'='*80}\n"
    )
    
    with open(log_file, 'a') as f:
        f.write(log_entry)
    
    return human_readable, train_metrics, test_metrics





def train_symbolic_regressor(X, y, blacklist_features=None, population_size=10000, generations=12):
    """
    Train a symbolic regressor with smart feature handling and robust validation.
    """
    # Create a clean copy of features
    X = X.copy()
    
    if blacklist_features:
        existing_blacklist = [col for col in blacklist_features if col in X.columns]
        if existing_blacklist:
            X = X.drop(columns=existing_blacklist)
    
    # Store filtered feature columns
    feature_cols = X.columns.tolist()
    
    # Sample sizes
    train_size = int(0.05 * len(X))
    test_size = int(0.02 * len(X))
    
    # Set random seed
    np.random.seed(33013301)
    
    # Create train/test splits
    all_indices = np.arange(len(X))
    train_indices = np.random.choice(all_indices, size=train_size, replace=False)
    remaining_indices = np.setdiff1d(all_indices, train_indices)
    test_indices = np.random.choice(remaining_indices, size=test_size, replace=False)
    
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    
    # Configure and train the model
    est = SymbolicRegressor(
        population_size=population_size,
        generations=generations,
        metric=combined_fitness_function,
        function_set=('add', 'sub', 'mul', 'div', 'abs'),
        init_depth=(3, 6),
        parsimony_coefficient=0.03,
        p_crossover=0.7,
        p_subtree_mutation=0.2,
        p_hoist_mutation=0.05,
        p_point_mutation=0.05,
        max_samples=0.7,
        feature_names=X.columns,
        n_jobs=32,
        verbose=1,
        const_range=None,
    )
    
    print("\nTraining symbolic regressor...")
    est.fit(X_train, y_train)
    
    # Log results
    program_str = str(est._program)
    human_readable, train_metrics, test_metrics = log_solution(
        formula=program_str,
        feature_cols=feature_cols,
        model=est,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )
    
    return est, human_readable, feature_cols




def DataSpeedup(df):
    for col in df.columns:
        if df[col].dtype == 'float64':
            # First standardize the column to prevent overflow
            if col != 'Target':  # Preserve target variable
                std = np.std(df[col])
                if std != 0:
                    df[col] = (df[col] - np.mean(df[col])) / std
            
            # Round to reasonable precision
            df[col] = df[col].round(4)
            
            # Clip extreme values
            df[col] = np.clip(df[col], -10, 10)
            
            # Optimize data types
            if df[col].max() < np.finfo(np.float32).max:
                df[col] = df[col].astype(np.float32)
    return df



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Symbolic Regression')
    parser.add_argument('--data_dir', type=str, default='Data/PriceData', help='Directory containing the data files')
    parser.add_argument('--sample_size', type=str, help='Number or percentage of files to process (e.g., "100" or "10%")')
    parser.add_argument('--population_size', type=int, default=30000, help='Population size for genetic programming')
    parser.add_argument('--generations', type=int, default=5, help='Number of generations')
    parser.add_argument('--min_history', type=int, default=100, help='Minimum history length')
    args = parser.parse_args()

    # Load or create dataset
    if os.path.exists("Z_Genetic_DataSet.parquet"):
        print("Loading existing dataset...")
        df = pd.read_parquet("Z_Genetic_DataSet.parquet")
        ##print the number of rows and not the columns 
        print(f"Loaded dataset with {len(df)} rows")
        
    else:
        print("Creating new dataset...")
        data_prep = DataPrep(args.data_dir, min_history=args.min_history)
        
        # Get list of all parquet files
        all_files = [f for f in os.listdir(args.data_dir) if f.endswith('.parquet')]
        
        if args.sample_size:
            if args.sample_size.endswith('%'):
                # Handle percentage
                percentage = float(args.sample_size.rstrip('%'))
                num_files = int(len(all_files) * (percentage / 100))
                files = all_files[:num_files]
                print(f"Processing {percentage}% of files ({num_files} files)")
            else:
                # Handle absolute number
                num_files = min(int(args.sample_size), len(all_files))
                files = all_files[:num_files]
                print(f"Processing {num_files} files")
        else:
            files = all_files
            print(f"Processing all {len(files)} files")
        
        df = data_prep.prepare_dataset(files=files)
        df.to_parquet("Z_Genetic_DataSet.parquet")

    # Identify feature columns and optimize data types
    feature_cols = [col for col in df.columns if col not in ['Date', 'Return', 'Target']]
    df = DataSpeedup(df)
    
    # Update blacklist to match actual EMD feature names
    blacklist = [
        # Core OHLCV data
        'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',

        # Simple derivatives
        'High_Low', 'High_Close', 'Low_Close',

        # Basic lags
        'High_Lag2', 'High_Lag5',
        'Low_Lag2', 'Low_Lag5',
        'Volume_Lag2', 'Volume_Lag5',


        #other more complex blacklisted features underperfoming in the model
        "permutation_entropy", "lyapunov_exponent"
    ]

    X = df[feature_cols]
    y = df['Target']
    
    # Train model with command line arguments
    model, human_readable, used_features = train_symbolic_regressor(
        X, 
        y, 
        blacklist_features=blacklist,
        population_size=args.population_size,
        generations=args.generations
    )
    
    # Use filtered features for final logging
    X_filtered = X[used_features]
    
    log_solution(
        formula=str(model._program),
        feature_cols=used_features,
        model=model,
        X_train=X_filtered,
        X_test=X_filtered,
        y_train=y,
        y_test=y
    )

    ##print the final ic ev and pf
    print("Final IC:", round(abs(spearmanr(y, model.predict(X_filtered))[0]), 3))
    print("Final EV:", round(calculate_trading_metrics_from_predictions(y, model.predict(X_filtered))['expected_value'], 3))
    print("Final PF:", round(calculate_trading_metrics_from_predictions(y, model.predict(X_filtered))['profit_factor'], 3))
    print("Final Formula:", human_readable)

