import os
import pandas as pd
import numpy as np
from gplearn.genetic import SymbolicRegressor
from gplearn.fitness import make_fitness
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import datetime
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import random

CONFIG = {
    'input_directory': 'Data/PriceData',
    'output_directory': 'Results',
}

GENCONFIG = {
    'population_size': 10000,
    'generations': 3,
    'tournament_size': 10,
    'stopping_criteria': 0.1,
    'const_range': (-100, 100),
    'init_depth': (1, 5),
    'init_method': 'half and half',
    'function_set': ('add', 'sub', 'mul', 'div', 'log', 'abs', 'neg', 'sqrt', 'max', 'min', "inv"),
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

def plot_fitness_evolution(est_gp):
    plt.figure(figsize=(12, 6))
    plt.plot(est_gp.run_details_['generation'], est_gp.run_details_['average_fitness'], label='Average Fitness')
    plt.plot(est_gp.run_details_['generation'], est_gp.run_details_['best_fitness'], label='Best Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Evolution of Fitness Scores')
    plt.legend()
    plt.savefig(os.path.join(CONFIG['output_directory'], 'fitness_evolution.png'))
    plt.close()

def plot_program_length(est_gp):
    plt.figure(figsize=(12, 6))
    plt.plot(est_gp.run_details_['generation'], est_gp.run_details_['average_length'], label='Average Length')
    plt.plot(est_gp.run_details_['generation'], est_gp.run_details_['best_length'], label='Best Length')
    plt.xlabel('Generation')
    plt.ylabel('Program Length')
    plt.title('Evolution of Program Length')
    plt.legend()
    plt.savefig(os.path.join(CONFIG['output_directory'], 'program_length_evolution.png'))
    plt.close()

def plot_prediction_vs_actual(y_true, y_pred, title):
    plt.figure(figsize=(12, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Returns')
    plt.ylabel('Predicted Returns')
    plt.title(title)
    plt.savefig(os.path.join(CONFIG['output_directory'], f'{title.lower().replace(" ", "_")}.png'))
    plt.close()

def load_data(file_path):
    df = pd.read_parquet(file_path)
    if 'Date' not in df.columns and df.index.name == 'Date':
        df = df.reset_index()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.iloc[10:-10]  # Remove first and last 10 rows
    return df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

def prepare_data(df):
    df['Future_Volume_Change'] = df['Volume'].pct_change(7).shift(-1)
    #df['Future_Volume_Change'] = df['Future_Volume_Change'].ewm(span=5).mean()
    
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[f'{col}_Lag1'] = df[col].shift(3)
        df[f'{col}_Lag2'] = df[col].shift(7)
        df[f'{col}_Pct_Change'] = df[col].pct_change()
    
    df = df.dropna().replace([np.inf, -np.inf], np.nan).dropna()
    df = df.iloc[10:-10]  # Remove additional 10 rows from start and end

    feature_columns = [col for col in df.columns if col not in ['Date', 'Future_Volume_Change']]
    X = df[feature_columns]
    y = df['Future_Volume_Change']
    
    return X, y

def safe_operation(operation, default_value):
    try:
        result = operation()
        return default_value if np.isnan(result) or np.isinf(result) else result
    except:
        return default_value

def preprocess_data(y, y_pred):
    mask = np.isfinite(y) & np.isfinite(y_pred)
    return y[mask], y_pred[mask]


def _custom_fitness(y, y_pred, w):
    y, y_pred = preprocess_data(y, y_pred)

    if len(y) == 0 or len(y_pred) == 0:
        return 1e10

    if np.all(y_pred == y_pred[0]):
        return 1e9

    # Direction accuracy
    direction_accuracy = np.mean(np.sign(y) == np.sign(y_pred))
    
    # Magnitude similarity
    magnitude_similarity = 1 - np.mean(np.abs(y - y_pred) / (np.abs(y) + 1e-8))
    
    # Correlation
    correlation = np.corrcoef(y, y_pred)[0, 1]
    
    # Consistency of predictions
    std_pred = np.std(y_pred)
    consistency = 1 / (1 + np.exp(-100 * (std_pred - 0.001)))
    
    # Penalize extreme predictions
    extreme_penalty = np.mean(np.abs(y_pred) > 5 * np.std(y))
    
    # Reward for predicting large moves correctly
    large_move_accuracy = np.mean((np.abs(y) > np.std(y)) & (np.sign(y) == np.sign(y_pred)))

    combined_score = (
        0.30 * direction_accuracy +
        0.20 * magnitude_similarity +
        0.20 * correlation +
        0.10 * consistency +
        0.10 * (1 - extreme_penalty) +
        0.10 * large_move_accuracy
    )

    return 1 - combined_score  # We want to minimize, so we return 1 - score

custom_fitness = make_fitness(function=_custom_fitness, greater_is_better=False)




def translate_program(program, feature_names):
    program_str = str(program)
    for i, name in enumerate(feature_names):
        program_str = program_str.replace(f'X{i}', name)
    return program_str



def calculate_metrics(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    
    if len(y_true) == 0:
        return None
    
    mse = safe_operation(lambda: mean_squared_error(y_true, y_pred), np.inf)
    mae = safe_operation(lambda: mean_absolute_error(y_true, y_pred), np.inf)
    correlation = safe_operation(lambda: np.corrcoef(y_true, y_pred)[0, 1], 0)
    
    # Calculate R-squared
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    ss_res = np.sum((y_true - y_pred)**2)
    r_squared = safe_operation(lambda: 1 - (ss_res / ss_tot), 0)
    
    # Calculate prediction accuracy within different thresholds
    accuracy_5percent = np.mean(np.abs((y_pred - y_true) / y_true) <= 0.05)
    accuracy_10percent = np.mean(np.abs((y_pred - y_true) / y_true) <= 0.10)
    
    # Calculate directional accuracy (increase vs. decrease)
    directional_accuracy = np.mean((y_true > 0) == (y_pred > 0))
    
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': np.sqrt(mse),
        'Correlation': correlation,
        'R-squared': r_squared,
        'Accuracy (±5%)': accuracy_5percent,
        'Accuracy (±10%)': accuracy_10percent,
        'Directional Accuracy': directional_accuracy
    }




def validate_out_of_sample(program, feature_names):
    random.seed(42)
    file_paths = [os.path.join(CONFIG['input_directory'], f) for f in os.listdir(CONFIG['input_directory']) if f.endswith('.parquet')]
    random.shuffle(file_paths)
    file_paths = file_paths[:500]

    all_data = pd.concat([load_data(file) for file in file_paths])
    X, y = prepare_data(all_data)

    y_pred = program.predict(X)
    metrics = calculate_metrics(y, y_pred)
    if metrics:
        print_metrics(metrics, "Out-of-Sample Performance")


def print_metrics(metrics, title):
    print(f"\n{title}:")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"MAE: {metrics['MAE']:.4f}")
    print(f"Correlation: {metrics['Correlation']:.4f}")
    print(f"R-squared: {metrics['R-squared']:.4f}")
    print(f"Accuracy (±5%): {metrics['Accuracy (±5%)']:.4f}")
    print(f"Accuracy (±10%): {metrics['Accuracy (±10%)']:.4f}")
    print(f"Directional Accuracy: {metrics['Directional Accuracy']:.4f}")



def main():
    file_paths = [os.path.join(CONFIG['input_directory'], f) for f in os.listdir(CONFIG['input_directory']) if f.endswith('.parquet')]
    file_paths = file_paths[:50]
    all_data = pd.concat([load_data(file) for file in file_paths])

    X, y = prepare_data(all_data)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {len(X_train)}")
    est_gp = SymbolicRegressor(**GENCONFIG, metric=custom_fitness)
    est_gp.fit(X_train, y_train)
    
    y_pred_train = est_gp.predict(X_train)
    y_pred_test = est_gp.predict(X_test)
    
    print("Best program:", est_gp._program)
    
    train_metrics = calculate_metrics(y_train, y_pred_train)
    test_metrics = calculate_metrics(y_test, y_pred_test)
    
    if train_metrics:
        print_metrics(train_metrics, "Genetic Programming (Train) Performance")

    if test_metrics:
        print_metrics(test_metrics, "Genetic Programming (Test) Performance")


    
    os.makedirs(CONFIG['output_directory'], exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    feature_names = X.columns.tolist()
    
    # Update the writing to file in main():
    with open(os.path.join(CONFIG['output_directory'], 'best_programs.txt'), 'a') as f:
        f.write(f"\n\n--- New Best Program ({timestamp}) ---\n")
        f.write(f"Program: {translate_program(est_gp._program, feature_names)}\n")
        if test_metrics:
            for metric, value in test_metrics.items():
                f.write(f"Test Set {metric}: {value:.6f}\n")
        else:
            f.write("Unable to calculate test metrics (no valid data)\n")
    plot_fitness_evolution(est_gp)
    plot_program_length(est_gp)
    plot_prediction_vs_actual(y_test, y_pred_test, 'Predicted vs Actual Returns (Test Set)')
    
    validate_out_of_sample(est_gp, feature_names)

if __name__ == "__main__":
    main()
