import os
import numpy as np
import pandas as pd
import re
from scipy.optimize import minimize, dual_annealing
from tqdm import tqdm
from sklearn.metrics import mutual_info_score, mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
from gplearn.fitness import make_fitness
from sklearn.model_selection import train_test_split
import joblib
import datetime

# Configuration
CONFIG = {
    'log_file': 'Results/best_programs_plain_text.txt',
    'dataset_cache_path': 'cached_dataset.pkl',
    'output_directory': 'OptimizationResults',
    'decimal_places': 4  # New configuration for decimal places
}

def load_functions_from_log(log_file):
    """Load the best functions from the log file."""
    functions = []
    with open(log_file, 'r') as f:
        content = f.read()
        matches = re.findall(r'Program: (.+)\n', content)
        for match in matches:
            functions.append(match.strip())
    return functions

# Custom fitness function with weighted components
def custom_fitness(y_true, y_pred, w=None):
    ic = abs(spearmanr(y_true, y_pred)[0])
    mi = abs(mutual_info_score(np.digitize(y_true, np.linspace(np.min(y_true), np.max(y_true), 10)),
                                np.digitize(y_pred, np.linspace(np.min(y_pred), np.max(y_pred), 10))))
    return -(0.7 * ic + 0.3 * mi)





def round_constants(function_str, decimal_places):
    """Round constants in the function string to specified decimal places."""
    def round_match(match):
        return f"{float(match.group()):.{decimal_places}f}"
    return re.sub(r'-?\d+\.\d+', round_match, function_str)


def modify_formula(function_str, column_names, mode="constants"):
    """Modify either constants or operations in the formula."""
    if mode == "constants":
        constants = re.findall(r'(-?\d+\.\d+)', function_str)
        return constants
    elif mode == "operations":
        operators = ['add', 'sub', 'mul', 'div', 'abs']
        func_parts = re.split(r'(\W+)', function_str)
        modifiable_ops = [op for op in func_parts if op in operators]
        return modifiable_ops

def create_lambda_function(function_str, column_names):
    function_str = function_str.replace('add', 'np.add')
    function_str = function_str.replace('sub', 'np.subtract')
    function_str = function_str.replace('mul', 'np.multiply')
    function_str = function_str.replace('div', 'np.divide')
    function_str = function_str.replace('abs', 'np.abs')
    
    lambda_str = f"lambda {', '.join(column_names)}: {function_str}"
    try:
        return eval(lambda_str)
    except SyntaxError as e:
        raise ValueError(f"Error in function generation: {e}")

def optimize_formula(func, X, y, column_names, mode="constants"):
    best_func = func
    best_fitness = float('inf')

    if mode == "constants":
        constants = modify_formula(func, column_names, mode)
        initial_guess = [float(c) for c in constants]

        def objective(params):
            nonlocal best_func, best_fitness
            optimized_func = func
            for i, param in enumerate(params):
                optimized_func = optimized_func.replace(constants[i], f"{param:.{CONFIG['decimal_places']}f}")
            try:
                lambda_func = create_lambda_function(optimized_func, column_names)
                predictions = lambda_func(**X)
                fitness = custom_fitness(y, predictions)
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_func = optimized_func
                return fitness
            except Exception:
                return 1e10

        minimize(objective, initial_guess, method='Nelder-Mead')
        return best_func

    elif mode == "operations":
        operations = modify_formula(func, column_names, mode)

        def objective(ops_idx):
            nonlocal best_func, best_fitness
            new_func = func
            for i, op in enumerate(operations):
                new_op = ['add', 'sub', 'mul', 'div', 'abs'][int(ops_idx[i])]
                new_func = new_func.replace(op, new_op, 1)
            try:
                lambda_func = create_lambda_function(new_func, column_names)
                predictions = lambda_func(**X)
                fitness = custom_fitness(y, predictions)
                if fitness < best_fitness:
                    best_fitness = fitness
                    best_func = new_func
                return fitness
            except Exception:
                return 1e10

        bounds = [(0, 4)] * len(operations)
        dual_annealing(objective, bounds)
        return best_func





def hybrid_optimize(func, X, y, column_names):
    func_optimized_constants = optimize_formula(func, X, y, column_names, mode="constants")
    func_optimized_hybrid = optimize_formula(func_optimized_constants, X, y, column_names, mode="operations")
    return round_constants(func_optimized_hybrid, CONFIG['decimal_places'])

def calculate_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    ic = abs(spearmanr(y_true, y_pred)[0])
    mi = abs(mutual_info_score(np.digitize(y_true, np.linspace(np.min(y_true), np.max(y_true), 10)),
                                np.digitize(y_pred, np.linspace(np.min(y_pred), np.max(y_pred), 10))))
    con = np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred)))
    return mse, mae, correlation, ic, mi, con


def evaluate_function(func, X, y, column_names):
    try:
        lambda_func = create_lambda_function(func, column_names)
        predictions = lambda_func(**X)
        mse, mae, correlation, ic, mi, con = calculate_metrics(y, predictions)
        fitness = -(0.7 * abs(ic) + 0.3 * mi)
        return fitness, mse, mae, correlation, ic, mi, con
    except Exception:
        return 1e10, 1e10, 1e10, 0, 0, 0, 0


def evaluate_function(func, X, y, column_names):
    try:
        lambda_func = create_lambda_function(func, column_names)
        predictions = lambda_func(**X)
        mse, mae, correlation, ic, mi, con = calculate_metrics(y, predictions)
        fitness = -(0.7 * ic + 0.3 * mi)  # Note: ic and mi are already absolute here
        return fitness, mse, mae, correlation, ic, mi, con
    except Exception:
        return 1e10, 1e10, 1e10, 0, 0, 0, 0

def optimize_and_evaluate(functions, X_train, y_train, X_test, y_test, column_names):
    results = []
    for func in tqdm(functions, desc="Optimizing Formulas"):
        optimized_func = hybrid_optimize(func, X_train, y_train, column_names)
        
        _, orig_mse, orig_mae, orig_corr, orig_ic, orig_mi, orig_con = evaluate_function(func, X_test, y_test, column_names)
        opt_fitness, opt_mse, opt_mae, opt_corr, opt_ic, opt_mi, opt_con = evaluate_function(optimized_func, X_test, y_test, column_names)
        
        # If optimized performance is worse, keep the original
        if opt_fitness > _:
            optimized_func = func
            opt_mse, opt_mae, opt_corr, opt_ic, opt_mi, opt_con = orig_mse, orig_mae, orig_corr, orig_ic, orig_mi, orig_con
        
        # Calculate percentage changes
        ic_change = ((opt_ic - orig_ic) / orig_ic) * 100 if orig_ic != 0 else 0
        mi_change = ((opt_mi - orig_mi) / orig_mi) * 100 if orig_mi != 0 else 0
        con_change = ((opt_con - orig_con) / orig_con) * 100 if orig_con != 0 else 0
        overall_change = (ic_change * 0.7 + mi_change * 0.3)
        
        results.append({
            'Original Function': func,
            'Optimized Function': optimized_func,
            'Original Metrics': (orig_mse, orig_mae, orig_corr, orig_ic, orig_mi, orig_con),
            'Optimized Metrics': (opt_mse, opt_mae, opt_corr, opt_ic, opt_mi, opt_con),
            'Changes': (ic_change, mi_change, con_change, overall_change)
        })

    return results






def main():
    X, y = joblib.load(CONFIG['dataset_cache_path'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    column_names = X.columns.tolist()

    functions = load_functions_from_log(CONFIG['log_file'])
    results = optimize_and_evaluate(functions, X_train, y_train, X_test, y_test, column_names)

    os.makedirs(CONFIG['output_directory'], exist_ok=True)
    with open(os.path.join(CONFIG['output_directory'], 'optimization_results.txt'), 'w') as f:
        for result in results:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"\n\n--- New Optimized Program ({timestamp}) ---\n")
            f.write(f"Original Program: {result['Original Function']}\n")
            f.write("Original Test Set Metrics:\n")
            f.write(f"  MSE: {result['Original Metrics'][0]:.6f}\n")
            f.write(f"  MAE: {result['Original Metrics'][1]:.6f}\n")
            f.write(f"  Correlation: {result['Original Metrics'][2]:.6f}\n")
            f.write(f"  Information Coefficient: {result['Original Metrics'][3]:.6f}\n")
            f.write(f"  Mutual Information: {result['Original Metrics'][4]:.6f}\n")
            f.write(f"  Concordance: {result['Original Metrics'][5]:.6f}\n\n")
            
            f.write(f"Optimized Program: {result['Optimized Function']}\n")
            f.write("Optimized Test Set Metrics:\n")
            f.write(f"  MSE: {result['Optimized Metrics'][0]:.6f}\n")
            f.write(f"  MAE: {result['Optimized Metrics'][1]:.6f}\n")
            f.write(f"  Correlation: {result['Optimized Metrics'][2]:.6f}\n")
            f.write(f"  Information Coefficient: {result['Optimized Metrics'][3]:.6f}\n")
            f.write(f"  Mutual Information: {result['Optimized Metrics'][4]:.6f}\n")
            f.write(f"  Concordance: {result['Optimized Metrics'][5]:.6f}\n\n")
            
            f.write("Percentage Changes:\n")
            f.write(f"  IC Change: {result['Changes'][0]:.2f}%\n")
            f.write(f"  MI Change: {result['Changes'][1]:.2f}%\n")
            f.write(f"  CON Change: {result['Changes'][2]:.2f}%\n")
            f.write(f"  Overall Change: {result['Changes'][3]:.2f}%\n")

if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()