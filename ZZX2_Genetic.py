import os
import numpy as np
import pandas as pd
import re
from scipy.optimize import minimize
from tqdm import tqdm
from sklearn.metrics import mutual_info_score, mean_squared_error
from scipy.stats import spearmanr
from gplearn.fitness import make_fitness
from sklearn.model_selection import train_test_split
import joblib

# Configuration
CONFIG = {
    'log_file': 'Results/best_programs_plain_text.txt',
    'dataset_cache_path': 'cached_dataset.pkl',
    'output_directory': 'OptimizationResults'
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

# Custom fitness function as per gplearn section
def custom_fitness(y_true, y_pred, w=None):
    ic = abs(spearmanr(y_true, y_pred)[0])
    mi = mutual_info_score(np.digitize(y_true, np.linspace(np.min(y_true), np.max(y_true), 10)),
                           np.digitize(y_pred, np.linspace(np.min(y_pred), np.max(y_pred), 10)))
    return -ic - mi

# Modify function to allow operator replacement in addition to constant optimization
def modify_formula(function_str, column_names, mode="constants"):
    """Modify either constants or operations in the formula."""
    if mode == "constants":
        constants = re.findall(r'(-?\d+\.\d+)', function_str)
        return constants  # Return constants as before
    elif mode == "operations":
        operators = ['add', 'sub', 'mul', 'div', 'abs']
        func_parts = re.split(r'(\W+)', function_str)
        modifiable_ops = [op for op in func_parts if op in operators]
        return modifiable_ops

# Create lambda function for eval
def create_lambda_function(function_str, column_names):
    function_str = function_str.replace('add', 'np.add')
    function_str = function_str.replace('sub', 'np.subtract')
    function_str = function_str.replace('mul', 'np.multiply')
    function_str = function_str.replace('div', 'np.divide')
    function_str = function_str.replace('abs', 'np.abs')
    
    lambda_str = f"lambda {', '.join(column_names)}: {function_str}"
    return eval(lambda_str)

# Optimizer that swaps operations or constants
def optimize_formula(func, X, y, column_names, mode="constants"):
    if mode == "constants":
        constants = modify_formula(func, column_names, mode)
        initial_guess = [float(c) for c in constants]

        def objective(params):
            optimized_func = func
            for i, param in enumerate(params):
                optimized_func = optimized_func.replace(constants[i], str(param))
            lambda_func = create_lambda_function(optimized_func, column_names)
            predictions = lambda_func(**X)
            return custom_fitness(y, predictions)

        result = minimize(objective, initial_guess, method='Nelder-Mead')
        optimized_func = func
        for i, param in enumerate(result.x):
            optimized_func = optimized_func.replace(constants[i], str(param))

        return optimized_func

    elif mode == "operations":
        operations = modify_formula(func, column_names, mode)
        new_func = func
        for op in operations:
            # Randomly swap one operator with another
            new_op = np.random.choice(['add', 'sub', 'mul', 'div', 'abs'])
            new_func = new_func.replace(op, new_op, 1)
        return new_func

# Evaluate function using custom fitness
def evaluate_function(func, X, y, column_names):
    lambda_func = create_lambda_function(func, column_names)
    predictions = lambda_func(**X)
    
    # Use custom fitness function
    return custom_fitness(y, predictions)

# Main MCTS optimization with progress bar (tqdm)
def mcts_optimize(functions, X_train, y_train, X_test, y_test, column_names, optimize_mode='constants'):
    results = []
    for i, func in enumerate(tqdm(functions, desc="Optimizing Formulas")):

        # Perform constant optimization
        optimized_func = optimize_formula(func, X_train, y_train, column_names, mode=optimize_mode)
        train_score = evaluate_function(optimized_func, X_train, y_train, column_names)
        test_score = evaluate_function(optimized_func, X_test, y_test, column_names)
        
        results.append({
            'Original Function': func,
            'Optimized Function': optimized_func,
            'Train Fitness': train_score,
            'Test Fitness': test_score
        })

    return results

# Main function to load functions and optimize them
def main():
    # Load dataset (assuming X and y are already preprocessed)
    X, y = joblib.load(CONFIG['dataset_cache_path'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    column_names = X.columns.tolist()

    # Load functions from the log file
    functions = load_functions_from_log(CONFIG['log_file'])

    # Optimize and evaluate each function with progress bar and more robust optimization
    results = mcts_optimize(functions, X_train, y_train, X_test, y_test, column_names, optimize_mode="constants")

    # Save results as before
    os.makedirs(CONFIG['output_directory'], exist_ok=True)
    with open(os.path.join(CONFIG['output_directory'], 'optimization_results.txt'), 'w') as f:
        for result in results:
            f.write(f"Original Function: {result['Original Function']}\n")
            f.write(f"Optimized Function: {result['Optimized Function']}\n")
            f.write(f"Train Fitness: {result['Train Fitness']:.6f}\n")
            f.write(f"Test Fitness: {result['Test Fitness']:.6f}\n")
            f.write("\n")

if __name__ == "__main__":
    main()