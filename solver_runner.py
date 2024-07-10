from Utility import Utility
import os
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from JacobiExecuter import JacobiExecuter
from ConjugateGradientExecuter import ConjugateGradientExecuter
import matplotlib as mpl

PATH = 'matrici'
RESULTS_DIR = 'results'

def run_matrix_solvers():
    matrix_files = Utility.get_matrix_paths(PATH)
    tolerances = [1e-4, 1e-6, 1e-8, 1e-10]
    max_iterations = 20000
    results = {}

    for matrix in matrix_files:
        A = Utility.read(matrix)
        
        if Utility.matrix_checks(A) is False:
            print(f'Incorrect matrix: {matrix}')
            continue
        
        if A is None:
            print(f'Failed to read matrix: {matrix}')
            continue 
        
        matrix_results = {}
        for tol in tolerances:
            tol_results = {}

            solvers = [
                ('Jacobi', JacobiExecuter(A, tol, max_iterations)),
                ('Conjugate Gradient', ConjugateGradientExecuter(A, tol, max_iterations))
            ]
            
            for solver_name, solver in solvers:
                print(f'\nSolving {matrix} with tolerance {tol} using {solver_name}')
                
                start_time = time.perf_counter()
                counter, residual_norm, list_of_residuals = solver.method_execution()
                end_time = time.perf_counter()
                time_usage = end_time - start_time
                print(f'{solver_name} solver finished in {time_usage:.6f} seconds with residual {residual_norm} and iterations {counter}')

                tol_results[solver_name] = {
                    'Iterations': counter,
                    'Residual': residual_norm,
                    'Time Usage (seconds)': time_usage,
                    'List of residuals': list_of_residuals
                }

            matrix_results[tol] = tol_results
        results[matrix] = matrix_results

    # Write the results to CSV
    Utility.write_usage_csv(os.path.join(RESULTS_DIR, 'usage_data.csv'), results)
    
    return results

def parse_data(results):
    parsed_data = []
    for matrix, tol_results in results.items():
        for tol, solver_results in tol_results.items():
            for solver, metrics in solver_results.items():
                parsed_data.append({
                    'Matrix': matrix,
                    'Solver': solver,
                    'Tolerance': tol,
                    'Residual': metrics['Residual'],
                    'Time Usage (seconds)': metrics['Time Usage (seconds)'],
                    'List of residuals': metrics['List of residuals']
                })
    return pd.DataFrame(parsed_data)

def plot_results(df, specific_tolerance=1e-6):
    # Ensure the results directory exists
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    sns.set(style="whitegrid")

    # Get unique matrices
    matrices = df['Matrix'].unique()

    # Adjusting rcParams to handle large data points
    mpl.rcParams['agg.path.chunksize'] = 10000

    for matrix in matrices:
        matrix_df = df[df['Matrix'] == matrix]
        
        # Filter for a specific tolerance
        tolerance_df = matrix_df[matrix_df['Tolerance'] == specific_tolerance]

        if tolerance_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle(f'Residual Progression for Matrix: {matrix} - Tolerance: {specific_tolerance}', fontsize=16)

        # Iterate over each solver
        for solver_name in tolerance_df['Solver'].unique():
            solver_residuals = []

            # Accumulate residuals for the specific solver and tolerance
            for _, row in tolerance_df[tolerance_df['Solver'] == solver_name].iterrows():
                residuals = np.array(row['List of residuals'])  # Convert to numpy array
                residuals = residuals.flatten()  # Ensure it's flattened to 1D array
                solver_residuals.extend(residuals)

            # Plot residuals for the solver
            iterations = np.arange(1, len(solver_residuals) + 1)
            ax.plot(iterations, solver_residuals, label=f'{solver_name}')

        ax.set_ylabel('Residual')
        ax.set_xlabel('Iteration')
        ax.set_yscale('log')
        ax.legend(loc='upper right')  # Specify legend location

        plot_path = os.path.join(RESULTS_DIR, f'{os.path.basename(matrix)}_residuals_tolerance_{specific_tolerance}.png')
        plt.savefig(plot_path)
        plt.close(fig)



if __name__ == "__main__":
    results = run_matrix_solvers()
    data = parse_data(results)
    plot_results(data)
