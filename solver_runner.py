from Utility import Utility
import os
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from JacobiExecuter import JacobiExecuter
from ConjugateGradientExecuter import ConjugateGradientExecuter

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
            print('incorrect matrix')
            continue
        
        if A is None:
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
                    'Iterations': metrics['Iterations'],
                    'Residual': metrics['Residual'],
                    'Time Usage (seconds)': metrics['Time Usage (seconds)'],
                    'List of residuals': metrics['List of residuals']
                })
    return pd.DataFrame(parsed_data)

def plot_results(df):
    # Ensure the results directory exists
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    sns.set(style="whitegrid")

    # Get unique matrices
    matrices = df['Matrix'].unique()

    for matrix in matrices:
        matrix_df = df[df['Matrix'] == matrix]

        fig, axs = plt.subplots(3, 1, figsize=(12, 18))  # Time, Residual, Iterations
        fig.suptitle(f'Results for Matrix: {matrix}', fontsize=16)

        # Sort the tolerances in descending order
        matrix_df = matrix_df.sort_values(by='Tolerance', ascending=False)

        # Plot time usage
        sns.barplot(x='Tolerance', y='Time Usage (seconds)', hue='Solver', data=matrix_df, ax=axs[0])
        axs[0].set_title('Time Usage by Solver and Tolerance')
        axs[0].set_ylabel('Time Usage (seconds)')
        axs[0].set_xlabel('Tolerance')

        # Plot residual
        sns.barplot(x='Tolerance', y='Residual', hue='Solver', data=matrix_df, ax=axs[1])
        axs[1].set_title('Residual by Solver and Tolerance')
        axs[1].set_ylabel('Residual')
        axs[1].set_xlabel('Tolerance')
        axs[1].set_yscale('log')
        axs[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1e}'))

        # Plot iterations
        sns.barplot(x='Tolerance', y='Iterations', hue='Solver', data=matrix_df, ax=axs[2])
        axs[2].set_title('Iterations by Solver and Tolerance')
        axs[2].set_ylabel('Iterations')
        axs[2].set_xlabel('Tolerance')

        fig.subplots_adjust(top=0.96, bottom=0.04, left=0.06, right=0.98, hspace=0.3)
        
        # Save the time, residual, and iterations plot
        plot_path = os.path.join(RESULTS_DIR, f'{os.path.basename(matrix)}_summary.png')
        plt.savefig(plot_path)
        plt.close(fig)

        # Plot cumulative residuals over iterations for each tolerance level
        for tol in matrix_df['Tolerance'].unique():
            tol_df = matrix_df[matrix_df['Tolerance'] == tol]

            fig, ax = plt.subplots(figsize=(12, 8))
            ax.set_title(f'Cumulative Residual Progression for Matrix: {matrix} with Tolerance: {tol}', fontsize=16)

            for solver in tol_df['Solver'].unique():
                solver_df = tol_df[tol_df['Solver'] == solver]
                combined_residuals = np.concatenate(solver_df['List of residuals'].values)
                cumulative_residuals = np.cumsum(combined_residuals)

                ax.plot(cumulative_residuals, label=solver)
                ax.set_ylabel('Cumulative Residual')
                ax.set_xlabel('Iteration')

            ax.legend(loc="upper right")
            plot_path = os.path.join(RESULTS_DIR, f'{os.path.basename(matrix)}_cumulative_residuals_tol_{tol}.png')
            plt.savefig(plot_path)
            plt.close(fig)

if __name__ == "__main__":
    results = run_matrix_solvers()
    data = parse_data(results)
    plot_results(data)
