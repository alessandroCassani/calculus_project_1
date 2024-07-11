from Utility import Utility
import os
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from JacobiExecuter import JacobiExecuter
from ConjugateGradientExecuter import ConjugateGradientExecuter
from GaussSeidelExecuter import GaussSeidelExecuter
from GradientExecuter import GradientExecuter
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
                ('Conjugate Gradient', ConjugateGradientExecuter(A, tol, max_iterations)),
                ('Gaus Seidel', GaussSeidelExecuter(A, tol, max_iterations)),
                ('Gradient', GradientExecuter(A, tol, max_iterations))
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

import matplotlib as mpl

def plot_results(df, specific_tolerance=1e-6):
    # Ensure the results directory exists
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    sns.set(style="whitegrid")

    # Get unique matrices
    matrices = df['Matrix'].unique()

    # Adjusting rcParams to handle large data points
    plt.rcParams['agg.path.chunksize'] = 10000

    for idx, matrix in enumerate(matrices):
        matrix_df = df[df['Matrix'] == matrix]

        fig, axs = plt.subplots(4, 1, figsize=(12, 32))
        fig.suptitle(f'Results for Matrix: {matrix}', fontsize=16)

        # Sort the tolerances in descending order
        matrix_df = matrix_df.sort_values(by='Tolerance', ascending=False)

        # Plot time usage
        max_time = matrix_df['Time Usage (seconds)'].max()
        barplot = sns.barplot(x='Tolerance', y='Time Usage (seconds)', hue='Solver', data=matrix_df, ax=axs[0])
        axs[0].set_title('Time Usage by Solver and Tolerance')
        axs[0].set_ylabel('Time Usage (seconds)')
        axs[0].set_xlabel('Tolerance')
        axs[0].set_ylim(0, max_time * 1.1)  # Set ylim slightly above max time for better visualization

        for container in axs[0].containers:
            axs[0].bar_label(container)

        for patch in barplot.patches:
            patch.set_edgecolor(patch.get_facecolor())

        # Plot residual
        max_residual = matrix_df['Residual'].max()
        min_residual = matrix_df['Residual'].min()
        barplot = sns.barplot(x='Tolerance', y='Residual', hue='Solver', data=matrix_df, ax=axs[1])
        axs[1].set_title('Residual by Solver and Tolerance')
        axs[1].set_ylabel('Residual')
        axs[1].set_xlabel('Tolerance')
        axs[1].set_yscale('log')
        axs[1].set_ylim(min_residual * 0.1, max_residual * 1.1)
        axs[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1e}'))

        for container in axs[1].containers:
            axs[1].bar_label(container)

        for patch in barplot.patches:
            patch.set_edgecolor(patch.get_facecolor())

        # Plot iterations
        max_iterations = matrix_df['Iterations'].max()
        barplot = sns.barplot(x='Tolerance', y='Iterations', hue='Solver', data=matrix_df, ax=axs[2])
        axs[2].set_title('Iterations by Solver and Tolerance')
        axs[2].set_ylabel('Iterations')
        axs[2].set_xlabel('Tolerance')
        axs[2].set_ylim(0, max_iterations * 1.1)  # Set ylim slightly above max iterations for better visualization

        for container in axs[2].containers:
            axs[2].bar_label(container)

        for patch in barplot.patches:
            patch.set_edgecolor(patch.get_facecolor())

        # Plot residual progression for specific tolerance
        tolerance_df = matrix_df[matrix_df['Tolerance'] == specific_tolerance]

        if not tolerance_df.empty:
            # Iterate over each solver
            for solver_name in tolerance_df['Solver'].unique():
                solver_residuals = []

                # Accumulate residuals for the specific solver and tolerance
                for _, row in tolerance_df[tolerance_df['Solver'] == solver_name].iterrows():
                    residuals = np.array(row['List of residuals'])  # Convert to numpy array
                    residuals = residuals.flatten()  # Ensure it's flattened to 1D array
                    solver_residuals.extend(residuals)

                # Smooth the residuals
                solver_residuals = pd.Series(solver_residuals).rolling(window=100, min_periods=1).mean().values

                # Plot residuals for the solver
                iterations = np.arange(1, len(solver_residuals) + 1)
                axs[3].plot(iterations, solver_residuals, label=f'{solver_name}', alpha=0.7)

            axs[3].set_ylabel('Residual')
            axs[3].set_xlabel('Iteration')
            axs[3].set_yscale('log')
            axs[3].legend(loc='upper right')  # Specify legend location

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # Save the plot
        plot_path = os.path.join(RESULTS_DIR, f'{os.path.basename(matrix)}_results.png')
        plt.savefig(plot_path)
        plt.close(fig)

if __name__ == "__main__":
    results = run_matrix_solvers()
    data = parse_data(results)
    plot_results(data)