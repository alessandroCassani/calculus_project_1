import os
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.ticker as mticker  # Import ticker for scientific notation formatting
from Utility import Utility
from JacobiExecuter import JacobiExecuter
from ConjugateGradientExecuter import ConjugateGradientExecuter
from GaussSeidelExecuter import GaussSeidelExecuter
from GradientExecuter import GradientExecuter

PATH = 'matrici'
RESULTS_DIR = 'results'

def run_matrix_solvers():
    matrix_files = Utility.get_matrix_paths(PATH)
    tolerances = [1e-4, 1e-6, 1e-8, 1e-10]
    max_iterations = 30000
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
                ('Gauss Seidel', GaussSeidelExecuter(A, tol, max_iterations)),
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
                    'Time Usage (seconds)': max(time_usage, 1e-10),  # Ensure no zero values
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
    df = pd.DataFrame(parsed_data)
    # Convert columns to numeric to avoid plotting issues
    df['Time Usage (seconds)'] = pd.to_numeric(df['Time Usage (seconds)'], errors='coerce')
    df['Residual'] = pd.to_numeric(df['Residual'], errors='coerce')
    df['Iterations'] = pd.to_numeric(df['Iterations'], errors='coerce')
    return df

def plot_results(df):
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    sns.set(style="whitegrid")
    matrices = df['Matrix'].unique()
    plt.rcParams['agg.path.chunksize'] = 10000

    for idx, matrix in enumerate(matrices):
        matrix_df = df[df['Matrix'] == matrix]

        fig, axs = plt.subplots(4, 1, figsize=(12, 32))
        fig.suptitle(f'Results for Matrix: {matrix}', fontsize=16)
        matrix_df = matrix_df.sort_values(by='Tolerance', ascending=False)

        # Plot time usage
        max_time = matrix_df['Time Usage (seconds)'].max()
        barplot = sns.barplot(x='Tolerance', y='Time Usage (seconds)', hue='Solver', data=matrix_df, ax=axs[0])
        axs[0].set_title('Time Usage by Solver and Tolerance')
        axs[0].set_ylabel('Time Usage (seconds)')
        axs[0].set_xlabel('Tolerance')
        axs[0].set_yscale('log')  # Set y-axis to logarithmic scale to make all bars visible
        axs[0].set_ylim(1e-10, max_time * 1.2)  # Set y-axis limits starting from 1e-10 for better visibility
        
        # Format y-axis with exponential notation
        axs[0].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'$10^{{{int(np.log10(x))}}}$' if x > 0 else ''))

        # Round time usage to 3 decimal places for labels
        for container in axs[0].containers:
            labels = [f'{v.get_height():.3f}' for v in container]  # Use 3 decimal places for time usage
            axs[0].bar_label(container, labels=labels)

        for patch in barplot.patches:
            patch.set_edgecolor(patch.get_facecolor())

        # Position the legend outside the plot
        axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0.)

        # Plot residual
        max_residual = matrix_df['Residual'].max()
        min_residual = matrix_df['Residual'].min()
        barplot = sns.barplot(x='Tolerance', y='Residual', hue='Solver', data=matrix_df, ax=axs[1])
        axs[1].set_title('Residual by Solver and Tolerance')
        axs[1].set_ylabel('Residual')
        axs[1].set_xlabel('Tolerance')
        axs[1].set_yscale('log')
        axs[1].set_ylim(min_residual * 0.1, max_residual * 1.1)
        
        # Use scientific notation for the y-axis labels
        axs[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'$10^{{{int(np.log10(x))}}}$' if x > 0 else ''))

        # Add labels only for the maximum residuals in each tolerance level
        for tolerance, group in matrix_df.groupby('Tolerance'):
            max_residual = group['Residual'].max()
            max_index = group['Residual'].idxmax()
            row = group.loc[max_index]
            axs[1].text(row['Tolerance'], row['Residual'] + max_residual * 0.1, f'{row["Residual"]:.2e}', 
                        color='black', ha='center', fontsize=12, weight='bold')

        for patch in barplot.patches:
            patch.set_edgecolor(patch.get_facecolor())

        # Position the legend outside the plot
        axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0.)

        # Plot iterations
        max_iterations = matrix_df['Iterations'].max()
        barplot = sns.barplot(x='Tolerance', y='Iterations', hue='Solver', data=matrix_df, ax=axs[2])
        axs[2].set_title('Iterations by Solver and Tolerance')
        axs[2].set_ylabel('Iterations')
        axs[2].set_xlabel('Tolerance')
        axs[2].set_yscale('log')  # Set y-axis to logarithmic scale to ensure visibility of all bars
        axs[2].set_ylim(1, max_iterations * 1.1)

        for container in axs[2].containers:
            labels = [f'{int(v.get_height())}' for v in container]  # Use integer values for iterations
            axs[2].bar_label(container, labels=labels)

        for patch in barplot.patches:
            patch.set_edgecolor(patch.get_facecolor())

        # Position the legend outside the plot
        axs[2].legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0.)

        # Plot residual progression for specific tolerance
        tolerance_df = matrix_df[matrix_df['Tolerance'] == 1e-10]

        if not tolerance_df.empty:
            for solver_name in tolerance_df['Solver'].unique():
                for _, row in tolerance_df[tolerance_df['Solver'] == solver_name].iterrows():
                    residuals = np.array(row['List of residuals'])
                    iterations = np.arange(1, len(residuals) + 1)
                    axs[3].plot(iterations, residuals, label=f'{solver_name}', alpha=0.7)

            axs[3].set_ylabel('Residual')
            axs[3].set_xlabel('Iteration')
            axs[3].set_yscale('log')
            # Position the legend outside the plot
            axs[3].legend(loc='upper left', bbox_to_anchor=(1, 1), borderaxespad=0.)

        plt.tight_layout(rect=[0, 0, 0.85, 0.96])  # Adjust rect to give space for the legend outside

        # Save the plot
        plot_path = os.path.join(RESULTS_DIR, f'{os.path.basename(matrix)}_results.png')
        plt.savefig(plot_path, bbox_inches='tight')  # Save figure with adjusted bbox
        plt.close(fig)


if __name__ == "__main__":
    results = run_matrix_solvers()
    data = parse_data(results)
    plot_results(data)
