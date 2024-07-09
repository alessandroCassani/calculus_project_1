from Utility import Utility
import os
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from memory_profiler import memory_usage
from JacobiExecuter import JacobiExecuter
from GaussSeidelExecuter import GaussSeidelExecuter
from GradientExecuter import GradientExecuter
from ConjugateGradientExecuter import ConjugateGradientExecuter

PATH = 'matrici'

def run_matrix_solvers():
    matrix_files = Utility.get_matrix_paths(PATH)
    tolerances = [1e-4, 1e-6, 1e-8, 1e-10]
    max_iterations = 20000
    results = {}

    for matrix in matrix_files:
        A = Utility.read(matrix)
        
        if Utility.matrix_checks(A) is False:
            print('incorrect matrix')
            return
        
        if A is None:
            continue 
        
        matrix_results = {}
        for tol in tolerances:
            tol_results = {}

            solvers = [
                ('Jacobi', JacobiExecuter(A, tol, max_iterations)),
                ('Gauss-Seidel', GaussSeidelExecuter(A, tol, max_iterations)),
                ('Gradient Descent', GradientExecuter(A, tol, max_iterations)),
                ('Conjugate Gradient', ConjugateGradientExecuter(A, tol, max_iterations))
            ]
            
            for solver_name, solver in solvers:
                print(f'\nSolving {matrix} with tolerance {tol} using {solver_name}')
                
                start_time = time.time()
                mem_usage = memory_usage((solver.methodExecution,), max_iterations=1)[0]
                end_time = time.time()
                _, iterations, residual_norm = solver.methodExecution()

                tol_results[solver_name] = {
                    'Memory Usage (MB)': mem_usage,
                    'Time Usage (seconds)': end_time - start_time,
                    'Iterations': iterations,
                    'Residual Norm': residual_norm
                }

            matrix_results[tol] = tol_results
        results[matrix] = matrix_results

    # Write the results to CSV
    Utility.write_usage_csv('results1/usage_data.csv', results)
    
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
                    'Memory Usage (MB)': metrics['Memory Usage (MB)'],
                    'Time Usage (seconds)': metrics['Time Usage (seconds)'],
                    'Iterations': metrics['Iterations'],
                    'Residual Norm': metrics['Residual Norm']
                })
    return pd.DataFrame(parsed_data)

def plot_results(df):
    plt.figure(figsize=(12, 16))
    
    sns.set(style="whitegrid")

    # Plot time usage
    plt.subplot(4, 1, 1)
    sns.barplot(x='Matrix', y='Time Usage (seconds)', hue='Solver', data=df)
    plt.title('Time Usage by Solver and Matrix')
    plt.ylabel('Time Usage (seconds)')
    plt.xlabel('Matrix')

    # Plot memory usage
    plt.subplot(4, 1, 2)
    sns.barplot(x='Matrix', y='Memory Usage (MB)', hue='Solver', data=df)
    plt.title('Memory Usage by Solver and Matrix')
    plt.ylabel('Memory Usage (MB)')
    plt.xlabel('Matrix')

    # Plot number of iterations
    plt.subplot(4, 1, 3)
    sns.barplot(x='Matrix', y='Iterations', hue='Solver', data=df)
    plt.title('Iterations by Solver and Matrix')
    plt.ylabel('Iterations')
    plt.xlabel('Matrix')

    # Plot residual norm
    plt.subplot(4, 1, 4)
    sns.barplot(x='Matrix', y='Residual Norm', hue='Solver', data=df)
    plt.title('Residual Norm by Solver and Matrix')
    plt.ylabel('Residual Norm')
    plt.xlabel('Matrix')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    results = run_matrix_solvers()
    
    data = parse_data(results)
    plot_results(data)
