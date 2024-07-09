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
                #('Gauss-Seidel', GaussSeidelExecuter(A, tol, max_iterations)),
                #('Gradient Descent', GradientExecuter(A, tol, max_iterations)),
                ('Conjugate Gradient', ConjugateGradientExecuter(A, tol, max_iterations))
            ]
            
            for solver_name, solver in solvers:
                print(f'\nSolving {matrix} with tolerance {tol} using {solver_name}')
                
                start_time = time.time()
                counter, residual_norm = solver.method_execution()
                end_time = time.time()

                tol_results[solver_name] = {
                    'Iterations': counter,
                    'Residual': residual_norm,
                    'Time Usage (seconds)': end_time - start_time
                }

            matrix_results[tol] = tol_results
        results[matrix] = matrix_results

    # Write the results to CSV
    Utility.write_usage_csv('results1/usage_data1.csv', results)
    
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
                    'Time Usage (seconds)': metrics['Time Usage (seconds)']
                })
    return pd.DataFrame(parsed_data)

def plot_results(df):
    # Increase figure height to accommodate three subplots
    plt.figure(figsize=(12, 18))
    
    sns.set(style="whitegrid")

    # Plot time usage
    plt.subplot(3, 1, 1)
    max_time = df['Time Usage (seconds)'].max()
    sns.barplot(x='Matrix', y='Time Usage (seconds)', hue='Solver', data=df)
    plt.title('Time Usage by Solver and Matrix')
    plt.ylabel('Time Usage (seconds)')
    plt.xlabel('Matrix')
    plt.ylim(0, max_time * 1.1)  # Set ylim slightly above max time for better visualization
    
    # Plot residual
    plt.subplot(3, 1, 2)
    max_residual = df['Residual'].max()
    sns.barplot(x='Matrix', y='Residual', hue='Solver', data=df)
    plt.title('Residual by Solver and Matrix')
    plt.ylabel('Residual')
    plt.xlabel('Matrix')
    plt.ylim(0, max_residual * 1.1)  # Set ylim slightly above max residual for better visualization

    # Plot iterations
    plt.subplot(3, 1, 3)
    max_iterations = df['Iterations'].max()
    sns.barplot(x='Matrix', y='Iterations', hue='Solver', data=df)
    plt.title('Iterations by Solver and Matrix')
    plt.ylabel('Iterations')
    plt.xlabel('Matrix')
    plt.ylim(0, max_iterations * 1.1)  # Set ylim slightly above max iterations for better visualization
    
    # Adjust spacing between subplots
    plt.tight_layout(pad=3.0)
    
    plt.show()


if __name__ == "__main__":
    results = run_matrix_solvers()
    
    data = parse_data(results)
    plot_results(data)
