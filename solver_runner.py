from Utility import Utility
import os
import numpy as np
import time
import matplotlib.pyplot as plt
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

                tol_results[solver_name] = {
                    'Memory Usage (MB)': mem_usage,
                    'Time Usage (seconds)': end_time - start_time
                }

            matrix_results[tol] = tol_results
        results[matrix] = matrix_results
        Utility.write_usage_csv('results/usage_data.csv', results)  


def plot_memory_time_usage(memory_usage_data, time_usage_data):
    for matrix_file in memory_usage_data:
        for tol in memory_usage_data[matrix_file]:
            solvers = list(memory_usage_data[matrix_file][tol].keys())
            memory_usages = [memory_usage_data[matrix_file][tol][solver] for solver in solvers]
            time_usages = [time_usage_data[matrix_file][tol][solver] for solver in solvers]

            # Plot memory usage
            plt.figure(figsize=(12, 6))
            plt.bar(solvers, memory_usages, color='skyblue')
            plt.xlabel('Solver')
            plt.ylabel('Memory Usage (MB)')
            plt.title(f'Memory Usage for {matrix_file} with tolerance {tol}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'memory_usage_{os.path.basename(matrix_file)}_tol_{tol}.png')
            plt.close()

            # Plot time usage
            plt.figure(figsize=(12, 6))
            plt.bar(solvers, time_usages, color='salmon')
            plt.xlabel('Solver')
            plt.ylabel('Time Usage (seconds)')
            plt.title(f'Time Usage for {matrix_file} with tolerance {tol}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f'time_usage_{os.path.basename(matrix_file)}_tol_{tol}.png')
            plt.close()

if __name__ == "__main__":
    run_matrix_solvers()
