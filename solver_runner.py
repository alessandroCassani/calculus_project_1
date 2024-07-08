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

def run_matrix_solvers():
    root_dir = "matrici"
    matrix_files = Utility.build_matrix_paths_list('matrici')
    tolerances = [1e-4, 1e-6, 1e-8, 1e-10]
    max_iterations = 20000

    for matrix in matrix_files:
        A = Utility.read(matrix)
        if A is None:
            continue  # Skip this matrix if it couldn't be read

        for tol in tolerances:
            print(f'\nSolving {matrix} with tolerance {tol}')

            print('Jacobi resolution method')
            jacobi = JacobiExecuter(A, tol, max_iterations)
            jacobi.methodExecution()

            print('\nGauss-Seidel resolution method')
            gs = GaussSeidelExecuter(A, tol, max_iterations)
            gs.methodExecution()

            print('\nGradient Descent resolution method')
            gr = GradientExecuter(A, tol, max_iterations)
            gr.methodExecution()

            print('\nConjugate Gradient resolution method')
            gr_con = ConjugateGradientExecuter(A, tol, max_iterations)
            gr_con.methodExecution()

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
