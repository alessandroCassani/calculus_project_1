from Utility import Utility
from IterativeSolvers import IterativeSolvers
import os
import numpy as np
import time
import matplotlib.pyplot as plt
from memory_profiler import memory_usage

def run_solvers_on_matrices(matrix_files: list, tolerances: list, max_iter: int):
    memory_usage_data = {}
    time_usage_data = {}

    for matrix_file in matrix_files:
        A = Utility.read_sparse_matrix(matrix_file)
        b = np.ones(A.shape[0])  # Assuming b is a vector of ones for simplicity
        
        for tol in tolerances:
            print(f"\n Solving {matrix_file} with tolerance {tol}")
            
            solvers = {
                "Jacobi": IterativeSolvers.jacobi,
                "Gauss-Seidel": IterativeSolvers.gauss_seidel,
                "Gradient Descent": IterativeSolvers.gradient_descent,
                "Conjugate Gradient": IterativeSolvers.conjugate_gradient
            }

            for solver_name, solver in solvers.items():
                try:
                    mem_usage = memory_usage((solver, (A, b, tol, max_iter)), interval=0.1)
                    start_time = time.time()
                    x = solver(A, b, tol, max_iter)
                    end_time = time.time()
                    
                    residual = Utility.relative_residual_norm(A, x, b)
                    elapsed_time = end_time - start_time
                    
                    print(f"{solver_name} converged to solution with residual {residual}")

                    # Save memory and time usage data
                    if matrix_file not in memory_usage_data:
                        memory_usage_data[matrix_file] = {}
                        time_usage_data[matrix_file] = {}
                    if tol not in memory_usage_data[matrix_file]:
                        memory_usage_data[matrix_file][tol] = {}
                        time_usage_data[matrix_file][tol] = {}

                    memory_usage_data[matrix_file][tol][solver_name] = max(mem_usage)
                    time_usage_data[matrix_file][tol][solver_name] = elapsed_time

                except RuntimeError as e:
                    print(e)

    # Plot memory and time usage
    plot_memory_time_usage(memory_usage_data, time_usage_data)

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
    root_dir = "matrici"
    matrix_files = [os.path.join(root_dir, "spa1.mtx"),
                    os.path.join(root_dir, "spa2.mtx"),
                    os.path.join(root_dir, "vem1.mtx"),
                    os.path.join(root_dir, "vem2.mtx")]
    tolerances = [1e-4, 1e-6, 1e-8, 1e-10]
    max_iter = 20000
    
    run_solvers_on_matrices(matrix_files, tolerances, max_iter)
