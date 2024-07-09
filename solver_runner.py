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

if __name__ == "__main__":
    run_matrix_solvers()
