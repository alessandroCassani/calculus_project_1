from Utility import Utility
from IterativeSolvers import IterativeSolvers
import os
import numpy as np

def run_solvers_on_matrices(matrix_files: list, tolerances: list, max_iter: int):
    for matrix_file in matrix_files:
        A = Utility.read_sparse_matrix(matrix_file)
        b = np.ones(A.shape[0])  # Assuming b is a vector of ones for simplicity
        
        for tol in tolerances:
            print(f"Solving {matrix_file} with tolerance {tol}")
            try:
                x_jacobi = IterativeSolvers.jacobi(A, b, tol, max_iter)
                print(f"Jacobi converged to solution with residual {Utility.relative_residual_norm(A, x_jacobi, b)}")
            except RuntimeError as e:
                print(e)
            
            try:
                x_gs = IterativeSolvers.gauss_seidel(A, b, tol, max_iter)
                print(f"Gauss-Seidel converged to solution with residual {Utility.relative_residual_norm(A, x_gs, b)}")
            except RuntimeError as e:
                print(e)
            
            try:
                x_gd = IterativeSolvers.gradient_descent(A, b, tol, max_iter)
                print(f"Gradient Descent converged to solution with residual {Utility.relative_residual_norm(A, x_gd, b)}")
            except RuntimeError as e:
                print(e)
            
            try:
                x_cg = IterativeSolvers.conjugate_gradient(A, b, tol, max_iter)
                print(f"Conjugate Gradient converged to solution with residual {Utility.relative_residual_norm(A, x_cg, b)}")
            except RuntimeError as e:
                print(e)
                

if __name__ == "__main__":
    root_dir = "matrici"
    matrix_files = [os.path.join(root_dir, "spa1.mtx"),
                    os.path.join(root_dir, "spa2.mtx"),
                    os.path.join(root_dir, "vem1.mtx"),
                    os.path.join(root_dir, "vem2.mtx")]
    tolerances = [1e-4, 1e-6, 1e-8, 1e-10]
    max_iter = 20000
    
    run_solvers_on_matrices(matrix_files, tolerances, max_iter)
