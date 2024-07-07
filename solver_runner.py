import IterativeSolvers
import utility
import numpy as np

def run_solvers_on_matrices(matrix_files: list, tolerances: list, max_iter: int):
    for matrix_file in matrix_files:
        A = utility.read_sparse_matrix(matrix_file)
        b = np.ones(A.shape[0])  # Assuming b is a vector of ones for simplicity
        
        for tol in tolerances:
            print(f"Solving {matrix_file} with tolerance {tol}")
            try:
                x_jacobi = IterativeSolvers.jacobi(A, b, tol, max_iter)
                print(f"Jacobi converged to solution with residual {utility.relative_residual_norm(A, x_jacobi, b)}")
            except RuntimeError as e:
                print(e)
            
            try:
                x_gs = IterativeSolvers.gauss_seidel(A, b, tol, max_iter)
                print(f"Gauss-Seidel converged to solution with residual {utility.relative_residual_norm(A, x_gs, b)}")
            except RuntimeError as e:
                print(e)
            
            try:
                x_gd = IterativeSolvers.gradient_descent(A, b, tol, max_iter)
                print(f"Gradient Descent converged to solution with residual {utility.relative_residual_norm(A, x_gd, b)}")
            except RuntimeError as e:
                print(e)
            
            try:
                x_cg = IterativeSolvers.conjugate_gradient(A, b, tol, max_iter)
                print(f"Conjugate Gradient converged to solution with residual {utility.relative_residual_norm(A, x_cg, b)}")
            except RuntimeError as e:
                print(e)
