from Utility import Utility
import numpy as np  # Importing NumPy for numerical operations
import scipy.sparse as sp  # Importing SciPy sparse matrix functions
from scipy.sparse import csc_matrix  # Importing compressed sparse column matrix format

class IterativeSolvers:
    """
    A class containing implementations of various iterative solvers for solving Ax = b.
    """
    
    @staticmethod
    def check_matrix_properties(A: csc_matrix):
        """
        Check if the matrix A is symmetric and positive definite.
        
        Parameters:
        - A (csc_matrix): The sparse matrix to check.
        
        Raises:
        - ValueError: If A is not symmetric or not positive definite.
        """
    
        if not Utility.is_symmetric(A):
            raise ValueError("Matrix A is not symmetric.")
        if not Utility.is_positive_definite(A):
            raise ValueError("Matrix A is not positive definite.")

    @staticmethod
    def jacobi(A: csc_matrix, b: np.ndarray, tol: float, max_iter: int) -> np.ndarray:
        """
        Solves Ax = b using the Jacobi iterative method.
        
        Parameters:
        A (csc_matrix): Sparse matrix A
        b (np.ndarray): Vector b
        tol (float): Tolerance for convergence
        max_iter (int): Maximum number of iterations
        
        Returns:
        np.ndarray: Solution vector x
        """
        IterativeSolvers.check_matrix_properties(A)
        
        x = Utility.init_zero_vector(A.shape[0])  # Initialize solution vector x with zeros
        D = A.diagonal()  # Extract the diagonal elements of A
        R = A - sp.diags(D)  # Calculate the remainder matrix R (A without diagonal)

        for k in range(max_iter):
            # Compute the next iteration vector x_new
            x_new = (b - R @ x) / D
            # Check the convergence criterion
            if Utility.relative_residual_norm(A, x_new, b) < tol:
                return x_new  # Return solution if within tolerance
            x = x_new  # Update x for the next iteration

        raise RuntimeError("Jacobi method did not converge within the maximum number of iterations.")
    
    @staticmethod
    def gauss_seidel(A: csc_matrix, b: np.ndarray, tol: float, max_iter: int) -> np.ndarray:
        """
        Solves Ax = b using the Gauss-Seidel iterative method.
        
        Parameters:
        A (csc_matrix): Sparse matrix A
        b (np.ndarray): Vector b
        tol (float): Tolerance for convergence
        max_iter (int): Maximum number of iterations
        
        Returns:
        np.ndarray: Solution vector x
        """
        IterativeSolvers.check_matrix_properties(A)
        x = Utility.init_zero_vector(A.shape[0])  # Initialize solution vector x with zeros
        
        for k in range(max_iter):
            x_new = np.copy(x)  # Make a copy of x to update values
            for i in range(A.shape[0]):
                # Compute the sum of A[i, :] * x excluding the diagonal element
                sigma = A[i, :].dot(x) - A[i, i] * x[i]
                # Update the ith element of x_new
                x_new[i] = (b[i] - sigma) / A[i, i]
            # Check the convergence criterion
            if Utility.relative_residual_norm(A, x_new, b) < tol:
                return x_new  # Return solution if within tolerance
            x = x_new  # Update x for the next iteration

        raise RuntimeError("Gauss-Seidel method did not converge within the maximum number of iterations.")
    
    @staticmethod
    def gradient_descent(A: csc_matrix, b: np.ndarray, tol: float, max_iter: int) -> np.ndarray:
        """
        Solves Ax = b using the Gradient Descent method.
        
        Parameters:
        A (csc_matrix): Sparse matrix A
        b (np.ndarray): Vector b
        tol (float): Tolerance for convergence
        max_iter (int): Maximum number of iterations
        
        Returns:
        np.ndarray: Solution vector x
        """
        IterativeSolvers.check_matrix_properties(A)
        x = Utility.init_zero_vector(A.shape[0])  # Initialize solution vector x with zeros
        r = b - A @ x  # Compute the initial residual vector r

        for k in range(max_iter):
            # Compute the step size alpha
            alpha = r.dot(r) / (r.dot(A @ r))
            # Update the solution vector x
            x = x + alpha * r
            # Update the residual vector r
            r = b - A @ x
            # Check the convergence criterion
            if np.linalg.norm(r) / np.linalg.norm(b) < tol:
                return x  # Return solution if within tolerance

        raise RuntimeError("Gradient Descent method did not converge within the maximum number of iterations.")
    
    @staticmethod
    def conjugate_gradient(A: csc_matrix, b: np.ndarray, tol: float, max_iter: int) -> np.ndarray:
        """
        Solves Ax = b using the Conjugate Gradient method.
        
        Parameters:
        A (csc_matrix): Sparse matrix A
        b (np.ndarray): Vector b
        tol (float): Tolerance for convergence
        max_iter (int): Maximum number of iterations
        
        Returns:
        np.ndarray: Solution vector x
        """
        IterativeSolvers.check_matrix_properties(A)
        x = Utility.init_zero_vector(A.shape[0])  # Initialize solution vector x with zeros
        r = b - A @ x  # Compute the initial residual vector r
        p = np.copy(r)  # Initialize the conjugate direction vector p
        rs_old = r.dot(r)  # Compute the initial dot product of r

        for k in range(max_iter):
            Ap = A @ p  # Compute A*p
            # Compute the step size alpha
            alpha = rs_old / (p.dot(Ap))
            # Update the solution vector x
            x = x + alpha * p
            # Update the residual vector r
            r = r - alpha * Ap
            rs_new = r.dot(r)  # Compute the new dot product of r
            # Check the convergence criterion
            if np.sqrt(rs_new) / np.linalg.norm(b) < tol:
                return x  # Return solution if within tolerance
            # Update the conjugate direction vector p
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new  # Update rs_old for the next iteration

        raise RuntimeError("Conjugate Gradient method did not converge within the maximum number of iterations.")
