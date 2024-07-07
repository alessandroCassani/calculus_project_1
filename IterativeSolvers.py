import utility
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csc_matrix

class IterativeSolvers:
    
    @staticmethod
    def jacobi(A: csc_matrix, b: np.ndarray, tol: float, max_iter: int) -> np.ndarray:
        """
        Solves Ax = b using the Jacobi iterative method.
        """
        x = utility.init_zero_vector(A.shape[0])
        D = A.diagonal()
        R = A - sp.diags(D)
        
        for k in range(max_iter):
            x_new = (b - R @ x) / D
            if utility.relative_residual_norm(A, x_new, b) < tol:
                return x_new
            x = x_new
        
        raise RuntimeError("Jacobi method did not converge within the maximum number of iterations.")
    
    @staticmethod
    def gauss_seidel(A: csc_matrix, b: np.ndarray, tol: float, max_iter: int) -> np.ndarray:
        """
        Solves Ax = b using the Gauss-Seidel iterative method.
        """
        x = utility.init_zero_vector(A.shape[0])
        
        for k in range(max_iter):
            x_new = np.copy(x)
            for i in range(A.shape[0]):
                sigma = A[i, :].dot(x) - A[i, i] * x[i]
                x_new[i] = (b[i] - sigma) / A[i, i]
            if utility.relative_residual_norm(A, x_new, b) < tol:
                return x_new
            x = x_new
        
        raise RuntimeError("Gauss-Seidel method did not converge within the maximum number of iterations.")
    
    @staticmethod
    def gradient_descent(A: csc_matrix, b: np.ndarray, tol: float, max_iter: int) -> np.ndarray:
        """
        Solves Ax = b using the Gradient Descent method.
        """
        x = utility.init_zero_vector(A.shape[0])
        r = b - A @ x
        
        for k in range(max_iter):
            alpha = r.dot(r) / (r.dot(A @ r))
            x = x + alpha * r
            r = b - A @ x
            if np.linalg.norm(r) / np.linalg.norm(b) < tol:
                return x
        
        raise RuntimeError("Gradient Descent method did not converge within the maximum number of iterations.")
    
    @staticmethod
    def conjugate_gradient(A: csc_matrix, b: np.ndarray, tol: float, max_iter: int) -> np.ndarray:
        """
        Solves Ax = b using the Conjugate Gradient method.
        """
        x = utility.init_zero_vector(A.shape[0])
        r = b - A @ x
        p = np.copy(r)
        rs_old = r.dot(r)
        
        for k in range(max_iter):
            Ap = A @ p
            alpha = rs_old / (p.dot(Ap))
            x = x + alpha * p
            r = r - alpha * Ap
            rs_new = r.dot(r)
            if np.sqrt(rs_new) / np.linalg.norm(b) < tol:
                return x
            p = r + (rs_new / rs_old) * p
            rs_old = rs_new
        
        raise RuntimeError("Conjugate Gradient method did not converge within the maximum number of iterations.")
